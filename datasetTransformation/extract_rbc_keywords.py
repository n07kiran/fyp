#!/usr/bin/env python3
"""
RBC Morphology Keyword Extractor
================================

A deterministic, regex-based system for extracting Red Blood Cell (RBC) morphology
keywords from hematology peripheral blood smear morphology reports.

ARCHITECTURE OVERVIEW
---------------------

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     MAIN PROCESSING PIPELINE                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
    │  │   Directory  │───▶│     File     │───▶│      Section       │    │
    │  │   Scanner    │    │    Reader    │    │     Extractor      │    │
    │  └──────────────┘    └──────────────┘    └────────────────────┘    │
    │                                                    │               │
    │                                                    ▼               │
    │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
    │  │    Output    │◀───│   Keyword    │◀───│    RBC Keyword     │    │
    │  │    Writer    │    │  Normalizer  │    │     Extractor      │    │
    │  └──────────────┘    └──────────────┘    └────────────────────┘    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

DESIGN DECISIONS
----------------

1. **Section-Based Extraction**: We isolate the RBC morphology section before
   keyword matching to prevent false positives from WBC/platelet/clinical sections.

2. **Regex over ML**: Ensures deterministic, reproducible results without training
   data or model dependencies. Medical terminology is well-defined and finite.

3. **Canonical Keyword Mapping**: Handles spelling variations (e.g., "eliptocytes"
   → "elliptocytes") and synonyms (e.g., "hypochromia" → "hypochromic").

4. **Ordered Output**: Keywords maintain a clinically-meaningful order (primary
   morphology first, then secondary features) for consistent, comparable results.

DATA FLOW
---------

1. Scan dataset for all Morphology_reports directories
2. For each directory:
   a. Enumerate all .txt files (excluding output file)
   b. For each file:
      - Read raw text content
      - Extract RBC section (between RBC header and WBC/platelet header)
      - Apply keyword regex patterns against RBC section only
      - Normalize and deduplicate keywords
      - Store results
   c. Write morphological_keywords.txt to the directory

TARGET KEYWORDS
---------------

Primary (label-defining):
    normocytic, microcytic, macrocytic, target cells, elliptocytes

Secondary (supporting):
    hypochromic, polychromasia, anisocytosis, poikilocytosis,
    anisopoikilocytosis, nucleated rbcs, schistocytes, tear drop cells, spherocytes

USAGE
-----
    Run from project root:
        python extract_rbc_keywords.py

    Or with verbose output:
        python extract_rbc_keywords.py --verbose

OUTPUT FORMAT
-------------
    Each line in morphological_keywords.txt:
        file_name, ["keyword1", "keyword2", ...]

    Example:
        001_a, ["microcytic", "hypochromic", "elliptocytes"]
        002_a, ["normocytic", "hypochromic", "target cells"]

Author: Senior Python Engineer / Medical NLP Practitioner
Version: 2.0.0
Date: 2026-02-05
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# =============================================================================
# CONFIGURATION: KEYWORD DEFINITIONS AND PATTERNS
# =============================================================================

# Canonical keywords in clinically-meaningful order.
# Primary morphology descriptors first, then secondary features.
KEYWORD_ORDER: List[str] = [
    # Primary (label-defining) - describe overall RBC size/shape
    "normocytic",
    "microcytic",
    "macrocytic",
    "target cells",
    "elliptocytes",
    # Secondary (supporting) - additional morphological features
    "hypochromic",
    "polychromasia",
    "anisocytosis",
    "poikilocytosis",
    "anisopoikilocytosis",
    "nucleated rbcs",
    "schistocytes",
    "tear drop cells",
    "spherocytes",
]

# Regex patterns mapped to canonical keywords.
# Each pattern handles common spelling variations and synonyms found in medical reports.
# Patterns are case-insensitive and match word boundaries to prevent partial matches.
KEYWORD_PATTERNS: Dict[str, str] = {
    # === PRIMARY KEYWORDS ===
    
    # Normocytic: normal-sized RBCs
    "normocytic": r"\bnormo[-\s]?cytic\b",
    
    # Microcytic: smaller than normal RBCs (common in iron deficiency, thalassemia)
    # Also matches "microcytosis" - the condition of having microcytic cells
    "microcytic": r"\bmicro[-\s]?cytic\b|\bmicrocytosis\b",
    
    # Macrocytic: larger than normal RBCs (common in B12/folate deficiency)
    "macrocytic": r"\bmacro[-\s]?cytic\b|\bmacrocytosis\b",
    
    # Target cells (codocytes): cells with bullseye appearance
    # Common in thalassemia, liver disease, hemoglobinopathies
    "target cells": r"\btarget\s+cells?\b|\bcodocytes?\b",
    
    # Elliptocytes (ovalocytes): oval/elliptical RBCs
    # Handles common typo "eliptocytes" seen in some reports
    "elliptocytes": (
        r"\bel{1,2}iptocytes?\b|"        # elliptocytes, eliptocytes (typo)
        r"\belliptocyt(?:ic|osis)\b|"    # elliptocytic, elliptocytosis
        r"\bovalocytes?\b"                # synonym: ovalocytes
    ),
    
    # === SECONDARY KEYWORDS ===
    
    # Hypochromic/Hypochromia: pale RBCs with increased central pallor
    # Common in iron deficiency anemia
    "hypochromic": r"\bhypochrom(?:ic|ia)\b",
    
    # Polychromasia: bluish-gray tint indicating young RBCs (reticulocytes)
    # Indicates active red cell production
    "polychromasia": (
        r"\bpolychromasia\b|"
        r"\bpolychromatic\b|"
        r"\bpolychromatophil(?:ic|ia)\b"
    ),
    
    # Anisocytosis: variation in RBC size
    "anisocytosis": r"\banisocytosis\b|\banisocytic\b",
    
    # Poikilocytosis: variation in RBC shape
    "poikilocytosis": r"\bpoikilocytosis\b|\bpoikilocytic\b",
    
    # Anisopoikilocytosis: combined variation in both size AND shape
    # Kept separate from poikilocytosis as it's a distinct finding
    "anisopoikilocytosis": r"\banisopoikilocytosis\b",
    
    # Nucleated RBCs (NRBCs): RBCs with retained nucleus (normally absent)
    # Indicates stress erythropoiesis or bone marrow infiltration
    "nucleated rbcs": (
        r"\bnucleated\s+rbcs?\b|"
        r"\bnrbc(?:s)?\b|"
        r"\bnucleated\s+red\s+(?:blood\s+)?cells?\b"
    ),
    
    # Schistocytes: fragmented RBCs
    # Seen in microangiopathic hemolytic anemia, DIC, mechanical hemolysis
    "schistocytes": (
        r"\bschistocytes?\b|"
        r"\bfragmented\s+rbcs?\b|"
        r"\bfragmented\s+red\s+(?:blood\s+)?cells?\b|"
        r"\bhelmet\s+cells?\b"            # synonym
    ),
    
    # Tear drop cells (dacrocytes): teardrop-shaped RBCs
    # Common in myelofibrosis, myelophthisic anemia
    "tear drop cells": (
        r"\btear\s*drop\s+cells?\b|"
        r"\bteardrop\s+cells?\b|"
        r"\bdacrocytes?\b"                # medical term
    ),
    
    # Spherocytes: sphere-shaped RBCs lacking central pallor
    # Seen in hereditary spherocytosis, autoimmune hemolytic anemia
    "spherocytes": r"\bspherocytes?\b|\bspherocytic\b|\bspherocytosis\b",
}

# Compile patterns once at module load for efficiency
COMPILED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (keyword, re.compile(KEYWORD_PATTERNS[keyword], re.IGNORECASE))
    for keyword in KEYWORD_ORDER
]


# =============================================================================
# SECTION BOUNDARY DETECTION PATTERNS
# =============================================================================

# Pattern to identify RBC morphology section headers
# Matches various formats: "RBC MORPHOLOGY:", "RBC:", "Red Cell Morphology:", etc.
RBC_HEADING_RE = re.compile(
    r"^\s*("
    r"rbc\s*morphology\s*/?\s*peripheral\s+smear|"
    r"rbc\s*morphology|"
    r"rbcs?\s*:|"
    r"red\s+cells?\s*morphology|"
    r"red\s+cells?|"
    r"peripheral\s+smear\s*:|"
    r"erythrocytes?\s*morphology|"
    r"erythrocytes?"
    r")\s*[:\-/]",
    re.IGNORECASE | re.MULTILINE,
)

# Patterns that indicate the END of RBC section (start of other sections)
# These should NOT be searched for keywords
OTHER_HEADING_RE = re.compile(
    r"^\s*("
    r"wbc|wbcs|wbc\s*morphology|white\s+cells?|white\s+blood\s+cells?|leukocytes?|"
    r"platelets?|platelet\s*morphology|plt|thrombocytes?|"
    r"differential|"
    r"comments?|impression|note|notes|"
    r"clinical|history|interpretation|conclusion"
    r")\s*[:\-]",
    re.IGNORECASE | re.MULTILINE,
)


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class FileResult:
    """Result of processing a single morphology report file."""
    file_name: str                          # Filename without extension
    keywords: List[str] = field(default_factory=list)  # Extracted keywords
    rbc_section_found: bool = True          # Whether RBC section was identified
    error: Optional[str] = None             # Error message if processing failed


@dataclass
class DirectoryResult:
    """Result of processing a Morphology_reports directory."""
    directory_path: Path
    file_results: List[FileResult] = field(default_factory=list)
    total_files: int = 0
    files_with_keywords: int = 0
    total_keywords: int = 0
    
    @property
    def success_rate(self) -> float:
        """Percentage of files that had at least one keyword extracted."""
        return (self.files_with_keywords / self.total_files * 100) if self.total_files > 0 else 0.0


@dataclass
class ProcessingStats:
    """Overall statistics from processing the entire dataset."""
    directories_processed: int = 0
    total_files: int = 0
    total_keywords: int = 0
    keyword_frequency: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# CORE FUNCTIONS: SECTION EXTRACTION
# =============================================================================

def extract_rbc_section(text: str) -> Tuple[str, bool]:
    """
    Extract the RBC morphology section from a full report.
    
    This function isolates text between the RBC morphology header and the
    next section header (WBC, platelets, etc.), ensuring we don't capture
    content from other sections that might contain RBC-related terminology.
    
    Strategy:
    1. Search for RBC section header
    2. If found, capture text until the next section header
    3. Also capture any trailing text after all recognized section headers,
       since some reports place RBC findings (e.g., "microcytic hypochromic
       picture of RBCs") after the WBC/Platelet sections rather than inline.
    4. If no header found, fall back to using full text (with warning flag)
    
    Args:
        text: Full text content of the morphology report
        
    Returns:
        Tuple of (extracted_section, section_found_flag)
    """
    lines = text.splitlines()
    start_idx: Optional[int] = None
    
    # Find the start of RBC section
    for i, line in enumerate(lines):
        if RBC_HEADING_RE.search(line):
            start_idx = i
            break
    
    # If no RBC header found, return full text with warning flag
    if start_idx is None:
        return text, False
    
    # Collect RBC section lines (between RBC header and next section header)
    section_lines: List[str] = []
    
    # Include any inline content after the heading (e.g., "RBC: normocytic...")
    heading_line = lines[start_idx]
    inline_match = re.search(r"[:\-/]\s*(.*)$", heading_line)
    if inline_match and inline_match.group(1).strip():
        section_lines.append(inline_match.group(1).strip())
    
    # Collect subsequent lines until we hit another section header
    first_other_idx: Optional[int] = None
    for i, line in enumerate(lines[start_idx + 1:], start=start_idx + 1):
        if OTHER_HEADING_RE.search(line):
            first_other_idx = i
            break
        section_lines.append(line)
    
    # Collect trailing text after all recognized section headers.
    # Some reports place RBC description (e.g. "microcytic hypochromic picture
    # of RBCs") after the WBC/Platelet sections as a trailing paragraph.
    if first_other_idx is not None:
        last_section_end: Optional[int] = None
        for i in range(first_other_idx, len(lines)):
            if OTHER_HEADING_RE.search(lines[i]):
                last_section_end = i
        # Gather all lines after the last recognized section header's block
        if last_section_end is not None:
            # Skip past the last section header to find where its content ends
            trailing_start = last_section_end + 1
            # Advance past lines that belong to the last non-RBC section
            # (lines until a blank line or end of file)
            in_section = True
            for i in range(trailing_start, len(lines)):
                line_stripped = lines[i].strip()
                if in_section:
                    # Still in the last non-RBC section — look for a blank line
                    # that signals the end of that section's content
                    if not line_stripped:
                        in_section = False
                    continue
                # We are past the last non-RBC section. Remaining non-blank
                # text is orphan/trailing content that likely describes RBCs.
                if line_stripped:
                    section_lines.append(lines[i])
    
    # Join and return
    extracted = "\n".join(section_lines) if section_lines else text
    return extracted, True


# =============================================================================
# CORE FUNCTIONS: KEYWORD EXTRACTION
# =============================================================================

def extract_keywords(text: str) -> List[str]:
    """
    Extract RBC morphology keywords from text using regex patterns.
    
    Uses pre-compiled regex patterns to identify medical terminology.
    Patterns handle common spelling variations and synonyms.
    
    Args:
        text: Text to search (typically the extracted RBC section)
        
    Returns:
        List of canonical keywords in predefined order, deduplicated
    """
    text_lower = text.lower()
    found: Set[str] = set()
    
    for keyword, pattern in COMPILED_PATTERNS:
        if pattern.search(text_lower):
            found.add(keyword)
    
    # Return keywords in the predefined clinical order
    return [kw for kw in KEYWORD_ORDER if kw in found]


# =============================================================================
# FILE AND DIRECTORY PROCESSING
# =============================================================================

def process_single_file(file_path: Path) -> FileResult:
    """
    Process a single morphology report file.
    
    Pipeline:
    1. Read file content
    2. Extract RBC section
    3. Extract keywords from RBC section
    4. Return structured result
    
    Args:
        file_path: Path to the .txt morphology report file
        
    Returns:
        FileResult with extracted keywords and metadata
    """
    file_name = file_path.stem
    
    try:
        # Read file with UTF-8 encoding, fallback gracefully for encoding issues
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Extract RBC-specific section
        rbc_section, section_found = extract_rbc_section(content)
        
        # Extract keywords from RBC section only
        keywords = extract_keywords(rbc_section)
        
        return FileResult(
            file_name=file_name,
            keywords=keywords,
            rbc_section_found=section_found
        )
        
    except OSError as e:
        return FileResult(
            file_name=file_name,
            keywords=[],
            error=f"Read error: {e}"
        )


def process_directory(report_dir: Path, verbose: bool = False) -> DirectoryResult:
    """
    Process all morphology report files in a directory.
    
    Args:
        report_dir: Path to Morphology_reports directory
        verbose: If True, print progress for each file
        
    Returns:
        DirectoryResult with all file results and statistics
    """
    result = DirectoryResult(directory_path=report_dir)
    
    # Get all .txt files, excluding our output file
    txt_files = sorted([
        f for f in report_dir.glob("*.txt")
        if f.name != "morphological_keywords.txt"
    ])
    
    result.total_files = len(txt_files)
    
    for file_path in txt_files:
        file_result = process_single_file(file_path)
        result.file_results.append(file_result)
        
        if file_result.keywords:
            result.files_with_keywords += 1
            result.total_keywords += len(file_result.keywords)
        
        if verbose and file_result.error:
            print(f"    Warning: {file_result.file_name}: {file_result.error}")
    
    return result


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def write_output_file(
    directory_result: DirectoryResult,
    verbose: bool = False
) -> Path:
    """
    Write the morphological_keywords.txt output file.
    
    Output format: file_name, ["keyword1", "keyword2", ...]
    
    Args:
        directory_result: Processed results for the directory
        verbose: If True, print confirmation message
        
    Returns:
        Path to the written output file
    """
    output_path = directory_result.directory_path / "morphological_keywords.txt"
    
    lines: List[str] = []
    for file_result in directory_result.file_results:
        # Format keywords as JSON array for proper quoting
        keywords_json = json.dumps(file_result.keywords)
        lines.append(f"{file_result.file_name}, {keywords_json}")
    
    # Write with trailing newline
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8"
    )
    
    if verbose:
        print(f"    Output: {output_path.name} ({len(lines)} entries)")
    
    return output_path


# =============================================================================
# DIRECTORY DISCOVERY
# =============================================================================

def find_morphology_directories(root: Path) -> List[Path]:
    """
    Find all Morphology_reports directories under the dataset root.
    
    Searches recursively for directories named 'Morphology_reports'
    under AneRBC_dataset, including all subdatasets and cohorts.
    
    Args:
        root: Project root path (parent of AneRBC_dataset)
        
    Returns:
        Sorted list of Morphology_reports directory paths
    """
    return sorted(root.glob("AneRBC_dataset/**/Morphology_reports"))


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def calculate_statistics(
    directory_results: List[DirectoryResult]
) -> ProcessingStats:
    """
    Calculate aggregate statistics from all processed directories.
    
    Args:
        directory_results: List of results from all directories
        
    Returns:
        ProcessingStats with aggregate metrics
    """
    stats = ProcessingStats()
    stats.directories_processed = len(directory_results)
    
    # Initialize keyword frequency counter
    for keyword in KEYWORD_ORDER:
        stats.keyword_frequency[keyword] = 0
    
    for dir_result in directory_results:
        stats.total_files += dir_result.total_files
        stats.total_keywords += dir_result.total_keywords
        
        for file_result in dir_result.file_results:
            for keyword in file_result.keywords:
                stats.keyword_frequency[keyword] += 1
    
    return stats


def print_summary(stats: ProcessingStats, verbose: bool = False) -> None:
    """
    Print processing summary to stdout.
    
    Args:
        stats: Aggregate statistics from processing
        verbose: If True, print detailed keyword frequency breakdown
    """
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Directories processed:    {stats.directories_processed}")
    print(f"Report files processed:   {stats.total_files}")
    print(f"Total keywords extracted: {stats.total_keywords}")
    
    if stats.total_files > 0:
        avg = stats.total_keywords / stats.total_files
        print(f"Average keywords/file:    {avg:.2f}")
    
    if verbose and stats.keyword_frequency:
        print("\nKeyword Frequency:")
        print("-" * 40)
        for keyword in KEYWORD_ORDER:
            count = stats.keyword_frequency.get(keyword, 0)
            pct = (count / stats.total_files * 100) if stats.total_files > 0 else 0
            print(f"  {keyword:<22} {count:>5}  ({pct:5.1f}%)")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Main entry point for the RBC Morphology Keyword Extractor.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract RBC morphology keywords from hematology reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_rbc_keywords.py
    python extract_rbc_keywords.py --verbose
    python extract_rbc_keywords.py --root /path/to/project
        """
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress and statistics"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    args = parser.parse_args()
    
    verbose: bool = args.verbose
    root: Path = args.root
    
    if verbose:
        print("=" * 60)
        print("RBC Morphology Keyword Extractor")
        print("=" * 60)
        print(f"Project root: {root}")
    
    # Find all Morphology_reports directories
    report_dirs = find_morphology_directories(root)
    
    if not report_dirs:
        print("Error: No Morphology_reports directories found under AneRBC_dataset/",
              file=sys.stderr)
        print("Please ensure the script is run from the project root.", file=sys.stderr)
        return 1
    
    if verbose:
        print(f"Found {len(report_dirs)} Morphology_reports directories\n")
    
    # Process each directory
    all_results: List[DirectoryResult] = []
    
    for report_dir in report_dirs:
        # Display relative path for clarity
        try:
            rel_path = report_dir.relative_to(root / "AneRBC_dataset")
        except ValueError:
            rel_path = report_dir
        
        if verbose:
            print(f"Processing: {rel_path}")
        
        # Process directory
        dir_result = process_directory(report_dir, verbose=verbose)
        all_results.append(dir_result)
        
        # Write output file
        write_output_file(dir_result, verbose=verbose)
        
        if verbose:
            print(f"    Files: {dir_result.total_files}, "
                  f"Keywords: {dir_result.total_keywords}, "
                  f"Success rate: {dir_result.success_rate:.1f}%\n")
    
    # Calculate and print summary
    stats = calculate_statistics(all_results)
    
    if verbose:
        print_summary(stats, verbose=True)
    else:
        print(f"Processed {stats.total_files} files across "
              f"{stats.directories_processed} directories. "
              f"Extracted {stats.total_keywords} keywords.")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
