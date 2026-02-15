#!/usr/bin/env python3
"""
Morphology Reports Comparison Script
=====================================

Compares corresponding morphology report files between AneRBC-I and AneRBC-II
datasets to identify if they are identical or have differences.

This script performs:
1. File-by-file content comparison
2. Identifies matching and differing files
3. Reports files present in one dataset but not the other
4. Provides detailed statistics and optionally shows differences

Usage:
    python compare_morphology_reports.py
    python compare_morphology_reports.py --verbose
    python compare_morphology_reports.py --show-diff

Author: Comparison Analysis Script
Date: 2026-02-05
"""

import argparse
import difflib
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing a single file pair."""
    filename: str
    identical: bool
    file1_exists: bool = True
    file2_exists: bool = True
    file1_hash: Optional[str] = None
    file2_hash: Optional[str] = None
    file1_size: int = 0
    file2_size: int = 0
    diff_lines: List[str] = field(default_factory=list)


@dataclass
class DirectoryComparisonResult:
    """Result of comparing two directories."""
    dir1_name: str
    dir2_name: str
    total_files: int = 0
    identical_files: int = 0
    different_files: int = 0
    only_in_dir1: List[str] = field(default_factory=list)
    only_in_dir2: List[str] = field(default_factory=list)
    file_results: List[ComparisonResult] = field(default_factory=list)
    
    @property
    def identical_percentage(self) -> float:
        """Calculate percentage of identical files."""
        if self.total_files == 0:
            return 0.0
        return (self.identical_files / self.total_files) * 100


# =============================================================================
# CORE COMPARISON FUNCTIONS
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for quick comparison."""
    hash_md5 = hashlib.md5()
    try:
        content = file_path.read_bytes()
        hash_md5.update(content)
        return hash_md5.hexdigest()
    except OSError:
        return ""


def compare_files(file1: Path, file2: Path, compute_diff: bool = False) -> ComparisonResult:
    """
    Compare two files and return detailed comparison result.
    
    Args:
        file1: Path to first file
        file2: Path to second file
        compute_diff: If True, compute line-by-line differences
        
    Returns:
        ComparisonResult with comparison details
    """
    filename = file1.name
    result = ComparisonResult(filename=filename, identical=False)
    
    # Check file existence
    result.file1_exists = file1.exists()
    result.file2_exists = file2.exists()
    
    if not result.file1_exists or not result.file2_exists:
        return result
    
    # Get file sizes
    result.file1_size = file1.stat().st_size
    result.file2_size = file2.stat().st_size
    
    # Quick size comparison
    if result.file1_size != result.file2_size:
        result.identical = False
        if compute_diff:
            result.diff_lines = _compute_diff(file1, file2)
        return result
    
    # Compute hashes for content comparison
    result.file1_hash = compute_file_hash(file1)
    result.file2_hash = compute_file_hash(file2)
    
    result.identical = (result.file1_hash == result.file2_hash)
    
    if not result.identical and compute_diff:
        result.diff_lines = _compute_diff(file1, file2)
    
    return result


def _compute_diff(file1: Path, file2: Path) -> List[str]:
    """Compute unified diff between two files."""
    try:
        content1 = file1.read_text(encoding='utf-8', errors='ignore').splitlines()
        content2 = file2.read_text(encoding='utf-8', errors='ignore').splitlines()
        
        diff = difflib.unified_diff(
            content1, content2,
            fromfile=str(file1.name),
            tofile=str(file2.name),
            lineterm=''
        )
        return list(diff)
    except OSError:
        return ["Error reading files for diff"]


def compare_directories(
    dir1: Path,
    dir2: Path,
    file_pattern: str = "*.txt",
    exclude_files: Set[str] = None,
    compute_diff: bool = False
) -> DirectoryComparisonResult:
    """
    Compare all matching files between two directories.
    
    Args:
        dir1: First directory path
        dir2: Second directory path
        file_pattern: Glob pattern for files to compare
        exclude_files: Set of filenames to exclude
        compute_diff: If True, compute line-by-line differences
        
    Returns:
        DirectoryComparisonResult with all comparison details
    """
    if exclude_files is None:
        exclude_files = {"morphological_keywords.txt"}
    
    result = DirectoryComparisonResult(
        dir1_name=dir1.name,
        dir2_name=dir2.name
    )
    
    # Get file sets
    files1 = {f.name for f in dir1.glob(file_pattern) if f.name not in exclude_files}
    files2 = {f.name for f in dir2.glob(file_pattern) if f.name not in exclude_files}
    
    # Files only in one directory
    result.only_in_dir1 = sorted(files1 - files2)
    result.only_in_dir2 = sorted(files2 - files1)
    
    # Common files to compare
    common_files = sorted(files1 & files2)
    result.total_files = len(common_files)
    
    # Compare each common file
    for filename in common_files:
        file1 = dir1 / filename
        file2 = dir2 / filename
        
        file_result = compare_files(file1, file2, compute_diff=compute_diff)
        result.file_results.append(file_result)
        
        if file_result.identical:
            result.identical_files += 1
        else:
            result.different_files += 1
    
    return result


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def print_comparison_summary(result: DirectoryComparisonResult, verbose: bool = False) -> None:
    """Print summary of directory comparison."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {result.dir1_name} vs {result.dir2_name}")
    print(f"{'='*60}")
    
    print(f"\nTotal files compared:  {result.total_files}")
    print(f"Identical files:       {result.identical_files} ({result.identical_percentage:.1f}%)")
    print(f"Different files:       {result.different_files}")
    
    if result.only_in_dir1:
        print(f"\nFiles only in {result.dir1_name}: {len(result.only_in_dir1)}")
        if verbose:
            for f in result.only_in_dir1[:10]:  # Show first 10
                print(f"  - {f}")
            if len(result.only_in_dir1) > 10:
                print(f"  ... and {len(result.only_in_dir1) - 10} more")
    
    if result.only_in_dir2:
        print(f"\nFiles only in {result.dir2_name}: {len(result.only_in_dir2)}")
        if verbose:
            for f in result.only_in_dir2[:10]:
                print(f"  - {f}")
            if len(result.only_in_dir2) > 10:
                print(f"  ... and {len(result.only_in_dir2) - 10} more")
    
    # List different files if any
    if result.different_files > 0:
        print(f"\nDifferent files:")
        different = [r for r in result.file_results if not r.identical]
        for fr in different[:20]:  # Show first 20
            size_diff = fr.file2_size - fr.file1_size
            print(f"  - {fr.filename} (size diff: {size_diff:+d} bytes)")
        if len(different) > 20:
            print(f"  ... and {len(different) - 20} more")


def print_detailed_diff(result: DirectoryComparisonResult, max_files: int = 5) -> None:
    """Print detailed diff for different files."""
    different = [r for r in result.file_results if not r.identical and r.diff_lines]
    
    if not different:
        return
    
    print(f"\n{'='*60}")
    print("DETAILED DIFFERENCES")
    print(f"{'='*60}")
    
    for i, fr in enumerate(different[:max_files]):
        print(f"\n--- {fr.filename} ---")
        for line in fr.diff_lines[:50]:  # Limit diff lines shown
            print(line)
        if len(fr.diff_lines) > 50:
            print(f"... ({len(fr.diff_lines) - 50} more lines)")


# =============================================================================
# MAIN COMPARISON LOGIC
# =============================================================================

def compare_anerbc_datasets(
    root: Path,
    verbose: bool = False,
    show_diff: bool = False
) -> Dict[str, DirectoryComparisonResult]:
    """
    Compare all corresponding Morphology_reports directories between
    AneRBC-I and AneRBC-II datasets.
    
    Args:
        root: Project root directory
        verbose: Print verbose output
        show_diff: Show detailed file differences
        
    Returns:
        Dictionary mapping comparison names to results
    """
    dataset_root = root / "AneRBC_dataset"
    
    # Define directory pairs to compare
    comparisons = [
        (
            "Anemic_individuals",
            dataset_root / "AneRBC-I" / "Anemic_individuals" / "Morphology_reports",
            dataset_root / "AneRBC-II" / "Anemic_individuals" / "Morphology_reports"
        ),
        (
            "Healthy_individuals", 
            dataset_root / "AneRBC-I" / "Healthy_individuals" / "Morphology_reports",
            dataset_root / "AneRBC-II" / "Healthy_individuals" / "Morphology_reports"
        ),
    ]
    
    results = {}
    all_identical = True
    
    for name, dir1, dir2 in comparisons:
        if not dir1.exists():
            print(f"Warning: {dir1} does not exist")
            continue
        if not dir2.exists():
            print(f"Warning: {dir2} does not exist")
            continue
        
        print(f"\nComparing {name}...")
        result = compare_directories(dir1, dir2, compute_diff=show_diff)
        results[name] = result
        
        print_comparison_summary(result, verbose=verbose)
        
        if show_diff and result.different_files > 0:
            print_detailed_diff(result)
        
        if result.different_files > 0 or result.only_in_dir1 or result.only_in_dir2:
            all_identical = False
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_compared = sum(r.total_files for r in results.values())
    total_identical = sum(r.identical_files for r in results.values())
    total_different = sum(r.different_files for r in results.values())
    
    print(f"\nTotal files compared across all directories: {total_compared}")
    print(f"Total identical: {total_identical}")
    print(f"Total different: {total_different}")
    
    if all_identical and total_compared > 0:
        print("\n✅ RESULT: AneRBC-I and AneRBC-II Morphology_reports are IDENTICAL")
    elif total_different > 0:
        print(f"\n⚠️  RESULT: Found {total_different} DIFFERENT files between datasets")
    else:
        print("\n❓ RESULT: Could not complete comparison")
    
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Morphology_reports between AneRBC-I and AneRBC-II datasets"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output including file lists"
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Show detailed line-by-line differences for differing files"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("AneRBC Dataset Morphology Reports Comparison")
    print("="*60)
    print(f"Project root: {args.root}")
    
    results = compare_anerbc_datasets(
        root=args.root,
        verbose=args.verbose,
        show_diff=args.show_diff
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
