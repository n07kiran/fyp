#!/usr/bin/env python3
"""
Full CBC Clinical Data Extractor for AneRBC-II
==============================================

Extracts complete blood count (CBC) parameters from AneRBC-II CBC reports
and classifies anemia based on MCV values.

Extracts 10 core CBC parameters:
    WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, MPV, RDW-CV

For each parameter, also extracts abnormal flag (marked with * in reports).

ANEMIA CLASSIFICATION
---------------------
    Class 0: Healthy individuals
    Class 1: Anemic with MCV < 80 (Microcytic)
    Class 2: Anemic with 80 <= MCV <= 100 (Normocytic)
    Class 3: Anemic with MCV > 100 (Macrocytic)
    Class -1: Anemic with missing MCV

USAGE
-----
    Run from project root:
        python3 datasetTransformation/extract_full_cbc_data.py

OUTPUT
------
    Code/Fusion_Model/transformedDataset/AneRBC_II_Full_Clinical_Data.csv
"""

import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# =============================================================================
# FIELD NAME MAPPINGS
# =============================================================================

# Map variations in field names to standardized names
FIELD_MAPPINGS = {
    "WBC": "WBC",
    "RBC": "RBC",
    "HGB": "HGB",
    "Hemoglobin": "HGB",
    "HCT": "HCT",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "PLT": "PLT",
    "MPV": "MPV",
    "RDW-CV": "RDW_CV",
    "RDW---CV": "RDW_CV",
    "RDW": "RDW_CV",
}

# Core CBC parameters to extract (in order)
CBC_PARAMETERS = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "MPV", "RDW_CV"]


# =============================================================================
# CBC PARSING
# =============================================================================

def parse_cbc_value(result_text: str) -> Tuple[Optional[float], bool]:
    """
    Parse a CBC result value, extracting the numeric value and abnormal flag.
    
    Args:
        result_text: The result text (e.g., "* 10.1 g/dL" or "8.49 x10.e 3/µl")
        
    Returns:
        Tuple of (value as float, abnormal flag as bool)
        Returns (None, False) if parsing fails
    """
    # Check for abnormal flag (asterisk prefix)
    abnormal = result_text.strip().startswith("*")
    
    # Remove asterisk and whitespace
    cleaned = result_text.replace("*", "").strip()
    
    # Extract numeric value using regex
    # Match patterns like: "10.1", "8.49 x10.e 3/µl", etc.
    match = re.search(r"(\d+\.?\d*)", cleaned)
    
    if match:
        try:
            value = float(match.group(1))
            return value, abnormal
        except ValueError:
            return None, abnormal
    
    return None, abnormal


def parse_cbc_report(file_path: Path) -> Dict[str, any]:
    """
    Parse a single CBC report file and extract all parameters.
    
    Args:
        file_path: Path to the CBC report .txt file
        
    Returns:
        Dictionary with extracted CBC parameters and abnormal flags
    """
    result = {param: None for param in CBC_PARAMETERS}
    abnormal_flags = {f"{param}_abnormal": False for param in CBC_PARAMETERS}
    
    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Split into lines and skip header
        lines = content.strip().split("\n")
        for line in lines[1:]:  # Skip header line
            # Parse as CSV (split on comma)
            parts = [p.strip() for p in line.split(",")]
            
            if len(parts) < 2:
                continue
            
            test_name = parts[0].strip()
            result_value = parts[1].strip() if len(parts) > 1 else ""
            
            # Map test name to standardized field name
            field_name = FIELD_MAPPINGS.get(test_name)
            
            if field_name and field_name in CBC_PARAMETERS:
                value, abnormal = parse_cbc_value(result_value)
                result[field_name] = value
                abnormal_flags[f"{field_name}_abnormal"] = abnormal
    
    except Exception as e:
        print(f"Warning: Error parsing {file_path.name}: {e}", file=sys.stderr)
    
    # Merge result and abnormal flags
    result.update(abnormal_flags)
    
    return result


# =============================================================================
# ANEMIA CLASSIFICATION
# =============================================================================

def classify_anemia(cohort: str, mcv_value: Optional[float]) -> int:
    """
    Classify anemia category based on cohort and MCV value.
    
    Args:
        cohort: Either "Anemic" or "Healthy"
        mcv_value: MCV value in fL (or None if missing)
        
    Returns:
        Anemia category (0, 1, 2, 3, or -1)
    """
    if cohort == "Healthy":
        return 0
    
    # Anemic individuals
    if mcv_value is None:
        return -1  # Missing MCV
    
    if mcv_value < 80:
        return 1  # Microcytic
    elif mcv_value <= 100:
        return 2  # Normocytic
    else:
        return 3  # Macrocytic


# =============================================================================
# DIRECTORY PROCESSING
# =============================================================================

def find_cbc_directories(root: Path) -> List[Tuple[Path, str]]:
    """
    Find all CBC_reports directories under AneRBC-II.
    
    Args:
        root: Project root path
        
    Returns:
        List of tuples (directory_path, cohort) where cohort is "Anemic" or "Healthy"
    """
    directories = []
    
    # Find AneRBC-II directories
    anerbc_ii_path = root / "AneRBC_dataset" / "AneRBC-II"
    
    if not anerbc_ii_path.exists():
        return directories
    
    # Find Anemic_individuals/CBC_reports
    anemic_path = anerbc_ii_path / "Anemic_individuals" / "CBC_reports"
    if anemic_path.exists():
        directories.append((anemic_path, "Anemic"))
    
    # Find Healthy_individuals/CBC_reports
    healthy_path = anerbc_ii_path / "Healthy_individuals" / "CBC_reports"
    if healthy_path.exists():
        directories.append((healthy_path, "Healthy"))
    
    return directories


def process_cbc_directory(directory: Path, cohort: str) -> List[Dict[str, any]]:
    """
    Process all CBC report files in a directory.
    
    Args:
        directory: Path to CBC_reports directory
        cohort: "Anemic" or "Healthy"
        
    Returns:
        List of dictionaries containing extracted data for each file
    """
    results = []
    
    # Find all .txt files
    txt_files = sorted(directory.glob("*.txt"))
    
    for txt_file in txt_files:
        # Parse CBC report
        cbc_data = parse_cbc_report(txt_file)
        
        # Classify anemia
        mcv_value = cbc_data.get("MCV")
        anemia_category = classify_anemia(cohort, mcv_value)
        
        # Build result row
        row = {
            "File_Name": txt_file.name,
            "Cohort": cohort,
            **cbc_data,
            "Anemia_Category": anemia_category
        }
        
        results.append(row)
    
    return results


# =============================================================================
# OUTPUT WRITING
# =============================================================================

def write_csv_output(data: List[Dict[str, any]], output_path: Path) -> None:
    """
    Write extracted CBC data to CSV file.
    
    Args:
        data: List of dictionaries containing CBC data
        output_path: Path to output CSV file
    """
    if not data:
        print("Warning: No data to write", file=sys.stderr)
        return
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define column order
    columns = ["File_Name", "Cohort"]
    
    # Add CBC parameters with alternating value and abnormal flag
    for param in CBC_PARAMETERS:
        columns.append(param)
        columns.append(f"{param}_abnormal")
    
    columns.append("Anemia_Category")
    
    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for row in data:
            # Convert boolean abnormal flags to int (0/1) for CSV
            csv_row = {}
            for key, value in row.items():
                if key.endswith("_abnormal"):
                    csv_row[key] = 1 if value else 0
                elif value is None and key not in ["File_Name", "Cohort", "Anemia_Category"]:
                    csv_row[key] = ""  # Empty string for missing numeric values
                else:
                    csv_row[key] = value
            
            writer.writerow(csv_row)
    
    print(f"Output written to: {output_path}")


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def print_statistics(data: List[Dict[str, any]]) -> None:
    """
    Print summary statistics about processed data.
    
    Args:
        data: List of dictionaries containing CBC data
    """
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    # Count by cohort
    anemic_count = sum(1 for row in data if row["Cohort"] == "Anemic")
    healthy_count = sum(1 for row in data if row["Cohort"] == "Healthy")
    
    print(f"Total files processed:    {len(data)}")
    print(f"  Anemic individuals:     {anemic_count}")
    print(f"  Healthy individuals:    {healthy_count}")
    
    # Count by anemia category
    print("\nAnemia Category Distribution:")
    print("-" * 40)
    category_names = {
        0: "Class 0 (Healthy)",
        1: "Class 1 (Microcytic, MCV < 80)",
        2: "Class 2 (Normocytic, 80 <= MCV <= 100)",
        3: "Class 3 (Macrocytic, MCV > 100)",
        -1: "Class -1 (Missing MCV)"
    }
    
    for category in [0, 1, 2, 3, -1]:
        count = sum(1 for row in data if row["Anemia_Category"] == category)
        if count > 0:
            print(f"  {category_names[category]:<40} {count:>5}")
    
    # Count files with missing critical fields
    print("\nMissing Critical Fields:")
    print("-" * 40)
    critical_fields = ["MCV", "HGB", "RBC"]
    for field in critical_fields:
        missing_count = sum(1 for row in data if row.get(field) is None)
        if missing_count > 0:
            pct = (missing_count / len(data) * 100)
            print(f"  {field:<10} {missing_count:>5} files ({pct:5.1f}%)")
    
    print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Main entry point for CBC data extraction.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Determine project root (parent of datasetTransformation directory)
    script_path = Path(__file__).resolve()
    root = script_path.parent.parent
    
    print("=" * 60)
    print("Full CBC Clinical Data Extractor for AneRBC-II")
    print("=" * 60)
    print(f"Project root: {root}\n")
    
    # Find CBC directories
    cbc_dirs = find_cbc_directories(root)
    
    if not cbc_dirs:
        print("Error: No CBC_reports directories found under AneRBC-II/", file=sys.stderr)
        print("Please ensure the script is run from the project root.", file=sys.stderr)
        return 1
    
    print(f"Found {len(cbc_dirs)} CBC_reports directories\n")
    
    # Process all directories
    all_data = []
    
    for directory, cohort in cbc_dirs:
        rel_path = directory.relative_to(root / "AneRBC_dataset")
        print(f"Processing: {rel_path} ({cohort})")
        
        data = process_cbc_directory(directory, cohort)
        all_data.extend(data)
        
        print(f"  Processed {len(data)} files\n")
    
    # Write output
    output_path = root / "Code" / "Fusion_Model" / "transformedDataset" / "AneRBC_II_Full_Clinical_Data.csv"
    write_csv_output(all_data, output_path)
    
    # Print statistics
    print_statistics(all_data)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
