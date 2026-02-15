#!/usr/bin/env python3
"""
Build a centralized CSV of morphology keywords.

- Reads all morphological_keywords.txt files under AneRBC_dataset/**/Morphology_reports
- Adds cohort information (anemic_individuals | healthy_individuals)
- Deduplicates entries across AneRBC-I and AneRBC-II (they are identical)
- Writes all_morphology_keywords.csv with columns: cohort,file_name,keywords

Usage:
    python build_morphology_keywords_csv.py
    python build_morphology_keywords_csv.py --output my_keywords.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


def detect_cohort(path: Path) -> str:
    """Infer cohort from the path."""
    parts = {p.lower() for p in path.parts}
    if "anemic_individuals" in parts:
        return "anemic_individuals"
    if "healthy_individuals" in parts:
        return "healthy_individuals"
    return "unknown"


def parse_keywords_line(line: str) -> Tuple[str, str]:
    """Parse a single line: file_name, ["kw1", ...] -> (file_name, json_string)."""
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    try:
        file_name, kw_part = line.split(",", 1)
    except ValueError:
        raise ValueError(f"cannot split line: {line!r}")
    file_name = file_name.strip()
    kw_part = kw_part.strip()
    # Ensure the keywords portion is valid JSON array
    try:
        json.loads(kw_part)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid keywords JSON for {file_name}: {e}")
    return file_name, kw_part


def collect_keywords(dataset_root: Path) -> Dict[Tuple[str, str], str]:
    """Collect keywords from all morphology_keywords.txt files.

    Returns a mapping (cohort, file_name) -> keywords_json_string
    """
    results: Dict[Tuple[str, str], str] = {}
    for keywords_file in dataset_root.rglob("morphological_keywords.txt"):
        cohort = detect_cohort(keywords_file)
        try:
            text = keywords_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                file_name, keywords_json = parse_keywords_line(line)
            except ValueError:
                continue
            key = (cohort, file_name)
            # Deduplicate: first occurrence wins (AneRBC-I vs AneRBC-II are identical)
            if key not in results:
                results[key] = keywords_json
    return results


def write_csv(output_path: Path, data: Dict[Tuple[str, str], str]) -> None:
    """Write consolidated CSV with header: cohort,file_name,keywords"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort", "file_name", "keywords"])
        for (cohort, file_name) in sorted(data.keys()):
            writer.writerow([cohort, file_name, data[(cohort, file_name)]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build centralized morphology keywords CSV")
    parser.add_argument("--output", type=Path, default=Path("all_morphology_keywords.csv"), help="Output CSV path")
    args = parser.parse_args()

    project_root = Path.cwd()
    dataset_root = project_root / "AneRBC_dataset"
    if not dataset_root.exists():
        print(f"Dataset not found at {dataset_root}")
        return 1

    data = collect_keywords(dataset_root)
    write_csv(args.output, data)
    print(f"Wrote {len(data)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
