# Guidance for AI coding agents — AneRBC project

This file contains concise, actionable guidance to help AI agents be immediately productive in this repository.

Overview
- The repo is a Python-based data processing / dataset prep project for RBC morphology (AneRBC dataset).
- Major components:
  - data scripts at repo root: `build_morphology_keywords_csv.py`, `extract_rbc_keywords.py`, `compare_morphology_reports.py`.
  - master keywords CSV: `all_morphology_keywords.csv` (generated/consumed by scripts).
  - dataset layout: `AneRBC_dataset/` with subfolders `AneRBC-I/` and `AneRBC-II/`, each containing `Anemic_individuals/` and `Healthy_individuals/` and subfolders like `CBC_reports/`, `Morphology_reports/`, `Original_images/`, `RGB_segmented/`, `Binary_segmented/`.
  - notebooks and environment: `Code/main.ipynb` and `Code/requirements.txt`.

Big picture & data flow
- `build_morphology_keywords_csv.py` aggregates or normalizes morphology keywords into `all_morphology_keywords.csv`.
- `extract_rbc_keywords.py` reads `all_morphology_keywords.csv` and the `CBC_reports/` text files (e.g. `AneRBC_dataset/AneRBC-I/Anemic_individuals/CBC_reports/001_a.txt`) to extract mentions and produce downstream CSVs/reports.
- `compare_morphology_reports.py` compares generated keyword summaries with existing `Morphology_reports/` to measure coverage/consistency.

Key conventions and patterns (do not assume defaults)
- File layout is data-first: scripts expect dataset folders relative to repo root. Use absolute paths only when necessary.
- CBC report filenames follow an index + suffix pattern (e.g. `001_a.txt`) — suffix `_a` indicates anemic; mirror patterns for healthy individuals.
- Keywords are CSV-based; avoid introducing alternate storage formats without updating the three main scripts.
- There are no unit tests or CI in the repo; changes to parsing logic should be verified by running scripts on a small subset of `AneRBC_dataset/` and inspecting outputs.

Developer workflows and commands
- Create environment and install deps (work from repo root):

  python -m venv .venv
  source .venv/bin/activate
  pip install -r Code/requirements.txt

- Run the keyword builder (writes/updates `all_morphology_keywords.csv`):

  python build_morphology_keywords_csv.py

- Example: extract keywords from a dataset split (adjust paths if needed):

  python extract_rbc_keywords.py --data-dir AneRBC_dataset/AneRBC-I/Anemic_individuals --keywords all_morphology_keywords.csv

- Compare outputs to morphology reports:

  python compare_morphology_reports.py --predictions out/keywords_summary.csv --ground-truth AneRBC_dataset/AneRBC-I/Anemic_individuals/Morphology_reports

Integration points & external dependencies
- Primary dependency: Python packages listed in `Code/requirements.txt` (used by notebooks and scripts).
- Data: the `AneRBC_dataset/` directory is authoritative; scripts read raw `.txt` CBC reports and existing morphology reports.
- Notebooks (in `Code/`) are used for interactive data exploration and training; prefer reproducing key steps from the notebook in scripts for automation.

Editing guidance for AI agents
- When modifying parsing logic, make small, reversible changes and add a short runnable example script or snippet demonstrating the change.
- Preserve CSV headers and column ordering used by downstream scripts; inspect `all_morphology_keywords.csv` before changing columns.
- Avoid large refactors: this repo uses simple script-based orchestration rather than a service architecture.

Examples from the codebase
- Input example path: `AneRBC_dataset/AneRBC-I/Anemic_individuals/CBC_reports/001_a.txt` — use this pattern for unit/manual tests.
- Config-like file: `all_morphology_keywords.csv` — treat it as the single source of truth for morphology terms.

If anything is ambiguous
- Ask for a sample CBC report or a small subset of `AneRBC_dataset/` to validate parsing changes.
- Request clarification on desired output CSV schema before altering `extract_rbc_keywords.py` outputs.

Next steps / How to iterate
- After implementing changes, run the three main scripts on a small subset and share the produced CSVs so we can verify correctness.
- If you want, I can also add a tiny `scripts/validate_sample.sh` runner and a short README describing quick checks.

5 classes 
1. Normocytes (Normal RBCs)
2. Microcytes (Small RBCs)
3. Macrocytes (Large RBCs)
4. Elliptocytes (Oval or elongated RBCs)
5. Target Cells (Target RBCs)
— End of guidance —
