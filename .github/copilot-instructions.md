# Guidance for AI coding agents — AneRBC project

This file contains concise, actionable guidance to help AI agents be immediately productive in this repository.

Overview
- The repo is a Python-based data processing / dataset prep project for RBC morphology (AneRBC dataset).
  - notebooks and environment: `Code/main.ipynb` and `Code/requirements.txt`.

Key conventions and patterns (do not assume defaults)
- File layout is data-first: scripts expect dataset folders relative to repo root. Use absolute paths only when necessary.
- CBC report filenames follow an index + suffix pattern (e.g. `001_a.txt`) — suffix `_a` indicates anemic; mirror patterns for healthy individuals.
- Keywords are CSV-based; avoid introducing alternate storage formats without updating the three main scripts.
- There are no unit tests or CI in the repo; changes to parsing logic should be verified by running scripts on a small subset of `AneRBC_dataset/` and inspecting outputs.

Developer workflows and commands
- Create environment and install deps (work from repo root):
  source .venv/bin/activate
  pip install -r Code/requirements.txt

Integration points & external dependencies
- Primary dependency: Python packages listed in `Code/requirements.txt` (used by notebooks and scripts).
- Data: the `AneRBC_dataset/` directory is authoritative; scripts read raw `.txt` CBC reports and existing morphology reports.
- Notebooks (in `Code/`) are used for interactive data exploration and training; prefer reproducing key steps from the notebook in scripts for automation.

Editing guidance for AI agents
- When modifying parsing logic, make small, reversible changes and add a short runnable example script or snippet demonstrating the change.
- Avoid large refactors: this repo uses simple script-based orchestration rather than a service architecture.

Examples from the codebase
- Input example path: `AneRBC_dataset/AneRBC-I/Anemic_individuals/CBC_reports/001_a.txt` — use this pattern for unit/manual tests.
- Config-like file: `all_morphology_keywords.csv` — treat it as the single source of truth for morphology terms.

If anything is ambiguous
- Ask for a sample CBC report or a small subset of `AneRBC_dataset/` to validate parsing changes.
- Request clarification on desired output CSV schema before altering `extract_rbc_keywords.py` outputs.

Next steps / How to iterate
- After implementing changes, run the three main scripts on a small subset and share the produced CSVs so we can verify correctness.
