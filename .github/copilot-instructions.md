AneRBC project — quick developer guide

Overview

Quick environment setup (macOS, zsh)

```bash
source .venv/bin/activate

- To confirm activation, run `which python` — path should include `.venv`.
# AneRBC project — quick developer guide

## Overview

- **AneRBC_dataset:** contains the original image dataset and reports used for training and analysis.
- **Code:** contains project code, notebooks and utilities. Key folder: `Code/Fusion_Model` (dataset prep, training, evaluation, utils).
- **research papers:** referenced papers and the dataset description (see `AneRBC_Image_Dataset_Description.pdf`).
 - **AneRBC_dataset:** contains the original image dataset and reports used for training and analysis. See [AneRBC_dataset](AneRBC_dataset/).
 - **Code:** contains project code, notebooks and utilities. Key folder: [Code/Fusion_Model](Code/Fusion_Model/) (dataset prep, training, evaluation, utils). See [Code](Code/) for top-level scripts and [Code/requirements.txt](Code/requirements.txt) for dependencies.
 - **research papers:** referenced papers and the dataset description. See [research papers](research%20papers/).

## Quick environment setup (macOS, zsh)

- Create and activate a virtualenv (recommended name: `.venv`):

```bash
source .venv/bin/activate
```

- To confirm activation, run `which python` — path should include `.venv`.

## Notes & conventions

- Notebooks are the canonical, reproducible workflow; prefer notebooks for step-by-step experiments and the scripts for automation.
- Keep virtualenvs out of version control (add to `.gitignore`).
- Prefer simple, straightforward code; avoid complex patterns and abstractions.
Notes & conventions
- Notebooks are the canonical, reproducible workflow; prefer notebooks for step-by-step experiments and the scripts for automation.
- Keep virtualenvs out of version control (add to `.gitignore`).
- I want the code to be very simple and straightforward, so - I avoid complex patterns and abstractions.
- Always create a respective .md file to explain each file changes you make, Also add simple markdown cells to the parent notebook to explain the code you write in the respective code cell.
- There is a folder named, `Viva_questions1`, If your changes can be of a potential question, You need analyze you changes and the viva files and add or update the viva questions and answers if needed. 
- Use .ipynb notebooks, avoid .py scripts unless necessary for automation or utility functions.
- 4 pretrained modeles : VGG16, Resnet152V2, MobileNetv2, InceptionV3
## Checklist for new code
- [ ] is the venv activated and dependencies installed?
- [ ] is the code working and tested in a notebook?
- [ ] is the code simple and straightforward?
- [ ] have you written a proper markdown cell to the parent cell for your code? 
