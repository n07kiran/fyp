# Changes — 2026-05-17 (Explainable AneRBC-II VGG16 uses Kaggle artifacts folder)

## Goal

Make the explainability notebook load the trained VGG16 fusion model **directly** from the local artifacts folder:

`kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts/`

This avoids the previous zip-based lookup.

## What changed

### Notebook updated

File: `explainable-multiclass-transformed-anerbc-ii-vgg16-ipynb.ipynb`

- **Removed zip-first model discovery** (no `transformed_anerbc_ii_vgg16_saved_output.zip` dependency).
- **Added direct artifacts path constants**:
  - `ARTIFACTS_DIRNAME = "kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts"`
  - `EXPECTED_MODEL_FILENAME = "multiClass_transformed_aneRBC_ii_vgg16_fusion_best.keras"`
- **Model path now resolves to**:
  - `kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts/models/multiClass_transformed_aneRBC_ii_vgg16_fusion_best.keras`
  - If that exact file is missing, the notebook falls back to the first model file found under the artifacts folder.
- **Cleaned imports**: removed `zipfile` and `shutil` since they were only used for zip extraction.
- **Updated markdown cells** to reflect the new artifacts-based loading and removed outdated “zip” instructions.

## How to run

1. Ensure the folder exists at repo root:
   - `kaggle_vgg16_transformed_AneRBC_II_multiClass_fusion_model_artifacts/`
2. Open `explainable-multiclass-transformed-anerbc-ii-vgg16-ipynb.ipynb`.
3. Run cells from top to bottom.

If you moved/renamed the artifacts folder, update `ARTIFACTS_DIRNAME` in **Section 2** of the notebook.

## Viva Q&A updates

- Updated `viva_questions/medium_viva_questions.md` with a question explaining why we use `compile=False` when loading a model for inference/explainability.
- Updated `viva_questions/hard_viva_questions.md` with a question explaining how to load `.keras` artifacts that include custom layers.
