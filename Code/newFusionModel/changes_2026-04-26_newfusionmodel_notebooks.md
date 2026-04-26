# New Fusion Model Notebook Rewrite (2026-04-26)

## Summary

`Code/newFusionModel` was rewritten from Python scripts into four standalone Jupyter notebooks, one per pretrained backbone.

## Files Added

- `vgg16_fusion.ipynb`
- `mobilenetv2_fusion.ipynb`
- `inceptionv3_fusion.ipynb`
- `resnet152v2_fusion.ipynb`

## Files Removed

- `fusion_common.py`
- `vgg16_fusion.py`
- `mobilenetv2_fusion.py`
- `inceptionv3_fusion.py`
- `resnet152v2_fusion.py`
- `__init__.py`

## What Each Notebook Implements

Each notebook is self-contained and does not import a shared local module.

- Uses train split from:
  - `Code/DataAugmentation/outputs/train_split_macrocytic_augmented.csv`
- Uses validation and test splits unchanged from:
  - `Code/Fusion_Model/transformedDataset/val_split.csv`
  - `Code/Fusion_Model/transformedDataset/test_split.csv`
- Loads images from `image_path` and validates file existence.
- Uses CBC features:
  - `WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, MPV, RDW_CV`
- Uses labels from `final_class` with mapping:
  - `0 Healthy, 1 Microcytic, 2 Normocytic, 3 Macrocytic`

## Fusion Architecture Used

- Image branch:
  - model-specific preprocessing
  - pretrained CNN base extracted from old multiclass `.keras` checkpoint
  - `GlobalAveragePooling2D`
  - `Dense(256, relu, L2=1e-4)`
  - `Dropout(0.30)`
- CBC branch:
  - train-median imputation
  - train-fitted `Normalization`
  - `Dense(64, relu, L2=1e-4)`
  - `Dropout(0.25)`
  - `Dense(32, relu, L2=1e-4)`
  - `Dropout(0.10)`
- Fusion head:
  - `Concatenate`
  - `Dense(128, relu, L2=1e-4)`
  - `Dropout(0.40)`
  - `Dense(64, relu, L2=1e-4)`
  - `Dropout(0.20)`
  - `Dense(4, softmax, dtype=float32)`

## Two-Stage Training Policy

- Stage 1:
  - Freeze entire CNN base
  - Train new fusion layers with `Adam(1e-4)`
- Stage 2:
  - Keep BatchNorm layers frozen
  - Fine-tune top CNN layers with `Adam(1e-5)`
  - Backbone-specific unfreeze policy:
    - VGG16: `block5_*`
    - MobileNetV2: last 20 non-BN layers
    - InceptionV3: last 40 non-BN layers
    - ResNet152V2: last 40 non-BN layers

## Built-In Validation And Tests

- Verifies all `image_path` files resolve for train/val/test.
- Verifies no CBC NaNs remain after train-median imputation.
- Verifies output shape is `(batch, 4)`.
- Verifies old classifier head layers are not reused.
- Verifies stage-1 freeze and stage-2 unfreeze policies.
- Runs a 1-epoch smoke train on a small subset before full training.

## Artifacts Saved

All notebooks write to `Code/newFusionModel/artifacts/`:

- `models/<backbone>_fusion_best.keras`
- `history/<backbone>_history.csv`
- `metrics/<backbone>_classification_report.txt`
- `plots/<backbone>_confusion_matrix.png`
- `fusion_results_summary.csv`
