# Change Notes — Kaggle multiclass transformed_AneRBC-II fusion (ResNet152V2)

Date: 2026-05-16

## Added
- `kaggle_multiclass-transformed-anerbc-ii-resnet152v2-ipynb.ipynb`

## What this notebook is
A Kaggle-ready notebook that trains a **two-input fusion model** (image + CBC) for **4-class anemia classification** on `transformed_AneRBC-II`, using **ResNet152V2** as the image backbone.

## Key differences vs the MobileNetV2 Kaggle notebook
- Backbone config switched to ResNet152V2:
  - `MODEL_SLUG = "resnet152v2"`
  - `MODEL_DISPLAY_NAME = "ResNet152V2"`
  - `BASE_LAYER_NAME = "resnet152v2"`
  - `PREPROCESS_MODE = "tf_minus_one_to_one"`
- Cleared all outputs/execution counts so the notebook is a clean template.
- Early Kaggle dataset “sanity check” cells were made non-fatal and standardized to `/kaggle/input`.

## How to run on Kaggle
- Attach dataset: `n07kiran/transformed-AneRBC-dataset`
- Run all cells top-to-bottom.
