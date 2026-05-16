# Change Notes — Kaggle multiclass transformed_AneRBC-II fusion (InceptionV3)

Date: 2026-05-16

## Added
- `kaggle_multiclass-transformed-anerbc-ii-inceptionv3-ipynb.ipynb`

## What this notebook is
A Kaggle-ready notebook that trains a **two-input fusion model** (image + CBC) for **4-class anemia classification** on `transformed_AneRBC-II`, using **InceptionV3** as the image backbone.

## Key differences vs the MobileNetV2 Kaggle notebook
- Backbone config switched to InceptionV3:
  - `MODEL_SLUG = "inceptionv3"`
  - `MODEL_DISPLAY_NAME = "InceptionV3"`
  - `BASE_LAYER_NAME = "inception_v3"`
  - `PREPROCESS_MODE = "tf_minus_one_to_one"`
- Cleared all outputs/execution counts so the notebook is a clean template.
- Early Kaggle dataset path cells now use `/kaggle/input/datasets/n07kiran/transformed-anerbc-dataset/...` (with a fallback to `/kaggle/input/<dataset-slug>` if needed).

## How to run on Kaggle
- Attach dataset: `n07kiran/transformed-AneRBC-dataset`
- Run all cells top-to-bottom.
