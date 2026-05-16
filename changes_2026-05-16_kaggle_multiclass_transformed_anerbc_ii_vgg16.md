# Change Notes — Kaggle multiclass transformed_AneRBC-II fusion (VGG16)

Date: 2026-05-16

## Added
- `kaggle_multiclass-transformed-anerbc-ii-vgg16-ipynb.ipynb`

## What this notebook is
A Kaggle-ready notebook that trains a **two-input fusion model** (image + CBC) for **4-class anemia classification** on `transformed_AneRBC-II`, using **VGG16** as the image backbone.

## Key differences vs the MobileNetV2 Kaggle notebook
- Backbone config switched to VGG16:
  - `MODEL_SLUG = "vgg16"`
  - `MODEL_DISPLAY_NAME = "VGG16"`
  - `BASE_LAYER_NAME = "vgg16"`
  - `PREPROCESS_MODE = "vgg16_caffe"`
- Cleared all outputs/execution counts so the notebook is a clean template.
- Early Kaggle dataset path cells now use `/kaggle/input/datasets/n07kiran/transformed-anerbc-dataset/...` (with a fallback to `/kaggle/input/<dataset-slug>` if needed).

## How to run on Kaggle
- Attach dataset: `n07kiran/transformed-AneRBC-dataset`
- Run all cells top-to-bottom.
