# 2026-05-11 Transformed AneRBC Dataset Notebook Update

Updated the `newFusionModel` notebooks to use the public Kaggle dataset:

- Kaggle dataset id: `n07kiran/transformed-AneRBC-dataset`
- Local dataset folder: `transformed_AneRBC_dataset`
- Split files: `train_split.csv`, `val_split.csv`, `test_split.csv`

## Multiclass Notebooks

Replaced the old four multiclass fusion notebooks with dataset-specific files:

- `multiClass_transformed_aneRBC_i_vgg16.ipynb`
- `multiClass_transformed_aneRBC_i_mobilenetv2.ipynb`
- `multiClass_transformed_aneRBC_i_inceptionv3.ipynb`
- `multiClass_transformed_aneRBC_i_resnet152v2.ipynb`
- `multiClass_transformed_aneRBC_ii_vgg16.ipynb`
- `multiClass_transformed_aneRBC_ii_mobilenetv2.ipynb`
- `multiClass_transformed_aneRBC_ii_inceptionv3.ipynb`
- `multiClass_transformed_aneRBC_ii_resnet152v2.ipynb`

Each notebook now resolves image paths relative to the selected transformed subset:

- `transformed_AneRBC-I`
- `transformed_AneRBC-II`

## Kaggle And Local Path Handling

The notebooks now separate:

- read-only dataset input under `/kaggle/input` on Kaggle
- writable model/history/metrics output under `/kaggle/working` on Kaggle
- local repo output under `Code/newFusionModel/.../artifacts`

Optional overrides:

- `TRANSFORMED_ANERBC_DATASET_ROOT`
- `ANERBC_REPO_ROOT`
- `ANERBC_CHECKPOINT_ROOT`
- `ANERBC_OUTPUT_ROOT`
- `ANERBC_USE_LOCAL_METAL=1` to opt into Apple Metal locally

## Binary Fusion Notebooks

The existing binary fusion notebooks were updated in place to use `transformed_AneRBC-I` split CSVs and the same local/Kaggle path resolver. They still collapse multiclass labels to binary labels: `0=Healthy`, `1=Anemia`.

Local Apple Silicon runs default to CPU because `tensorflow-metal` can terminate the Python process during these training/evaluation graphs. Kaggle GPU behavior is unchanged. The project `venv` should not install `tensorflow-metal` unless you are explicitly testing local Metal acceleration.

## Checkpoint Behavior

When compatible local checkpoints are available, notebooks reuse only the image backbone from those checkpoints. If a checkpoint is not available in Kaggle, the notebooks fall back to a Keras application image backbone so training can still run with the public transformed dataset attached.
