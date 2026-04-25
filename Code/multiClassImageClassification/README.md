# Multiclass Image Classification

This folder contains image-only multiclass classification notebooks for the
AneRBC-I anemia classes:

- Healthy
- Microcytic
- Normocytic
- Macrocytic

Only Case 2 is included: transfer learning with a frozen ImageNet base. Case 1
training from scratch is intentionally excluded because it is slow and not a
good fit for this dataset size.

## Data Sources

The notebooks use the Macrocytic-augmented training CSV by default:

```text
Code/DataAugmentation/outputs/train_split_macrocytic_augmented.csv
```

Validation and test data stay unchanged:

```text
Code/Fusion_Model/transformedDataset/val_split.csv
Code/Fusion_Model/transformedDataset/test_split.csv
```

This keeps evaluation on real, unseen images while improving the training
signal for the tiny Macrocytic class.

## Notebooks

- `multiclass_anerbc_i_vgg16_transfer_learning.ipynb`
- `multiclass_anerbc_i_resnet152v2_transfer_learning.ipynb`
- `multiclass_anerbc_i_mobilenetv2_transfer_learning.ipynb`
- `multiclass_anerbc_i_inceptionv3_transfer_learning.ipynb`

VGG16, ResNet152V2, and MobileNetV2 use `224x224` images. InceptionV3 uses
`299x299` images.

## Outputs

Each notebook writes outputs under:

```text
Code/multiClassImageClassification/artifacts/
```

The artifact folders are:

- `models/`: best `.keras` checkpoint
- `metrics/`: metric CSV and classification report
- `plots/`: training curves, confusion matrix, accuracy/F1 chart
- `splits/`: resolved train/validation/test CSV copies

## Running

Open one notebook and run cells top-to-bottom. The early cells validate image
paths, split counts, label mapping, batch shape, frozen base configuration, and
4-class output shape before training starts.
