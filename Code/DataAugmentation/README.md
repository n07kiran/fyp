# Macrocytic Data Augmentation

This folder contains a simple Macrocytic-only augmentation script for the
AneRBC training split.

## Why only Macrocytic?

The training split has only 6 Macrocytic images, while the next-smallest class
has 61 Normocytic images. The script therefore creates 55 new Macrocytic PNG
images so the Macrocytic training count also becomes 61.

Validation and test split files are not changed.

## Commands

From the project root:

```bash
venv/bin/python Code/DataAugmentation/macrocytic_augmentation.py preview
```

This saves:

```text
Code/DataAugmentation/outputs/macrocytic_augmentation_preview.png
```

Generate the augmented training data:

```bash
venv/bin/python Code/DataAugmentation/macrocytic_augmentation.py generate
```

This saves:

```text
Code/DataAugmentation/outputs/macrocytic_augmented_images/
Code/DataAugmentation/outputs/train_split_macrocytic_augmented.csv
```

## Important note

The generated CSV stores real augmented PNG paths in `image_path`. For
augmented rows, `processed_image_path` is left blank because this script does
not create `.pt` tensors. If you want to train the current fusion model directly
from this augmented CSV, run the existing preprocessing step later to create
matching tensor files.
