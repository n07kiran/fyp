"""
MultimodalDataset — loads pre-processed image tensors + CBC tabular features.

Images are expected as unnormalised tensors in [0, 1].
Training applies augmentation first, then ImageNet normalization.
Validation / test apply deterministic resize+crop, then normalization.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
# Image preprocessing and augmentation
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_AUGMENTATION = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(8),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

EVAL_PREPROCESS = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(224),
])

NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class MultimodalDataset(Dataset):
    """
    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain columns: ``processed_image_path``, the 10 CBC feature
        columns, and ``final_class``.
    cbc_features : list[str]
        Column names of the 10 numeric CBC features.
    augment : bool
        If True, apply training-time augmentation to images.
    project_root : str | None
        If given, ``processed_image_path`` values are resolved relative to
        this directory.  Otherwise they are treated as absolute paths.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        cbc_features: list[str],
        augment: bool = False,
        project_root: str | None = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.cbc_features = cbc_features
        self.augment = augment
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- Image tensor (pre-processed .pt) ---
        img_path = row["processed_image_path"]
        if self.project_root is not None:
            img_path = os.path.join(self.project_root, img_path)
        image = torch.load(img_path, weights_only=True)

        if self.augment:
            image = TRAIN_AUGMENTATION(image)
        else:
            image = EVAL_PREPROCESS(image)

        image = NORMALIZE(image)

        # --- CBC tabular features ---
        cbc = torch.tensor(
            row[self.cbc_features].values.astype("float32"),
            dtype=torch.float32,
        )

        # --- Label ---
        label = int(row["final_class"])

        return image, cbc, label
