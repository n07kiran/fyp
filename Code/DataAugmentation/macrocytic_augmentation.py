"""
Macrocytic-only data augmentation for the AneRBC training split.

Modes
-----
preview
    Saves a visual grid showing how rotation, flip, translation, and a combined
    transform look on the Macrocytic training images.

generate
    Saves augmented Macrocytic PNG images and writes a new augmented train CSV.
    The original train/validation/test CSV files are not modified.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "Code/Fusion_Model/transformedDataset/train_split.csv"
OUTPUT_ROOT = PROJECT_ROOT / "Code/DataAugmentation/outputs"
AUGMENTED_IMAGE_DIR = OUTPUT_ROOT / "macrocytic_augmented_images"
AUGMENTED_TRAIN_CSV = OUTPUT_ROOT / "train_split_macrocytic_augmented.csv"
PREVIEW_PATH = OUTPUT_ROOT / "macrocytic_augmentation_preview.png"

CLASS_COLUMN = "final_class"
MACROCYTIC_CLASS = 3
NORMOCYTIC_CLASS = 2
EXPECTED_MACROCYTIC_TRAIN_ROWS = 6
SEED = 42
MAX_ROTATION_DEGREES = 15
MAX_TRANSLATION_FRACTION = 0.10

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - compatibility for older Pillow
    BICUBIC = Image.BICUBIC

try:
    AFFINE = Image.Transform.AFFINE
except AttributeError:  # pragma: no cover - compatibility for older Pillow
    AFFINE = Image.AFFINE


def resolve_path(path_value: str) -> Path:
    """Resolve repo-relative paths from the CSV."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def relative_to_project(path: Path) -> str:
    """Store output paths in CSVs relative to the project root."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def estimate_fill_color(image: Image.Image, corner_size: int = 32) -> tuple[int, int, int]:
    """
    Estimate a bright slide background from image corners.

    This avoids black triangles after rotation/translation and keeps the added
    border close to the original smear background color.
    """
    width, height = image.size
    corner_boxes = [
        (0, 0, corner_size, corner_size),
        (width - corner_size, 0, width, corner_size),
        (0, height - corner_size, corner_size, height),
        (width - corner_size, height - corner_size, width, height),
    ]

    corner_arrays = []
    for box in corner_boxes:
        corner = image.crop(box).convert("RGB")
        corner_arrays.append(np.asarray(corner, dtype=np.uint8).reshape(-1, 3))

    pixels = np.concatenate(corner_arrays, axis=0).astype(np.int16)
    return tuple(int(np.percentile(pixels[:, channel], 90)) for channel in range(3))


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    fill_color = estimate_fill_color(image)
    return image.rotate(
        angle,
        resample=BICUBIC,
        expand=False,
        fillcolor=fill_color,
    )


def translate_image(image: Image.Image, dx: int, dy: int) -> Image.Image:
    fill_color = estimate_fill_color(image)
    return image.transform(
        image.size,
        AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=BICUBIC,
        fillcolor=fill_color,
    )


def random_macrocytic_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply safe geometry-only augmentation."""
    width, height = image.size
    augmented = image.convert("RGB")

    angle = rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
    augmented = rotate_image(augmented, angle)

    if rng.random() < 0.5:
        augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.25:
        augmented = augmented.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    max_dx = int(width * MAX_TRANSLATION_FRACTION)
    max_dy = int(height * MAX_TRANSLATION_FRACTION)
    dx = rng.randint(-max_dx, max_dx)
    dy = rng.randint(-max_dy, max_dy)
    augmented = translate_image(augmented, dx, dy)

    return augmented


def load_training_split(train_csv: Path = TRAIN_CSV) -> pd.DataFrame:
    if not train_csv.exists():
        raise FileNotFoundError(f"Training split not found: {train_csv}")
    return pd.read_csv(train_csv)


def get_macrocytic_train_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    macrocytic_df = train_df[train_df[CLASS_COLUMN] == MACROCYTIC_CLASS].copy()
    if len(macrocytic_df) != EXPECTED_MACROCYTIC_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MACROCYTIC_TRAIN_ROWS} Macrocytic training rows, "
            f"found {len(macrocytic_df)}."
        )

    missing_images = [
        row.image_path
        for row in macrocytic_df.itertuples(index=False)
        if not resolve_path(row.image_path).exists()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing Macrocytic image files: {missing_images}")

    return macrocytic_df.reset_index(drop=True)


def target_macrocytic_count(train_df: pd.DataFrame) -> int:
    """Match the next-smallest class: Normocytic."""
    return int((train_df[CLASS_COLUMN] == NORMOCYTIC_CLASS).sum())


def save_preview() -> None:
    """Create a preview grid for all Macrocytic training images."""
    train_df = load_training_split()
    macrocytic_df = get_macrocytic_train_rows(train_df)

    columns = ["Original", "Rotation", "Flip", "Translation", "Combined"]
    fig, axes = plt.subplots(
        len(macrocytic_df),
        len(columns),
        figsize=(18, 3.2 * len(macrocytic_df)),
    )

    for row_index, row in enumerate(macrocytic_df.itertuples(index=False)):
        image = Image.open(resolve_path(row.image_path)).convert("RGB")
        width, height = image.size

        examples = [
            image,
            rotate_image(image, angle=12),
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            translate_image(image, dx=int(width * 0.08), dy=int(-height * 0.06)),
            translate_image(
                rotate_image(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), angle=-10),
                dx=int(-width * 0.06),
                dy=int(height * 0.05),
            ),
        ]

        for column_index, example in enumerate(examples):
            ax = axes[row_index, column_index]
            ax.imshow(example)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(columns[column_index])
            if column_index == 0:
                ax.set_ylabel(row.patient_id, rotation=0, labelpad=38, va="center")

    fig.suptitle("Macrocytic-Only Augmentation Preview", fontsize=16)
    fig.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Macrocytic training images found: {len(macrocytic_df)}")
    print(f"Preview saved to: {relative_to_project(PREVIEW_PATH)}")


def generate_augmented_dataset() -> None:
    """Generate Macrocytic augmented images and a new train CSV."""
    train_df = load_training_split()
    macrocytic_df = get_macrocytic_train_rows(train_df)

    current_macrocytic_count = len(macrocytic_df)
    target_count = target_macrocytic_count(train_df)
    images_to_generate = target_count - current_macrocytic_count
    if images_to_generate <= 0:
        raise ValueError("Macrocytic class is already at or above the target count.")

    rng = random.Random(SEED)
    AUGMENTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    augmented_rows = []
    per_patient_counts: dict[str, int] = {}

    for global_index in range(images_to_generate):
        row = macrocytic_df.iloc[global_index % len(macrocytic_df)]
        patient_id = str(row["patient_id"])
        per_patient_counts[patient_id] = per_patient_counts.get(patient_id, 0) + 1

        new_patient_id = f"{patient_id}_aug_{per_patient_counts[patient_id]:03d}"
        source_image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        augmented_image = random_macrocytic_augmentation(source_image, rng)

        output_image_path = AUGMENTED_IMAGE_DIR / f"{new_patient_id}.png"
        augmented_image.save(output_image_path)

        new_row = row.copy()
        new_row["patient_id"] = new_patient_id
        new_row["image_path"] = relative_to_project(output_image_path)
        new_row["processed_image_path"] = ""
        new_row[CLASS_COLUMN] = MACROCYTIC_CLASS
        augmented_rows.append(new_row)

    augmented_train_df = pd.concat(
        [train_df, pd.DataFrame(augmented_rows)],
        ignore_index=True,
    )
    augmented_train_df.to_csv(AUGMENTED_TRAIN_CSV, index=False)

    counts = augmented_train_df[CLASS_COLUMN].value_counts().sort_index()
    written_images = [AUGMENTED_IMAGE_DIR / f"{row['patient_id']}.png" for row in augmented_rows]
    missing_written_images = [path for path in written_images if not path.exists()]
    if missing_written_images:
        raise FileNotFoundError(f"Generated image paths missing: {missing_written_images}")

    print(f"Macrocytic training images found: {current_macrocytic_count}")
    print(f"Target Macrocytic training count: {target_count}")
    print(f"Augmented images generated: {len(augmented_rows)}")
    print(f"Augmented image folder: {relative_to_project(AUGMENTED_IMAGE_DIR)}")
    print(f"Augmented train CSV: {relative_to_project(AUGMENTED_TRAIN_CSV)}")
    print("\nFinal augmented train class counts:")
    for class_id, count in counts.items():
        print(f"  Class {class_id}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or generate Macrocytic-only RBC smear augmentations."
    )
    parser.add_argument(
        "mode",
        choices=["preview", "generate"],
        help="Use 'preview' to save a visual grid or 'generate' to save images and CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preview":
        save_preview()
    elif args.mode == "generate":
        generate_augmented_dataset()


if __name__ == "__main__":
    main()
