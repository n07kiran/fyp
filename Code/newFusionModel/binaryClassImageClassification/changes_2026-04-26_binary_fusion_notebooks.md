# Binary Fusion Notebook Implementation Notes (2026-04-26)

## Summary

Implemented binary Image + CBC fusion notebooks for all four backbones in
`Code/newFusionModel/binaryClassImageClassification` by adapting the existing
multiclass fusion workflow.

Target classes:
- `0`: Healthy
- `1`: Anemia (collapsed from original `final_class` values `1, 2, 3`)

## Files Updated

- `vgg16_fusion.ipynb`
- `mobilenetv2_fusion.ipynb`
- `inceptionv3_fusion.ipynb`
- `resnet152v2_fusion.ipynb`

## Backbone-Specific Configuration

- VGG16
  - Checkpoint: `Code/ImageClassification/artifacts/models/vgg16_transfer_frozen_anerbc_i_best.keras`
  - Base layer: `vgg16`
  - Preprocessing mode: `vgg16_caffe`
  - Fine-tuning policy: prefix-based (`block5`)
- MobileNetV2
  - Checkpoint: `Code/ImageClassification/artifacts/models/mobilenetv2_transfer_frozen_anerbc_i_best.keras`
  - Base layer: `mobilenetv2_1.00_224`
  - Preprocessing mode: `unit_range` (`image / 255.0`)
  - Fine-tuning policy: last `20` non-BN layers
- InceptionV3
  - Checkpoint: `Code/ImageClassification/artifacts/models/inceptionv3_transfer_frozen_anerbc_i_best.keras`
  - Base layer: `inception_v3`
  - Preprocessing mode: `unit_range` (`image / 255.0`)
  - Fine-tuning policy: last `40` non-BN layers
- ResNet152V2
  - Checkpoint: `Code/ImageClassification/artifacts/models/resnet152v2_transfer_frozen_anerbc_i_best.keras`
  - Base layer: `resnet152v2`
  - Preprocessing mode: `unit_range` (`image / 255.0`)
  - Fine-tuning policy: last `40` non-BN layers

## Core Pipeline Changes

1. Label handling
- Read original `final_class` from CSV.
- Validate expected source IDs (`0, 1, 2, 3`).
- Collapse to binary using:
  - `df["final_class"] = (raw_labels != 0).astype("int64")`

2. Model head and loss
- Replaced multiclass head with binary head:
  - `Dense(1, activation="sigmoid", dtype="float32")`
- Replaced multiclass loss with binary loss:
  - `binary_crossentropy`

3. Prediction and evaluation
- Replaced argmax prediction with thresholded probability:
  - `probabilities >= 0.5`
- Updated report and confusion matrix to labels `[0, 1]` and class names
  `Healthy`, `Anemia`.
- Added binary-focused metrics in summary row:
  - `binary_f1`, `precision_anemia`, `recall_anemia`

4. Artifact separation
- Moved binary fusion outputs to:
  - `Code/newFusionModel/binaryClassImageClassification/artifacts`
- Preserved model-specific output naming by `MODEL_SLUG`.

5. Notebook usability updates
- Added/updated markdown cells to explain each major code block
  (data loading, model build, assertions, training, evaluation).
- Cleared notebook outputs and execution counts to keep the notebooks clean.

## Validation Performed

- Verified all four notebooks contain binary constants, binary checkpoint paths,
  binary artifact path, and binary evaluation logic.
- Ran syntax validation on every Python code cell across all four notebooks
  using `ast.parse`.
- Result: all notebook code cells parse successfully.
