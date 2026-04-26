# Project vs Paper Binary Accuracy Comparison (AneRBC)

This document is the restored comparison file for binary image classification (Anemic vs Healthy).

It compares:

- Our project results: Case 2 transfer learning (frozen base) saved in the local artifacts.
- Paper benchmark results: Table 6 (AneRBC-I) and Table 7 (AneRBC-II) from the AneRBC research paper.

## Sources used for this comparison

- Project summary: Code/ImageClassification/case2_transfer_learning_accuracy_comparison.md
- Project metrics CSVs: Code/ImageClassification/artifacts/metrics/
- Paper PDF: research papers/AneRBC_Image_Dataset_Research_Paper.pdf (Table 6, Table 7)
- Linked paper codebase: https://github.com/shahzadmscs/AneRBC_Segmentation_Classification_code

## AneRBC-I test accuracy (paper Table 6)

| Model | Our project (Case 2: transfer frozen) | Paper (without TL) | Paper (with TL) | Our - paper (without TL) | Our - paper (with TL) |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.7850 | 0.83 | 0.71 | -0.0450 | +0.0750 |
| ResNet152V2 | 0.8100 | 0.77 | 0.70 | +0.0400 | +0.1100 |
| VGG16 | 0.6600 | 0.91 | 0.86 | -0.2500 | -0.2000 |
| InceptionV3 | 0.8150 | 0.89 | 0.45 | -0.0750 | +0.3650 |

## AneRBC-II test accuracy (paper Table 7)

| Model | Our project (Case 2: transfer frozen) | Paper (without TL) | Paper (with TL) | Our - paper (without TL) | Our - paper (with TL) |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.8517 | 0.88 | 0.84 | -0.0283 | +0.0117 |
| ResNet152V2 | 0.8333 | 0.82 | 0.62 | +0.0133 | +0.2133 |
| VGG16 | 0.8329 | 0.94 | 0.76 | -0.1071 | +0.0729 |
| InceptionV3 | 0.8650 | 0.89 | 0.70 | -0.0250 | +0.1650 |

## Why the accuracy gap exists (now using the linked research codebase)

The key point is: this is not exactly the same training recipe.

### 1) Case mismatch can make our result look lower

- Our saved values are from one setup: transfer learning with frozen base (Case 2).
- The paper reports two setups: without transfer learning and with transfer learning.
- The paper itself states that several models perform better without transfer learning on AneRBC.

So comparing our frozen-transfer result to the paper's no-transfer result is a tougher baseline.

### 2) Shared research notebooks in the linked repo use a different transfer workflow

From the linked classification notebooks in the paper repo:

- Inputs are loaded at image_size=(256, 256).
- They run both phases: first frozen base_model.trainable = False, then a fine-tuning phase with base_model.trainable = True.
- Fine-tuning uses a low learning rate (for example, Adam with learning_rate=1e-5).
- Binary head/loss uses Dense(1) with BinaryCrossentropy(from_logits=True).

In our local VGG16 binary notebook:

- Inputs are resized to (224, 224).
- Pixels are explicitly normalized to [0, 1] by dividing by 255.
- Case 2 keeps the backbone frozen; there is no second-stage fine-tuning phase in that case.
- Head/loss uses sigmoid output + binary_crossentropy.

These differences alone can materially shift final test accuracy.

### 3) Our VGG16 AneRBC-I scratch run collapsed

In the local metrics CSV, VGG16 Case 1 (scratch) on AneRBC-I reached test accuracy 0.50 and F1 0.00.

That pattern usually means one-class prediction collapse, so it is not a stable scratch baseline against the paper's stronger scratch result.

## Practical conclusion

- Compared to paper without-transfer numbers, our Case 2 is often lower (especially VGG16 on AneRBC-I).
- Compared to paper with-transfer numbers, our Case 2 is often comparable or better for several models.
- The largest actionable gap is VGG16 AneRBC-I, where adding a controlled fine-tuning phase (after frozen training) is the first high-value experiment.

