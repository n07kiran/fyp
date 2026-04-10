# Case 2 Transfer Learning Accuracy Comparison

This table compares only **Case 2** (transfer learning with frozen base) across both datasets.

| CNN Model | AneRBC-I Accuracy | AneRBC-II Accuracy | Notes |
|---|---:|---:|---|
| VGG16 | 0.6600 | 0.8329 | From saved comparison CSV files |
| InceptionV3 | 0.8150 | 0.8650 | From saved comparison CSV files |
| MobileNetV2 | 0.7850 | 0.8517 | From saved comparison CSV files |
| ResNet152V2 | 0.8100 | N/A | AneRBC-II comparison CSV not found in artifacts/metrics |

## Metric Source Files

- artifacts/metrics/comparison_metrics.csv (VGG16, AneRBC-I)
- artifacts/metrics/comparison_metrics_anerbc_ii.csv (VGG16, AneRBC-II)
- artifacts/metrics/comparison_metrics_inceptionv3_anerbc_i.csv
- artifacts/metrics/comparison_metrics_inceptionv3_anerbc_ii.csv
- artifacts/metrics/comparison_metrics_mobilenetv2_anerbc_i.csv
- artifacts/metrics/comparison_metrics_mobilenetv2_anerbc_ii.csv
- artifacts/metrics/comparison_metrics_resnet152v2_anerbc_i.csv
