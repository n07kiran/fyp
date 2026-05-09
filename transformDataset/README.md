# Transformed AneRBC Dataset Pipeline

This folder contains `01_build_transformed_dataset.ipynb`, a notebook-first
pipeline that builds a 4-class transformed dataset at project root:

```text
transformed_dataset/
  transformed_AneRBC-I/
  transformed_AneRBC-II/
```

Each dataset folder contains the same four class folders:

```text
Healthy/
Microcytic/
Normocytic/
Macrocytic/
```

Each class folder contains:

```text
transformed_Original_images/
transformed_CBC_reports/
```

Only `Original_images` are used. `Binary_segmented` and `RGB_segmented` are
ignored.

## Labels

The notebook reuses the class-threshold logic from
`datasetTransformation/extract_full_cbc_data.py`:

- `0 Healthy`: healthy cohort
- `1 Microcytic`: anemic cohort with `MCV < 80`
- `2 Normocytic`: anemic cohort with `80 <= MCV <= 100`
- `3 Macrocytic`: anemic cohort with `MCV > 100`

An anemic sample with missing or unparsable MCV is skipped and reported.

## CBC CSV Format

For every output PNG there is one same-base CSV containing these columns:

```text
WBC,RBC,HGB,HCT,MCV,MCH,MCHC,PLT,MPV,RDW_CV,final_class
```

The CBC parser keeps only numeric values from the report result field and maps
RDW variants such as `RDW-CV`, `RDW---CV`, and `%RDW---CV` to `RDW_CV`.

## Naming

AneRBC-I input images are renamed from `NNN_a.png` or `NNN_h.png` to:

```text
NNN_{ClassName}.png
NNN_{ClassName}.csv
```

AneRBC-II input images are renamed from `NNNN_SS_a.png` or `NNNN_SS_h.png` to:

```text
NNNN_SS_{ClassName}.png
NNNN_SS_{ClassName}.csv
```

For AneRBC-II, CBC reports are patient-level. The notebook derives the 3-digit
patient CBC ID from the 4-digit image ID (`0001 -> 001`), prefers the matching
AneRBC-I CBC report, and falls back to AneRBC-II CBC reports if needed. The same
patient CBC values are replicated into one CSV per serial image.

## Macrocytic Augmentation

After writing the base transformed dataset, the notebook augments only original
Macrocytic images. It imports the safe geometry-only transform from
`Code/DataAugmentation/macrocytic_augmentation.py`: rotation up to +/-15
degrees, optional flips, translation up to +/-10%, and corner-estimated fill
color to avoid black borders.

The default fixed count is:

```text
MACRO_AUGMENTATIONS_PER_IMAGE = 9
```

Augmented files are named:

```text
NNN_Macrocytic_aug_001.png
NNNN_SS_Macrocytic_aug_001.png
```

Each augmented PNG gets a same-base CSV with the same CBC values and
`final_class=3`.

## Verification

The notebook verifies:

- every output PNG has exactly one same-base CSV
- every output CSV has exactly one same-base PNG
- class counts split by original and augmented records
- a small anemic MCV spot check against the class thresholds
