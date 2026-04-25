# Transformed Dataset — Contents

This folder contains the prepared CSV files used for modeling and a full clinical CSV.

Files
- `AneRBC_I_Full_Clinical_Data.csv`: Full clinical data from AneRBC-I. Columns include `File_Name`, `Cohort`, CBC test values (`WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `PLT`, `MPV`, `RDW_CV`) and `Anemia_Category` (original label).
- `curated_dataset.csv`: The cleaned, merged dataset used for modeling. Columns: `patient_id`, `image_path` (original image), `processed_image_path` (preprocessed image tensor path), CBC features, and `final_class` (numeric label used for training).
- `train_split.csv`, `val_split.csv`, `test_split.csv`: Patient-level stratified splits exported from `curated_dataset.csv`. Use these for training, validation, and testing respectively (split produced as ~70% train / 15% val / 15% test).
- `processed_images/` (referenced by `processed_image_path`): preprocessed image tensors (not always checked in); paths in `processed_image_path` point here.

Class mapping (numeric labels)
- 0: Healthy
- 1: Microcytic
- 2: Normocytic
- 3: Macrocytic

Notes
- `final_class` is the column used for stratification and training. Decode with the mapping above.
- `AneRBC_I_Full_Clinical_Data.csv` contains the original clinical lab values; `curated_dataset.csv` joins those values with image paths and preprocessing outputs.
- To regenerate splits or preprocessing, see `Code/Fusion_Model/01_dataset_preparation.ipynb`.
