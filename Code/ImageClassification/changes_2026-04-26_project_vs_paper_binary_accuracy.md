# Change log (2026-04-26) — Project vs Paper Binary Accuracy

## Summary

- Added a single markdown report that compares our **binary image-classification test accuracy** against the AneRBC paper’s reported results.
- Added one viva-ready Q&A explaining why our accuracy can differ from the paper (transfer learning vs scratch, preprocessing, fine-tuning choices).
- Restored the comparison markdown after accidental deletion and expanded the explanation using evidence from the linked paper repository code.

## Files changed

- Created: `Code/ImageClassification/project_vs_paper_binary_accuracy_comparison.md`
- Updated: `viva_questions/medium_viva_questions.md`
- Updated: `viva_questions/viva_update_notes_2026-04-26.md`

## Recovery/update details

- Rebuilt content in `Code/ImageClassification/project_vs_paper_binary_accuracy_comparison.md` (file had become empty).
- Added repo-backed notes from `https://github.com/shahzadmscs/AneRBC_Segmentation_Classification_code`:
	- classification notebooks use `image_size=(256, 256)`
	- transfer workflow includes frozen phase and fine-tune phase (`base_model.trainable = True` with low LR)
	- binary head/loss uses `Dense(1)` with `BinaryCrossentropy(from_logits=True)`

## Notes

- No notebook code or model-training logic was changed in this update.
- Project accuracies were taken from the already-saved Case 2 metrics tables/CSVs.
- Paper accuracies were taken from Tables 6–7 in `research papers/AneRBC_Image_Dataset_Research_Paper.pdf`.
