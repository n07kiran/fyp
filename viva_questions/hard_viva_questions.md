# Hard Viva Questions

## Why did we avoid color jitter for RBC smear augmentation?

Color in peripheral blood smear images can carry staining and morphology
information. Strong brightness, contrast, saturation, or hue changes may distort
clinically useful visual cues. For the first offline augmentation version, only
geometry-based transforms were used.

## Why are augmented rows given blank `processed_image_path` values?

The augmentation script stores real PNG images and updates `image_path`.
However, the fusion model pipeline currently trains from preprocessed `.pt`
tensors. Leaving `processed_image_path` blank avoids pretending tensor files
exist. A later preprocessing step should create matching `.pt` tensors before
using the augmented CSV directly with the current fusion loader.

## Why does the multiclass model use softmax with sparse categorical crossentropy?

The task is single-label multiclass classification: each image belongs to exactly
one of four classes. Softmax produces a probability distribution across the four
classes, and sparse categorical crossentropy works directly with integer labels
like `0`, `1`, `2`, and `3`.
