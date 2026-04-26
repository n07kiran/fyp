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

## What is sparse categorical crossentropy loss? Why is it helpful here?

**Sparse categorical crossentropy** is the standard loss for *single-label*
multiclass classification when labels are integers (not one-hot vectors).

For one sample, if the true class index is `y` and the model predicts a softmax
probability vector `p`, the loss is:

`L = -log(p[y])`

Why it fits our AneRBC-I multiclass setup:

- Our classes are encoded as integers `0..3` in the CSV, so we can train without
  converting labels to one-hot (simpler + less memory).
- It works naturally with a 4-unit softmax output.
- With class imbalance, we can combine it with `class_weight` so minority-class
  mistakes have higher impact during training.

## What is softmax activation? How is it used and helpful in our project?

**Softmax** converts raw model outputs (logits) into probabilities across all
classes:

`p_i = exp(z_i) / sum_j exp(z_j)`

Key properties:

- Outputs are in `[0, 1]` and sum to 1 (a probability distribution).
- It matches our problem assumption: each AneRBC-I image belongs to exactly one
  of the four anemia classes.
- It makes prediction straightforward: we take `argmax` to choose the class,
  and we can also report “confidence” via the predicted probability.

## What are other options for loss functions and activations, and why choose ours?

Common alternatives and when they apply:

- **`categorical_crossentropy` + softmax**: correct if labels are one-hot
  encoded (`[1,0,0,0]`, …). We didn’t choose it because our pipeline already
  stores labels as integers `0..3`.
- **Logits output (no activation) + `SparseCategoricalCrossentropy(from_logits=True)`**:
  also correct and sometimes numerically preferred. We chose an explicit softmax
  output for easier interpretation and simpler evaluation code.
- **Focal loss** (multiclass): useful when imbalance is severe and the model
  over-focuses on easy majority examples. In our project, we already address
  imbalance using Macrocytic augmentation + `class_weight`, so plain crossentropy
  is a simpler baseline.
- **`binary_crossentropy` + sigmoid**: used for *binary* classification or
  *multi-label* tasks (an image can belong to multiple classes at once). This is
  not our case; our classes are mutually exclusive.

Why our choice is appropriate:

- Problem type: single-label 4-class classification → **softmax**.
- Label format: integer class ids → **sparse categorical crossentropy**.
- Data imbalance: minority classes → add **class weights** (and augmentation).
