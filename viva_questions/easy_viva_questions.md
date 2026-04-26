# Easy Viva Questions

## Why did we augment only the Macrocytic class?

The Macrocytic class has very few samples compared with the other classes. In
the training split, it has only 6 images, while Normocytic has 61 images.
Augmenting only Macrocytic helps reduce class imbalance without unnecessarily
changing the majority classes.

## Which augmentations were used?

Rotation, flip, and translation were used. These are safe for blood smear
images because slide orientation and cell position can naturally vary during
image capture.

## What are the four classes in multiclass image classification?

The four classes are Healthy, Microcytic, Normocytic, and Macrocytic. The model
uses one output neuron per class with softmax activation.

## What is an epoch?

An **epoch** means one full pass through the entire training dataset.

In our AneRBC-I multiclass notebook, one epoch means the model has seen all
training images (the train split CSV) once, in mini-batches.

## What is batch size?

**Batch size** is the number of training images processed together before the
model updates its weights once.

In our notebook, `BATCH_SIZE = 32`, so the model learns using groups of 32
images at a time (the last batch in an epoch can be smaller).

## What is the difference between train, validation, and test sets (AneRBC-I)?

- **Train set**: used to *learn* model parameters (weights) via backpropagation.
  In our project, training uses the Macrocytic-augmented train CSV.
- **Validation set**: used during training to check generalization and guide
  decisions like early stopping and saving the best checkpoint. Validation
  images are not used to update weights.
- **Test set**: used only at the end for a final, unbiased performance report.
  Test images are never used to tune training.

For AneRBC-I, we keep validation and test splits as the original fusion-model
splits (no augmentation) to keep evaluation honest.

## What is resizing and normalization of images here?

- **Resizing** means converting every image to the same spatial resolution.
  We resize to `224×224` because VGG16 expects a fixed input size.
- **Normalization** means scaling pixel values to a consistent numeric range.
  We convert pixels from `[0, 255]` (uint8) to `[0, 1]` (float32) by dividing
  by 255. This improves training stability and makes learning more consistent.

## What does “batched and prefetched” mean?

- **Batched**: groups individual samples into a batch (size 32 here) so the GPU
  trains efficiently.
- **Prefetched**: prepares the next batch while the model is training on the
  current batch, improving throughput and reducing input pipeline bottlenecks.
