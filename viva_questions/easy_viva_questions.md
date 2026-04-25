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
