# Medium Viva Questions

## Why were validation and test images not augmented?

Validation and test sets should represent unseen real data. If augmented
versions are added to validation or test sets, evaluation can become biased and
over-optimistic. Therefore, only the training Macrocytic images were augmented.

## Why did we match Macrocytic to Normocytic instead of Microcytic?

Macrocytic is extremely small, so increasing it up to the next-smallest class is
a conservative balancing strategy. Matching the biggest minority class would
create many variants from only 6 originals and could increase overfitting.

## Why is Case 1 excluded from multiclass image classification?

Case 1 trains a CNN from scratch, which is slow and usually needs much more data.
For this dataset, transfer learning is more practical because the ImageNet base
already has useful low-level visual features.

## Why are class weights used in multiclass training?

Even after Macrocytic augmentation, the training data is still imbalanced.
Class weights make the loss pay more attention to minority classes instead of
letting the model mainly optimize for Healthy and Microcytic samples.
