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

## In our AneRBC-I notebook, how many steps are there in one epoch?

A **step** (also called an iteration) is one gradient update using one batch.

Our AneRBC-I multiclass train split has 755 images, and we use `BATCH_SIZE = 32`.
So each epoch has:

`steps_per_epoch = ceil(755 / 32) = 24`

That means the model updates its weights 24 times per epoch (the last batch is
smaller than 32).

## How is the validation split used during training in our project (AneRBC-I)?

After each epoch, Keras evaluates the model on the **validation** dataset.
We use validation results for two key training controls:

- **EarlyStopping** monitors `val_loss` and stops training if validation loss
  stops improving for many epochs (patience). This helps reduce overfitting.
- **ModelCheckpoint** monitors `val_accuracy` and saves the best-performing
  model on validation. This is the checkpoint we later load for test evaluation.

The important point: validation data is used for monitoring/selection, not for
weight updates.

## What is `class_weight`? Explain the given class weights in our project.

`class_weight` changes how much each class contributes to the training loss.
Higher weight means mistakes on that class are penalized more.

In our notebook run, the computed class weights were:

`{0: 0.5392857142857143, 1: 0.6669611307420494, 2: 3.0942622950819674, 3: 3.0942622950819674}`

In our project we compute weights using the balanced formula:

`weight(class_i) = N / (K * count(class_i))`

where `N` is total training samples and `K` is number of classes.

For AneRBC-I (train counts: Healthy=350, Microcytic=283, Normocytic=61,
Macrocytic=61; `N=755`, `K=4`):

- Healthy (0): `755 / (4*350) = 0.5392857142857143`
- Microcytic (1): `755 / (4*283) = 0.6669611307420494`
- Normocytic (2): `755 / (4*61) = 3.0942622950819674`
- Macrocytic (3): `755 / (4*61) = 3.0942622950819674`

So the rare classes (Normocytic/Macrocytic) get ~3.09× more importance in the
loss than a typical sample, which helps the model not ignore them even though
Healthy/Microcytic are much more common.
