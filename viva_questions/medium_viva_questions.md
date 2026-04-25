# Medium Viva Questions

## Why were validation and test images not augmented?

Validation and test sets should represent unseen real data. If augmented
versions are added to validation or test sets, evaluation can become biased and
over-optimistic. Therefore, only the training Macrocytic images were augmented.

## Why did we match Macrocytic to Normocytic instead of Microcytic?

Macrocytic is extremely small, so increasing it up to the next-smallest class is
a conservative balancing strategy. Matching the biggest minority class would
create many variants from only 6 originals and could increase overfitting.
