# Viva updates (2026-04-26)

## What changed

Added viva-ready Q&A for common deep-learning training terms, explicitly tied to
**AneRBC-I multiclass classification with VGG16 transfer learning**:

- Epoch, batch size
- Train vs validation vs test usage in our AneRBC-I pipeline
- Image resizing + normalization (`224×224`, `[0,1]` scaling)
- `tf.data` batching + prefetching
- `class_weight` concept + the exact weights used in the notebook (with formula)
- Sparse categorical crossentropy + softmax (what/why)
- Alternative loss/activation options and why the notebook’s choices fit

Added one more medium-level Q&A tied to our recent binary-classification results:

- Why our project’s binary accuracy can differ from the AneRBC paper (transfer
	learning vs scratch + preprocessing/fine-tuning differences)
- Sigmoid output vs logits (`from_logits=True`) in binary classification, and
	why both are valid when paired with the correct loss

## Files updated

- `viva_questions/easy_viva_questions.md`
- `viva_questions/medium_viva_questions.md`
- `viva_questions/hard_viva_questions.md`

## Additional update (newFusionModel notebook rewrite)

Added medium-level viva Q&A tied to the new image + CBC fusion workflow:

- Why feature-level multimodal fusion is used (image branch + CBC branch)
- Why BatchNorm is kept frozen during stage-2 fine-tuning while unfreezing only
	selected top CNN layers

File updated:

- `viva_questions/medium_viva_questions.md`

## Additional update (binary newFusionModel implementation)

Added medium-level viva Q&A tied to the new binary Image + CBC fusion notebooks:

- How multiclass `final_class` was collapsed into binary labels (`0` vs `1,2,3`)
- Why sigmoid + `binary_crossentropy` + `0.5` threshold was used
- Why backbone preprocessing differs (`vgg16_caffe` vs `tf_minus_one_to_one`)
- Why only the old CNN base is reused while replacing the old classifier head

File updated:

- `viva_questions/medium_viva_questions.md`
