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

## Files updated

- `viva_questions/easy_viva_questions.md`
- `viva_questions/medium_viva_questions.md`
- `viva_questions/hard_viva_questions.md`
