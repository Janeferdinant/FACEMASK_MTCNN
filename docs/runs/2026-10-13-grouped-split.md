# Run log: photo-grouped split, seeded re-run (2026-10-13)

A new, dated run of notebooks 1 and 2. It does **not** replace or confirm the 94.85% figure reported by the February 2022 notebook output; the two used different splits and are not directly comparable (see "How this differs from the 2022 figure").

## Setup

| Item | Value |
|---|---|
| Date | 2026-10-13 |
| Hardware | CPU only, Windows 11, no GPU |
| Python / TensorFlow / Keras | 3.11.9 / 2.21.0 / 3.15.1 |
| Other pins | numpy 2.4.6, scikit-learn 1.9.1, pandas 3.0.6, mtcnn 1.0.0, opencv-python 5.0.0.93 (full list in `requirements.txt`) |
| Install | fresh venv, `pip install -r requirements.txt`: 5 min 6 s, `pip check` clean |
| Notebook 1 | `jupyter nbconvert --to notebook --execute 1_data-preprocessing.ipynb`: 33 s |
| Notebook 2 | `jupyter nbconvert --to notebook --execute 2_training-face-mask-model.ipynb`: 328 s in total, of which 291 s training (20 epochs) |
| **Seed** | **42** (`tf.keras.utils.set_random_seed(42)`, TensorFlow op determinism on, and `random_state=42` in both splits) |
| Model | InceptionV3 (ImageNet weights, all layers frozen, cut at `mixed7`) + Flatten, Dense 1024, Dropout 0.3, Dense 3. 21,562,275 parameters, 12,587,011 trainable. Adam, categorical cross-entropy, 20 epochs, batch size 32 (Keras default) |

## Split

Photos are identified by the SHA-1 of the image file, so the 5 byte-identical duplicate photo pairs in the dataset also stay on the same side. Split with `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`: fold 1 is the test set, then the same call on the rest gives the validation set (fold 1). An assertion in the notebook checks that no photo appears in more than one split.

| Split | Face crops | Source photos | without_mask | with_mask | mask_weared_incorrect |
|---|---|---|---|---|---|
| train | 3,299 | 700 | 581 | 2,619 | 99 |
| validation | 366 | 70 | 64 | 290 | 12 |
| **test** | **407** | **78** | **72** | **323** | **12** |
| total | 4,072 | 848 distinct photos (853 files) | 717 | 3,232 | 123 |

## Results: final (epoch 20) model, the headline

This is the same evaluation protocol as the original notebook (last-epoch model scored on the test set).

**Overall accuracy: 0.9631 (392 of 407 test crops).** Test loss 0.5154. The majority-class baseline (always predict `with_mask`) is 0.7936.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| without_mask | 0.9559 | 0.9028 | 0.9286 | 72 |
| with_mask | 0.9668 | 0.9907 | 0.9786 | 323 |
| mask_weared_incorrect | 0.8750 | 0.5833 | 0.7000 | 12 |
| macro average | 0.9325 | 0.8256 | 0.8691 | 407 |
| weighted average | 0.9621 | 0.9631 | 0.9615 | 407 |

Confusion matrix (rows = true label, columns = predicted label):

| true \ predicted | without_mask | with_mask | mask_weared_incorrect |
|---|---|---|---|
| without_mask | 65 | 7 | 0 |
| with_mask | 2 | 320 | 1 |
| mask_weared_incorrect | 1 | 4 | 7 |

## Secondary: best-validation checkpoint (epoch 10, val accuracy 0.9508)

Chosen on the validation set only, then scored once on the same test set. Validation loss rises from 0.277 (epoch 1) to 0.804 (epoch 20), so the later epochs overfit.

Test accuracy 0.9730 (396 of 407), test loss 0.1661.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| without_mask | 0.9714 | 0.9444 | 0.9577 | 72 |
| with_mask | 0.9786 | 0.9907 | 0.9846 | 323 |
| mask_weared_incorrect | 0.8000 | 0.6667 | 0.7273 | 12 |
| macro average | 0.9167 | 0.8673 | 0.8899 | 407 |

| true \ predicted | without_mask | with_mask | mask_weared_incorrect |
|---|---|---|---|
| without_mask | 68 | 3 | 1 |
| with_mask | 2 | 320 | 1 |
| mask_weared_incorrect | 0 | 4 | 8 |

## How this differs from the 2022 figure

The notebook output from February 2022 reports 0.9485 on 408 test crops (confusion matrix read from the notebook image: 80/84, 300/311, 7/13). The two numbers are **not directly comparable**:

- **Different test sets.** 2022 used an unseeded random split by face crop, which cannot be recreated. This run splits by photo, so the test crops are from 78 photos that contributed nothing to training. The audit estimated that about 91% of the 2022 test crops had a sibling crop from the same photo in training, and the 2022 split also put duplicate photos on both sides.
- **The grouped score is not lower.** A photo-grouped split would normally be expected to score lower than a leaky one. Here it did not. With one split, 407 test crops and only 12 `mask_weared_incorrect` crops, a difference of 1.5 points is about 6 crops and is within what a different split could change. This run does not show that the 2022 number was inflated, and it does not show that it was not.
- **Different software.** TensorFlow 2.21 / Keras 3.15.1 here, an older TensorFlow in 2022; different input pipeline details (float32 scaling, an explicit validation set instead of `validation_split`).
- **Small rarest class.** `mask_weared_incorrect` is 3.0% of the dataset. Its recall (7 of 12 for the final model) rests on a handful of crops and is the weakest result in both runs.
- **One run, one split, one seed.** No confidence interval was computed. Treat the figures as a single dated measurement.

Photos in this dataset were also not checked for near-duplicates beyond exact byte-identical files, and the test photos come from the same public dataset as the training photos, so neither figure says how the model performs on other cameras, populations or mask styles.

## Artifacts

Model weights are not committed. They were saved outside the repository for a later demo. `data/run_metrics.json` (written by notebook 2, git-ignored) holds the full per-epoch history.

## Demo check (`demo/app.py`, local only, nothing deployed)

`python demo/app.py --weights <weights.h5> --check <image> ...` and the Gradio UI (called through `gradio_client`) were run on three photos from the test split, with the final (epoch 20) model. This is a smoke test of the code path, not an evaluation.

| Photo | Faces annotated in the dataset | Faces MTCNN found | Labels returned |
|---|---|---|---|
| maksssksksss3.png | 9 with_mask | 5 | 5 with_mask |
| maksssksksss124.png | 3 with_mask, 2 without_mask | 6 | 2 with_mask, 4 without_mask |
| maksssksksss250.png | 1 with_mask, 2 without_mask, 2 mask_weared_incorrect | 5 | 1 with_mask, 4 without_mask |

MTCNN missed small faces in the first photo (4 of 9) and, in the second and third, the labels do not match the annotations everywhere (the third has no `mask_weared_incorrect` result). The demo shows what the pipeline does on these images, including its mistakes.
