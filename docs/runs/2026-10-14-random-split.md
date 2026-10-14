# Run log: re-run with the original random-by-crop split (2026-10-14)

**Status: Verified (re-run 2026-10-14, random crop split, seeds 42, 43, 44).** Final epoch-20 test accuracy: 95.59%, 97.55%, 96.32%; **mean 96.49%, range 95.59% to 97.55%** (three runs).

This is a new, dated measurement. It does **not** replace, recreate or confirm the **Reported 94.85%** from the February 2022 notebook output. That run used no seed, so it cannot be repeated exactly; this log only re-does the same *style* of split. The three numbers in the README are three different measurements:

| Figure | Status | What the test set is |
|---|---|---|
| 94.85% (387 of 408 crops) | Reported (notebook output, Feb 2022, unseeded) | a random 10% of the face crops |
| 96.49% mean, 95.59 to 97.55% (this log) | Verified (re-run 2026-10-14, random crop split, seeds 42, 43, 44) | a random 10% of the face crops, three seeds |
| 96.31% (392 of 407 crops) | Verified (re-run 2026-10-13, photo-grouped split, seed 42) | 78 whole photos held out (see `2026-10-13-grouped-split.md`) |

## What was run

`python scripts/random_split_rerun.py --seeds 42 43 44` (console output: `2026-10-14-random-split-console.log`; per-seed numbers: `2026-10-14-random-split-seed<seed>.json`). It follows the 2022 notebook 2 cell by cell:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)   # random, by face crop
model.fit(X_train, Y_train, epochs=20, validation_split=0.1)                                 # last 10% of the train crops
model.evaluate(X_test, Y_test)                                                               # final epoch-20 model
```

| Item | Value |
|---|---|
| Hardware | CPU only, AMD Ryzen 7 5800X (8 cores, 16 threads), Windows 11 |
| Software | Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, scikit-learn 1.9.1 |
| Data | `data/X.npy`, `Y.npy`, `groups.npy` from notebook 1: 4,072 face crops from 848 distinct photos (853 files) |
| Model | the notebook 2 model: InceptionV3 (ImageNet, frozen, cut at `mixed7`) + Flatten, Dense 1024, Dropout 0.3, Dense 3; Adam, categorical cross-entropy, 20 epochs, batch size 32 |
| Split sizes | train 3,297, validation 367 (last 10% of train, not shuffled further), **test 408** (same size as the 2022 test set) |
| Seeds | 42, 43, 44: `set_random_seed(seed)`, TensorFlow op determinism on, `random_state=seed` in the split |
| Training time | 517 s, 295 s, 285 s per run (all inside the 20-minute limit). The first run was slowed by other work on the machine at the same time. |
| Weights | not saved, not committed |

## Results

| Seed | Test accuracy | Correct | Test loss | without_mask F1 | with_mask F1 | mask_weared_incorrect F1 (support) | Macro F1 | Test crops from a photo that is also in train |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.9559 | 390 / 408 | 0.3401 | 0.924 | 0.974 | 0.600 (10) | 0.833 | 377 (92.4%) |
| 43 | 0.9755 | 398 / 408 | 0.3254 | 0.961 | 0.985 | 0.667 (10) | 0.871 | 373 (91.4%) |
| 44 | 0.9632 | 393 / 408 | 0.5690 | 0.940 | 0.978 | 0.720 (16) | 0.879 | 379 (92.9%) |
| **mean** | **0.9649** | | | | | | | |

Range 0.9559 to 0.9755: a spread of 2.0 points, about 8 test crops. Confusion matrices (rows = true, columns = predicted; order without_mask, with_mask, mask_weared_incorrect):

| Seed | Confusion matrix |
|---|---|
| 42 | [[67, 5, 0], [5, 317, 4], [1, 3, 6]] |
| 43 | [[62, 3, 0], [2, 331, 0], [0, 5, 5]] |
| 44 | [[70, 6, 0], [2, 314, 0], [1, 6, 9]] |

## What each split measures, and why the random-crop figure can overstate generalisation

- **Random crop split (2022 style).** Each face crop is put in train or test at random. The dataset has about 4.8 faces per photo, so a test crop usually has a sibling from the *same photo* in the training set: typically the same camera, lighting and background. In all three seeds, 91% to 93% of the test crops came from a photo that also contributed training crops (and 5 pairs of byte-identical photos can sit on opposite sides). The score then partly measures how well the model recognises a scene it has seen, not only how well it recognises masks on unseen people. The expected bias is upward (optimistic), not downward.
- **Photo-grouped split (2026-10-13).** Whole photos are held out, so no test crop shares a photo with training. It asks the question a user cares about: how does the model do on a new scene? It is the fairer test of generalisation, but it is a single split with a different, smaller set of photos and only 12 `mask_weared_incorrect` crops.
- **What these runs show.** On this dataset the random-crop re-runs (mean 96.49%) are *not* clearly higher than the one grouped run (96.31%): the gap of 0.2 points is far smaller than the 2-point spread between seeds of the same split. So the leakage is real (91% to 93% of test crops share a photo with training) but its effect on accuracy is not visible at this sample size. Neither split establishes the other's number, and no figure here applies to other cameras, people or lighting.
- **The Reported 94.85%** lies 0.7 points below the lowest of the three re-runs and 1.6 points below their mean. With an unseeded run and a 2-point seed-to-seed spread, that difference is not something these runs can explain, and it is not evidence that the 2022 figure was wrong or right. It stays in the README as Reported, unchanged.
- **Majority-class baseline:** always predicting `with_mask` gives 78% to 82% (326, 333 and 316 of 408 for the three seeds), so accuracy alone flatters every split. Recall on `mask_weared_incorrect` (5 to 9 of 10 to 16 test crops) is the weak spot in every run.
