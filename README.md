# FACEMASK_MTCNN: face-mask detection with MTCNN and a transfer-learned InceptionV3 classifier

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow.svg)](https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn)

## Overview

A webcam detector that finds faces with MTCNN and labels each one `with_mask`, `without_mask` or `mask_weared_incorrect` using an InceptionV3 network fine-tuned by transfer learning. Three notebooks cover data preparation, training and webcam detection, and a small Gradio app runs the same pipeline on an uploaded photo. `demo/webcam.py` runs the pipeline on a webcam or a video file and shows the measured FPS: **real-time on a desktop CPU, 23.1 FPS mean** (Verified, run 2026-10-14, AMD Ryzen 7 5800X, MTCNN every 10 frames, 640x480; see [Results](#results) for the settings and caveats, including a short pause on each frame where MTCNN runs). It is a teaching project.

## Demo

**Live demo: not available yet.** A public Hugging Face Space is prepared (`demo/build_space.py`), but Hugging Face refused to create it on 2026-10-14 because a Gradio Space on free hardware needs a PRO subscription ([details](docs/runs/2026-10-14-space-check.md)). Until it exists, run the same app on your own computer. You do not need to train anything: the weights are downloaded once (about 190 MB) from the public [model repository](https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn).

```bash
pip install -r requirements.txt -r demo/requirements.txt
python demo/app.py
```

### Webcam or video file

```bash
python demo/webcam.py                         # default camera; q or Esc quits
python demo/webcam.py --source clip.mp4       # a video file (add --no-display --save out.mp4 to run headless)
```

It draws a labelled box on every face and the measured FPS in the corner. Options: `--detect-every N` (MTCNN on every N-th frame, default 10), `--detect-width`, `--mirror`, `--source` (camera index or file). No camera was available while this was written, so it was timed on a video file (see Results); it has not been run on a live webcam.

The Gradio app opens a page where you upload an image or take a webcam snapshot and get labelled boxes back. The app does not save images. The example images come from the CC0 dataset described below. More options are in [`demo/README.md`](demo/README.md). The project page, [janeferdinant.github.io/FACEMASK_MTCNN](https://janeferdinant.github.io/FACEMASK_MTCNN/), summarises the pipeline and results.

## Problem statement

Decide, for every face in an image, whether the person is wearing a mask, wearing none, or wearing one incorrectly (for example under the nose). The "worn incorrectly" class is the interesting one: it is rare, and the hardest to get right.

## Approach

![Pipeline: MTCNN face detection, face crop, InceptionV3 transfer-learned classifier, three class labels](docs/pipeline.svg)

1. **Detect.** MTCNN (pretrained, from the `mtcnn` package) proposes a box for each face. It is not trained in this repository.
2. **Crop.** Each box is cut out, resized to 100 x 100 pixels and scaled to 0-1.
3. **Classify.** InceptionV3 with frozen ImageNet weights, cut at layer `mixed7`, followed by Flatten, Dense 1024, Dropout 0.3 and Dense 3 with softmax. Only the new head is trained (12.6 million of 21.6 million parameters are trainable), with Adam and categorical cross-entropy for 20 epochs.

4. **Track (webcam script only).** MTCNN is the slow stage (about 0.25 s per 640x480 frame here). `demo/webcam.py` runs it on every 10th frame, moves the boxes in between with Lucas-Kanade optical flow (shift and zoom, no rotation), keeps each face's last label, forces a new detection when the picture changes abruptly, and classifies all faces of a frame in one batched call. A new face, or a face that changes its mask state, is labelled at the next detection (up to 9 frames, 0.3 s at 30 fps). MTCNN and the InceptionV3 classifier are unchanged; only how often they run changes, so the classifier's accuracy is the one in Results.

Notebook 2 splits the data **by source photo**, not by face. Many photos contain several faces, and a face-level random split would put faces from the same photo in both the training and the test set.

## Data

| | |
|---|---|
| Source | Kaggle, [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) by andrewmvd. The images and annotations come from the Make ML [Mask Dataset](https://makeml.app/datasets/mask). |
| License | CC0 1.0 (public domain dedication) |
| Size | 853 PNG photos with 853 PASCAL VOC XML files (about 400 MB, in `data/`), 4,072 annotated faces |
| Classes | `with_mask` 3,232 (79.4%) · `without_mask` 717 (17.6%) · `mask_weared_incorrect` 123 (3.0%) |
| Split (2026-10-13 run) | train 3,299 crops / 700 photos · validation 366 / 70 · test 407 / 78; seed 42; no photo in two splits |

**Privacy.** The images show real people's faces from a public dataset. CC0 describes the dataset's copyright status; it does not say whether the people shown consented. The images are used here only to train and test the classifier, and the demo's example images come from this dataset. If you appear in a photo and want it removed from this repository, contact the author through the links below.

## Results

| Metric | Value | Status | How measured | Date |
|---|---|---|---|---|
| Test accuracy | 94.85% (387 of 408 crops) | Reported (notebook output, unseeded split; ~91% of test crops share a source photo with training crops, so it overstates generalisation) | Original notebook 2 output: random face-level 10% split, no seed, so it cannot be recreated | Feb 2022 |
| Test accuracy | 96.31% (392 of 407 crops) | Verified (re-run 2026-10-13, photo-grouped split) | `model.evaluate` on 78 held-out photos, final epoch-20 model, seed 42 | 2026-10-13 |
| Per-class F1 | without_mask 0.929 · with_mask 0.979 · mask_weared_incorrect 0.700 (macro F1 0.869) | Verified (re-run 2026-10-13, photo-grouped split) | `sklearn` classification report on the same test set | 2026-10-13 |
| Majority-class baseline | 79.4% | Verified (re-run 2026-10-13, photo-grouped split) | Always predicting `with_mask` on the test set (323 of 407) | 2026-10-13 |
| Test accuracy | 96.49% mean (95.59%, 97.55%, 96.32%) | Verified (re-run 2026-10-14, random crop split, seeds 42, 43, 44) | Same split style as 2022 (random 10% of face crops, final epoch-20 model, 408 test crops), three fixed seeds; [log](docs/runs/2026-10-14-random-split.md) | 2026-10-14 |
| Webcam speed | 23.1 FPS mean (runs 22.3, 23.1, 23.9) | Verified (run 2026-10-14, AMD Ryzen 7 5800X, 8 cores / 16 threads, CPU only) | `demo/webcam.py` settings: MTCNN every 10 frames, 640x480, one batched classifier call; 470 timed frames of a video made from dataset photos; read, process and draw, no window; 60% of the CPU was busy with other programs; [log](docs/runs/2026-10-14-fps.md) | 2026-10-14 |
| Webcam latency | pipeline call: mean 43 ms, p95 260 ms | Verified (same run) | The p95 frames are those where MTCNN runs (one in ten): the picture pauses for about a quarter of a second each time | 2026-10-14 |
| Same labels as MTCNN on every frame | 93.4% of 1,618 faces (94.0% found, 93.4% found with the same label) | Verified (same run) | Box overlap IoU >= 0.5 and same class, against MTCNN + classifier on every frame of the same clip; N=5 gives 14.1 FPS and 95.9% (sweep in the log) | 2026-10-14 |
| Webcam speed, notebook 3 style | 1.8 FPS (58 frames) | Verified (same run) | MTCNN every frame and `model.predict` per face, as in notebook 3. Notebook 3's own 2022 note says about 1.3 FPS (**Reported**; machine not recorded) | 2026-10-14 |

Full numbers (per-class precision and recall, the confusion matrix, split sizes and the seed) are in [`docs/runs/2026-10-13-grouped-split.md`](docs/runs/2026-10-13-grouped-split.md).

**What each accuracy measures.** *94.85% (Reported, 2022)* and *96.49% (re-run, random crop split)* both come from a random split by face crop: a test face usually has siblings from the same photo in the training set (91% to 93% of test crops, counted in the re-runs), so the test is partly "have I seen this scene" and the figure can overstate how well the model works on new photos. *96.31% (photo-grouped split)* holds out whole photos, the fairer test of generalisation. The re-runs are new measurements: they neither recreate nor confirm the unseeded 2022 number, which sits 0.7 points below the lowest of the three re-runs; the re-runs differ among themselves by 2 points (about 8 crops). The random-crop mean (96.49%) is not clearly above the grouped result (96.31%): the leakage is real, but its effect is not visible at this sample size.

**Why the 2022 and grouped accuracies cannot be compared directly.** The 2022 figure comes from one unseeded random split by face (five duplicate photos also sat on both sides). The 2026-10-13 figure comes from a split by photo. You might expect the grouped one to be lower; it is not. Both are single runs on one split of a small test set. With only 12 `mask_weared_incorrect` test crops, one crop moves that class's recall by 8 points, and 1.5 points of overall accuracy is about 6 crops. The new run neither confirms nor refutes the 2022 number: it is a different, dated measurement on a different test set. The weakest result in both runs is recall on `mask_weared_incorrect` (7 of 12 here, 7 of 13 in 2022).

## Tech stack

Python 3.11, TensorFlow 2.21 / Keras 3.15, `mtcnn` 1.0.0, OpenCV 5.0, scikit-learn 1.9, NumPy, pandas, seaborn, Matplotlib, Jupyter and Gradio (demo only); pytest and ruff for the tests. Exact pins are in `requirements.txt`, `demo/requirements.txt` and `requirements-dev.txt`.

## Getting started

Prerequisites: Python 3.11, Git, a CPU (no GPU needed) and about 1.5 GB of free disk for the data and generated files, plus the Python environment. Installing TensorFlow takes about 5 minutes.

```bash
git clone https://github.com/Janeferdinant/FACEMASK_MTCNN.git
cd FACEMASK_MTCNN
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks in order, either in `jupyter notebook` or headless (`--inplace` overwrites the saved outputs in the notebook files):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 1_data-preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 2_training-face-mask-model.ipynb
```

Expected on a CPU laptop (2026-10-13 run): notebook 1 takes about 30 seconds and prints `Face crops : 4072 | distinct source photos : 848`. Notebook 2 takes about 5.5 minutes, prints `Test loss: 0.5154  Test accuracy: 0.9631  (407 face crops)` with seed 42, and writes `data/inceptionV3-model.h5`. A different TensorFlow version or CPU may change the last digits.

Try the Gradio demo on the model you just trained:

```bash
pip install -r demo/requirements.txt
python demo/app.py --weights data/inceptionV3-model.h5
```

Open http://127.0.0.1:7860. To check the pipeline without a browser, run `python demo/app.py --weights data/inceptionV3-model.h5 --check data/images/maksssksksss3.png`, which prints the detections as JSON. The weights path can also be given through the `FACEMASK_WEIGHTS` environment variable. Without `--weights` and without `data/inceptionV3-model.h5`, the app downloads the published weights instead (see [Demo](#demo)).

Webcam script, tests and benchmarks (the weights come from the notebooks above or are downloaded, see [Demo](#demo)):

```bash
python scripts/make_test_clip.py                      # 8-photo, 480-frame test video from the dataset -> data/benchmark_clip.mp4
python demo/webcam.py --source data/benchmark_clip.mp4 --no-display --save data/annotated.mp4
pip install -r requirements-dev.txt
python -m pytest                                      # tests on mocked frames, no weights needed
ruff check .
python scripts/benchmark_fps.py --log docs/runs/2026-10-14-fps.md   # about 10 minutes; rewrites that log
python scripts/random_split_rerun.py --seeds 42 43 44               # about 15 to 25 minutes on a CPU; rewrites docs/runs/2026-10-14-random-split-seed*.json
```

The `webcam.py` command ends by printing the frame count, mean FPS and pipeline latency. FPS depends on your CPU and on what else is running.

Notebook 3 (`3_detecting-mask-w-mtcnn.ipynb`) opens a webcam window. It was not re-run in 2026 because it needs a camera. It loads `data/inceptionV3-model.h5` and `data/category2label.pkl`, which notebooks 1 and 2 create.

## Project structure

```
1_data-preprocessing.ipynb        VOC XML -> 100x100 face crops -> data/X.npy, Y.npy, groups.npy
2_training-face-mask-model.ipynb  seeded photo-grouped split, InceptionV3 training, evaluation
3_detecting-mask-w-mtcnn.ipynb    MTCNN + classifier on webcam frames (original 2022 notebook)
demo/app.py                       Gradio app: image in, labelled boxes out
demo/webcam.py                    webcam / video detector: MTCNN every N frames, optical-flow tracking, measured FPS
demo/realtime.py                  frame skipping, box tracking, FPS counter (no TensorFlow needed)
scripts/                          benchmark_fps.py, make_test_clip.py, random_split_rerun.py
tests/                            pytest tests for demo/realtime.py (mocked frames)
demo/README.md                    how to run the demo; how the Hugging Face Space is set up
demo/build_space.py, demo/space/  files for the (not yet created) Hugging Face Space
docs/pipeline.svg                 pipeline diagram
docs/runs/                        dated run logs
index.html                        project page served by GitHub Pages
archive/                          unrelated legacy site files (see archive/README.md)
data/images, data/annotations     the dataset (CC0)
requirements.txt                  pinned versions that install on Python 3.11
```

Generated files (`data/*.npy`, `*.pkl`, `*.h5`, `*.keras`) are git-ignored. Model weights are not stored in this repository; they are published on the [Hugging Face Hub](https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn) with a model card.

## Real-world applications

The same detect, crop, classify pattern applies to any task that needs a per-face or per-object label: safety-equipment checks (helmets, goggles), occupancy counting, and teaching examples of two-stage vision pipelines. This repository is a learning project and has not been tested for any such use.

## Limitations

- **Small, single-source data.** 853 photos from one public dataset, with 123 `mask_weared_incorrect` faces in total. The classifier is untested on other cameras, lighting, face angles, skin tones or mask styles.
- **Rare class.** `mask_weared_incorrect` has only 12 test crops, so its scores are noisy.
- **One split, one seed.** The 2026 results are one measurement without a confidence interval. The best-checkpoint variant (97.3% on the same test set, in the run log) was chosen on the validation set; the headline uses the final epoch, as the original notebook did.
- **Overfitting.** Validation loss rises from 0.28 at epoch 1 to 0.80 at epoch 20.
- **Speed.** Notebook 3 calls the model once per face per frame and runs MTCNN on every frame (1.8 FPS here). The webcam script reaches 23.1 FPS on one desktop CPU by running MTCNN every 10 frames. It was timed on a video made of dataset photos (small faces, smooth motion, no motion blur), not on a camera, and a laptop will be slower. The picture pauses for about 0.25 s on each frame where MTCNN runs, so it is real-time in throughput but not perfectly smooth. Because of the shortcut, 6.6% of the faces that the per-frame baseline finds get a different label or no box.
- **Tracking.** Boxes between detections follow optical flow with shift and zoom only; fast motion, rotation or an occluded face can leave a box in the wrong place until the next detection.
- **Detector misses.** MTCNN can miss small faces (it found 5 of 9 on one dataset photo in the local demo check), and a missed face gets no label.
- **"Person Number"** in notebook 3's log counts detections per frame, not distinct people.
- **Faces of real people.** See the privacy note under Data.

## Future scope

- Group-aware cross-validation with confidence intervals instead of a single split.
- Class weighting or more `mask_weared_incorrect` examples, and a test on a second dataset.
- Run MTCNN in a background thread so the picture never pauses, test on a live webcam and a laptop CPU, and share one `predict()` function between the notebook, the Gradio demo and the webcam script (notebook 3 is still the 2022 code).
- A hosted, upload-only demo: the weights now have a public home, and the Space files are ready (see [Demo](#demo)); what is missing is a hosting plan that works for a free account.

## Credits & citation

- **Dataset.** Kaggle "Face Mask Detection" by andrewmvd (CC0), from the Make ML [Mask Dataset](https://makeml.app/datasets/mask).
- **MTCNN.** Zhang, K., Zhang, Z., Li, Z. and Qiao, Y., "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks", IEEE Signal Processing Letters, 2016. Implementation: [ipazc/mtcnn](https://github.com/ipazc/mtcnn) (MIT).
- **InceptionV3.** Szegedy, C. et al., "Rethinking the Inception Architecture for Computer Vision", CVPR 2016. ImageNet weights from Keras.
- **Third-party licenses.** Every third-party component and its license text is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original work © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0. Third-party components retain their own licenses; see THIRD_PARTY_NOTICES.md.

## Author

Jane Rose Ferdinant · [Portfolio](https://janeferdinant.vercel.app) · [LinkedIn](https://www.linkedin.com/in/janerosenf/)
