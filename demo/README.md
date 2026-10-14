# demo: Gradio app for the face-mask pipeline

`app.py` runs the same pipeline as notebook 3 on an uploaded photo or a webcam snapshot: MTCNN finds the faces, each face is cropped to 100 x 100 and scaled to 0-1, and the InceptionV3 classifier labels it `with_mask`, `without_mask` or `mask_weared_incorrect`. The result is the photo with labelled boxes plus a table of labels, classifier probabilities and MTCNN confidences.

The app does not save images and displays no accuracy figures. Gradio keeps uploads in a temporary cache, which the app clears every 10 minutes. This is a teaching demo: it is not for safety-critical or surveillance use.

## Run it locally

Prerequisites: Python 3.11, about 2 GB of disk for TensorFlow, and an internet connection for the first start (the weights are downloaded once, about 190 MB).

```bash
git clone https://github.com/Janeferdinant/FACEMASK_MTCNN.git
cd FACEMASK_MTCNN
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r demo/requirements.txt
python demo/app.py
```

Open http://127.0.0.1:7860. Where the weights come from, in this order:

1. `--weights <file>` or the `FACEMASK_WEIGHTS` environment variable (a `.h5` or `.keras` file);
2. `data/inceptionV3-model.h5`, which notebook 2 writes when you train the model yourself;
3. otherwise the app downloads `inceptionV3-model.keras` from the public model repository [`JaneAaronNF/face-mask-inceptionv3-mtcnn`](https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn).

To run the pipeline without a browser and print the detections as JSON:

```bash
python demo/app.py --check data/images/maksssksksss3.png
```

Other options: `--host`, `--port` (defaults: `$GRADIO_SERVER_NAME` / `$GRADIO_SERVER_PORT`, else 127.0.0.1 and 7860).

## Webcam or video file

`webcam.py` is the live version of the same pipeline (see the main README for the measured speed):

```bash
python demo/webcam.py                          # default camera; q or Esc quits
python demo/webcam.py --source clip.mp4 --no-display --save out.mp4
```

MTCNN runs on every 10th frame (`--detect-every N`), optical flow moves the boxes in between, and all faces of a frame are classified in one batch. `realtime.py` holds the logic (frame skipping, tracking, FPS counter) and is covered by `tests/`. Weights are found as above. Nothing is stored unless you pass `--save`.

## The Hugging Face Space (not yet live)

The files for a public Gradio Space named `JaneAaronNF/face-mask-demo` are ready, but **the Space has not been created**: on 2026-10-14 Hugging Face refused (HTTP 402) because a Gradio Space on free `cpu-basic` hardware needs a PRO subscription on this account. See [`docs/runs/2026-10-14-space-check.md`](../docs/runs/2026-10-14-space-check.md) for the exact message and for a local rehearsal of the Space package.

How the Space is set up, so it can be created as soon as that is resolved:

| File | Role |
|---|---|
| `demo/build_space.py` | Assembles the Space folder and uploads it with `huggingface_hub`. It uses your saved Hugging Face login (`hf auth login`); never pass a token on a command line. |
| `app.py` (copy of `demo/app.py`) | Same app. With no local weights it downloads `inceptionV3-model.keras` from the model repository at start-up. |
| `demo/space/README.md` | The Space's configuration (Gradio 6.30.0, Python 3.11) and its page text: privacy, example images, the roughly 30-second first load, limitations, and "not for safety-critical or surveillance use". |
| `demo/space/requirements.txt` | Pinned for Linux and Python 3.11: TensorFlow 2.21.0, Keras 3.15.1, mtcnn 1.0.0, OpenCV (headless) 5.0.0.93, NumPy 2.4.6. |
| `examples/` | Three test-split photos from the CC0 dataset, copied from `data/images`. No other images are used. |

```bash
python demo/build_space.py --dry-run     # assemble and list the files, upload nothing
python demo/build_space.py               # create the Space (public, CPU basic) and upload
```

## Model weights

Public model repository: https://huggingface.co/JaneAaronNF/face-mask-inceptionv3-mtcnn (model card with the verified grouped-split result and the limitations). `inceptionV3-model.keras` there is the final epoch-20 model from the 2026-10-13 run.
