"""Face-mask detection demo: MTCNN face detector + transfer-learned InceptionV3 classifier.

It runs the same pipeline as notebook 3, but on an uploaded image (or a webcam snapshot) instead of a
live webcam loop:

    MTCNN finds the faces -> each face is cropped, resized to 100x100 RGB and scaled to 0-1
    -> the InceptionV3 classifier trained in notebook 2 labels it
       with_mask / without_mask / mask_weared_incorrect

The trained weights are not stored in git. There are three ways to get them:

    1. Do nothing: if no weights are given and data/inceptionV3-model.h5 does not exist, the app downloads
       inceptionV3-model.keras (about 190 MB) from the Hugging Face model repository
       JaneAaronNF/face-mask-inceptionv3-mtcnn.
    2. Train them with notebooks 1 and 2 (notebook 2 writes data/inceptionV3-model.h5).
    3. Point at any weights file:

           python demo/app.py --weights data/inceptionV3-model.h5
           FACEMASK_WEIGHTS=path/to/inceptionV3-model.h5 python demo/app.py

The same file is the app of the Hugging Face Space JaneAaronNF/face-mask-demo (see demo/README.md).

Check mode runs the pipeline on image files, prints the detections as JSON and exits without
starting a server:

    python demo/app.py --weights data/inceptionV3-model.h5 --check data/images/maksssksksss0.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO / "data" / "inceptionV3-model.h5"
WEIGHTS_ENV = "FACEMASK_WEIGHTS"

# Class ids from notebook 1 (label2category); box colours from notebook 3, converted from BGR to RGB.
LABELS = {0: "without_mask", 1: "with_mask", 2: "mask_weared_incorrect"}
COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (255, 255, 0)}
IMG_SIZE = (100, 100)

# Where to download the weights from when there are no local ones (the public model repository).
HF_REPO = "JaneAaronNF/face-mask-inceptionv3-mtcnn"
HF_FILENAME = "inceptionV3-model.keras"

# Example photos from the Kaggle "Face Mask Detection" dataset (CC0). They are looked up in an
# `examples/` folder next to this file (the Hugging Face Space) or in data/images (this repository).
# All three are in the held-out test split of the 2026-10-13 re-run.
EXAMPLE_NAMES = ("maksssksksss3.png", "maksssksksss124.png", "maksssksksss250.png")
EXAMPLE_DIRS = (Path(__file__).resolve().parent / "examples", REPO / "data" / "images")
EXAMPLE_IMAGES = next(
    ([d / n for n in EXAMPLE_NAMES] for d in EXAMPLE_DIRS if all((d / n).is_file() for n in EXAMPLE_NAMES)), []
)


def resolve_weights(cli_value: str | None) -> Path:
    """Weights from --weights, else $FACEMASK_WEIGHTS, else data/inceptionV3-model.h5, else the Hub copy."""
    value = cli_value or os.environ.get(WEIGHTS_ENV)
    if value:
        path = Path(value).expanduser()
        if not path.is_file():
            sys.exit(f"Model weights not found: {path}")
        return path
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    print(f"No local weights at {DEFAULT_WEIGHTS}; downloading {HF_FILENAME} from {HF_REPO} ...", file=sys.stderr)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(HF_REPO, HF_FILENAME))
    except Exception as error:  # offline, repo unavailable, huggingface_hub missing
        sys.exit(
            f"Could not download the weights ({error}).\n"
            f"Train them with notebooks 1 and 2, then pass --weights <file> or set {WEIGHTS_ENV}."
        )


class Pipeline:
    """MTCNN detection followed by InceptionV3 classification of every detected face."""

    def __init__(self, weights: str | os.PathLike):
        # Imported here so that `--help` and a missing-weights error stay fast.
        from keras.models import load_model
        from mtcnn import MTCNN

        self.detector = MTCNN()
        self.model = load_model(str(weights), compile=False)

    def predict(self, rgb: np.ndarray) -> list[dict]:
        """Return one dict per detected face: box (x, y, w, h), label, score and MTCNN confidence."""
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an RGB image with shape (height, width, 3)")
        height, width = rgb.shape[:2]
        boxes, crops = [], []
        for face in self.detector.detect_faces(rgb):
            x, y, w, h = (int(v) for v in face["box"])
            # Keep the box inside the image (mtcnn >= 1.0 already clips; this guards older versions).
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w <= 1 or h <= 1:
                continue
            roi = rgb[y : y + h, x : x + w]
            crops.append(cv2.resize(roi, IMG_SIZE).astype("float32") / 255.0)
            boxes.append(((x, y, w, h), float(face["confidence"])))
        if not crops:
            return []
        # One batched call instead of notebook 3's call per face; the model and inputs are the same.
        scores = self.model.predict(np.stack(crops), verbose=0)
        results = []
        for ((x, y, w, h), confidence), row in zip(boxes, scores, strict=True):
            target = int(np.argmax(row))
            results.append(
                {
                    "box": [x, y, w, h],
                    "label": LABELS[target],
                    "score": round(float(row[target]), 4),
                    "face_confidence": round(confidence, 4),
                }
            )
        return results

    @staticmethod
    def annotate(rgb: np.ndarray, faces: list[dict]) -> np.ndarray:
        """Draw notebook 3's labelled boxes on a copy of the image."""
        out = rgb.copy()
        thickness = max(1, round(max(out.shape[:2]) / 400))
        font_scale = max(0.4, max(out.shape[:2]) / 1200)
        ids = {name: i for i, name in LABELS.items()}
        for face in faces:
            x, y, w, h = face["box"]
            color = COLORS[ids[face["label"]]]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
            text = f"{face['label']}: {face['score']:.2f}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            top = max(0, y - th - base - 4)
            cv2.rectangle(out, (x, top), (x + tw + 4, top + th + base + 4), color, -1)
            cv2.putText(out, text, (x + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return out


ABOUT = """
**Model:** MTCNN face detector (pretrained, from the `mtcnn` package) followed by an InceptionV3
classifier transfer-learned on the Kaggle *Face Mask Detection* dataset (CC0).
It labels each face `with_mask`, `without_mask` or `mask_weared_incorrect`.

**Please note**
- **Privacy.** Images are processed in memory and the app never stores them. (Gradio briefly keeps an
  upload in a temporary cache, which this app clears every 10 minutes.)
- **Examples.** The example images come only from the licensed (CC0) dataset above. It shows real
  people's faces; CC0 does not say whether they consented.
- **First load.** The first request after the app has been idle can take about 30 seconds while it wakes up.
- **Results are imperfect.** MTCNN can miss faces (it found 5 of 9 in one test photo), and the
  `mask_weared_incorrect` class has only 12 test crops and an F1 of 0.70. Photos unlike the training data
  will give more mistakes.
- **Not for safety-critical or surveillance use.** This is a teaching demo.
"""

HOW_IT_WORKS = """
1. **Detect.** MTCNN, a cascade of three small CNNs (P-Net, R-Net, O-Net), proposes face boxes and keeps
   the confident ones.
2. **Crop.** Each box is cut out of the image, resized to 100 x 100 pixels and scaled to the 0-1 range,
   exactly as the training crops were prepared in notebook 1.
3. **Classify.** InceptionV3 with frozen ImageNet weights (up to layer `mixed7`) extracts features; a small
   trained head (Dense 1024 -> Dropout 0.3 -> Dense 3, softmax) gives a probability per class.
   The box shows the most likely class and its probability.

The classifier saw only faces from this one dataset, so expect mistakes on very different photos
(lighting, angles, skin tones, mask styles), and it is least reliable on `mask_weared_incorrect`,
the rarest class.
"""


def build_ui(pipe: Pipeline):
    import gradio as gr

    def run(image):
        if image is None:
            raise gr.Error("Please upload an image or take a webcam snapshot first.")
        faces = pipe.predict(image)
        rows = [
            [i + 1, f["label"], f["score"], f["face_confidence"], ", ".join(map(str, f["box"]))]
            for i, f in enumerate(faces)
        ]
        if not faces:
            gr.Info("No faces found in this image.")
        return pipe.annotate(image, faces), rows

    with gr.Blocks(title="Face-mask detection: MTCNN + InceptionV3", delete_cache=(600, 600)) as ui:
        gr.Markdown("# Face-mask detection: MTCNN + InceptionV3")
        gr.Markdown(ABOUT)
        with gr.Row():
            image_in = gr.Image(
                type="numpy", image_mode="RGB", sources=["upload", "webcam", "clipboard"], label="Photo"
            )
            image_out = gr.Image(type="numpy", label="Detected faces", interactive=False)
        button = gr.Button("Detect masks", variant="primary")
        table = gr.Dataframe(
            headers=["face", "label", "classifier probability", "MTCNN confidence", "box (x, y, w, h)"],
            label="Faces",
            interactive=False,
        )
        button.click(run, inputs=image_in, outputs=[image_out, table])
        examples = [[str(p)] for p in EXAMPLE_IMAGES if p.is_file()]
        if examples:
            gr.Examples(examples=examples, inputs=image_in, label="Examples from the dataset")
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW_IT_WORKS)
    return ui


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help=f"trained Keras model (.h5 or .keras); default: ${WEIGHTS_ENV}, else {DEFAULT_WEIGHTS.relative_to(REPO)}, else the Hugging Face copy")
    parser.add_argument("--check", nargs="+", metavar="IMAGE", help="run on these image files, print JSON and exit")
    parser.add_argument("--host", default=None, help="server address (default: $GRADIO_SERVER_NAME, else 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="server port (default: $GRADIO_SERVER_PORT, else 7860)")
    args = parser.parse_args(argv)

    pipe = Pipeline(resolve_weights(args.weights))

    if args.check:
        report = []
        for name in args.check:
            bgr = cv2.imread(name, cv2.IMREAD_COLOR)
            if bgr is None:
                sys.exit(f"Cannot read image: {name}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            report.append({"image": Path(name).name, "size": [rgb.shape[1], rgb.shape[0]], "faces": pipe.predict(rgb)})
        print(json.dumps(report, indent=2))
        return

    host = args.host or os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = args.port or int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    build_ui(pipe).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
