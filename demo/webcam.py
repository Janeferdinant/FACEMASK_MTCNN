"""Webcam / video face-mask detector: MTCNN + transfer-learned InceptionV3, tuned to run fast on a CPU.

    python demo/webcam.py                                  # default webcam (device 0), press q or Esc to quit
    python demo/webcam.py --source 1                       # another camera
    python demo/webcam.py --source clip.mp4 --save out.mp4 --no-display

What makes it fast (see demo/realtime.py and docs/runs/2026-10-14-fps.md for the measurements):

    * MTCNN is the slow stage, so it runs on every N-th frame only (--detect-every, default 10); in between,
      the boxes are moved with optical flow and keep their last label.
    * MTCNN looks at a copy of the frame no wider than --detect-width pixels (default 640).
    * All faces of a frame are classified in one batched call (`predict_on_batch`). Notebook 3 calls
      `model.predict` once per face, which costs about 0.1 s per call on its own.

The detector (MTCNN) and the classifier (InceptionV3, same weights as the Gradio demo) are the pipeline of
notebook 3. What changes is how often they run. Faces that appear between two detections are picked up at
the next detection, and a face that changes its mask state is relabelled at the next detection too.

The trained weights are not in git; see demo/app.py for how they are found or downloaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from app import COLORS, LABELS, resolve_weights
from realtime import IMG_SIZE, FpsCounter, StreamPipeline, percentile

DEFAULT_DETECT_EVERY = 10
DEFAULT_DETECT_WIDTH = 640
DEFAULT_FRAME_WIDTH = 640


def load_models(weights):
    """Load MTCNN and the classifier."""
    from keras.models import load_model
    from mtcnn import MTCNN

    return MTCNN(), load_model(str(weights), compile=False)


def make_detect(detector, detect_width: int = DEFAULT_DETECT_WIDTH):
    """A detect(rgb) function: MTCNN on a copy no wider than `detect_width`, boxes mapped back to the frame."""

    def detect(rgb: np.ndarray):
        height, width = rgb.shape[:2]
        scale = 1.0
        if detect_width and width > detect_width:
            scale = detect_width / width
            rgb = cv2.resize(rgb, (detect_width, round(height * scale)), interpolation=cv2.INTER_AREA)
        return [(x / scale, y / scale, w / scale, h / scale) for x, y, w, h in (f["box"] for f in detector.detect_faces(rgb))]

    return detect


def make_classify(model):
    """A classify(crops) function: one batched call for all faces of a frame."""

    def classify(crops: np.ndarray) -> np.ndarray:
        return model.predict_on_batch(crops)

    return classify


def load_components(weights, detect_width: int = DEFAULT_DETECT_WIDTH):
    """Load the models; return (detect, classify) functions for StreamPipeline."""
    detector, model = load_models(weights)
    return make_detect(detector, detect_width), make_classify(model)


def warm_up(detect, classify, frame_shape=(480, 640, 3)) -> None:
    """Run both models once so TensorFlow's first-call tracing is not counted in the FPS."""
    detect(np.zeros(frame_shape, dtype=np.uint8))
    classify(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32))


def build_pipeline(weights=None, detect_every=DEFAULT_DETECT_EVERY, detect_width=DEFAULT_DETECT_WIDTH, classify_every=None):
    """Load the models and return a warmed-up StreamPipeline."""
    detect, classify = load_components(resolve_weights(weights), detect_width)
    warm_up(detect, classify)
    return StreamPipeline(detect, classify, detect_every=detect_every, classify_every=classify_every)


def open_source(source: str) -> cv2.VideoCapture:
    """A camera index ("0", "1", ...) or a video file path."""
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        sys.exit(f"Cannot open video source: {source}")
    return capture


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if not width or frame.shape[1] == width:
        return frame
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_faces(frame_bgr: np.ndarray, faces, fps: float | None = None) -> np.ndarray:
    """Draw labelled boxes (notebook 3's colours) and, if given, the measured FPS on the frame, in place."""
    for face in faces:
        x, y, w, h = face.box
        if face.label is None:
            color, text = (200, 200, 200), "..."
        else:
            r, g, b = COLORS[face.label]
            color, text = (b, g, r), f"{LABELS[face.label]}: {face.score:.2f}"
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y - th - base - 4)
        cv2.rectangle(frame_bgr, (x, top), (x + tw + 4, top + th + base + 4), color, -1)
        cv2.putText(frame_bgr, text, (x + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help='camera index ("0") or a video file path (default: 0)')
    parser.add_argument("--weights", help="trained Keras model (.h5 or .keras); default as in demo/app.py")
    parser.add_argument("--detect-every", type=int, default=DEFAULT_DETECT_EVERY, metavar="N",
                        help=f"run MTCNN on every N-th frame (default {DEFAULT_DETECT_EVERY}; 1 = every frame)")
    parser.add_argument("--classify-every", type=int, default=None, metavar="M",
                        help="also re-classify tracked faces on every M-th frame (default: only when MTCNN runs)")
    parser.add_argument("--detect-width", type=int, default=DEFAULT_DETECT_WIDTH, metavar="PX",
                        help=f"MTCNN sees a copy no wider than this (default {DEFAULT_DETECT_WIDTH}; 0 = full frame)")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, metavar="PX",
                        help=f"resize frames to this width before processing (default {DEFAULT_FRAME_WIDTH}; 0 = keep)")
    parser.add_argument("--mirror", action="store_true", help="flip frames horizontally (like a mirror), as notebook 3 does")
    parser.add_argument("--no-display", action="store_true", help="do not open a window (headless / benchmarking)")
    parser.add_argument("--save", metavar="PATH", help="write the annotated video to this .mp4 file")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (default: until the end)")
    args = parser.parse_args(argv)

    pipe = build_pipeline(args.weights, args.detect_every, args.detect_width, args.classify_every)
    capture = open_source(args.source)
    counter, latencies, writer = FpsCounter(), [], None
    try:
        while args.max_frames <= 0 or counter.frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame = resize_to_width(frame, args.width)
            if args.mirror:
                frame = cv2.flip(frame, 1)
            start = time.perf_counter()
            faces = pipe.process(frame)
            latencies.append(time.perf_counter() - start)
            counter.tick()
            draw_faces(frame, faces, counter.fps)
            if args.save:
                if writer is None:
                    fps_out = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, frame.shape[1::-1])
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Face mask detection (q or Esc to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    if counter.frames:
        print(
            f"{counter.frames} frames, mean {counter.mean_fps:.1f} FPS, pipeline latency "
            f"mean {1000 * float(np.mean(latencies)):.0f} ms, p95 {1000 * percentile(latencies, 95):.0f} ms "
            f"(detect every {args.detect_every}, detect width {args.detect_width or 'full'}, {Path(args.source).name})"
        )


if __name__ == "__main__":
    main()
