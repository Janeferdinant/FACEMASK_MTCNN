"""Make a short test video from photos of the dataset (CC0), so the webcam detector can be tried and timed without a camera.

Each photo is shown for a couple of seconds with a slow zoom and pan (a "Ken Burns" move), cropped to 4:3 and
resized to 640x480 at 30 frames per second, so the faces move smoothly across the frame like a hand-held camera.
It is a stand-in for a webcam: real footage has other lighting, motion blur and faces closer to the lens.

    python scripts/make_test_clip.py                  # writes data/benchmark_clip.mp4 (480 frames, 16 s)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Photos with 3 to 9 faces, mixed mask states; chosen by hand from data/annotations, in this order.
PHOTOS = (148, 131, 250, 327, 29, 346, 244, 249)
FRAME_SIZE = (640, 480)
FPS = 30


def ken_burns_frames(image: np.ndarray, frames: int, direction: int, zoom_end: float = 1.3):
    """Yield `frames` 640x480 crops of `image`: zooming in from the full 4:3 window while panning sideways."""
    height, width = image.shape[:2]
    win_w = min(width, round(height * 4 / 3))
    win_h = round(win_w * 3 / 4)
    for i in range(frames):
        t = i / max(1, frames - 1)
        zoom = 1.0 + (zoom_end - 1.0) * t
        cw, ch = round(win_w / zoom), round(win_h / zoom)
        slack_x = width - cw
        cx = round(slack_x * (t if direction > 0 else 1 - t))
        cy = round((height - ch) / 2)
        yield cv2.resize(image[cy : cy + ch, cx : cx + cw], FRAME_SIZE, interpolation=cv2.INTER_CUBIC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "benchmark_clip.mp4")
    parser.add_argument("--images", type=Path, default=REPO / "data" / "images")
    parser.add_argument("--frames-per-photo", type=int, default=60)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    total = 0
    for k, number in enumerate(PHOTOS):
        path = args.images / f"maksssksksss{number}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Cannot read {path}")
        for frame in ken_burns_frames(image, args.frames_per_photo, 1 if k % 2 == 0 else -1):
            writer.write(frame)
            total += 1
    writer.release()
    print(f"wrote {args.out.name} ({total} frames, {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, {FPS} fps, {len(PHOTOS)} photos)")


if __name__ == "__main__":
    main()
