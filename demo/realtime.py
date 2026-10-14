"""Frame-skipping, box tracking and FPS measurement for the webcam detector (no TensorFlow needed).

MTCNN is the slow part of the pipeline (about 0.2 s for a 640x480 frame on a desktop CPU). Running it on every
frame caps the loop at about 5 FPS, so the streaming pipeline runs it only on every N-th frame and, in between,
moves the boxes it already has with Lucas-Kanade optical flow:

    frame i % N == 0   detect faces (MTCNN) -> match them to the existing tracks -> classify all crops in ONE batch
    other frames       move each track's box with optical flow; keep its last label
    scene cut          a large change between two frames (a cut in a video, the camera covered) forces a detection

The detector and the classifier are passed in as plain functions, so this module can be tested with fakes
(see tests/test_realtime.py) and reused by demo/webcam.py and scripts/benchmark_fps.py.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

IMG_SIZE = (100, 100)  # classifier input, as in notebooks 1-3
SCENE_CUT_THRESHOLD = 35.0  # mean grey-level change (0-255) between 32x24 thumbnails of two frames that counts as a cut
MAX_SCALE_STEP = 1.25  # a tracked box may grow or shrink by at most this factor from one frame to the next

Box = tuple[float, float, float, float]  # x, y, width, height


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation; 0.0 for an empty list."""
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


class FpsCounter:
    """Measured frames per second: over the last `window` frames (`fps`) and since the start (`mean_fps`).

    Call `tick()` once per frame. Both rates are 0.0 until two frames have been seen. The clock can be replaced
    for tests.
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 frames")
        self._clock = clock
        self._stamps: deque[float] = deque(maxlen=window)
        self._first: float | None = None
        self._last: float | None = None
        self.frames = 0

    def tick(self) -> None:
        now = self._clock()
        if self._first is None:
            self._first = now
        self._last = now
        self._stamps.append(now)
        self.frames += 1

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    @property
    def mean_fps(self) -> float:
        if self.frames < 2 or self._first is None or self._last is None:
            return 0.0
        span = self._last - self._first
        return (self.frames - 1) / span if span > 0 else 0.0


@dataclass(frozen=True)
class FramePlan:
    detect: bool
    classify: bool


class FrameSkipper:
    """Decides which frames run the face detector and which re-run the classifier.

    The detector runs on frames 0, N, 2N, ... Detection frames always classify. `classify_every=M` also
    re-classifies the tracked boxes on every M-th frame in between (default: only on detection frames).
    """

    def __init__(self, detect_every: int = 5, classify_every: int | None = None) -> None:
        if detect_every < 1:
            raise ValueError("detect_every must be at least 1")
        if classify_every is not None and classify_every < 1:
            raise ValueError("classify_every must be at least 1")
        self.detect_every = detect_every
        self.classify_every = classify_every

    def plan(self, frame_index: int) -> FramePlan:
        detect = frame_index % self.detect_every == 0
        classify = detect or (self.classify_every is not None and frame_index % self.classify_every == 0)
        return FramePlan(detect=detect, classify=classify)


@dataclass
class Track:
    """One face followed across frames."""

    id: int
    box: np.ndarray  # float32 (x, y, w, h)
    label: int | None = None
    score: float = 0.0
    misses: int = 0  # detection frames in a row on which no detection matched this track
    fresh: bool = False  # True when `box` came from the detector on the current frame
    meta: dict = field(default_factory=dict)


class BoxTracker:
    """Keeps face identities across detections and moves boxes between detections with optical flow.

    `update(detections)` greedily pairs detections with tracks by IoU (highest first, at least `iou_threshold`).
    A matched track takes the new box and keeps its label. An unmatched detection starts a new track. A track
    that goes unmatched on more than `max_misses` detection frames in a row is dropped.

    `advance(prev_gray, gray)` follows corner points inside every box with Lucas-Kanade optical flow and moves and
    rescales the box by the similarity transform (shift and zoom, no rotation) that fits them. A box with too few
    trackable points (a flat region) stays where it is.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 1, min_points: int = 4) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_points = min_points
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Forget all tracks (their ids are not reused)."""
        self.tracks = []

    def update(self, detections: Sequence[Box]) -> list[Track]:
        pairs = sorted(
            (
                (iou(track.box, det), ti, di)
                for ti, track in enumerate(self.tracks)
                for di, det in enumerate(detections)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for overlap, ti, di in pairs:
            if overlap < self.iou_threshold:
                break
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = self.tracks[ti]
            track.box = np.asarray(detections[di], dtype=np.float32)
            track.misses = 0
            track.fresh = True
        survivors = []
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.misses += 1
                track.fresh = False
            if track.misses <= self.max_misses:
                survivors.append(track)
        for di, det in enumerate(detections):
            if di not in used_dets:
                survivors.append(Track(id=self._next_id, box=np.asarray(det, dtype=np.float32), fresh=True))
                self._next_id += 1
        self.tracks = survivors
        return self.tracks

    def advance(self, prev_gray: np.ndarray, gray: np.ndarray) -> None:
        """Move every track from the previous grayscale frame to the current one."""
        height, width = gray.shape[:2]
        for track in self.tracks:
            track.fresh = False
            track.box = clamp_box(track.box, width, height)
            x, y, w, h = (int(round(float(v))) for v in track.box)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            mask = np.zeros(prev_gray.shape[:2], dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.01, minDistance=3, mask=mask)
            if points is None or len(points) < self.min_points:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, winSize=(21, 21), maxLevel=2)
            if moved is None:
                continue
            ok = status.reshape(-1) == 1
            if int(ok.sum()) < self.min_points:
                continue
            track.box = clamp_box(self._moved_box(track.box, points[ok].reshape(-1, 2), moved[ok].reshape(-1, 2)), width, height)

    def _moved_box(self, box: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Move and rescale `box` by the motion of the points: a similarity transform, else the median shift."""
        x, y, w, h = (float(v) for v in box)
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if matrix is not None and int(inliers.sum()) >= self.min_points:
            scale = float(np.clip(np.hypot(matrix[0, 0], matrix[1, 0]), 1 / MAX_SCALE_STEP, MAX_SCALE_STEP))
            cx, cy = matrix @ np.array([x + w / 2, y + h / 2, 1.0])
            w, h = w * scale, h * scale
            return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)
        dx, dy = np.median(dst - src, axis=0)
        return np.array([x + dx, y + dy, w, h], dtype=np.float32)


def clamp_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    """Move a box so it lies inside a width x height frame (shrinking it only if it is larger than the frame)."""
    x, y, w, h = (float(v) for v in box)
    w, h = min(w, width), min(h, height)
    x = min(max(0.0, x), width - w)
    y = min(max(0.0, y), height - h)
    return np.array([x, y, w, h], dtype=np.float32)


def crop_batch(rgb: np.ndarray, boxes: Sequence[Sequence[float]]) -> tuple[np.ndarray, list[int]]:
    """Cut every box out of an RGB frame, resize to 100x100 and scale to 0-1, as in the training crops.

    Returns the (N, 100, 100, 3) float32 batch and the indices of the boxes that were kept (boxes thinner
    than 2 pixels are skipped).
    """
    height, width = rgb.shape[:2]
    crops, kept = [], []
    for index, box in enumerate(boxes):
        x, y, w, h = (round(float(v)) for v in clamp_box(box, width, height))
        if w <= 1 or h <= 1:
            continue
        crops.append(cv2.resize(rgb[y : y + h, x : x + w], IMG_SIZE).astype("float32") / 255.0)
        kept.append(index)
    batch = np.stack(crops) if crops else np.empty((0, *IMG_SIZE, 3), dtype=np.float32)
    return batch, kept


@dataclass(frozen=True)
class Face:
    """One labelled face on the current frame."""

    track_id: int
    box: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    label: int | None  # class id, or None if the face has not been classified yet
    score: float
    fresh: bool  # the box comes from the detector on this very frame (not tracked)


DetectFn = Callable[[np.ndarray], Sequence[Box]]  # RGB frame -> face boxes (x, y, w, h) in frame pixels
ClassifyFn = Callable[[np.ndarray], np.ndarray]  # (N, 100, 100, 3) float32 in 0-1 -> (N, 3) class scores


class StreamPipeline:
    """Webcam/video pipeline: detect every N-th frame, track in between, classify all crops in one batch."""

    def __init__(
        self,
        detect: DetectFn,
        classify: ClassifyFn,
        detect_every: int = 5,
        classify_every: int | None = None,
        tracker: BoxTracker | None = None,
        scene_cut_threshold: float | None = SCENE_CUT_THRESHOLD,
    ) -> None:
        self.detect = detect
        self.classify = classify
        self.skipper = FrameSkipper(detect_every, classify_every)
        self.tracker = tracker or BoxTracker()
        self.scene_cut_threshold = scene_cut_threshold
        self.frame_index = 0
        self._prev_gray: np.ndarray | None = None
        self._prev_thumb: np.ndarray | None = None

    def process(self, frame_bgr: np.ndarray) -> list[Face]:
        """Return the labelled faces on one BGR frame."""
        plan = self.skipper.plan(self.frame_index)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._is_scene_cut(gray):  # a different picture: old boxes are meaningless, so look again now
            self.tracker.reset()
            plan = FramePlan(detect=True, classify=True)
        if self._prev_gray is not None and self.tracker.tracks and self._prev_gray.shape == gray.shape:
            self.tracker.advance(self._prev_gray, gray)
        if plan.detect:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.tracker.update([tuple(map(float, box)) for box in self.detect(rgb)])
        if plan.classify:
            self._classify_tracks(frame_bgr)
        self._prev_gray = gray
        self.frame_index += 1
        height, width = frame_bgr.shape[:2]
        faces = []
        for track in self.tracker.tracks:
            x, y, w, h = clamp_box(track.box, width, height)
            faces.append(
                Face(track.id, (int(x), int(y), int(round(w)), int(round(h))), track.label, track.score, track.fresh)
            )
        return faces

    def _is_scene_cut(self, gray: np.ndarray) -> bool:
        thumb = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        previous, self._prev_thumb = self._prev_thumb, thumb
        if self.scene_cut_threshold is None or previous is None:
            return False
        return float(np.abs(thumb - previous).mean()) > self.scene_cut_threshold

    def _classify_tracks(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crops, kept = crop_batch(rgb, [track.box for track in self.tracker.tracks])
        if not kept:
            return
        scores = np.asarray(self.classify(crops))  # ONE call for every face on the frame
        for index, row in zip(kept, scores, strict=True):
            track = self.tracker.tracks[index]
            track.label = int(np.argmax(row))
            track.score = float(row[track.label])
