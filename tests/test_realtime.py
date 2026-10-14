"""Tests for demo/realtime.py: FPS counter, frame skipping, box tracking and the streaming pipeline.

Frames are synthetic (random textures and constants) and the detector and classifier are fakes, so the tests
need neither TensorFlow nor the trained weights.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from realtime import BoxTracker, FpsCounter, FrameSkipper, StreamPipeline, crop_batch, iou, percentile


class FakeClock:
    """A clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def textured_frame(shift: tuple[int, int] = (0, 0), size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """A BGR frame of blurred random texture, rolled by `shift` = (dx, dy) pixels."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(size[0], size[1]), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    rolled = np.roll(base, shift=(shift[1], shift[0]), axis=(0, 1))
    return cv2.cvtColor(rolled, cv2.COLOR_GRAY2BGR)


def zoomed_frame(scale: float, size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """The textured frame zoomed by `scale` about its centre, as if the camera moved closer."""
    base = textured_frame(size=size)
    matrix = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), 0.0, scale)
    return cv2.warpAffine(base, matrix, (size[1], size[0]), borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- iou and percentile


def test_iou_identical_disjoint_and_half_overlap():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    # two 10x10 boxes overlapping by 5x10: intersection 50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_degenerate_box_is_zero():
    assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_percentile_interpolates_and_handles_empty():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert percentile([], 95) == 0.0


def test_crop_batch_resizes_scales_and_skips_thin_boxes():
    rgb = np.full((100, 200, 3), 255, dtype=np.uint8)
    batch, kept = crop_batch(rgb, [(10, 10, 50, 40), (0, 0, 1, 30), (150, 60, 90, 90)])
    assert kept == [0, 2]  # the 1-pixel-wide box is skipped; the third box is pulled back inside the frame
    assert batch.shape == (2, 100, 100, 3) and batch.dtype == np.float32
    assert batch.min() == pytest.approx(1.0) and batch.max() == pytest.approx(1.0)
    empty, none = crop_batch(rgb, [])
    assert empty.shape == (0, 100, 100, 3) and none == []


# ---------------------------------------------------------------- FpsCounter


def test_fps_counter_is_zero_before_two_ticks():
    clock = FakeClock()
    counter = FpsCounter(clock=clock)
    assert counter.fps == 0.0 and counter.mean_fps == 0.0
    counter.tick()
    assert counter.fps == 0.0 and counter.mean_fps == 0.0
    assert counter.frames == 1


def test_fps_counter_steady_rate():
    clock = FakeClock()
    counter = FpsCounter(window=10, clock=clock)
    for _ in range(21):
        counter.tick()
        clock.advance(0.05)  # 20 frames per second
    assert counter.fps == pytest.approx(20.0)
    assert counter.mean_fps == pytest.approx(20.0)


def test_fps_counter_window_follows_recent_rate_while_mean_keeps_history():
    clock = FakeClock()
    counter = FpsCounter(window=5, clock=clock)
    for _ in range(20):  # slow start: 10 FPS
        counter.tick()
        clock.advance(0.1)
    for _ in range(20):  # then fast: 50 FPS
        counter.tick()
        clock.advance(0.02)
    assert counter.fps == pytest.approx(50.0)
    assert 10.0 < counter.mean_fps < 50.0


def test_fps_counter_rejects_tiny_window():
    with pytest.raises(ValueError):
        FpsCounter(window=1)


# ---------------------------------------------------------------- FrameSkipper


def test_skipper_detects_on_first_frame_and_every_nth():
    skipper = FrameSkipper(detect_every=5)
    detects = [i for i in range(21) if skipper.plan(i).detect]
    assert detects == [0, 5, 10, 15, 20]


def test_skipper_every_frame_when_n_is_one():
    skipper = FrameSkipper(detect_every=1)
    assert all(skipper.plan(i).detect for i in range(10))


def test_skipper_detect_frames_always_classify():
    skipper = FrameSkipper(detect_every=4, classify_every=3)
    for i in range(30):
        plan = skipper.plan(i)
        if plan.detect:
            assert plan.classify
    classified = [i for i in range(13) if skipper.plan(i).classify]
    assert classified == [0, 3, 4, 6, 8, 9, 12]  # multiples of 3 plus detect frames (multiples of 4)


def test_skipper_classify_defaults_to_detect_frames_only():
    skipper = FrameSkipper(detect_every=5)
    assert [i for i in range(11) if skipper.plan(i).classify] == [0, 5, 10]


@pytest.mark.parametrize("bad", [0, -3])
def test_skipper_rejects_non_positive_intervals(bad):
    with pytest.raises(ValueError):
        FrameSkipper(detect_every=bad)
    with pytest.raises(ValueError):
        FrameSkipper(detect_every=5, classify_every=bad)


# ---------------------------------------------------------------- BoxTracker


def test_tracker_creates_tracks_with_distinct_ids():
    tracker = BoxTracker()
    tracks = tracker.update([(10, 10, 40, 40), (200, 100, 50, 50)])
    assert len(tracks) == 2
    assert {t.id for t in tracks} == {1, 2}
    assert all(t.label is None for t in tracks)


def test_tracker_keeps_id_and_label_for_overlapping_detection():
    tracker = BoxTracker()
    (track,) = tracker.update([(10, 10, 40, 40)])
    track.label, track.score = 1, 0.9
    (again,) = tracker.update([(14, 12, 40, 40)])  # moved a little
    assert again.id == track.id
    assert again.label == 1 and again.score == 0.9
    assert tuple(again.box) == (14, 12, 40, 40)  # snaps to the new detection
    assert again.fresh


def test_tracker_new_far_detection_is_a_new_track():
    tracker = BoxTracker()
    tracker.update([(10, 10, 40, 40)])
    tracks = tracker.update([(10, 10, 40, 40), (200, 150, 40, 40)])
    assert sorted(t.id for t in tracks) == [1, 2]


def test_tracker_drops_a_track_after_max_misses():
    tracker = BoxTracker(max_misses=1)
    tracker.update([(10, 10, 40, 40)])
    assert len(tracker.update([])) == 1  # one missed detection is tolerated
    assert tracker.update([]) == []  # the second removes it


def test_tracker_matches_each_detection_to_at_most_one_track():
    tracker = BoxTracker()
    tracker.update([(10, 10, 40, 40), (30, 10, 40, 40)])  # two overlapping tracks
    tracks = tracker.update([(20, 10, 40, 40)])  # one detection between them
    assert len(tracks) == 2
    assert sum(t.fresh for t in tracks) == 1
    assert sum(t.misses for t in tracks) == 1


def test_tracker_advance_follows_a_shifted_texture():
    tracker = BoxTracker()
    prev, cur = textured_frame((0, 0)), textured_frame((6, 3))
    (track,) = tracker.update([(100, 80, 80, 80)])
    tracker.advance(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    x, y, w, h = track.box
    assert (x, y) == (pytest.approx(106, abs=1.5), pytest.approx(83, abs=1.5))
    assert (w, h) == (80, 80)
    assert not track.fresh


def test_tracker_advance_follows_a_zoom():
    tracker = BoxTracker()
    prev, cur = textured_frame(), zoomed_frame(1.1)
    (track,) = tracker.update([(120, 80, 80, 80)])  # centred on the zoom centre (160, 120)
    tracker.advance(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    x, y, w, h = track.box
    assert w == pytest.approx(88, abs=3) and h == pytest.approx(88, abs=3)
    assert (x + w / 2, y + h / 2) == (pytest.approx(160, abs=2), pytest.approx(120, abs=2))


def test_tracker_advance_limits_the_scale_change_per_frame():
    tracker = BoxTracker()
    prev, cur = textured_frame(), zoomed_frame(1.6)  # a jump no real camera makes in one frame
    (track,) = tracker.update([(120, 80, 80, 80)])
    tracker.advance(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    assert track.box[2] <= 80 * 1.25 + 1e-3


def test_tracker_advance_keeps_box_on_a_featureless_frame():
    tracker = BoxTracker()
    flat = np.full((240, 320), 128, dtype=np.uint8)
    (track,) = tracker.update([(100, 80, 80, 80)])
    tracker.advance(flat, flat)
    assert tuple(track.box) == (100, 80, 80, 80)


def test_tracker_advance_clamps_boxes_to_the_frame():
    tracker = BoxTracker()
    flat = np.full((100, 100), 128, dtype=np.uint8)
    (track,) = tracker.update([(90, 90, 40, 40)])  # sticks out of a 100x100 frame
    tracker.advance(flat, flat)
    x, y, w, h = track.box
    assert x + w <= 100 and y + h <= 100


# ---------------------------------------------------------------- StreamPipeline


class FakeDetector:
    """Returns one fixed face, moving 2 px right on every call, and counts its calls."""

    def __init__(self, faces_per_frame: int = 2) -> None:
        self.calls = 0
        self.faces_per_frame = faces_per_frame

    def __call__(self, rgb: np.ndarray):
        self.calls += 1
        return [(20 + 150 * i + 2 * self.calls, 40, 60, 60) for i in range(self.faces_per_frame)]


class FakeClassifier:
    """Predicts class 1 for every crop; records the batch sizes it was called with."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(len(crops))
        assert crops.shape[1:] == (100, 100, 3)
        assert crops.dtype == np.float32 and 0.0 <= crops.min() and crops.max() <= 1.0
        scores = np.zeros((len(crops), 3), dtype=np.float32)
        scores[:, 1] = 0.8
        scores[:, 0] = 0.2
        return scores


def mocked_frames(count: int):
    """`count` frames of a static scene (a still camera), so optical flow finds no motion."""
    for _ in range(count):
        yield textured_frame(size=(240, 400))


def test_pipeline_runs_detector_only_every_nth_frame():
    detector, classifier = FakeDetector(), FakeClassifier()
    pipe = StreamPipeline(detector, classifier, detect_every=5)
    for frame in mocked_frames(20):
        pipe.process(frame)
    assert detector.calls == 4  # frames 0, 5, 10, 15
    assert pipe.frame_index == 20


def test_pipeline_classifies_all_faces_in_one_batch_per_detection():
    detector, classifier = FakeDetector(faces_per_frame=3), FakeClassifier()
    pipe = StreamPipeline(detector, classifier, detect_every=5)
    for frame in mocked_frames(11):
        pipe.process(frame)
    assert classifier.batch_sizes == [3, 3, 3]  # frames 0, 5, 10: one batch of 3, never three batches of 1


def test_pipeline_keeps_labels_on_frames_between_detections():
    pipe = StreamPipeline(FakeDetector(), FakeClassifier(), detect_every=5)
    outputs = [pipe.process(frame) for frame in mocked_frames(6)]
    assert [f.label for f in outputs[0]] == [1, 1]
    for faces in outputs[1:5]:  # frames 1..4: no detector call, boxes tracked, labels kept
        assert len(faces) == 2
        assert all(f.label == 1 and not f.fresh for f in faces)
    assert all(f.fresh for f in outputs[5])
    assert [f.track_id for f in outputs[5]] == [f.track_id for f in outputs[0]]  # same identities


def test_pipeline_extra_classification_between_detections():
    classifier = FakeClassifier()
    pipe = StreamPipeline(FakeDetector(), classifier, detect_every=6, classify_every=2)
    for frame in mocked_frames(7):
        pipe.process(frame)
    # frames 0 (detect), 2, 4 (re-classify tracked boxes), 6 (detect)
    assert classifier.batch_sizes == [2, 2, 2, 2]


def test_pipeline_without_faces_does_not_call_the_classifier():
    detector, classifier = FakeDetector(faces_per_frame=0), FakeClassifier()
    pipe = StreamPipeline(detector, classifier, detect_every=3)
    for frame in mocked_frames(7):
        assert pipe.process(frame) == []
    assert classifier.batch_sizes == []


def test_pipeline_skips_degenerate_boxes_and_clamps_to_the_frame():
    def detector(_rgb):
        return [(380, 220, 100, 100), (-30, -30, 60, 60), (10, 10, 1, 1)]

    classifier = FakeClassifier()
    pipe = StreamPipeline(detector, classifier, detect_every=1)
    faces = pipe.process(next(mocked_frames(1)))
    assert classifier.batch_sizes == [2]  # the 1x1 box is not classified
    for face in faces:
        x, y, w, h = face.box
        assert x >= 0 and y >= 0 and x + w <= 400 and y + h <= 240


def test_pipeline_hands_the_detector_an_rgb_frame():
    seen = []

    def detector(rgb):
        seen.append(rgb[0, 0].tolist())
        return []

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # blue in BGR
    StreamPipeline(detector, FakeClassifier(), detect_every=1).process(frame)
    assert seen == [[0, 0, 255]]


def test_pipeline_detects_at_once_after_a_scene_cut_and_drops_old_tracks():
    detector, classifier = FakeDetector(), FakeClassifier()
    pipe = StreamPipeline(detector, classifier, detect_every=10)
    scene = textured_frame(size=(240, 400))
    first = pipe.process(scene)
    assert detector.calls == 1
    pipe.process(scene)  # same scene: nothing new, no detection
    assert detector.calls == 1
    cut = pipe.process(np.full_like(scene, 255))  # a hard cut: the picture turns white
    assert detector.calls == 2  # detected on frame 2, not on frame 10
    assert {f.track_id for f in cut}.isdisjoint({f.track_id for f in first})  # old identities are gone


def test_pipeline_scene_cut_check_can_be_switched_off():
    detector = FakeDetector()
    pipe = StreamPipeline(detector, FakeClassifier(), detect_every=10, scene_cut_threshold=None)
    scene = textured_frame(size=(240, 400))
    for frame in (scene, np.full_like(scene, 255), scene):
        pipe.process(frame)
    assert detector.calls == 1


def test_pipeline_ignores_small_frame_to_frame_changes():
    detector = FakeDetector()
    pipe = StreamPipeline(detector, FakeClassifier(), detect_every=10)
    for shift in range(6):  # a slow pan of 2 px per frame
        pipe.process(textured_frame((2 * shift, 0), size=(240, 400)))
    assert detector.calls == 1
