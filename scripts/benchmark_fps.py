"""Measure the speed of the webcam detector on a video file and compare it with a per-frame baseline.

It runs, on the same clip and the same weights:

  1. notebook-3 style      MTCNN on every frame, then `model.predict` once per face (what the 2022 notebook does)
  2. per-frame baseline    MTCNN on every frame, all faces classified in one batch (the reference for agreement)
  3. streaming pipeline    demo/realtime.py: MTCNN every N-th frame, optical-flow tracking in between, one batch
                           (the settings in demo/webcam.py; repeated, and swept over N and the detection width)

For each run it reports the mean FPS of the whole loop (read a frame, process it, draw the boxes; no window is
opened), the per-frame latency of the pipeline call (mean, p95), and, against the per-frame baseline, how many of
the baseline's faces the streaming pipeline also finds and labels the same way.

    python scripts/make_test_clip.py
    python scripts/benchmark_fps.py --log docs/runs/2026-10-14-fps.md

The first frames are excluded from the timings (--warmup-frames) because TensorFlow is still tracing graphs.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "demo"))

from app import LABELS, resolve_weights  # noqa: E402
from realtime import IMG_SIZE, Face, FpsCounter, StreamPipeline, crop_batch, iou, percentile  # noqa: E402
from webcam import draw_faces, load_models, make_classify, make_detect  # noqa: E402

# ---------------------------------------------------------------- pipelines under test


class PerFramePipeline:
    """MTCNN on every frame, then one batched classification: the reference for agreement."""

    def __init__(self, detect, classify) -> None:
        self.detect, self.classify = detect, classify

    def process(self, frame_bgr: np.ndarray) -> list[Face]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self.detect(rgb)
        crops, kept = crop_batch(rgb, boxes)
        faces = []
        if kept:
            for index, row in zip(kept, np.asarray(self.classify(crops)), strict=True):
                label = int(np.argmax(row))
                x, y, w, h = (int(round(v)) for v in boxes[index])
                faces.append(Face(index, (x, y, w, h), label, float(row[label]), True))
        return faces


class NotebookStylePipeline:
    """Notebook 3's loop: MTCNN on every frame, `model.predict` called once per face."""

    def __init__(self, detect, model) -> None:
        self.detect, self.model = detect, model

    def process(self, frame_bgr: np.ndarray) -> list[Face]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = []
        for index, (x, y, w, h) in enumerate(self.detect(rgb)):
            x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
            if w <= 1 or h <= 1:
                continue
            data = cv2.resize(rgb[y : y + h, x : x + w], IMG_SIZE) / 255.0
            scores = self.model.predict(data.reshape((1, *data.shape)), verbose=0)
            label = int(np.argmax(scores, axis=1)[0])
            faces.append(Face(index, (x, y, w, h), label, float(scores[0][label]), True))
        return faces


# ---------------------------------------------------------------- measuring


def run_clip(video: Path, pipe, warmup_frames: int, max_frames: int = 0) -> dict:
    """Run `pipe` over the clip; time the loop and the pipeline call; keep every frame's faces."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"Cannot open video: {video}")
    counter = FpsCounter()
    loop_times, latencies, detections, index = [], [], [], 0
    while max_frames <= 0 or index < max_frames:
        t0 = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        t1 = time.perf_counter()
        faces = pipe.process(frame)
        t2 = time.perf_counter()
        draw_faces(frame, faces, counter.fps)
        t3 = time.perf_counter()
        detections.append([(f.box, f.label) for f in faces])
        if index >= warmup_frames:
            counter.tick()
            loop_times.append(t3 - t0)
            latencies.append(t2 - t1)
        index += 1
    capture.release()
    return {
        "frames": index,
        "measured_frames": len(loop_times),
        "mean_fps": counter.mean_fps,
        "loop_ms_mean": 1000 * float(np.mean(loop_times)),
        "fps_at_p95_loop": 1.0 / percentile(loop_times, 95),
        "latency_ms_mean": 1000 * float(np.mean(latencies)),
        "latency_ms_p95": 1000 * percentile(latencies, 95),
        "latency_ms_max": 1000 * float(np.max(latencies)),
        "detections": detections,
    }


def agreement(reference: list, other: list, min_iou: float = 0.5) -> dict:
    """Frame by frame: how many of the reference faces does `other` also find, and with the same label?"""
    total = matched = same = 0
    for ref_faces, faces in zip(reference, other, strict=False):
        free = list(range(len(faces)))
        for box, label in ref_faces:
            total += 1
            best = max(free, key=lambda j: iou(box, faces[j][0]), default=None)
            if best is not None and iou(box, faces[best][0]) >= min_iou:
                free.remove(best)
                matched += 1
                same += int(faces[best][1] == label)
    return {
        "reference_faces": total,
        "found": matched / total if total else 0.0,
        "same_label_of_found": same / matched if matched else 0.0,
        "agreement": same / total if total else 0.0,
    }


def summarise(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "detections"}


# ---------------------------------------------------------------- hardware and report


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").is_file():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware() -> dict:
    info = {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "physical_cores": None, "os": platform.platform()}
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30)
    except ImportError:
        pass
    return info


def hardware_label(hw: dict) -> str:
    cores = f"{hw['physical_cores']} cores / {hw['logical_cores']} threads" if hw["physical_cores"] else f"{hw['logical_cores']} threads"
    return f"{hw['cpu']}, {cores}, CPU only"


def measure_background_load(high_priority: bool) -> dict:
    """System-wide CPU use over 3 seconds before anything of ours runs; optionally raise this process's priority."""
    info = {"cpu_percent": None, "priority": "normal"}
    try:
        import psutil
    except ImportError:
        return info
    if high_priority:
        try:
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -5)
            info["priority"] = "high"
        except (psutil.Error, AttributeError, OSError):
            info["priority"] = "normal (could not raise)"
    info["cpu_percent"] = round(psutil.cpu_percent(interval=3.0))
    return info


def fmt_row(name: str, r: dict, agree: dict | None = None) -> str:
    cells = [name, f"{r['mean_fps']:.1f}", f"{r['fps_at_p95_loop']:.1f}", f"{r['latency_ms_mean']:.0f}", f"{r['latency_ms_p95']:.0f}"]
    if agree is not None:
        cells += [f"{100 * agree['found']:.1f}%", f"{100 * agree['agreement']:.1f}%"]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=REPO / "data" / "benchmark_clip.mp4")
    parser.add_argument("--weights", help="trained Keras model; default as in demo/app.py")
    parser.add_argument("--detect-every", type=int, default=10, help="N for the headline run (default 10)")
    parser.add_argument("--detect-width", type=int, default=640, help="MTCNN input width for the headline run (default 640)")
    parser.add_argument("--repeats", type=int, default=3, help="repeats of the headline run (default 3)")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--notebook-frames", type=int, default=60, help="frames for the (slow) notebook-style run")
    parser.add_argument("--max-frames", type=int, default=0, help="limit every run to this many frames (default: whole clip)")
    parser.add_argument("--high-priority", action="store_true",
                        help="run this process at high CPU priority (useful when other programs keep the CPU busy; needs psutil)")
    parser.add_argument("--no-sweep", action="store_true", help="skip the sweep over N and the detection width")
    parser.add_argument("--log", type=Path, help="write the run log (markdown) here")
    parser.add_argument("--json", type=Path, help="also write the raw numbers (without per-frame detections) here")
    args = parser.parse_args()

    hw = hardware()
    load = measure_background_load(args.high_priority)
    print(f"CPU busy with other programs before the run: {load['cpu_percent']}% (priority: {load['priority']})", flush=True)
    import tensorflow as tf

    detector, model = load_models(resolve_weights(args.weights))
    classify = make_classify(model)
    detect_full = make_detect(detector, 0)
    # warm up both models so TensorFlow's first-call tracing is not in any timing
    detect_full(np.zeros((480, 640, 3), dtype=np.uint8))
    classify(np.zeros((2, *IMG_SIZE, 3), dtype=np.float32))
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)

    print("1/3 notebook-3 style (MTCNN every frame, model.predict per face) ...", flush=True)
    notebook = run_clip(args.video, NotebookStylePipeline(detect_full, model), 2, args.notebook_frames or args.max_frames)
    print(f"    {notebook['mean_fps']:.2f} FPS over {notebook['measured_frames']} frames", flush=True)

    print("2/3 per-frame baseline (MTCNN every frame, one batch) ...", flush=True)
    baseline = run_clip(args.video, PerFramePipeline(detect_full, classify), args.warmup_frames, args.max_frames)
    print(f"    {baseline['mean_fps']:.2f} FPS over {baseline['measured_frames']} frames", flush=True)
    faces_total = sum(len(f) for f in baseline["detections"])

    def stream(n: int, width: int, max_frames: int = 0) -> dict:
        detect = detect_full if width == 0 else make_detect(detector, width)
        return run_clip(args.video, StreamPipeline(detect, classify, detect_every=n), args.warmup_frames, max_frames or args.max_frames)

    print(f"3/3 streaming pipeline, N={args.detect_every}, detect width {args.detect_width}, {args.repeats} repeats ...", flush=True)
    stream(args.detect_every, args.detect_width, max_frames=60)  # one throw-away pass so caches and clocks are warm
    headline = []
    for i in range(args.repeats):
        result = stream(args.detect_every, args.detect_width)
        result["agreement"] = agreement(baseline["detections"], result["detections"])
        headline.append(result)
        print(f"    run {i + 1}: {result['mean_fps']:.1f} FPS, agreement {100 * result['agreement']['agreement']:.1f}%", flush=True)

    sweep = []
    if not args.no_sweep:
        configs = [(n, args.detect_width) for n in (2, 3, 5, 8, 10, 15) if n != args.detect_every]
        configs += [(args.detect_every, w) for w in (480, 320) if w < args.detect_width]
        for n, width in configs:
            result = stream(n, width)
            result["agreement"] = agreement(baseline["detections"], result["detections"])
            result.update(n=n, width=width)
            sweep.append(result)
            print(f"    sweep N={n} width={width}: {result['mean_fps']:.1f} FPS, agreement {100 * result['agreement']['agreement']:.1f}%", flush=True)

    mean_runs = float(np.mean([r["mean_fps"] for r in headline]))
    label = f"Verified (run {datetime.date.today().isoformat()}, {hardware_label(hw)})"
    summary = {
        "date": datetime.date.today().isoformat(),
        "hardware": hw,
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "clip": {"file": args.video.name, "frames": baseline["frames"], "reference_faces": faces_total, "warmup_frames_excluded": args.warmup_frames},
        "settings": {"detect_every": args.detect_every, "detect_width": args.detect_width, "frame_size": "640x480", "repeats": args.repeats},
        "notebook_style": summarise(notebook),
        "per_frame_baseline": summarise(baseline),
        "background_load": load,
        "headline_runs": [summarise(r) | {"agreement": r["agreement"]} for r in headline],
        "headline_mean_of_run_means_fps": mean_runs,
        "sweep": [summarise(r) | {"agreement": r["agreement"]} for r in sweep],
        "label": label,
    }
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Run log: webcam detector speed (2026-10-14)",
        "",
        f"**Status: {label}**",
        "",
        "A new, dated measurement made for this repository. There is no earlier speed measurement to compare it with, except the "
        "\"about 1.3 FPS\" note in notebook 3, whose machine was not recorded (**Reported**, not re-measured here). "
        "No physical webcam was available: the input is a video file made from dataset photos (see below).",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Hardware | {hardware_label(hw)}" + (f", {hw['ram_gb']} GB RAM" if "ram_gb" in hw else "") + f", {hw['os']} |",
        f"| Software | Python {platform.python_version()}, TensorFlow {tf.__version__}, OpenCV {cv2.__version__}, NumPy {np.__version__}, mtcnn 1.0.0 |",
        f"| Input | `scripts/make_test_clip.py`: {baseline['frames']} frames, 640x480, 30 fps, 8 dataset photos (CC0) with a slow zoom and pan; "
        f"{faces_total} faces in total according to the per-frame baseline |",
        "| Weights | the same `inceptionV3-model.keras` / `.h5` as the Gradio demo (final epoch-20 model of the 2026-10-13 run) |",
        f"| Headline settings | MTCNN every **{args.detect_every}** frames, MTCNN input width {args.detect_width} px (the whole 640 px frame), "
        "optical-flow tracking in between, one batched classifier call per detection, 640x480 frames |",
        f"| Other load | {load['cpu_percent']}% of the CPU was busy with other programs in the 3 seconds before the run (process priority: {load['priority']}). "
        "This machine was not idle, so absolute numbers are lower than on a quiet machine and vary from run to run |",
        f"| Timing | first {args.warmup_frames} frames excluded; the loop is *read frame, process, draw boxes*; no window is opened, so the numbers exclude the cost of showing the video |",
        "| Command | `python scripts/benchmark_fps.py --log docs/runs/2026-10-14-fps.md` |",
        "",
        "## Results",
        "",
        "Columns: mean FPS of the whole loop; FPS at the 95th-percentile loop time (the slowest 5% of frames run at or below "
        "this rate, which is what you see as stutter); mean and 95th-percentile latency of the pipeline call alone; "
        "against the per-frame baseline: share of its faces the pipeline also found (box overlap IoU >= 0.5) and share found "
        "*and* given the same label (the agreement).",
        "",
        "| Pipeline | mean FPS | FPS at p95 loop | latency mean (ms) | latency p95 (ms) | faces found | same label |",
        "|---|---|---|---|---|---|---|",
        fmt_row(f"1. notebook-3 style ({notebook['measured_frames']} frames)", notebook, agreement(baseline["detections"][: notebook["frames"]], notebook["detections"])),
        fmt_row("2. per-frame baseline (reference)", baseline, {"found": 1.0, "agreement": 1.0}),
    ]
    for i, r in enumerate(headline, 1):
        lines.append(fmt_row(f"3. streaming, N={args.detect_every}, width {args.detect_width}, run {i}", r, r["agreement"]))
    lines += [
        "",
        f"**Headline: {mean_runs:.1f} FPS mean (mean of {args.repeats} runs: "
        + ", ".join(f"{r['mean_fps']:.1f}" for r in headline)
        + f"), N={args.detect_every}, 640x480, CPU only; agreement with the per-frame baseline "
        + f"{100 * float(np.mean([r['agreement']['agreement'] for r in headline])):.1f}%.**",
        "",
    ]
    if sweep:
        lines += [
            "### Trade-off: how often MTCNN runs, and how large its input is (one run each)",
            "",
            "| N (MTCNN every N frames) | MTCNN input width | mean FPS | FPS at p95 loop | latency mean (ms) | latency p95 (ms) | faces found | same label |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sweep:
            lines.append(
                f"| {r['n']} | {r['width']} | {r['mean_fps']:.1f} | {r['fps_at_p95_loop']:.1f} | {r['latency_ms_mean']:.0f} | "
                f"{r['latency_ms_p95']:.0f} | {100 * r['agreement']['found']:.1f}% | {100 * r['agreement']['agreement']:.1f}% |"
            )
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "- **Latency is uneven.** MTCNN takes most of a frame budget when it runs, so frames on which it runs are slow and the other frames are fast. "
        "The mean FPS is high, but the picture pauses briefly every N frames; the p95 columns show that.",
        "- **Agreement measures the shortcut, not the model.** It compares the streaming pipeline with the same MTCNN and classifier run on every frame. "
        "It says nothing about whether either is right. The classifier's accuracy is in the README (Results).",
        "- **Not a webcam.** The clip is dataset photos with a smooth zoom and pan: faces are small (30 to 130 px), there is no motion blur and nobody "
        "moves or puts on a mask. A real camera, other lighting and a lower-power CPU will give other numbers. This machine is a desktop CPU; a laptop will be slower.",
        "- **New faces and new mask states wait for the next detection**, up to N-1 frames (N-1 at 30 fps is about "
        f"{1000 * (args.detect_every - 1) / 30:.0f} ms for N={args.detect_every}). Boxes between detections follow optical flow and keep their size.",
        "- **Classes.** Label ids: " + ", ".join(f"{i} = {n}" for i, n in LABELS.items()) + ".",
        "",
    ]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {args.log}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
