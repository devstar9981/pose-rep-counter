"""Runs MediaPipe Pose Landmarker (lite) over every RepCount-A video and saves the landmarks.

    python -m repcount.extract --workers 6            # all splits; finished videos are skipped
    python -m repcount.extract --split test --limit 5

Settings match the browser demo: the same pose_landmarker_lite.task, VIDEO mode, one person, default
confidence thresholds. Frames are processed at up to 30 per second (higher frame rates are thinned)
and scaled so the longer side is at most 640 px, about what a laptop camera gives the browser.
Output: data/poses/<split>/<video_id>.npz with
    kp        float16 [N, 33, 4]   x, y (0-1 of the frame), z, visibility; NaN when no person was found
    t         float32 [N]          seconds from the start of the video
    width, height, fps, frames
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .data import POSES, ROOT, SPLITS, video_bytes, videos

MODEL = ROOT / "models" / "pose_landmarker_lite.task"
MAX_RATE = 30.0
MAX_SIDE = 640

_landmarker = None


def _init_worker() -> None:
    global _landmarker
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    _landmarker = vision.PoseLandmarker.create_from_options(options)


def _reset_landmarker() -> None:
    # VIDEO mode tracks across frames and needs rising timestamps, so each video gets a fresh one.
    global _landmarker
    if _landmarker is not None:
        _landmarker.close()
    _init_worker()


def landmarks_for_video(path: str):
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_SIDE / max(width, height, 1))
    _reset_landmarker()

    kps, times = [], []
    index = -1
    last_slot = -1
    last_ms = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        t = index / fps
        slot = int(t * MAX_RATE)
        if slot == last_slot:
            continue
        last_slot = slot
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ms = max(last_ms + 1, int(round(t * 1000)))
        last_ms = ms
        result = _landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if result.pose_landmarks:
            kps.append([(p.x, p.y, p.z, p.visibility) for p in result.pose_landmarks[0]])
        else:
            kps.append(np.full((33, 4), np.nan))
        times.append(t)
    capture.release()
    kp = np.asarray(kps, dtype=np.float16).reshape(-1, 33, 4)
    return kp, np.asarray(times, dtype=np.float32), width, height, fps, index + 1


def _process(job: tuple[str, int, str]) -> tuple[str, str, int, float]:
    split, row, video_id = job
    out = POSES / split / f"{video_id}.npz"
    if out.exists():
        return split, video_id, -1, 0.0
    started = time.perf_counter()
    [(_, ext, blob)] = video_bytes(split, [row])
    fd, path = tempfile.mkstemp(suffix=ext if ext.startswith(".") else f".{ext}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        kp, t, width, height, fps, frames = landmarks_for_video(path)
    finally:
        os.remove(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, kp=kp, t=t, width=width, height=height, fps=fps, frames=frames)
    tmp.replace(out)
    return split, video_id, len(t), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, nargs="+", default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    jobs = []
    for split in args.split:
        listed = videos(split)
        rows = list(enumerate(listed))[: args.limit or None]
        jobs += [(split, row, v.video_id) for row, v in rows]
    print(f"{len(jobs)} videos, {args.workers} workers", flush=True)
    started = time.perf_counter()
    frames = 0
    done = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for split, video_id, n, seconds in pool.imap_unordered(_process, jobs):
            done += 1
            if n >= 0:
                frames += n
            if done % 25 == 0 or done == len(jobs):
                rate = frames / max(1e-6, time.perf_counter() - started)
                print(f"{done}/{len(jobs)} videos, {frames} frames, {rate:.0f} frames/s", flush=True)


if __name__ == "__main__":
    main()
