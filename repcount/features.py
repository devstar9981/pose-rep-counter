"""Turns MediaPipe landmarks into the per-frame features the model reads.

The same steps run in the browser (src/repcounter.ts in the portfolio), so keep them simple and causal:
every value for a frame depends only on that frame and the frames before it.

1. Resample to a fixed 15 frames per second, so camera frame rates don't matter.
2. Keep 13 body joints: nose, shoulders, elbows, wrists, hips, knees, ankles.
3. Undo the frame's aspect ratio, center on the middle of the hips (or shoulders when the hips are
   hidden), and divide by the body's size: the median torso length over the last 64 frames.
4. Per frame: 13 x 2 coordinates, 13 visibilities and a "person found" flag = 40 numbers.
"""

from __future__ import annotations

import numpy as np

RATE = 15.0
# MediaPipe landmark numbers.
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# Where each joint goes when the picture is mirrored (left and right swap).
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
SHOULDERS = (1, 2)
HIPS = (7, 8)
SCALE_FRAMES = 64
FEATURES = len(JOINTS) * 3 + 1
MAX_GAP = 0.3  # seconds; detections further apart than this leave the frames between them empty


def resample(kp: np.ndarray, t: np.ndarray, speed: float = 1.0, rate: float = RATE) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks [N, 33, 4] at times t -> [M, 13, 3] (x, y, visibility) at a steady rate.

    A new frame between two detections is interpolated when they are at most MAX_GAP apart, and left
    empty otherwise (the person was missing). That rule only needs the detections on either side, so
    the browser can apply it live. `speed` > 1 plays the video faster (training augmentation).
    Returns the new frames and the source time of each one.
    """
    pts = kp[:, JOINTS, :].astype(np.float32)[:, :, [0, 1, 3]]
    if len(t) == 0:
        return np.zeros((0, len(JOINTS), 3), np.float32), np.zeros(0, np.float32)
    src = np.arange(0.0, float(t[-1]) + 1e-6, speed / rate, dtype=np.float64)
    valid = ~np.isnan(pts[:, 0, 0])
    out = np.full((len(src), len(JOINTS), 3), np.nan, np.float32)
    tv = t[valid].astype(np.float64)
    if len(tv) == 0:
        return out, src.astype(np.float32)
    flat = pts[valid].reshape(len(tv), -1)
    right = np.searchsorted(tv, src)  # first detection at or after each new frame
    exact = (right < len(tv)) & (tv[np.minimum(right, len(tv) - 1)] == src)
    inner = (right > 0) & (right < len(tv))
    span = np.where(inner, tv[np.minimum(right, len(tv) - 1)] - tv[np.maximum(right - 1, 0)], np.inf)
    keep = exact | (inner & (span <= MAX_GAP))
    for j in range(flat.shape[1]):
        out.reshape(len(src), -1)[:, j] = np.interp(src, tv, flat[:, j])
    out[~keep] = np.nan
    return out, src.astype(np.float32)


def featurize(pts: np.ndarray, aspect: float) -> np.ndarray:
    """[M, 13, 3] resampled joints -> [M, 40] model features (see module docstring)."""
    m = len(pts)
    feats = np.zeros((m, FEATURES), np.float32)
    if m == 0:
        return feats
    xy = pts[:, :, :2].copy()
    xy[:, :, 0] *= aspect
    vis = np.nan_to_num(pts[:, :, 2], nan=0.0)
    present = ~np.isnan(xy[:, 0, 0])

    mid_hip = xy[:, HIPS, :].mean(axis=1)
    mid_shoulder = xy[:, SHOULDERS, :].mean(axis=1)
    hip_seen = vis[:, HIPS].min(axis=1) > 0.3
    center = np.where(hip_seen[:, None], mid_hip, mid_shoulder)
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    torso[~present] = np.nan
    scale = rolling_median(torso, SCALE_FRAMES)
    scale = np.where(np.isnan(scale) | (scale < 0.02), 0.25, scale)

    coords = (xy - center[:, None, :]) / scale[:, None, None]
    coords = np.clip(np.nan_to_num(coords, nan=0.0), -4.0, 4.0)
    feats[:, : 2 * len(JOINTS)] = coords.reshape(m, -1)
    feats[:, 2 * len(JOINTS) : 3 * len(JOINTS)] = vis
    feats[:, -1] = present
    feats[~present] = 0.0
    return feats


def rolling_median(values: np.ndarray, frames: int) -> np.ndarray:
    """Median of each value and the `frames - 1` before it, ignoring NaN (causal)."""
    padded = np.concatenate([np.full(frames - 1, np.nan, np.float32), values.astype(np.float32)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, frames)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(windows, axis=1)


def mirror(pts: np.ndarray) -> np.ndarray:
    """Left-right flip of [M, 13, 3] joints (x in 0-1)."""
    out = pts[:, MIRROR, :].copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]
    return out
