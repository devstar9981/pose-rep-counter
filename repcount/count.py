"""From per-frame model outputs to a repetition count, frame by frame (the browser runs the same loop).

While a repetition is in progress, the phase angle moves forward once around the circle per rep. We
add up how far it moves (small backward steps count as negative, so jitter cancels out) and report
the completed laps. `offset` lets a rep count a little before the lap fully closes, since the last
few frames of a rep are often cut short.
"""

from __future__ import annotations

import math

import numpy as np


def count_reps(active: np.ndarray, phase: np.ndarray, threshold: float = 0.5, offset: float = 0.25) -> tuple[int, np.ndarray]:
    """active: [M] probability; phase: [M, 2] (cos, sin). Returns the count and the running count per frame."""
    laps = 0.0
    previous: float | None = None
    shown = 0
    running = np.zeros(len(active), np.int32)
    for i in range(len(active)):
        if active[i] >= threshold:
            angle = (math.atan2(float(phase[i, 1]), float(phase[i, 0])) / (2 * math.pi)) % 1.0
            if previous is not None:
                laps += ((angle - previous + 0.5) % 1.0) - 0.5
            previous = angle
        else:
            previous = None
        shown = max(shown, math.floor(laps + offset))
        running[i] = shown
    return shown, running


def metrics(pred: list[int], truth: list[int]) -> dict[str, float]:
    """The standard RepCount metrics: MAE (absolute error divided by the true count) and
    OBO (share of videos counted within one rep)."""
    p = np.asarray(pred, np.float64)
    g = np.asarray(truth, np.float64)
    keep = g > 0
    return {
        "mae": float(np.mean(np.abs(p[keep] - g[keep]) / g[keep])),
        "obo": float(np.mean(np.abs(p[keep] - g[keep]) <= 1)),
        "videos": int(keep.sum()),
    }
