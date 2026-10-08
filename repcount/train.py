"""Trains the repetition counter on RepCount-A train, picks the best epoch on validation.

    python -m repcount.train --epochs 40

Each training video is precomputed at several playback speeds (0.75x to 1.35x), mirrored and not,
each with a small random rotation and zoom, so a batch is just random 256-frame slices. Labels per
frame: whether a repetition is in progress, and how far through it we are (0 at the start, 1 at
the end). Writes checkpoints/best.pt with the validation-tuned counting settings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .count import count_reps, metrics
from .data import CLASSES, POSES, ROOT, load_labels
from .features import RATE, featurize, mirror, resample
from .model import RepTCN, split_outputs

CHECKPOINTS = ROOT / "checkpoints"
CROP = 256
SPEEDS = (0.75, 0.85, 1.0, 1.15, 1.35)


def load_split(split: str) -> list[dict]:
    items = []
    for v in load_labels()[split]:
        path = POSES / split / f"{v['video_id']}.npz"
        if not path.exists() or v["count"] <= 0:
            continue
        z = np.load(path)
        items.append(
            dict(
                v,
                kp=z["kp"].astype(np.float32),
                t=z["t"].astype(np.float32),
                aspect=float(z["width"]) / max(1.0, float(z["height"])),
                fps=float(z["fps"]),
            )
        )
    return items


def frame_labels(src: np.ndarray, cycles: list[tuple[int, int]], fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per resampled frame: 1 inside a repetition (else 0), and the progress through it (else NaN)."""
    active = np.zeros(len(src), np.float32)
    phase = np.full(len(src), np.nan, np.float32)
    for start, end in cycles:
        ts, te = start / fps, end / fps
        inside = (src >= ts) & (src < te)
        active[inside] = 1.0
        phase[inside] = (src[inside] - ts) / (te - ts)
    return active, phase


def jitter(pts: np.ndarray, aspect: float, rng: random.Random) -> np.ndarray:
    """Small random rotation (up to 10 degrees) and zoom (up to 10%) around the frame center."""
    out = pts.copy()
    angle = math.radians(rng.uniform(-10, 10))
    zoom = rng.uniform(0.9, 1.1)
    x = (out[:, :, 0] - 0.5) * aspect
    y = out[:, :, 1] - 0.5
    c, s = math.cos(angle), math.sin(angle)
    out[:, :, 0] = (zoom * (c * x - s * y)) / aspect + 0.5
    out[:, :, 1] = zoom * (s * x + c * y) + 0.5
    return out


def prepare(item: dict, speed: float, flip: bool, rng: random.Random | None, rate: float = RATE) -> dict:
    pts, src = resample(item["kp"], item["t"], speed, rate)
    if flip:
        pts = mirror(pts)
    if rng is not None:
        pts = jitter(pts, item["aspect"], rng)
    active, phase = frame_labels(src, item["cycles"], item["fps"])
    return {
        "x": featurize(pts, item["aspect"]),
        "active": active,
        "phase": phase,
        "action": CLASSES.index(item["action"]),
    }


class Batches:
    """Random 256-frame slices from the precomputed variants."""

    def __init__(self, variants: list[dict], seed: int = 0):
        self.variants = variants
        self.rng = np.random.default_rng(seed)
        # Longer videos are picked more often, in proportion to their length.
        lengths = np.array([max(1, len(v["x"])) for v in variants], np.float64)
        self.weights = lengths / lengths.sum()

    def sample(self, size: int):
        x = np.zeros((size, CROP, self.variants[0]["x"].shape[1]), np.float32)
        active = np.zeros((size, CROP), np.float32)
        phase = np.full((size, CROP), np.nan, np.float32)
        action = np.zeros(size, np.int64)
        for b, k in enumerate(self.rng.choice(len(self.variants), size=size, p=self.weights)):
            v = self.variants[k]
            n = len(v["x"])
            start = int(self.rng.integers(-CROP // 4, max(1, n - CROP // 2)))
            lo, hi = max(0, start), min(n, start + CROP)
            if hi > lo:
                x[b, lo - start : hi - start] = v["x"][lo:hi]
                active[b, lo - start : hi - start] = v["active"][lo:hi]
                phase[b, lo - start : hi - start] = v["phase"][lo:hi]
            action[b] = v["action"]
        # A little noise on the coordinates of frames with a person in them.
        present = x[:, :, -1:] > 0
        x[:, :, :26] += self.rng.normal(0, 0.02, x[:, :, :26].shape).astype(np.float32) * present
        return (
            torch.from_numpy(x).transpose(1, 2),
            torch.from_numpy(active),
            torch.from_numpy(phase),
            torch.from_numpy(action),
        )


def loss_fn(y: torch.Tensor, active: torch.Tensor, phase: torch.Tensor, action: torch.Tensor, present: torch.Tensor):
    a_logit, p_vec, c_logit = split_outputs(y)
    l_active = nn.functional.binary_cross_entropy_with_logits(a_logit, active)
    inside = ~torch.isnan(phase)
    target = torch.stack([torch.cos(2 * math.pi * phase), torch.sin(2 * math.pi * phase)], dim=1)
    unit = p_vec / (p_vec.norm(dim=1, keepdim=True) + 1e-6)
    cos_err = 1 - (unit * torch.nan_to_num(target)).sum(dim=1)
    l_phase = (cos_err * inside).sum() / inside.sum().clamp(min=1)
    ce = nn.functional.cross_entropy(c_logit, action[:, None].expand(-1, c_logit.shape[-1]), reduction="none")
    l_action = (ce * present).sum() / present.sum().clamp(min=1)
    return l_active + l_phase + 0.2 * l_action, (l_active.item(), l_phase.item(), l_action.item())


@torch.no_grad()
def predict(model: RepTCN, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-video causal pass: [M, F] -> active probability [M], phase [M, 2], action probabilities [M, C]."""
    device = next(model.parameters()).device
    y = model(torch.from_numpy(x).T[None].to(device))[0].cpu()
    a, p, c = split_outputs(y[None])
    return torch.sigmoid(a[0]).numpy(), p[0].T.numpy(), torch.softmax(c[0], dim=0).T.numpy()


def evaluate(model: RepTCN, prepared: list[dict], threshold: float = 0.5, offset: float = 0.25) -> dict:
    model.eval()
    preds, truths, rows = [], [], []
    for v in prepared:
        a, p, c = predict(model, v["x"])
        n, _ = count_reps(a, p, threshold, offset)
        preds.append(n)
        truths.append(v["count"])
        rows.append({"video_id": v["video_id"], "action": v["action_name"], "count": v["count"], "pred": n})
    out = metrics(preds, truths)
    out["rows"] = rows
    return out


def calibrate(model: RepTCN, prepared: list[dict]) -> tuple[float, float, dict]:
    """Picks the activity threshold and lap offset with the best validation OBO (then MAE)."""
    model.eval()
    outputs = [predict(model, v["x"])[:2] for v in prepared]
    truths = [v["count"] for v in prepared]
    best = None
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
        for offset in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            preds = [count_reps(a, p, threshold, offset)[0] for a, p in outputs]
            m = metrics(preds, truths)
            key = (m["obo"], -m["mae"])
            if best is None or key > best[0]:
                best = (key, threshold, offset, m)
    return best[1], best[2], best[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--name", default="best")
    parser.add_argument("--rate", type=float, default=RATE, help="frames per second the model works at")
    parser.add_argument("--ema", type=float, default=0.998, help="decay of the weight average that gets evaluated and saved")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dilations", default="1,2,4,8,16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)

    started = time.perf_counter()
    train_items = load_split("train")
    val_items = load_split("validation")
    dilations = tuple(int(d) for d in args.dilations.split(","))
    variants = [
        dict(prepare(item, speed, flip, rng, args.rate), video_id=item["video_id"])
        for item in train_items
        for speed in SPEEDS
        for flip in (False, True)
    ]
    val = [
        dict(prepare(item, 1.0, False, None, args.rate), video_id=item["video_id"], count=item["count"], action_name=item["action"])
        for item in val_items
    ]
    print(f"{len(train_items)} train videos ({len(variants)} variants), {len(val)} validation, "
          f"prepared in {time.perf_counter() - started:.0f}s", flush=True)

    model = RepTCN(args.channels, dropout=args.dropout, dilations=dilations).to(device)
    # An exponential moving average of the weights (BatchNorm statistics included) is what gets
    # evaluated and saved: it's steadier than the weights at any single step.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, training on {device}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total, pct_start=0.1)
    batches = Batches(variants, args.seed)
    CHECKPOINTS.mkdir(exist_ok=True)
    best_key = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        parts = np.zeros(3)
        for _ in range(args.steps):
            x, active, phase, action = (t.to(device) for t in batches.sample(args.batch))
            loss, detail = loss_fn(model(x), active, phase, action, x[:, -1, :])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            ema.update_parameters(model)
            parts += detail
        parts /= args.steps
        m = evaluate(ema.module, val)
        history.append(
            {"epoch": epoch, "active": float(parts[0]), "phase": float(parts[1]), "action": float(parts[2]),
             "val_mae": m["mae"], "val_obo": m["obo"]}
        )
        key = (m["obo"], -m["mae"])
        mark = ""
        if best_key is None or key > best_key:
            best_key = key
            weights = {k: v.detach().cpu() for k, v in ema.module.state_dict().items()}
            torch.save(
                {"model": weights, "channels": args.channels, "dilations": dilations, "rate": args.rate, "epoch": epoch},
                CHECKPOINTS / f"{args.name}.pt",
            )
            mark = "  *"
        print(f"epoch {epoch:>3}  loss active {parts[0]:.3f} phase {parts[1]:.3f} action {parts[2]:.3f}  "
              f"val MAE {m['mae']:.3f} OBO {m['obo']:.3f}  {time.perf_counter() - started:.0f}s{mark}", flush=True)

    state = torch.load(CHECKPOINTS / f"{args.name}.pt", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    threshold, offset, m = calibrate(model, val)
    state.update(threshold=threshold, offset=offset, val=m, history=history, args=vars(args))
    torch.save(state, CHECKPOINTS / f"{args.name}.pt")
    print(f"best epoch {state['epoch']}: threshold {threshold}, offset {offset} -> val MAE {m['mae']:.3f} OBO {m['obo']:.3f}")
    (CHECKPOINTS / f"{args.name}.json").write_text(json.dumps({k: v for k, v in state.items() if k != "model"}, indent=1))


if __name__ == "__main__":
    main()
