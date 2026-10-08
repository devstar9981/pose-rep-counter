"""Scores a checkpoint on the RepCount-A test set (never used for training or tuning).

    python -m repcount.evaluate                 # checkpoints/best.pt
    python -m repcount.evaluate --int8          # the same model with INT8 weights, as shipped

Writes results/test[-int8].json with the overall MAE and OBO, a breakdown by exercise, and every
video's predicted and true count.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from .count import metrics
from .data import ROOT
from .model import RepTCN
from .train import CHECKPOINTS, evaluate, load_split, prepare

RESULTS = ROOT / "results"


def load_model(name: str = "best", int8: bool = False) -> tuple[RepTCN, dict]:
    state = torch.load(CHECKPOINTS / f"{name}.pt", weights_only=False)
    model = RepTCN(state["channels"], dilations=state.get("dilations", (1, 2, 4, 8, 16)))
    model.load_state_dict(state["model"])
    model.eval()
    if int8:
        quantize_weights_(model)
    return model, state


@torch.no_grad()
def quantize_weights_(model: torch.nn.Module) -> None:
    """Rounds every conv weight to INT8 per output channel (symmetric), in place; activations stay float.

    This is exactly what the exported INT8 ONNX model and the browser engine do.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            w = module.weight
            scale = w.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12) / 127.0
            w.copy_(torch.round(w / scale).clamp(-127, 127) * scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    model, state = load_model(args.name, args.int8)
    items = load_split(args.split)
    prepared = [
        dict(
            prepare(item, 1.0, False, None, state.get("rate", 15.0)),
            video_id=item["video_id"],
            count=item["count"],
            action_name=item["action"],
        )
        for item in items
    ]
    result = evaluate(model, prepared, state["threshold"], state["offset"])
    by_action = defaultdict(lambda: ([], []))
    for row in result["rows"]:
        by_action[row["action"]][0].append(row["pred"])
        by_action[row["action"]][1].append(row["count"])
    result["by_action"] = {a: metrics(p, t) for a, (p, t) in sorted(by_action.items())}
    # PoseRAC and TransRAC report on the 151 test videos of the eight named exercises (all but "other").
    named = [r for r in result["rows"] if r["action"] != "other"]
    result["named_151"] = metrics([r["pred"] for r in named], [r["count"] for r in named])
    result.update(
        split=args.split,
        weights="int8" if args.int8 else "fp32",
        threshold=state["threshold"],
        offset=state["offset"],
        epoch=state["epoch"],
        rate=state.get("rate", 15.0),
        dilations=list(state.get("dilations", (1, 2, 4, 8, 16))),
        parameters=sum(p.numel() for p in model.parameters()),
        mean_abs_error_reps=float(np.mean([abs(r["pred"] - r["count"]) for r in result["rows"]])),
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.split}{'-int8' if args.int8 else ''}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"{args.split} ({result['weights']}): MAE {result['mae']:.3f}  OBO {result['obo']:.3f}  on {result['videos']} videos")
    n = result["named_151"]
    print(f"  the eight named exercises: MAE {n['mae']:.3f}  OBO {n['obo']:.3f}  on {n['videos']} videos")
    for action, m in result["by_action"].items():
        print(f"  {action:<13} MAE {m['mae']:.3f}  OBO {m['obo']:.3f}  ({m['videos']} videos)")


if __name__ == "__main__":
    main()
