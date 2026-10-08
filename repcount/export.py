"""Exports the trained model for deployment and measures it.

    python -m repcount.export

1. Folds each BatchNorm into the convolution before it (same math, fewer layers).
2. Exports ONNX (FP32) for one 64-frame window -> outputs for its last frame.
3. Makes an INT8-weight version: every conv weight stored as int8 with one scale per output channel,
   restored by DequantizeLinear at load. Activations stay float, so the browser engine can match it.
4. Checks both against PyTorch, times them with ONNX Runtime on one CPU thread, and records sizes.
5. Writes web/rep-counter.bin + web/rep-counter.json: the same int8 weights in a small format the
   portfolio's TypeScript engine reads, plus test vectors to check that engine against PyTorch.
"""

from __future__ import annotations

import json
import time

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from .data import CLASSES, ROOT
from .evaluate import RESULTS, load_model
from .features import FEATURES, JOINTS, MIRROR, SCALE_FRAMES
from .model import Block, RepTCN, receptive_field

EXPORT = ROOT / "export"
WEB = ROOT / "web"


@torch.no_grad()
def fold_batchnorm(model: RepTCN) -> RepTCN:
    for block in model.blocks:
        assert isinstance(block, Block)
        bn, conv = block.norm, block.conv
        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.mul_(factor[:, None, None])
        conv.bias.copy_((conv.bias - bn.running_mean) * factor + bn.bias)
        block.norm = torch.nn.Identity()
    return model


class LastFrame(torch.nn.Module):
    def __init__(self, model: RepTCN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, :, -1]


def int8_weights(fp32_path, out_path) -> None:
    """Replaces each Conv weight with int8 + per-channel scale behind a DequantizeLinear node."""
    m = onnx.load(str(fp32_path))
    inits = {i.name: i for i in m.graph.initializer}
    new_nodes = []
    for node in m.graph.node:
        if node.op_type == "Conv" and node.input[1] in inits:
            name = node.input[1]
            w = numpy_helper.to_array(inits[name]).astype(np.float32)
            scale = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-12) / 127.0
            q = np.clip(np.round(w / scale[:, None, None]), -127, 127).astype(np.int8)
            m.graph.initializer.remove(inits[name])
            m.graph.initializer.extend(
                [
                    numpy_helper.from_array(q, f"{name}_int8"),
                    numpy_helper.from_array(scale.astype(np.float32), f"{name}_scale"),
                    numpy_helper.from_array(np.zeros(w.shape[0], np.int8), f"{name}_zero"),
                ]
            )
            new_nodes.append(
                helper.make_node("DequantizeLinear", [f"{name}_int8", f"{name}_scale", f"{name}_zero"], [f"{name}_deq"], axis=0)
            )
            node.input[1] = f"{name}_deq"
    for n in reversed(new_nodes):
        m.graph.node.insert(0, n)
    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))


def time_onnx(path, x: np.ndarray, runs: int = 2000) -> float:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    for _ in range(100):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, {name: x})
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def run_onnx(path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def write_web(model: RepTCN, reference: RepTCN, state: dict, windows: np.ndarray) -> dict:
    """Packs int8 weights for the browser. Layout: all float32 data (scales, biases), then all int8 data."""
    floats: list[np.ndarray] = []
    ints: list[np.ndarray] = []
    layers = []
    f_off = 0
    i_off = 0

    def add(name: str, conv: torch.nn.Conv1d, dilation: int = 1):
        nonlocal f_off, i_off
        w = conv.weight.detach().numpy().astype(np.float32)
        b = conv.bias.detach().numpy().astype(np.float32)
        scale = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-12) / 127.0
        q = np.clip(np.round(w / scale[:, None, None]), -127, 127).astype(np.int8)
        layers.append(
            {"name": name, "out": w.shape[0], "in": w.shape[1], "kernel": w.shape[2], "dilation": dilation,
             "weights": i_off, "scales": f_off, "bias": f_off + w.shape[0]}
        )
        floats.extend([scale.astype(np.float32), b])
        ints.append(q.reshape(-1))
        f_off += 2 * w.shape[0]
        i_off += q.size

    add("inp", model.inp)
    for d, block in zip(model.dilations, model.blocks):
        add(f"block{d}.conv", block.conv, d)
        add(f"block{d}.mix", block.mix)
    add("head", model.head)
    float_block = np.concatenate(floats).astype("<f4")
    int_block = np.concatenate(ints).astype(np.int8)
    WEB.mkdir(exist_ok=True)
    (WEB / "rep-counter.bin").write_bytes(float_block.tobytes() + int_block.tobytes())

    # Reference outputs from `reference` (the INT8-weight model) for a few real windows, to test the browser engine.
    with torch.no_grad():
        ref = LastFrame(reference)(torch.from_numpy(windows)).numpy()
    manifest = {
        "version": 1,
        "rate": state["rate"],
        "window": receptive_field(model.dilations) + 1,
        "features": FEATURES,
        "joints": JOINTS,
        "mirror": MIRROR,
        "scaleFrames": SCALE_FRAMES,
        "classes": CLASSES,
        "threshold": state["threshold"],
        "offset": state["offset"],
        "floats": int(float_block.size),
        "ints": int(int_block.size),
        "layers": layers,
    }
    (WEB / "rep-counter.json").write_text(json.dumps(manifest, indent=1))
    (WEB / "test-vectors.json").write_text(
        json.dumps({"inputs": windows.round(6).tolist(), "outputs": ref.round(6).tolist()})
    )
    return {"bytes": (WEB / "rep-counter.bin").stat().st_size}


def write_video_vectors(model: RepTCN, state: dict) -> None:
    """One whole validation video: raw landmarks in, Python's features, outputs and count out."""
    from .count import count_reps
    from .features import featurize, resample
    from .train import load_split

    items = [i for i in load_split("validation") if 10 < i["t"][-1] < 40 and np.isfinite(i["kp"][:, 0, 0]).mean() > 0.9]
    item = items[0]
    pts, _ = resample(item["kp"], item["t"], 1.0, state["rate"])
    feats = featurize(pts, item["aspect"])
    with torch.no_grad():
        y = model(torch.from_numpy(feats).T[None])[0].T.numpy()
    active = 1 / (1 + np.exp(-y[:, 0]))
    count, running = count_reps(active, y[:, 1:3], state["threshold"], state["offset"])
    kp = item["kp"][:, :, [0, 1, 3]].astype(np.float32)
    landmarks = [None if np.isnan(f[0, 0]) else [[float(v) for v in p] for p in f] for f in kp]
    (WEB / "test-video.json").write_text(json.dumps({
        "video_id": item["video_id"], "true_count": item["count"], "aspect": item["aspect"],
        "t": [float(v) for v in item["t"]], "landmarks": landmarks,
        "features": feats.round(6).tolist(), "outputs": y.round(6).tolist(),
        "count": int(count), "running": running.tolist(),
    }))


def real_windows(window: int, rate: float, n: int = 8) -> np.ndarray:
    """A few feature windows (one receptive field plus the frame itself) from validation videos."""
    from .train import load_split, prepare

    WINDOW = window
    items = load_split("validation")[:n]
    out = []
    for item in items:
        x = prepare(item, 1.0, False, None, rate)["x"]
        start = max(0, len(x) // 2 - WINDOW // 2)
        w = np.zeros((WINDOW, FEATURES), np.float32)
        chunk = x[start : start + WINDOW]
        w[: len(chunk)] = chunk
        out.append(w.T)
    return np.stack(out).astype(np.float32)


def main() -> None:
    EXPORT.mkdir(exist_ok=True)
    model, state = load_model()
    state.setdefault("rate", 15.0)
    fold_batchnorm(model)
    window = receptive_field(model.dilations) + 1
    windows = real_windows(window, state["rate"])
    x1 = windows[:1]

    fp32 = EXPORT / "rep-counter-fp32.onnx"
    torch.onnx.export(LastFrame(model), (torch.from_numpy(x1),), str(fp32), input_names=["features"],
                      output_names=["outputs"], opset_version=18, dynamo=True, external_data=False, verbose=False)
    int8 = EXPORT / "rep-counter-int8.onnx"
    int8_weights(fp32, int8)

    with torch.no_grad():
        torch_out = LastFrame(model)(torch.from_numpy(windows)).numpy()
    diff_fp32 = max(float(np.abs(run_onnx(fp32, w[None]) - torch_out[i : i + 1]).max()) for i, w in enumerate(windows))

    from .evaluate import quantize_weights_

    quantized, _ = load_model()
    fold_batchnorm(quantized)
    quantize_weights_(quantized)
    with torch.no_grad():
        torch_int8 = LastFrame(quantized)(torch.from_numpy(windows)).numpy()
    diff_int8 = max(float(np.abs(run_onnx(int8, w[None]) - torch_int8[i : i + 1]).max()) for i, w in enumerate(windows))

    web = write_web(model, quantized, state, windows)
    write_video_vectors(quantized, state)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "window_frames": window,
        "rate": state["rate"],
        "onnx_fp32_kb": round(fp32.stat().st_size / 1024, 1),
        "onnx_int8_kb": round(int8.stat().st_size / 1024, 1),
        "web_bin_kb": round(web["bytes"] / 1024, 1),
        "max_abs_diff_onnx_fp32_vs_torch": diff_fp32,
        "max_abs_diff_onnx_int8_vs_torch_int8": diff_int8,
        "max_abs_diff_int8_vs_fp32": float(np.abs(torch_int8 - torch_out).max()),
        "ort_ms_fp32": time_onnx(fp32, x1),
        "ort_ms_int8": time_onnx(int8, x1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "export.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
