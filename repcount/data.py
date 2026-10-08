"""RepCount-A (the LLSP split of Hu et al., CVPR 2022), as packaged in Lance format on Hugging Face.

Each row is one video with its exercise, its repetition count and the start and end frame of every
repetition. The video bytes are stored in the row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPCOUNT = DATA / "repcount" / "data"
POSES = DATA / "poses"
SPLITS = ("train", "validation", "test")

HF_REPO = "lmms-lab-eval/repcounta-lance"
HF_REVISION = "ae2e0b0496124d900d4e54a8e92f7ec15a292afd"

# RepCount spells some exercises two ways ("push_up" and "pushups"); these are the names we use.
CLASSES = [
    "battle_rope",
    "bench_press",
    "front_raise",
    "jumping_jack",
    "pommel_horse",
    "pull_up",
    "push_up",
    "sit_up",
    "squat",
    "other",
]
ALIASES = {
    "battlerope": "battle_rope",
    "battle_rope": "battle_rope",
    "benchpressing": "bench_press",
    "bench_pressing": "bench_press",
    "frontraise": "front_raise",
    "front_raise": "front_raise",
    "jumpjacks": "jumping_jack",
    "jump_jack": "jumping_jack",
    "jumping_jack": "jumping_jack",
    "pommelhorse": "pommel_horse",
    "pommel_horse": "pommel_horse",
    "pullups": "pull_up",
    "pull_up": "pull_up",
    "pushups": "push_up",
    "push_up": "push_up",
    "situp": "sit_up",
    "sit_up": "sit_up",
    "squat": "squat",
    "squant": "squat",  # a typo in the RepCount labels
    "others": "other",
    "other": "other",
}


def canonical(action: str) -> str:
    key = action.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in ALIASES:
        raise KeyError(f"Unknown RepCount action {action!r}; add it to ALIASES")
    return ALIASES[key]


@dataclass
class Video:
    split: str
    video_id: str
    action: str
    count: int
    # (start, end) frame of each repetition, in the video's own frame numbers.
    cycles: list[tuple[int, int]]


def parse_cycles(raw: str | None) -> list[tuple[int, int]]:
    """RepCount stores markers L1, L2, L3, ... as start, end, start, end of each repetition."""
    if not raw:
        return []
    marks = [int(round(float(m))) for m in json.loads(raw) if m is not None and str(m) != "nan"]
    return [(marks[i], marks[i + 1]) for i in range(0, len(marks) - 1, 2) if marks[i + 1] > marks[i]]


def open_split(split: str):
    import lance

    return lance.dataset(str(REPCOUNT / f"{split}.lance"))


def videos(split: str) -> list[Video]:
    """Every video in a split, without the video bytes."""
    table = open_split(split).to_table(columns=["video_id", "action_type", "count", "cycle_bounds_json"])
    out = []
    for row in table.to_pylist():
        out.append(
            Video(
                split=split,
                video_id=str(row["video_id"]),
                action=canonical(str(row["action_type"])),
                count=int(row["count"]) if row["count"] is not None else -1,
                cycles=parse_cycles(row["cycle_bounds_json"]),
            )
        )
    return out


def video_bytes(split: str, indices: list[int]) -> list[tuple[str, str, bytes]]:
    """(video_id, extension, bytes) for the given row numbers of a split."""
    ds = open_split(split)
    meta = ds.take(indices, columns=["video_id", "video_ext"]).to_pylist()
    blobs = ds.take_blobs("video_blob", indices=indices)
    return [(str(m["video_id"]), "." + str(m["video_ext"] or "mp4").lstrip("."), b.read()) for m, b in zip(meta, blobs)]


LABELS = DATA / "labels.json"


def export_labels() -> None:
    """Saves every split's labels to data/labels.json, so training doesn't need the video files."""
    out = {split: [vars(v) for v in videos(split)] for split in SPLITS if (REPCOUNT / f"{split}.lance").exists()}
    LABELS.write_text(json.dumps(out, indent=1))
    for split, rows in out.items():
        print(split, len(rows), "videos")


def load_labels() -> dict[str, list[dict]]:
    return json.loads(LABELS.read_text())


if __name__ == "__main__":
    export_labels()
