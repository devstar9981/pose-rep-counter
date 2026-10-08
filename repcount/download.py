"""Downloads what the pipeline needs, at fixed versions:

    python -m repcount.download

1. Google's MediaPipe Pose Landmarker (lite), checked against the SHA-256 of the file the portfolio
   demo serves, so training and the browser use the same model.
2. RepCount-A (about 9.7 GB, Lance format) at the exact revision used for the results. RepCount-A
   comes from Hu et al., "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for
   Repetitive Action Counting" (CVPR 2022). The videos come from YouTube and are for research use;
   this repository doesn't redistribute them.
"""

import hashlib
import urllib.request

from .data import DATA, HF_REPO, HF_REVISION, ROOT

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
POSE_MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def pose_model() -> None:
    path = ROOT / "models" / "pose_landmarker_lite.task"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != POSE_MODEL_SHA256:
        raise SystemExit(f"{path.name} doesn't match the model the results used (sha256 {digest}).")
    print("MediaPipe pose model:", path)


def repcount() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=["data/*"],
        local_dir=DATA / "repcount",
    )
    print("RepCount-A saved to", path)


if __name__ == "__main__":
    pose_model()
    repcount()
