"""Turns a RepCount video into a .y4m file that Chrome can use as a fake webcam.

    python web/make_y4m.py <split> <video_id> <out.y4m> [lead-in seconds]

Chrome flags: --use-fake-device-for-media-stream --use-file-for-fake-video-capture=<out.y4m>
The frames are letterboxed into 640x480, the size the portfolio demo asks the camera for. An optional
lead-in of black frames gives the page time to load its models before the person appears, so a test
sees every repetition.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from repcount.data import load_labels, video_bytes  # noqa: E402

split, video_id, out = sys.argv[1:4]
lead_in = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
row = next(i for i, v in enumerate(load_labels()[split]) if v["video_id"] == video_id)
[(_, ext, blob)] = video_bytes(split, [row])
src = f"{out}.src{ext}"
with open(src, "wb") as f:
    f.write(blob)
capture = cv2.VideoCapture(src)
fps = capture.get(cv2.CAP_PROP_FPS) or 30
W, H = 640, 480
frames = 0
with open(out, "wb") as f:
    f.write(f"YUV4MPEG2 W{W} H{H} F{round(fps)}:1 Ip A1:1 C420jpeg\n".encode())
    black = cv2.cvtColor(np.zeros((H, W, 3), np.uint8), cv2.COLOR_BGR2YUV_I420).tobytes()
    for _ in range(round(lead_in * fps)):
        f.write(b"FRAME\n" + black)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = min(W / w, H / h)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
        canvas = np.zeros((H, W, 3), np.uint8)
        y0, x0 = (H - resized.shape[0]) // 2, (W - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        f.write(b"FRAME\n" + cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420).tobytes())
        frames += 1
print(f"{out}: {lead_in:.0f} s lead-in + {frames} video frames at {fps:.1f} fps")
