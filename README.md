# Pose rep counter

Counts exercise repetitions live in the browser from a laptop or phone camera. A 103K-parameter model with INT8 weights (a 106 KB download) reads MediaPipe body landmarks and runs in about a third of a millisecond per frame, entirely on the device. Trained and tested on [RepCount-A](https://github.com/SvipRepetitionCounting/TransRAC), the standard benchmark for repetition counting, and compared with published results.

**Try it:** the live demo on [ruijin-ai-dev.vercel.app](https://ruijin-ai-dev.vercel.app/#demo). Turn on your camera, step back so your whole body is in view, and do a few squats, jumping jacks or front raises. Nothing leaves your device.

## Results

RepCount-A test set, the same videos the published methods report on. MAE is the absolute counting error divided by the true count (lower is better). OBO is the share of videos counted within one repetition (higher is better).

| Method | Input | MAE | OBO |
| --- | --- | ---: | ---: |
| RepNet | video frames | 0.995 | 0.013 |
| TransRAC (CVPR 2022) | video frames | 0.443 | 0.291 |
| PoseRAC | MediaPipe (BlazePose) landmarks | **0.236** | 0.560 |
| **This model, INT8 weights (what the browser runs)** | MediaPipe landmarks | 0.255 | **0.620** |
| This model, FP32 | MediaPipe landmarks | 0.252 | 0.620 |

Published numbers are from Table 3 of the PoseRAC paper (151 test videos: every exercise except "other"). Here it's 150 of those 151: the 151st, `train143`, is labeled 0 repetitions in both public copies of the labels, so its MAE is undefined. Including the one "other" video, the INT8 model scores MAE 0.258 and OBO 0.616 on 151 videos. On average it's off by 3.0 repetitions per video.

By exercise (INT8):

| Exercise | Videos | OBO | MAE |
| --- | ---: | ---: | ---: |
| Jumping jack | 26 | 0.885 | 0.079 |
| Squat | 18 | 0.778 | 0.279 |
| Pull-up | 19 | 0.737 | 0.158 |
| Push-up | 16 | 0.688 | 0.190 |
| Front raise | 18 | 0.611 | 0.290 |
| Sit-up | 20 | 0.450 | 0.392 |
| Pommel horse | 15 | 0.333 | 0.091 |
| Bench press | 18 | 0.333 | 0.590 |

Pommel horse videos have many repetitions (often 30 or more), so the counts are close in relative terms (MAE 0.09) but rarely within one. Bench press and sit-ups are the hardest, probably because the person is lying down and partly hidden by the bench or the floor.

**Deployment numbers** (Intel Core i7-9700F; GTX 1050 Ti for the browser test):

| | |
| --- | --- |
| Parameters | 102,541 (BatchNorm folded) |
| Browser weights (INT8 + scales) | 106 KB |
| ONNX, FP32 / INT8 weights | 456 KB / 166 KB |
| ONNX Runtime, one 128-frame window, 1 thread | 0.34 ms (FP32), 0.39 ms (INT8) |
| Browser engine, per new frame (streaming) | 0.35 ms median |
| INT8 vs FP32 on the test set | MAE 0.255 vs 0.252, OBO identical |

INT8 here means weight-only quantization: it makes the download 2.7 times smaller and leaves accuracy unchanged, but it doesn't make inference faster, since the weights are restored to float when the model loads.

**End to end in a real browser.** `web/browser_test.mjs` opens the portfolio in headless Chrome with a test video as the camera (Chrome's fake-camera flags), presses "Turn on camera" and reads the count on screen. On test video `stu9_64` (14 squats) the page counted 15, the same number the offline evaluation gives for that video, with MediaPipe running at about 50 fps on the GPU and no console errors.

## How it works

```
camera frame → MediaPipe Pose (lite) → 33 landmarks → 15 fps resampling → 40 features per frame
             → causal TCN (103K parameters, INT8 weights) → rep in progress? how far through it? which exercise?
             → phase counter → reps
```

1. **Pose.** Google's MediaPipe Pose Landmarker (the lite model) finds 33 body landmarks per frame. Training uses the exact model file the website runs, in the same mode (`VIDEO`, one person, default thresholds), so the model learns from the same kind of landmarks it sees live.
2. **Features.** Landmarks are resampled to a steady 15 frames per second, so the camera's frame rate doesn't matter. A new frame between two detections is interpolated when they're at most 0.3 seconds apart and left empty otherwise; that rule needs nothing from the future, so the browser applies it live. Thirteen joints (nose, shoulders, elbows, wrists, hips, knees, ankles) are centered on the hips and divided by the median torso length over the last 64 frames, which removes where the person stands and how far they are from the camera. Each frame becomes 40 numbers: 26 coordinates, 13 visibilities and a "person found" flag.
3. **Model.** A causal temporal convolutional network: six residual blocks with dilated convolutions (dilations 1 to 32), so each prediction sees the last 127 frames, about 8.5 seconds, and nothing from the future. For every frame it predicts whether a repetition is in progress, how far through it we are (the cosine and sine of a phase angle that goes once around per rep) and which exercise it is.
4. **Counting.** While a rep is in progress, the counter adds up how far the phase angle moves, so small steps backward and forward cancel out, and reports completed laps. Fed the true labels, this rule scores MAE 0.007 on validation, so nearly all of the error comes from the model, not the counter.

**Training.** Labels come from RepCount-A's start and end frame of every repetition: frames inside a rep get its progress, from 0 at the start to 1 at the end. Each training video is used at five playback speeds (0.75x to 1.35x), mirrored and not, with a small random rotation and zoom, plus coordinate noise. AdamW with a one-cycle schedule, 40 epochs of 150 batches of 32 random 256-frame slices, and an exponential moving average of the weights, which is what gets evaluated and saved. About 3.5 minutes on a GTX 1050 Ti.

**Model selection.** Seven runs compared 15 and 30 frames per second, 63 and 127 frames of context, and averaged versus raw weights, on the validation set only. 30 fps was no better than 15, so the cheaper 15 fps model ships. The activity threshold (0.6) and the counting offset (0.7) were then tuned on validation, where the chosen checkpoint scores MAE 0.281 and OBO 0.615. The test set was scored once, at the end.

## Deployment

`python -m repcount.export` writes:

- `export/rep-counter-fp32.onnx` and `export/rep-counter-int8.onnx`: one 128-frame window in, the last frame's outputs out. The INT8 version stores each convolution weight as int8 with one scale per output channel (`DequantizeLinear`); activations stay float.
- `web/rep-counter.bin` and `web/rep-counter.json`: the same INT8 weights in a compact layout for the browser, with the settings the browser needs (frame rate, joints, threshold, offset).

The browser engine ([`rep-counter.ts`](https://github.com/ruijindev0801/Portfolio2026/blob/master/src/lib/rep-counter.ts), about 350 lines of TypeScript with no dependencies) reproduces the resampling, features, network and counter. It runs the network as a stream: each dilated layer keeps only the past inputs it needs, so a new frame costs about 100 thousand multiply-adds instead of re-running a whole window.

**Parity.** `web/parity.mjs` runs the TypeScript engine in Node against files the export writes:

| Check | Max absolute difference |
| --- | ---: |
| Network outputs on 8 real windows, TypeScript vs PyTorch (INT8 weights) | 0.000007 |
| Features on a full validation video, from raw landmarks | 0.000001 |
| Model outputs on that video, frame by frame | 0.000005 |
| Count on that video | identical (9) |

## Reproduce

Python 3.12. A CPU is enough; a GPU makes training a few minutes instead of an hour.

```bash
pip install -r requirements.txt
python -m repcount.download              # RepCount-A (9.7 GB) and the MediaPipe model, pinned and checked
python -m repcount.data                  # labels -> data/labels.json
python -m repcount.extract --workers 7   # MediaPipe landmarks for all 1,041 videos (about 95 minutes on 8 cores)
python -m repcount.train --dilations 1,2,4,8,16,32 --seed 2
python -m repcount.evaluate              # test set, FP32
python -m repcount.evaluate --int8       # test set, INT8 weights (what the browser runs)
python -m repcount.export                # ONNX, browser weights, test vectors
node --experimental-transform-types web/parity.mjs path/to/rep-counter.ts
node --experimental-transform-types web/bench.mjs path/to/rep-counter.ts
```

`checkpoints/best.pt` is the model behind every number above, and `results/` holds the full test results, including every video's predicted and true count.

## Limitations

- **The test set is small.** 150 videos means one video is worth 0.7 points of OBO, so differences of a few points between methods aren't meaningful on their own.
- **The comparison isn't perfectly matched.** PoseRAC trained on 487 videos of the eight exercises with its own pose annotations. This model trains on all 758 training videos with RepCount's original start and end labels. Both use MediaPipe landmarks and the same test videos.
- **Lying-down exercises are weak.** Bench press and sit-ups score lowest. The likely reason is that the body is partly hidden by the bench or the floor, so the landmarks the model reads are less reliable.
- **Live use works best with the whole body in view,** one person, and a camera roughly facing them.
- **The INT8 model isn't faster,** only smaller (see above).

## Credits

- RepCount-A: Hu et al., "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting", CVPR 2022. The videos come from YouTube and are used here for research only; this repository doesn't redistribute them. The copy used is the [Lance packaging on Hugging Face](https://huggingface.co/datasets/lmms-lab-eval/repcounta-lance), pinned to a fixed revision.
- PoseRAC: Yao, Cheng and Zou, "PoseRAC: Pose Saliency Transformer for Repetitive Action Counting", 2023. The comparison numbers come from its Table 3.
- MediaPipe Pose Landmarker (lite): Google. `repcount.download` fetches it from Google and checks it against the SHA-256 of the file the portfolio serves.

Code: [Apache 2.0](LICENSE).
