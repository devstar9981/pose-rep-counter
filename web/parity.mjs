// Checks the portfolio's TypeScript engine against PyTorch, using the files export.py writes.
//   node --experimental-transform-types web/parity.mjs <path to rep-counter.ts>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const engine = process.argv[2] ?? "../Portfolio/src/lib/rep-counter.ts";
const { RepModel, RepCounter } = await import(pathToFileURL(engine).href);
const here = new URL(".", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("rep-counter.json", here)));
const bin = readFileSync(new URL("rep-counter.bin", here));
const model = new RepModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));

// 1. The network alone: stream each 64-frame window from a clean state; compare the last outputs.
const vectors = JSON.parse(readFileSync(new URL("test-vectors.json", here)));
let worst = 0;
vectors.inputs.forEach((window, w) => {
  model.reset();
  let out;
  for (let f = 0; f < manifest.window; f++) out = model.step(Float32Array.from(window.map((row) => row[f])));
  out.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - vectors.outputs[w][i]))));
});
console.log(`network: ${vectors.inputs.length} windows, max abs diff ${worst.toExponential(2)}`);

// 2. Everything: raw landmarks of a whole video -> resampling, features, network, counting.
const video = JSON.parse(readFileSync(new URL("test-video.json", here)));
model.reset();
const counter = new RepCounter(model);
const frames = [];
counter.onFrame = (features, outputs) => frames.push([Float32Array.from(features), Float32Array.from(outputs)]);
let last;
video.t.forEach((t, i) => {
  const lm = video.landmarks[i];
  const landmarks = lm && Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
  if (lm) manifest.joints.forEach((index, j) => (landmarks[index] = { x: lm[index][0], y: lm[index][1], visibility: lm[index][2] }));
  last = counter.push(t * 1000, lm ? landmarks : null, video.aspect);
});
const n = Math.min(frames.length, video.features.length);
let featDiff = 0;
let outDiff = 0;
for (let f = 0; f < n; f++) {
  frames[f][0].forEach((v, i) => (featDiff = Math.max(featDiff, Math.abs(v - video.features[f][i]))));
  frames[f][1].forEach((v, i) => (outDiff = Math.max(outDiff, Math.abs(v - video.outputs[f][i]))));
}
console.log(`video ${video.video_id}: ${frames.length} frames in the browser engine, ${video.features.length} in Python`);
console.log(`  features max abs diff ${featDiff.toExponential(2)}, outputs max abs diff ${outDiff.toExponential(2)}`);
console.log(`  count: browser ${last.count}, Python ${video.running[n - 1]} at the same frame, true ${video.true_count}`);
