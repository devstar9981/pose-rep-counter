// Times the browser engine's streaming step (one new frame) in Node.
//   node --experimental-transform-types web/bench.mjs <path to rep-counter.ts>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const { RepModel } = await import(pathToFileURL(process.argv[2]).href);
const here = new URL(".", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("rep-counter.json", here)));
const bin = readFileSync(new URL("rep-counter.bin", here));
const model = new RepModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const frame = Float32Array.from({ length: manifest.features }, (_, i) => Math.sin(i));
for (let i = 0; i < 2000; i++) model.step(frame);
const times = [];
for (let i = 0; i < 20000; i++) {
  const t = performance.now();
  model.step(frame);
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
console.log(`streaming step: median ${(times[10000] * 1000).toFixed(1)} µs, p95 ${(times[19000] * 1000).toFixed(1)} µs per frame`);
