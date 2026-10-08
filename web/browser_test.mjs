// Runs the portfolio's pose demo in headless Chrome with a video file as the camera, and records
// the live rep count it shows.
//   node web/browser_test.mjs <site url> <video.y4m> <seconds> <out prefix>
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [url, video, seconds = "30", out = "browser-test"] = process.argv.slice(2);
const PORT = 9411;
const chrome = spawn(
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    "--hide-scrollbars",
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-video-capture=${resolve(video)}`,
    "--autoplay-policy=no-user-gesture-required",
    `--user-data-dir=${mkdtempSync(join(tmpdir(), "reps-"))}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let wsUrl;
for (let i = 0; i < 100 && !wsUrl; i++) {
  try {
    wsUrl = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page")?.webSocketDebuggerUrl;
  } catch {}
  if (!wsUrl) await sleep(200);
}
const ws = new WebSocket(wsUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
const logs = [];
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id) {
    pending.get(m.id)?.(m);
    pending.delete(m.id);
    return;
  }
  if (m.method === "Runtime.exceptionThrown") logs.push(`exception: ${m.params.exceptionDetails.exception?.description}`);
  if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type))
    logs.push(`${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 200)}`);
};
const send = (method, params = {}) =>
  new Promise((r) => {
    const n = ++id;
    pending.set(n, r);
    ws.send(JSON.stringify({ id: n, method, params }));
  });
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url });
await sleep(4000);
const clicked = await evaluate(`(() => {
  const b = [...document.querySelectorAll("#demo button")].find((el) => el.textContent.includes("Turn on camera"));
  b?.scrollIntoView({ block: "center" }); b?.click(); return Boolean(b);
})()`);
console.log("clicked Turn on camera:", clicked);
const readout = `(() => {
  const n = document.querySelector("#demo p.font-mono");
  const label = n?.nextElementSibling?.textContent ?? "";
  const stats = [...document.querySelectorAll("#demo p")].map((p) => p.textContent).find((t) => / fps /.test(t)) ?? "";
  return { count: n ? Number(n.textContent) : null, label, stats };
})()`;
const samples = [];
const started = Date.now();
while (Date.now() - started < Number(seconds) * 1000) {
  await sleep(1000);
  const r = await evaluate(readout);
  samples.push({ s: Math.round((Date.now() - started) / 1000), ...r });
  console.log(`${samples.at(-1).s}s  count ${r.count}  ${r.label}  ${r.stats}`);
}
const rect = await evaluate(`(() => { const r = document.querySelector("#demo").getBoundingClientRect(); return { x: r.left, y: r.top + scrollY, w: r.width, h: r.height }; })()`);
const shot = await send("Page.captureScreenshot", { format: "png", clip: { x: rect.x, y: rect.y, width: rect.w, height: rect.h, scale: 1 } });
writeFileSync(`${out}.png`, Buffer.from(shot.result.data, "base64"));
writeFileSync(`${out}.json`, JSON.stringify({ samples, logs }, null, 1));
console.log(logs.length ? `console:\n  ${logs.join("\n  ")}` : "console: clean");
ws.close();
chrome.kill();
