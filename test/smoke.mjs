// Smoke test: real models, real images, real HTTP. Run: npm test
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import sharp from "sharp";

const PORT = 8123, URL = `http://127.0.0.1:${PORT}`;
const server = spawn(process.execPath, ["server.js"], { env: { ...process.env, PORT }, stdio: "inherit" });

try {
  for (let i = 0; ; i++) { // wait for models
    const h = await fetch(`${URL}/api/health`).then((r) => r.json()).catch(() => null);
    if (h?.status === "ok") break;
    if (i > 600) throw new Error("server never became ready");
    await new Promise((r) => setTimeout(r, 500));
  }
  const post = (name, bytes, type = "image/jpeg") => {
    const form = new FormData();
    form.append("file", new Blob([bytes], { type }), name);
    return fetch(`${URL}/api/remove`, { method: "POST", body: form });
  };

  let r = await post("a.jpg", readFileSync("test/animal-1.jpg"));
  assert.equal(r.status, 200);
  const out = sharp(Buffer.from(await r.arrayBuffer()));
  assert.equal((await out.metadata()).format, "png");
  const alpha = await out.extractChannel(3).raw().toBuffer();
  assert.ok(alpha.includes(0) && alpha.includes(255), "expected both transparent and opaque pixels");

  // Background clutter (sign + shopping cart, left of the subject) must be dropped.
  r = await post("g.jpg", readFileSync("test/anime-girl-2.jpg"));
  const img = sharp(Buffer.from(await r.arrayBuffer()));
  const { width, height } = await img.metadata();
  const left = await img.extractChannel(3).extract({ left: 0, top: 0, width: Math.floor(width * 0.4), height }).raw().toBuffer();
  const leak = left.filter((v) => v > 128).length / left.length;
  assert.ok(leak < 0.01, `background clutter kept: ${(leak * 100).toFixed(1)}%`);

  assert.equal((await post("x.jpg", Buffer.from("not an image"))).status, 400);
  assert.equal((await post("x.gif", Buffer.from("GIF89a\x01\x00\x01\x00\x00\x00\x00;"), "image/gif")).status, 400);
  const gif = await sharp({ create: { width: 4, height: 4, channels: 3, background: "red" } }).gif().toBuffer();
  assert.equal((await post("x.gif", gif, "image/gif")).status, 415);
  assert.equal((await post("x.jpg", Buffer.alloc(0))).status, 400);
  assert.equal((await post("big.jpg", Buffer.alloc(11 * 1024 * 1024))).status, 413);
  console.log("ok");
} finally {
  server.kill();
}
