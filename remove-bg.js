// Background removal: port of rembg's isnet-general-use + u2net sessions, with u2net gating.
// isnet gives the sharp edges; u2net decides where the subject is. isnet alone sometimes
// keeps faint background text/logos, u2net alone has soft 320px edges.
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import path from "node:path";
import ort from "onnxruntime-node";
import sharp from "sharp";

const MODEL_DIR = process.env.MODEL_DIR || path.join(import.meta.dirname, "models");
const RELEASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0";
const MODELS = {
  isnet: { file: "isnet-general-use.onnx", md5: "fc16ebd8b0c10d971d3513d564d01e29", size: 1024, mean: [0.5, 0.5, 0.5], std: [1, 1, 1] },
  u2net: { file: "u2net.onnx", md5: "60024c5c889badc19c04ad937298a77b", size: 320, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};
export const MAX_SIDE = Number(process.env.MAX_SIDE || 4096);
const sessions = {};

async function ensureModel({ file, md5 }) {
  const dest = path.join(MODEL_DIR, file);
  if (existsSync(dest)) return dest;
  mkdirSync(MODEL_DIR, { recursive: true });
  console.log(`Downloading ${file}…`);
  const res = await fetch(`${RELEASE}/${file}`);
  if (!res.ok) throw new Error(`Model download failed: ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest + ".part"));
  const got = createHash("md5").update(readFileSync(dest + ".part")).digest("hex");
  if (got !== md5) throw new Error(`Checksum mismatch for ${file}`);
  renameSync(dest + ".part", dest);
  return dest;
}

export const downloadModels = () => Promise.all(Object.values(MODELS).map(ensureModel));

export async function loadModels() {
  for (const [name, m] of Object.entries(MODELS)) {
    sessions[name] = await ort.InferenceSession.create(await ensureModel(m), {
      enableCpuMemArena: false, // release memory between requests
      graphOptimizationLevel: "all",
    });
  }
}

export const modelsReady = () => Object.keys(sessions).length === Object.keys(MODELS).length;

// rembg BaseSession.normalize + predict: resize, scale by max pixel, mean/std, CHW → min-max mask.
async function predict(name, rgb, width, height) {
  const { size, mean, std } = MODELS[name];
  const px = await sharp(rgb, { raw: { width, height, channels: 3 } })
    .resize(size, size, { fit: "fill", kernel: "lanczos3" }).raw().toBuffer();
  let max = 1e-6;
  for (const v of px) if (v > max) max = v;
  const n = size * size, input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) input[c * n + i] = (px[i * 3 + c] / max - mean[c]) / std[c];

  const s = sessions[name];
  const out = (await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, size, size]) }))[s.outputNames[0]].data;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { if (out[i] < lo) lo = out[i]; if (out[i] > hi) hi = out[i]; }
  const mask = Buffer.alloc(n);
  for (let i = 0; i < n; i++) mask[i] = Math.min(255, Math.max(0, ((out[i] - lo) / (hi - lo)) * 255));
  return mask; // size × size, single channel
}

const resizeMask = (mask, from, w, h, kernel) =>
  sharp(mask, { raw: { width: from, height: from, channels: 1 } })
    .resize(w, h, { fit: "fill", kernel }).extractChannel(0).raw().toBuffer();

// Where u2net sees the subject, grown by 8px at ≤320px scale (~2.5%) so hair/edges survive.
async function subjectGate(rgb, width, height) {
  const scale = Math.min(1, 320 / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale));
  const small = await resizeMask(await predict("u2net", rgb, width, height), 320, w, h, "lanczos3");
  let cur = Uint8Array.from(small, (v) => (v > 25 ? 1 : 0));
  for (let it = 0; it < 8; it++) { // binary dilation, 4-connected (scipy default structure)
    const next = cur.slice();
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!cur[i] && ((x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w]))) next[i] = 1;
      }
    cur = next;
  }
  return sharp(Buffer.from(cur.map((v) => v * 255)), { raw: { width: w, height: h, channels: 1 } })
    .resize(width, height, { fit: "fill", kernel: "linear" }).extractChannel(0).raw().toBuffer();
}

export class BadImage extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}

export async function removeBackground(input) {
  let meta;
  try {
    meta = await sharp(input, { limitInputPixels: 60_000_000 }).metadata();
  } catch {
    throw new BadImage("The file is not a valid image, or it is too large.");
  }
  if (!["jpeg", "png", "webp"].includes(meta.format)) throw new BadImage("Only JPEG, PNG and WebP images are supported.", 415);

  const { data: rgb, info } = await sharp(input, { limitInputPixels: 60_000_000 })
    .rotate() // apply EXIF orientation
    .resize(MAX_SIDE, MAX_SIDE, { fit: "inside", withoutEnlargement: true })
    .toColourspace("srgb").removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = info;

  const edges = await resizeMask(await predict("isnet", rgb, width, height), 1024, width, height, "lanczos3");
  const gate = await subjectGate(rgb, width, height);
  const alpha = Buffer.alloc(width * height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = (edges[i] * gate[i]) / 255;

  return sharp(rgb, { raw: { width, height, channels: 3 } })
    .joinChannel(alpha, { raw: { width, height, channels: 1 } })
    .png({ compressionLevel: 3 }).toBuffer();
}
