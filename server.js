import express from "express";
import multer from "multer";
import path from "node:path";
import { BadImage, loadModels, modelsReady, removeBackground } from "./remove-bg.js";

const MAX_MB = Number(process.env.MAX_UPLOAD_MB || 10);
const PORT = Number(process.env.PORT || 8000);

const app = express();
app.disable("x-powered-by");
app.set("trust proxy", 1);
app.use((req, res, next) => {
  res.set({
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy":
      "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
  });
  next();
});

// Memory storage: uploads never touch disk, nothing to clean up.
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_MB * 1024 * 1024, files: 1 } });

// ponytail: one inference at a time per process (CPU-bound, caps memory); add a real queue if you scale out
let chain = Promise.resolve();
const serial = (fn) => (chain = chain.then(fn, fn));

app.post("/api/remove", upload.single("file"), async (req, res) => {
  if (!modelsReady()) return res.status(503).json({ detail: "The AI model is still loading. Try again in a minute." });
  if (!req.file?.size) return res.status(400).json({ detail: "Empty file." });
  try {
    const png = await serial(() => removeBackground(req.file.buffer));
    res.set("Cache-Control", "no-store").type("png").send(png);
  } catch (err) {
    if (err instanceof BadImage) return res.status(err.status).json({ detail: err.message });
    console.error(err);
    res.status(500).json({ detail: "Processing failed. Please try another image." });
  }
});

app.get("/api/health", (req, res) => res.json({ status: modelsReady() ? "ok" : "loading" }));
app.get("/api/config", (req, res) => res.json({ maxMb: MAX_MB }));
app.use(express.static(path.join(import.meta.dirname, "static"), { index: "index.html" }));

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError)
    return res.status(err.code === "LIMIT_FILE_SIZE" ? 413 : 400)
      .json({ detail: err.code === "LIMIT_FILE_SIZE" ? `File is larger than ${MAX_MB} MB.` : "Invalid upload." });
  next(err);
});

app.listen(PORT, () => console.log(`Listening on :${PORT}`));
loadModels().then(() => console.log("Models ready")).catch((e) => console.error("Model load failed:", e));
