# BG Remover

AI background remover: upload an image, get a transparent PNG. Node.js (Express) backend + a single static page (`static/index.html`), no build step.

**Engine:** the [rembg](https://github.com/danielgatis/rembg) `isnet-general-use` and `u2net` ONNX models on `onnxruntime-node`, with rembg's preprocessing ported to JS (`remove-bg.js`). isnet gives the sharp edges; u2net locates the subject, and isnet pixels far from it are zeroed. That removes leftover background text/logos/clutter that isnet alone keeps, at ~1.3 s/image on CPU. Benchmarked against `birefnet-general` (9 s, similar quality) and `bria-rmbg` (10 s, non-commercial license).

## Run locally

```bash
npm install
npm start
```

Open http://localhost:8000. The first start downloads both models (~355 MB, checksum-verified) into `./models`; `/api/health` reports `loading` until they're ready.

Smoke test (real models, real images): `npm test`.

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `PORT` | `8000` | |
| `MAX_UPLOAD_MB` | `10` | Upload size cap. |
| `MAX_SIDE` | `4096` | Larger images are downscaled before processing. |
| `MODEL_DIR` | `./models` | Where models are stored. |

## Deploy

Any Node ≥ 20 host (Hostinger Node.js hosting, VPS, Render, …): `npm install && npm start`, entry file `server.js`.

- **RAM:** ~1–1.5 GB. One inference runs at a time per process.
- `.npmrc` skips onnxruntime's optional CUDA download, so installs on Linux stay small.
- **Images are never written to disk** — uploads are held in memory and discarded after the response.
- Put HTTPS and rate limiting at the proxy/host level.
