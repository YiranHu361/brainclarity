# ClarityBrain.org MVP

Fast, assistive brain MRI tumor detection UI built with Next.js (App Router, TypeScript, Tailwind). The MVP includes a stubbed `/api/analyze` endpoint you can swap with your GPU/ONNX inference and demo overlays in `public/cases/`.

## Quick start

```bash
npm install   # already run once by create-next-app
npm run dev   # http://localhost:3000
npm run build # ensure it’s production-ready
npm test      # run vitest unit tests
```

## Customize
- Replace demo overlays: drop anonymized SVG/PNG files into `public/cases/` and adjust the sample case data in `src/app/page.tsx`.
- Real inference: `src/app/api/analyze/route.ts` uses the trained ONNX model in `public/model/brain_tumor.onnx` via `onnxruntime-node`. Swap in your own weights or endpoint if preferred.
- Branding/content: edit `src/app/page.tsx` hero copy, stats, and sections.

## Training
- Dataset expected at `public/Brain Tumor Data Set/{Brain Tumor,Healthy}` (ImageFolder layout).
- Train and export ONNX: `python3 scripts/train.py --epochs 1 --batch_size 32` (adjust epochs for quality—CPU will be slower).
- Outputs: `public/model/brain_tumor.onnx`, `brain_tumor_state_dict.pt`, and `class_map.json`.

## Deploy (Vercel)
1) Push to GitHub. 2) Import into Vercel (Node.js runtime, App Router). 3) Set any API URLs or secrets in Vercel env vars, then redeploy.
2b) Optional logging: set `NEON_DATABASE_URL` to log predictions into Neon (table `inference_logs` auto-created).

## Notes
- Assistive use only; radiologist oversight required.
- Inputs: DICOM/PNG/JPEG/TIFF accepted client-side; adjust accept list as needed.
