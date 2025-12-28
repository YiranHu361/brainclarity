import fs from "fs";
import path from "path";
import sharp from "sharp";
import * as ort from "onnxruntime-node";

export type Prediction = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  overlayUrl?: string;
  summary?: string;
  recommendations?: string[];
};

const MODEL_DIR = path.join(process.cwd(), "public", "model");
const MODEL_PATH = path.join(MODEL_DIR, "brain_tumor.onnx");
const CLASS_MAP_PATH = path.join(MODEL_DIR, "class_map.json");
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let classNames: string[] | null = null;

function loadClassNames(): string[] {
  if (classNames) return classNames;
  const raw = fs.readFileSync(CLASS_MAP_PATH, "utf8");
  const parsed = JSON.parse(raw);
  classNames = parsed.classes as string[];
  return classNames;
}

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["cpu"],
    });
  }
  return sessionPromise;
}

async function preprocess(buffer: Buffer): Promise<ort.Tensor> {
  const img = sharp(buffer).resize(256, 256, { fit: "cover" }).removeAlpha();
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const channels = info.channels;
  if (channels < 3) {
    throw new Error("Expected image with at least 3 channels (RGB).");
  }

  // If RGBA, drop alpha
  let rgb = data;
  if (channels > 3) {
    const filtered = Buffer.alloc((data.length / channels) * 3);
    for (let i = 0, j = 0; i < data.length; i += channels, j += 3) {
      filtered[j] = data[i];
      filtered[j + 1] = data[i + 1];
      filtered[j + 2] = data[i + 2];
    }
    rgb = filtered;
  }

  const width = info.width;
  const height = info.height;
  const floatData = new Float32Array(3 * width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 3;
      const r = rgb[idx] / 255;
      const g = rgb[idx + 1] / 255;
      const b = rgb[idx + 2] / 255;
      const outIdx = y * width + x;
      floatData[outIdx] = (r - MEAN[0]) / STD[0];
      floatData[width * height + outIdx] = (g - MEAN[1]) / STD[1];
      floatData[2 * width * height + outIdx] = (b - MEAN[2]) / STD[2];
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, height, width]);
}

function softmax(logits: ort.Tensor): number[] {
  const data = Array.from(logits.data as Float32Array);
  const maxLogit = Math.max(...data);
  const exps = data.map((v) => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sumExp);
}

export async function runInference(buffer: Buffer): Promise<Prediction> {
  const session = await getSession();
  const input = await preprocess(buffer);
  const outputs = await session.run({ input });
  const logits = outputs.logits;
  if (!logits) {
    throw new Error("Model output 'logits' missing.");
  }

  const probs = softmax(logits);
  const labels = loadClassNames();
  const maxIdx = probs.indexOf(Math.max(...probs));
  const label = labels[maxIdx] ?? `class_${maxIdx}`;
  const confidence = probs[maxIdx] ?? 0;
  const probabilities = labels.reduce<Record<string, number>>((acc, curr, idx) => {
    acc[curr] = probs[idx] ?? 0;
    return acc;
  }, {});

  const overlayUrl =
    label.toLowerCase().includes("tumor") || label.toLowerCase().includes("cancer")
      ? "/cases/case-glioma.svg"
      : "/cases/case-healthy.svg";

  const summary =
    label.toLowerCase().includes("tumor") || label.toLowerCase().includes("cancer")
      ? "AI suggests tumor presence in this slice; please review overlay and full study."
      : "No focal lesion detected in this slice; radiologist sign-off required.";

  const recommendations =
    label.toLowerCase().includes("tumor") || label.toLowerCase().includes("cancer")
      ? [
          "Review overlay with radiologist.",
          "Consider prioritizing for neuro consult or follow-up sequencing.",
        ]
      : ["Radiologist review recommended before clearing the study."];

  return { label, confidence, probabilities, overlayUrl, summary, recommendations };
}
