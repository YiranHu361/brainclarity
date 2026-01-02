"use client";

import * as ort from "onnxruntime-web";

export type Prediction = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  overlayUrl: string;
  summary: string;
  recommendations: string[];
};

const MODEL_PATH = "/model/brain_tumor.onnx";
const CLASS_MAP_PATH = "/model/class_map.json";
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let classNames: string[] | null = null;
let initialized = false;

async function initOrt(): Promise<void> {
  if (initialized) return;

  // Configure WASM settings before any session creation
  // Use exact version matching the package.json
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  // Disable multi-threading to avoid potential issues
  ort.env.wasm.numThreads = 1;
  // Disable SIMD if causing issues (uncomment if needed)
  // ort.env.wasm.simd = false;

  initialized = true;
}

async function loadClassNames(): Promise<string[]> {
  if (classNames) return classNames;
  const response = await fetch(CLASS_MAP_PATH);
  const parsed = await response.json();
  classNames = parsed.classes as string[];
  return classNames;
}

async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      await initOrt();

      const response = await fetch(MODEL_PATH);
      if (!response.ok) {
        throw new Error(`Failed to load model: ${response.status}`);
      }
      const modelBuffer = await response.arrayBuffer();

      // Create session with explicit options
      return ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "basic",
      });
    })();
  }
  return sessionPromise;
}

async function preprocessImage(file: File): Promise<ort.Tensor> {
  // Create an image bitmap from the file
  const bitmap = await createImageBitmap(file);

  // Create a canvas to resize and extract pixel data
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");

  // Draw and resize image to 256x256
  ctx.drawImage(bitmap, 0, 0, 256, 256);
  const imageData = ctx.getImageData(0, 0, 256, 256);
  const { data } = imageData;

  // Convert to float32 tensor with normalization
  // ONNX expects [1, 3, 256, 256] in CHW format
  const floatData = new Float32Array(3 * 256 * 256);

  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const pixelIdx = (y * 256 + x) * 4; // RGBA
      const r = data[pixelIdx] / 255;
      const g = data[pixelIdx + 1] / 255;
      const b = data[pixelIdx + 2] / 255;

      const outIdx = y * 256 + x;
      floatData[outIdx] = (r - MEAN[0]) / STD[0];                    // R channel
      floatData[256 * 256 + outIdx] = (g - MEAN[1]) / STD[1];        // G channel
      floatData[2 * 256 * 256 + outIdx] = (b - MEAN[2]) / STD[2];    // B channel
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, 256, 256]);
}

function softmax(logits: ort.Tensor): number[] {
  const data = Array.from(logits.data as Float32Array);
  const maxLogit = Math.max(...data);
  const exps = data.map((v) => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sumExp);
}

export async function runInference(file: File): Promise<Prediction> {
  const session = await getSession();
  const input = await preprocessImage(file);
  const outputs = await session.run({ input });
  const logits = outputs.logits;
  if (!logits) {
    throw new Error("Model output 'logits' missing.");
  }

  const probs = softmax(logits);
  const labels = await loadClassNames();
  const maxIdx = probs.indexOf(Math.max(...probs));
  const label = labels[maxIdx] ?? `class_${maxIdx}`;
  const confidence = probs[maxIdx] ?? 0;
  const probabilities = labels.reduce<Record<string, number>>((acc, curr, idx) => {
    acc[curr] = probs[idx] ?? 0;
    return acc;
  }, {});

  const isTumor = label.toLowerCase().includes("tumor") || label.toLowerCase().includes("cancer");

  const overlayUrl = isTumor ? "/cases/case-glioma.svg" : "/cases/case-healthy.svg";

  const summary = isTumor
    ? "AI suggests tumor presence in this slice; please review overlay and full study."
    : "No focal lesion detected in this slice; radiologist sign-off required.";

  const recommendations = isTumor
    ? [
        "Review overlay with radiologist.",
        "Consider prioritizing for neuro consult or follow-up sequencing.",
      ]
    : ["Radiologist review recommended before clearing the study."];

  return { label, confidence, probabilities, overlayUrl, summary, recommendations };
}
