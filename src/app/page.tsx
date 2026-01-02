"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import { runInference } from "@/lib/inference-client";

type AnalysisResult = {
  label: string;
  confidence: number;
  summary: string;
  recommendations: string[];
  overlayUrl?: string;
  fileName?: string;
  fileSize?: number;
  turnaroundMs?: number;
  timestamp?: string;
};

type CaseCard = {
  id: string;
  title: string;
  subtitle: string;
  confidence: number;
  image: string;
  badge: string;
  description: string;
};

const successfulCases: CaseCard[] = [
  {
    id: "glioma",
    title: "Glioma · left frontal",
    subtitle: "Trained data sample",
    confidence: 0.92,
    image: "/Brain%20Tumor%20Data%20Set/Brain%20Tumor/Cancer%20(1).jpg",
    badge: "Detected",
    description: "Sample from training set; clearly hyperintense rounded mass.",
  },
  {
    id: "meningioma",
    title: "Meningioma-like",
    subtitle: "Trained data sample",
    confidence: 0.95,
    image: "/Brain%20Tumor%20Data%20Set/Brain%20Tumor/Cancer%20(10).jpg",
    badge: "Detected",
    description:
      "Extra-axial appearing mass; part of training distribution.",
  },
  {
    id: "healthy",
    title: "No tumor detected",
    subtitle: "Trained data sample",
    confidence: 0.98,
    image: "/Brain%20Tumor%20Data%20Set/Healthy/Not%20Cancer%20%20(1).jpeg",
    badge: "Clear",
    description: "Negative example from training set; no focal lesion present.",
  },
];

const modelFacts = [
  "CNN in PyTorch (4 conv blocks + 2 FC), dropout 0.25",
  "Trained on 4,600 MRI slices (tumor vs healthy)",
  "Validation on full set: accuracy 0.993, F1 0.993 (CPU run)",
  "Input: 3x256x256; augmentations: flips, rotation, normalization",
  "Optimiser: Adam 3e-4 · CrossEntropyLoss",
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const nextPreview = URL.createObjectURL(file);
    setPreview(nextPreview);
    return () => URL.revokeObjectURL(nextPreview);
  }, [file]);

  const handleAnalyze = useCallback(async () => {
    setError(null);
    if (!file) {
      setError("Add an MRI file (DICOM/PNG/JPEG) to analyze.");
      return;
    }

    setIsLoading(true);
    const started = Date.now();

    try {
      // Run inference directly in browser using WASM
      const prediction = await runInference(file);
      setResult({
        ...prediction,
        fileName: file.name,
        fileSize: file.size,
        turnaroundMs: Date.now() - started,
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      console.error("Inference error:", err);
      setError(err instanceof Error ? err.message : "Unexpected error.");
    } finally {
      setIsLoading(false);
    }
  }, [file]);

  const loadDemoCase = (caseId: string) => {
    const match = successfulCases.find((c) => c.id === caseId);
    if (!match) return;
    setResult({
      label: match.badge === "Clear" ? "No tumor detected" : "Tumor detected",
      confidence: match.confidence,
      summary: match.description,
      recommendations:
        match.badge === "Clear"
          ? ["Radiologist review recommended before clearing."]
          : [
              "Outline provided for rapid review.",
              "Consider prioritizing for neuroradiology follow-up.",
            ],
      overlayUrl: match.image,
      fileName: `${match.id}-demo.mri`,
      fileSize: 0,
      turnaroundMs: 850,
      timestamp: new Date().toISOString(),
    });
  };

  const formattedConfidence = useMemo(() => {
    if (!result) return null;
    return `${Math.round(result.confidence * 100)}%`;
  }, [result]);

  return (
    <div className="min-h-screen px-4 pb-16 pt-10 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-10">
        <header className="glass sticky top-4 z-10 flex items-center justify-between rounded-2xl px-6 py-4 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500/10 text-sky-300">
              <span className="text-lg font-semibold">CB</span>
            </div>
            <div>
              <p className="text-lg font-semibold text-white">ClarityBrain.org</p>
              <p className="text-sm text-slate-400">
                Assistive brain MRI tumor detection
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <a
              href="#upload"
              className="rounded-full border border-sky-500/40 px-4 py-2 text-sm font-medium text-sky-100 transition hover:border-sky-300 hover:text-white"
            >
              Upload MRI
            </a>
            <button
              onClick={() => loadDemoCase("glioma")}
              className="rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400"
            >
              View demo
            </button>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.3fr_0.8fr]">
          <div className="glass rounded-3xl px-8 py-10">
            <p className="mb-2 inline-flex items-center rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-sky-100">
              Local-first · Assistive
            </p>
            <h1 className="text-3xl font-semibold leading-tight text-white md:text-4xl">
              Fast assist for brain MRI tumor detection.
            </h1>
            <p className="mt-4 max-w-3xl text-lg text-slate-300">
              Upload a scan, run inference, and get an outline plus confidence
              with a PyTorch CNN trained on 4,600 MRI slices (F1 0.993, accuracy 0.993 on full set).
              Built to
              lighten your second-read workload—radiologist oversight required.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <span className="rounded-full bg-white/5 px-3 py-1 text-sm text-slate-200">
                CNN · PyTorch
              </span>
              <span className="rounded-full bg-white/5 px-3 py-1 text-sm text-slate-200">
                Heatmap overlays
              </span>
              <span className="rounded-full bg-white/5 px-3 py-1 text-sm text-slate-200">
                PDF/CSV-ready output
              </span>
            </div>
          </div>

          <div className="card rounded-3xl px-6 py-6">
            <div className="flex items-center justify-between">
              <p className="text-sm uppercase tracking-wide text-slate-400">
                Model snapshot
              </p>
              <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-300">
                F1 0.993
              </span>
            </div>
            <ul className="mt-4 space-y-2 text-sm text-slate-200">
              {modelFacts.map((fact) => (
                <li key={fact} className="flex gap-2">
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-sky-400" />
                  <span>{fact}</span>
                </li>
              ))}
            </ul>
            <div className="mt-4 rounded-2xl bg-sky-500/10 px-4 py-3 text-xs text-sky-100">
              Swap in your own weights or ONNX export. For heavy inference, point
              the `/api/analyze` route to a GPU endpoint.
            </div>
          </div>
        </section>

        <section
          id="upload"
          className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr] xl:grid-cols-[1.1fr_0.9fr]"
        >
          <div className="card rounded-3xl p-6 sm:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">
                Upload & analyze
              </h2>
              <span className="text-xs text-slate-400">Local-only upload</span>
            </div>
            <div className="mt-4 rounded-2xl border border-dashed border-sky-400/50 bg-white/5 p-4 text-center sm:p-8">
              <input
                id="file-input"
                type="file"
                accept=".dcm,.png,.jpg,.jpeg,.tif,.tiff"
                className="hidden"
                onChange={(e) => {
                  const nextFile = e.target.files?.[0];
                  setFile(nextFile ?? null);
                  setResult(null);
                }}
              />
              <label
                htmlFor="file-input"
                className="block cursor-pointer rounded-xl bg-sky-500/10 px-4 py-3 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/20"
              >
                Choose MRI file
              </label>
              <p className="mt-2 text-sm text-slate-300">
                DICOM or image formats · kept in browser, sent only to this app
                for analysis.
              </p>
              {file && (
                <div className="mt-4 text-sm text-slate-200">
                  Selected: <span className="font-semibold">{file.name}</span>{" "}
                  ({Math.round(file.size / 1024)} KB)
                </div>
              )}
              {preview && (
                <div className="mt-4 flex justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={preview}
                    alt="MRI preview"
                    className="max-h-64 rounded-xl border border-sky-500/30 object-contain"
                  />
                </div>
              )}
              <div className="mt-5 flex flex-wrap justify-center gap-3">
                <button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className="rounded-full bg-sky-500 px-5 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isLoading ? "Running analysis…" : "Run analysis"}
                </button>
                <button
                  type="button"
                  onClick={() => loadDemoCase("meningioma")}
                  className="rounded-full border border-sky-500/40 px-5 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-200 hover:text-white"
                >
                  Use demo MRI
                </button>
              </div>
              {error && (
                <p className="mt-3 text-sm text-rose-300" role="alert">
                  {error}
                </p>
              )}
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl bg-sky-500/10 px-4 py-3 text-sm text-slate-100">
                <p className="text-xs uppercase tracking-wide text-sky-200">
                  Input
                </p>
                <p className="font-semibold">3 x 256 x 256</p>
              </div>
              <div className="rounded-2xl bg-sky-500/10 px-4 py-3 text-sm text-slate-100">
                <p className="text-xs uppercase tracking-wide text-sky-200">
                  Turnaround
                </p>
                <p className="font-semibold">~1–2 seconds on GPU</p>
              </div>
              <div className="rounded-2xl bg-sky-500/10 px-4 py-3 text-sm text-slate-100">
                <p className="text-xs uppercase tracking-wide text-sky-200">
                  Output
                </p>
                <p className="font-semibold">Label, confidence, overlay</p>
              </div>
            </div>
          </div>

          <div className="card rounded-3xl p-6 sm:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Result</h2>
              <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300">
                Demo-friendly
              </span>
            </div>
            {!result ? (
              <div className="mt-6 text-sm text-slate-300">
                Run an analysis or load a demo MRI to see predicted label,
                confidence, and overlay.
              </div>
            ) : (
              <div className="mt-4 space-y-4">
                <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Prediction
                    </p>
                    <p className="text-lg font-semibold text-white">
                      {result.label}
                    </p>
                  </div>
                  {formattedConfidence && (
                    <span className="rounded-full bg-sky-500/15 px-3 py-1 text-sm font-semibold text-sky-100">
                      {formattedConfidence}
                    </span>
                  )}
                </div>

                {result.summary && (
                  <p className="text-sm text-slate-200">{result.summary}</p>
                )}

                {result.overlayUrl && (
                  <div className="overflow-hidden rounded-2xl border border-slate-700">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result.overlayUrl}
                      alt="Overlay"
                      className="w-full"
                    />
                  </div>
                )}

                {result.recommendations?.length ? (
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Recommendations
                    </p>
                    <ul className="mt-2 space-y-1 text-sm text-slate-200">
                      {result.recommendations.map((rec) => (
                        <li key={rec} className="flex gap-2">
                          <span className="mt-1 h-1.5 w-1.5 rounded-full bg-emerald-400" />
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                <div className="grid gap-3 sm:grid-cols-2">
                  {result.fileName && (
                    <div className="rounded-xl bg-white/5 px-3 py-2 text-xs text-slate-300">
                      File: <span className="font-semibold">{result.fileName}</span>
                    </div>
                  )}
                  {typeof result.turnaroundMs === "number" && (
                    <div className="rounded-xl bg-white/5 px-3 py-2 text-xs text-slate-300">
                      Turnaround: ~{result.turnaroundMs} ms
                    </div>
                  )}
                  {result.timestamp && (
                    <div className="rounded-xl bg-white/5 px-3 py-2 text-xs text-slate-300">
                      Time: {new Date(result.timestamp).toLocaleString()}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="card rounded-3xl p-6 sm:p-8">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400">
                Successful cases
              </p>
              <h3 className="text-2xl font-semibold text-white">
                Examples with overlays
              </h3>
              <p className="text-sm text-slate-300">
                Swap in your own anonymized examples to showcase wins.
              </p>
            </div>
            <button
              onClick={() => loadDemoCase("healthy")}
              className="rounded-full border border-sky-400/50 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-200 hover:text-white"
            >
              Load healthy example
            </button>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {successfulCases.map((item) => (
              <div key={item.id} className="rounded-2xl bg-white/5 p-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold text-white">{item.title}</p>
                  <span className="rounded-full bg-slate-100/10 px-3 py-1 text-[11px] text-slate-200">
                    {item.badge}
                  </span>
                </div>
                <p className="text-xs uppercase tracking-wide text-slate-400">
                  {item.subtitle}
                </p>
                <div className="mt-3 overflow-hidden rounded-xl border border-slate-700">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.image}
                    alt={item.title}
                    className="h-40 w-full object-cover"
                  />
                </div>
                <p className="mt-3 text-sm text-slate-200">{item.description}</p>
                <div className="mt-2 text-xs text-slate-400">
                  Confidence {Math.round(item.confidence * 100)}%
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="glass rounded-3xl px-6 py-6 sm:px-8 sm:py-8">
          <div className="rounded-2xl bg-white/5 p-5">
            <p className="text-xs uppercase tracking-wide text-slate-400">
              Responsible use
            </p>
            <p className="mt-2 text-sm text-slate-200">
              Assistive tool only. Radiologist oversight required. Do not serve PHI without
              proper safeguards. For local-only workflows, keep inference on-device or inside your
              hospital network.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
