import { NextResponse } from "next/server";
import { runInference } from "@/lib/inference";
import { logPrediction } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;
export async function GET() {
  return NextResponse.json({ status: "ok", message: "Use POST with multipart/form-data" });
}

export async function OPTIONS() {
  return NextResponse.json({ status: "ok" });
}

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10MB

export async function POST(request: Request) {
  let fileName = "scan";
  let fileSize = 0;
  const started = Date.now();

  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "Expected a `file` field with an MRI image." },
        { status: 400 },
      );
    }

    fileName = file.name || "scan";
    fileSize = file.size || 0;

    if (fileSize === 0) {
      return NextResponse.json({ error: "File is empty." }, { status: 400 });
    }

    if (fileSize > MAX_FILE_BYTES) {
      return NextResponse.json(
        { error: "File too large. Please upload a file under 10MB." },
        { status: 413 },
      );
    }

    if (fileName.toLowerCase().endsWith(".dcm")) {
      return NextResponse.json(
        {
          error:
            "DICOM parsing is not enabled in this build. Please upload a PNG/JPEG slice or convert your DICOM.",
        },
        { status: 415 },
      );
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const prediction = await runInference(buffer);

    // log to Neon (optional)
    logPrediction({
      fileName,
      label: prediction.label,
      confidence: prediction.confidence,
      createdAt: new Date(),
    }).catch(() => {
      /* swallow logging errors */
    });

    return NextResponse.json({
      ...prediction,
      fileName,
      fileSize,
      turnaroundMs: Date.now() - started,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    const badRequest =
      message.includes("Content-Type was not one of") ||
      message.includes("Expected a `file` field");

    if (badRequest) {
      return NextResponse.json(
        { error: "Expected multipart form data with a file." },
        { status: 400 },
      );
    }

    console.error("Inference error:", error);
    return NextResponse.json(
      {
        error: "Unable to run analysis right now.",
        detail: process.env.NODE_ENV === "production" ? undefined : message,
      },
      { status: 500 },
    );
  }
}
