import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    message: "Inference runs client-side. This endpoint is for health checks only.",
  });
}

export async function POST() {
  return NextResponse.json({
    status: "ok",
    message: "Inference now runs client-side in the browser using WASM. No server call needed.",
  });
}
