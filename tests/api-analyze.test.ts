import { describe, expect, it } from "vitest";
import { POST } from "@/app/api/analyze/route";
import sharp from "sharp";

async function makePngBuffer() {
  return sharp({
    create: {
      width: 256,
      height: 256,
      channels: 3,
      background: { r: 20, g: 20, b: 20 },
    },
  })
    .png()
    .toBuffer();
}

describe("POST /api/analyze", () => {
  it(
    "rejects missing file",
    async () => {
      const req = new Request("http://localhost/api/analyze", { method: "POST" });
      const res = (await POST(req)) as Response;
      expect(res.status).toBe(400);
    },
    10000,
  );

  it(
    "returns prediction for png upload",
    async () => {
      const buffer = await makePngBuffer();
      const fd = new FormData();
      fd.append("file", new File([buffer], "test.png", { type: "image/png" }));
      const req = new Request("http://localhost/api/analyze", {
        method: "POST",
        body: fd,
      });
      const res = (await POST(req)) as Response;
      expect(res.status).toBe(200);
      const json = await res.json();
      expect(json.label).toBeDefined();
      expect(json.confidence).toBeGreaterThan(0);
    },
    20000,
  );
});
