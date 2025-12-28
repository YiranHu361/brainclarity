import { describe, expect, it } from "vitest";
import sharp from "sharp";
import { runInference } from "@/lib/inference";
import fs from "fs";

const MODEL_PATH = "public/model/brain_tumor.onnx";

describe("runInference", () => {
  it("loads model file", () => {
    expect(fs.existsSync(MODEL_PATH)).toBe(true);
  });

  it(
    "returns a prediction with probabilities",
    async () => {
      const buffer = await sharp({
        create: {
          width: 256,
          height: 256,
          channels: 3,
          background: { r: 10, g: 10, b: 10 },
        },
      })
        .png()
        .toBuffer();

      const result = await runInference(buffer);
      expect(result.label).toBeTypeOf("string");
      expect(result.confidence).toBeGreaterThan(0);
      expect(Object.keys(result.probabilities).length).toBeGreaterThan(0);
    },
    20000,
  );
});
