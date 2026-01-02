import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["sharp"],
  outputFileTracingIncludes: {
    "/api/analyze": [
      "./public/model/brain_tumor.onnx",
      "./public/model/class_map.json",
    ],
    "/app/api/analyze/route": [
      "./public/model/brain_tumor.onnx",
      "./public/model/class_map.json",
    ],
  },
};

export default nextConfig;
