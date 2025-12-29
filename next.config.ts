import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["onnxruntime-node", "sharp"],
  outputFileTracingIncludes: {
    "/api/analyze": [
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm64/*",
      "./public/model/**",
    ],
    "/app/api/analyze/route": [
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm64/*",
      "./public/model/**",
    ],
  },
};

export default nextConfig;
