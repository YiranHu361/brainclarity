import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["onnxruntime-node", "sharp"],
  outputFileTracingIncludes: {
    "/api/analyze": [
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*",
      "./public/model/brain_tumor.onnx",
      "./public/model/class_map.json",
    ],
    "/app/api/analyze/route": [
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*",
      "./public/model/brain_tumor.onnx",
      "./public/model/class_map.json",
    ],
  },
  outputFileTracingExcludes: {
    "/api/analyze": [
      "./node_modules/onnxruntime-node/bin/napi-v6/win32/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/darwin/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm64/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/ppc64le/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/aarch64/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/s390x/**",
      "./node_modules/@img/sharp-libvips-linuxmusl-x64/**",
      "./node_modules/@img/sharp-win32-*/**",
      "./node_modules/@img/sharp-darwin-*/**",
      "./public/model/brain_tumor_state_dict.pt",
    ],
    "/app/api/analyze/route": [
      "./node_modules/onnxruntime-node/bin/napi-v6/win32/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/darwin/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/arm64/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/ppc64le/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/aarch64/**",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/s390x/**",
      "./node_modules/@img/sharp-libvips-linuxmusl-x64/**",
      "./node_modules/@img/sharp-win32-*/**",
      "./node_modules/@img/sharp-darwin-*/**",
      "./public/model/brain_tumor_state_dict.pt",
    ],
  },
};

export default nextConfig;
