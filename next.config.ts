import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    serverComponentsExternalPackages: ["onnxruntime-node", "sharp"],
  },
};

export default nextConfig;
