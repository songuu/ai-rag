import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  
  // 排除某些原生模块，确保 pdf-parse 正常工作
  serverExternalPackages: ['pdf-parse', '@napi-rs/canvas', 'pdfjs-dist', 'canvas'],
  
  // Turbopack 配置（Next.js 16+ 默认使用 Turbopack）
  turbopack: {},
};

export default nextConfig;
