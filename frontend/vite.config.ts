import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/preview_chunks": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/upload_selected_chunks": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/upload_data": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/delete_chunks": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/upload_document": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/download_document": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
