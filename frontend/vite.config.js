import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Phase 7b: dev-server proxy so the Vite dev server (default port 5173) can call the
// FastAPI backend (port 8000) with plain relative paths ("/gap-analysis", not
// "http://127.0.0.1:8000/gap-analysis") -- no CORS workaround needed, and the same
// relative paths keep working unchanged once Phase 7c serves this build as static
// files from FastAPI itself (same origin at that point, proxy simply unused).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/gap-analysis": "http://127.0.0.1:8000",
      "/draft": "http://127.0.0.1:8000",
      "/history": "http://127.0.0.1:8000",
      "/auth": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
