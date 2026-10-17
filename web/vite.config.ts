import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` forwards API calls to `python3 server.py --cpu` on port 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/classify": "http://localhost:8000",
      "/info": "http://localhost:8000",
      "/shots": "http://localhost:8000",
    },
  },
});
