import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages subpath (repo name) so built asset URLs
// resolve under https://<user>.github.io/g-fold/.
export default defineConfig({
  base: "/g-fold/",
  plugins: [react()],
  // esnext target is required to support top-level await in main.tsx (wasm init).
  build: { target: "esnext" },
  // Pure-logic tests only (node env). Glob allows .tsx so component tests
  // aren't silently skipped if added later (they'd need a jsdom env then).
  test: { environment: "node", include: ["tests/**/*.test.{ts,tsx}"] },
});
