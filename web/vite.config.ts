import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the GitHub Pages subpath (repo name) so built asset URLs
// resolve under https://<user>.github.io/g-fold/.
export default defineConfig({
  base: "/g-fold/",
  plugins: [react()],
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
