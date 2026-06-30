import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ensureWasm } from "./wasm/init";

// The whole UI is useless without the solver, so block render on init. Surface
// a load failure as plain text rather than a silent blank page.
try {
  await ensureWasm();
} catch (err) {
  document.body.textContent =
    "Failed to load the solver (WASM). Check the browser console for details.";
  throw err;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
