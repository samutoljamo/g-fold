import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ensureWasm } from "./wasm/init";

await ensureWasm();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
