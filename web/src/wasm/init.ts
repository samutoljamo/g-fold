import init, { solve } from "./pkg/gfold_wasm.js";
import type { Config, Trajectory, Spacecraft, Environment, Solver } from "./pkg/gfold_wasm.js";

let ready: Promise<void> | null = null;

/** Initialize the wasm module exactly once. Safe to await repeatedly. */
export function ensureWasm(): Promise<void> {
  if (!ready) ready = init().then(() => undefined);
  return ready;
}

export { solve };
export type { Config, Trajectory, Spacecraft, Environment, Solver };
