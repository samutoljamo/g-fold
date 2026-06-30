import type { Spacecraft, Environment, Solver } from "../wasm/init";

/**
 * A fully-specified config used throughout the UI.
 *
 * The wasm `Config` is deeply optional — every section and field carries a
 * serde default on the Rust side, so tsify emits them all as `?`. That is
 * painful to consume in forms and plots (`config.spacecraft?.wet_mass`
 * everywhere). `AppConfig` is the app-internal shape where every section and
 * scalar is present. It is assignable to `Config`, so it passes straight to
 * `solve` with no conversion.
 *
 * Solver time-of-flight fields stay optional on purpose: omitting
 * `time_of_flight` makes the core solver search for the fuel-optimal ToF.
 */
export type AppConfig = {
  spacecraft: Required<Spacecraft>;
  environment: Required<Environment>;
  solver: { n: number } & Pick<Solver, "time_of_flight" | "tof_min" | "tof_max">;
};
