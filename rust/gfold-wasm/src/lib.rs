//! Typed WASM bindings. Config in / Trajectory out as JS objects; tsify emits
//! the .d.ts so TypeScript sees the real shapes.
use gfold_core::{config::Config, solve::solve as core_solve, solve::Trajectory};
use wasm_bindgen::prelude::*;

/// Solve the powered descent guidance problem.
///
/// # Errors
/// Throws a `JsError` when the solver reports an infeasible or failed status.
#[wasm_bindgen]
pub fn solve(config: Config) -> Result<Trajectory, JsError> {
    core_solve(&config).map_err(|e| JsError::new(&e))
}
