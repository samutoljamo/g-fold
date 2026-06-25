//! G-FOLD fuel-optimal powered-descent guidance, solved as a second-order cone
//! program with [Clarabel](https://clarabel.org).
//!
//! The min-fuel soft-landing problem is posed directly in conic form with no
//! modeling layer: the trajectory is discretized into `n` nodes with a
//! first-order hold on the thrust acceleration, lossless convexification turns
//! the thrust-magnitude bounds into second-order-cone constraints on a slack
//! variable, and the spacecraft mass is carried in log space so the dynamics
//! stay linear.
//!
//! Pipeline:
//! - [`config`] — problem inputs (spacecraft, environment, horizon).
//! - [`derive`] — precomputed log-mass linearization for the thrust bounds.
//! - [`assemble`] — builds the Clarabel problem (decision vector, cones, `A`/`b`).
//! - [`solve`] — runs Clarabel and unpacks a typed [`solve::Trajectory`].
//! - [`validate`] — independent physics/feasibility checks on a solution.
//!
//! See the `examples/` directory for analysis harnesses.

// The constraint builders index gravity/component arrays by the same loop
// variable used for the x/u component offsets; an indexed loop is clearer here
// than zipped iterators.
#![allow(clippy::needless_range_loop)]

pub mod config;
pub mod derive;
pub mod assemble;
pub mod solve;
pub mod validate;
