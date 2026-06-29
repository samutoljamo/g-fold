//! Ground-truth accuracy harness: how fine does the mesh need to be?
//!
//! The solver enforces boundary + dynamics as *node* equalities, so at the
//! nodes it "believes" it hits the target exactly. The physical question is
//! different: if you reconstruct the commanded acceleration u(t) and fly the
//! TRUE continuous dynamics
//!     p' = v,   v' = u(t) + g
//! (u is the physical acceleration in G-FOLD's convexified coords, so this is
//! mass-independent), how far does the real trajectory drift from what the
//! optimizer thinks, and how far does it miss the target?
//!
//! The control is piecewise-linear (first-order hold) between nodes, matching
//! the discretization, so we reconstruct it that way and integrate with
//! high-resolution RK4 (the "truth"). The terminal miss and the max node gap
//! are the discretization error, and they bound the smallest usable n.
//!
//! Run: cargo run --release --example accuracy -p gfold-core

use gfold_core::config::Config;
use gfold_core::solve::solve;

const SUBSTEPS: usize = 128; // RK4 substeps per node interval (the "truth")

/// piecewise-linear commanded acceleration at global time t
fn u_at(t: f64, u: &[[f64; 3]], dt: f64) -> [f64; 3] {
    let n = u.len();
    let s = t / dt;
    let mut i = s.floor() as usize;
    if i >= n - 1 {
        i = n - 2;
    }
    let frac = s - i as f64;
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = u[i][c] * (1.0 - frac) + u[i + 1][c] * frac;
    }
    out
}

/// state derivative y=(p,v) -> (v, u(t)+g)
fn deriv(t: f64, y: &[f64; 6], u: &[[f64; 3]], dt: f64, g: [f64; 3]) -> [f64; 6] {
    let a = u_at(t, u, dt);
    [y[3], y[4], y[5], a[0] + g[0], a[1] + g[1], a[2] + g[2]]
}

fn rk4_step(t: f64, y: &[f64; 6], h: f64, u: &[[f64; 3]], dt: f64, g: [f64; 3]) -> [f64; 6] {
    let k1 = deriv(t, y, u, dt, g);
    let mut y2 = *y;
    for c in 0..6 { y2[c] = y[c] + 0.5 * h * k1[c]; }
    let k2 = deriv(t + 0.5 * h, &y2, u, dt, g);
    let mut y3 = *y;
    for c in 0..6 { y3[c] = y[c] + 0.5 * h * k2[c]; }
    let k3 = deriv(t + 0.5 * h, &y3, u, dt, g);
    let mut y4 = *y;
    for c in 0..6 { y4[c] = y[c] + h * k3[c]; }
    let k4 = deriv(t + h, &y4, u, dt, g);
    let mut out = *y;
    for c in 0..6 {
        out[c] = y[c] + h / 6.0 * (k1[c] + 2.0 * k2[c] + 2.0 * k3[c] + k4[c]);
    }
    out
}

fn norm3(a: [f64; 3]) -> f64 { (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() }

fn main() {
    println!(
        "{:>5}  {:>10}  {:>12}  {:>12}  {:>12}  {:>10}",
        "n", "dt (s)", "term pos (m)", "term vel m/s", "max node (m)", "obj"
    );
    for &n in &[20usize, 30, 40, 50, 60, 80, 100, 150, 200, 400, 800] {
        let mut cfg = Config::default();
        cfg.solver.n = n;
        cfg.solver.time_of_flight = Some(44.63);
        let traj = match solve(&cfg) {
            Ok(t) => t,
            Err(e) => { println!("{n:5}  solve failed: {e}"); continue; }
        };
        let g = cfg.environment.gravity;
        let dt = cfg.dt();
        let u = &traj.u_values;

        // integrate from the exact initial state through all node intervals,
        // capturing the integrated state at each node to compare against what
        // the optimizer reported.
        let mut y = [
            traj.positions[0][0], traj.positions[0][1], traj.positions[0][2],
            traj.velocities[0][0], traj.velocities[0][1], traj.velocities[0][2],
        ];
        let h = dt / SUBSTEPS as f64;
        let mut max_node_gap = 0.0f64;
        for i in 0..n - 1 {
            let t0 = i as f64 * dt;
            for k in 0..SUBSTEPS {
                y = rk4_step(t0 + k as f64 * h, &y, h, u, dt, g);
            }
            // integrated state now at node i+1
            let gap = norm3([
                y[0] - traj.positions[i + 1][0],
                y[1] - traj.positions[i + 1][1],
                y[2] - traj.positions[i + 1][2],
            ]);
            max_node_gap = max_node_gap.max(gap);
        }

        let term_pos = norm3([
            y[0] - cfg.spacecraft.target_position[0],
            y[1] - cfg.spacecraft.target_position[1],
            y[2] - cfg.spacecraft.target_position[2],
        ]);
        let term_vel = norm3([
            y[3] - cfg.spacecraft.target_velocity[0],
            y[4] - cfg.spacecraft.target_velocity[1],
            y[5] - cfg.spacecraft.target_velocity[2],
        ]);

        println!(
            "{n:5}  {dt:10.4}  {term_pos:12.4}  {term_vel:12.5}  {max_node_gap:12.4}  {:10.3}",
            traj.objective
        );
    }
}
