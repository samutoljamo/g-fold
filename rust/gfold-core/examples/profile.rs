//! Profiling harness: where does the Clarabel solve time go, across n?
//!
//! Run: cargo run --release --example profile -p gfold-core

use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver};
use gfold_core::assemble::assemble;
use gfold_core::config::Config;
use std::time::Instant;

fn settings() -> DefaultSettings<f64> {
    DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    }
}

fn main() {
    for &n in &[20usize, 50, 100, 200, 400, 800] {
        let mut cfg = Config::default();
        cfg.solver.n = n;
        let prob = assemble(&cfg);

        // warm up
        for _ in 0..3 {
            let mut s = DefaultSolver::new(
                &prob.p_mat, &prob.q, &prob.a_mat, &prob.b, &prob.cones, settings(),
            )
            .unwrap();
            s.solve();
        }

        // timed fresh solve
        let t1 = Instant::now();
        let mut solver = DefaultSolver::new(
            &prob.p_mat, &prob.q, &prob.a_mat, &prob.b, &prob.cones, settings(),
        )
        .unwrap();
        solver.solve();
        let solve_ms = t1.elapsed().as_secs_f64() * 1e3;
        let iters = solver.info.iterations.max(1);

        // assemble timing
        let ta = Instant::now();
        for _ in 0..20 {
            let _ = assemble(&cfg);
        }
        let asm_ms = ta.elapsed().as_secs_f64() * 1e3 / 20.0;

        println!(
            "n={n:4}  vars={:5} rows={:5}  iters={iters:3}  assemble={asm_ms:7.3}ms  solve={solve_ms:8.3}ms  per-iter={:.3}ms",
            prob.q.len(),
            prob.b.len(),
            solve_ms / iters as f64
        );
        if n == 100 {
            println!("    --- n=100 Clarabel section breakdown ---");
            if let Some(ref t) = solver.timers {
                t.print();
            }
        }
    }
}
