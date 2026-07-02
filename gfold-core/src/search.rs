//! Time-of-flight search: auto-bracket, coarse feasibility sweep, golden-section.
use crate::coast;
use crate::config::Config;
use crate::solve::{solve_fixed, Trajectory};
use crate::validate;

/// Generous outer bracket for the feasible time-of-flight window. Heuristic —
/// only needs to *contain* the feasible interval; the sweep locates the actual
/// feasible sub-interval inside it.
pub fn auto_tof_bounds(cfg: &Config) -> (f64, f64) {
    let sc = &cfg.spacecraft;
    let m_dry = (sc.wet_mass - sc.fuel).max(1.0);
    let v0 = (sc.initial_velocity.iter().map(|c| c * c).sum::<f64>()).sqrt();
    let a_max = (cfg.max_thrust() / m_dry).max(1e-3);
    let t_lo = (v0 / a_max).max(1.0);
    let t_burn = sc.fuel / (sc.fuel_consumption * cfg.min_thrust()).max(1e-9);
    let t_hi = t_burn.max(t_lo * 4.0);
    (t_lo, t_hi)
}

/// Golden-section maximization of `f` over [a, b]. `f` returns `None` for
/// infeasible probes (treated as -infinity). Returns (argmax, value).
fn golden_max<F: FnMut(f64) -> Option<f64>>(mut f: F, mut a: f64, mut b: f64, iters: usize) -> (f64, f64) {
    let invphi = 0.618_033_988_749_895_f64;
    let mut c = b - (b - a) * invphi;
    let mut d = a + (b - a) * invphi;
    let mut fc = f(c).unwrap_or(f64::NEG_INFINITY);
    let mut fd = f(d).unwrap_or(f64::NEG_INFINITY);
    for _ in 0..iters {
        if fc >= fd {
            b = d; d = c; fd = fc;
            c = b - (b - a) * invphi; fc = f(c).unwrap_or(f64::NEG_INFINITY);
        } else {
            a = c; c = d; fc = fd;
            d = a + (b - a) * invphi; fd = f(d).unwrap_or(f64::NEG_INFINITY);
        }
    }
    if fc >= fd { (c, fc) } else { (d, fd) }
}

/// Coarse sweep + golden refine over [t_lo, t_hi]. Returns the best (tof,
/// trajectory) by final mass, or None if every probe is infeasible.
/// `samples` (>= 2) sets the coarse grid resolution and `golden_iters` the
/// refinement fidelity.
fn best_tof(
    cfg: &Config,
    t_lo: f64,
    t_hi: f64,
    samples: usize,
    golden_iters: usize,
) -> Option<(f64, Trajectory)> {
    debug_assert!(samples >= 2, "best_tof needs at least 2 grid samples");
    let step = (t_hi - t_lo) / ((samples - 1) as f64);
    // Sweep the whole grid, keeping the best *grid index* by final mass.
    // Infeasible probes are skipped but their grid slots still define the
    // bracket, so the refine can explore into a slot adjacent to the best
    // sample even if that neighbour itself was infeasible (the optimum often
    // sits between the lowest feasible ToF and the infeasible short-flight
    // regime).
    let mut best: Option<(usize, f64)> = None; // (grid index, final_mass)
    for i in 0..samples {
        let t = t_lo + step * (i as f64);
        if let Ok(tr) = solve_fixed(cfg, t) {
            if best.is_none_or(|(_, m)| tr.final_mass > m) {
                best = Some((i, tr.final_mass));
            }
        }
    }
    let (bi, _) = best?;
    let lo = t_lo + step * (bi.saturating_sub(1) as f64);
    let hi = t_lo + step * ((bi + 1).min(samples - 1) as f64);
    let (best_t, _) = golden_max(
        |t| solve_fixed(cfg, t).ok().map(|tr| tr.final_mass),
        lo,
        hi,
        golden_iters,
    );
    // A failure here (after a feasible sweep) is as good as infeasible for
    // the search: drop the solver's error string and let the caller report
    // its own range-level message.
    solve_fixed(cfg, best_t).ok().map(|tr| (best_t, tr))
}

/// A `min_thrust_realized` residual above this means the lossless
/// convexification went lossy (unflyable command), not merely marginally
/// loose from an active pointing cone — reject the candidate. Matches the
/// bound drawn in validate's tight_max_angle_is_feasible_and_binding test.
const LOSSY_RESIDUAL: f64 = 1e-2;
/// A refined coast shorter than this is search noise — return a plain burn.
const MIN_COAST: f64 = 0.05;
/// Outer grid over coast duration.
const COAST_SAMPLES: usize = 9;
/// Reduced inner-search effort once a burn-ToF hint exists (warm bracket).
const HINT_SAMPLES: usize = 7;
const HINT_ITERS: usize = 8;

/// Solve the burn that follows `t_c` seconds of coast. Returns the burn ToF
/// and trajectory, or None when infeasible or lossy. `tof_hint` (the previous
/// candidate's burn ToF) shrinks the inner search: the optimal burn time
/// moves smoothly with the coast duration.
///
/// The warm bracket assumes the optimal burn ToF drifts slowly with t_c
/// (~15-20% per grid step in practice). A stale hint can make a probe read
/// as infeasible, degrading the golden refine toward grid resolution — the
/// final full-bounds re-solve keeps the returned trajectory unbiased.
fn coast_candidate(cfg: &Config, t_c: f64, tof_hint: Option<f64>) -> Option<(f64, Trajectory)> {
    let ccfg = coast::coast_config(cfg, t_c);
    let (tof, traj) = match tof_hint {
        Some(h) => best_tof(&ccfg, 0.5 * h, 1.5 * h, HINT_SAMPLES, HINT_ITERS),
        None => {
            let (lo, hi) = match (ccfg.solver.tof_min, ccfg.solver.tof_max) {
                (Some(lo), Some(hi)) => (lo, hi),
                _ => auto_tof_bounds(&ccfg),
            };
            best_tof(&ccfg, lo, hi, 13, 20)
        }
    }?;
    let lossy = validate::validate(&ccfg, &traj, 1e-4)
        .iter()
        .any(|v| v.name == "min_thrust_realized" && v.residual > LOSSY_RESIDUAL);
    if lossy {
        None
    } else {
        Some((tof, traj))
    }
}

/// Search for the fuel-optimal ignition time (outer) and burn time-of-flight
/// (inner). Returns (coast duration, stitched coast+burn trajectory).
///
/// The latest feasible ignition is NOT assumed optimal: with lateral diverts
/// and state constraints, fuel vs. ignition time need not be monotone, so the
/// coast duration is swept and refined like the ToF itself. `t_c = 0` is
/// always sampled, so enabling coast can never lose fuel beyond search
/// resolution.
///
/// Cost: on the order of a few hundred SOCP solves (opt-in via
/// `allow_coast`). Note the stricter failure mode: when no admissible coast
/// exists and the pure burn is lossy (realized thrust below the floor),
/// this returns an error where `search_tof` would return the unflyable
/// trajectory.
pub fn search_coast(cfg: &Config) -> Result<(f64, Trajectory), String> {
    let t_max = coast::max_coast(cfg);
    let samples = if t_max > 0.0 { COAST_SAMPLES } else { 1 };
    let step = if t_max > 0.0 { t_max / (COAST_SAMPLES - 1) as f64 } else { 0.0 };
    let mut hint: Option<f64> = None;
    let mut best: Option<(usize, f64)> = None; // (grid index, final_mass)
    for i in 0..samples {
        if let Some((tof, tr)) = coast_candidate(cfg, step * i as f64, hint) {
            hint = Some(tof);
            if best.is_none_or(|(_, m)| tr.final_mass > m) {
                best = Some((i, tr.final_mass));
            }
        }
    }
    let (bi, _) = best.ok_or_else(|| {
        format!("infeasible: no flyable ignition time in [0.000, {t_max:.3}] (every candidate infeasible or below the throttle floor)")
    })?;
    let mut t_best = step * bi as f64;
    if t_max > 0.0 {
        let lo = step * bi.saturating_sub(1) as f64;
        let hi_idx = (bi + 1).min(COAST_SAMPLES - 1);
        let hi = step * hi_idx as f64;
        let mut h = hint;
        let (t, _) = golden_max(
            |t_c| {
                let r = coast_candidate(cfg, t_c, h);
                if let Some((tof, _)) = &r {
                    h = Some(*tof);
                }
                r.map(|(_, tr)| tr.final_mass)
            },
            lo,
            hi,
            10,
        );
        t_best = t;
    }
    if t_best < MIN_COAST {
        t_best = 0.0;
    }
    // full-quality re-solve at the chosen ignition time; fall back to the
    // best grid point if the golden pick came from a stale hint bracket
    let (t_c, burn) = match coast_candidate(cfg, t_best, None) {
        Some((_, tr)) => (t_best, tr),
        None => {
            let g = step * bi as f64;
            let (_, tr) = coast_candidate(cfg, g, None).ok_or_else(|| {
                "coast search failed to re-solve its best candidate".to_string()
            })?;
            (g, tr)
        }
    };
    if t_c < MIN_COAST {
        return Ok((0.0, burn));
    }
    Ok((t_c, coast::prepend_coast(cfg, t_c, &burn)))
}

/// Search for the fuel-optimal time-of-flight and return (tof, trajectory).
pub fn search_tof(cfg: &Config) -> Result<(f64, Trajectory), String> {
    let (t_lo, t_hi) = match (cfg.solver.tof_min, cfg.solver.tof_max) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => auto_tof_bounds(cfg),
    };
    best_tof(cfg, t_lo, t_hi, 13, 20).ok_or_else(|| {
        format!("infeasible: no feasible time-of-flight in [{t_lo:.3}, {t_hi:.3}]")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn lunar_trap_selects_a_coast() {
        // 24 kN engine on the Moon: the 20% throttle floor (4800 N) exceeds
        // hover thrust (2000·1.62 = 3240 N), so a continuous burn can't fly
        // slow and the pure-burn solution goes lossy (see validate's
        // catches_realized_thrust_below_min). With allow_coast the solver
        // must fall ballistically first and then fly a clean saturated burn.
        let mut cfg = Config::default();
        cfg.spacecraft.initial_position = [200.0, -150.0, 1500.0];
        cfg.spacecraft.initial_velocity = [-20.0, 5.0, -5.0];
        cfg.environment.gravity = [0.0, 0.0, -1.62];
        cfg.solver.n = 40; // keep the nested search fast in debug builds
        cfg.solver.allow_coast = true;
        let (t_c, traj) = search_coast(&cfg).expect("coast search");
        assert!(t_c > 1.0, "expected a real coast, got {t_c}");
        assert_eq!(traj.coast_duration, t_c);
        assert!(traj.time_points.len() > cfg.solver.n);
        assert_eq!(traj.thrusts[0], 0.0);
        // ends on target like any other solution
        let last = traj.positions.last().unwrap();
        assert!(last[2].abs() < 1e-2, "lands at target altitude");
    }

    #[test]
    fn coast_never_hurts_the_default_descent() {
        // Mars default: t_c = 0 is always in the sweep, so enabling the flag
        // can't lose fuel (beyond search resolution).
        let mut cfg = Config::default();
        cfg.solver.n = 40;
        let plain = search_tof(&cfg).expect("plain").1;
        cfg.solver.allow_coast = true;
        let coasted = search_coast(&cfg).expect("coast").1;
        assert!(
            coasted.final_mass >= plain.final_mass - 1.0,
            "coast search must not do worse: {} vs {}",
            coasted.final_mass,
            plain.final_mass
        );
    }

    #[test]
    fn search_finds_feasible_optimum() {
        let (tof, traj) = search_tof(&Config::default()).expect("search");
        let (lo, hi) = auto_tof_bounds(&Config::default());
        assert!(tof >= lo && tof <= hi, "tof {tof} outside [{lo},{hi}]");
        let fixed = solve_fixed(&Config::default(), 44.63).unwrap();
        assert!(traj.final_mass >= fixed.final_mass - 1.0);
    }

    #[test]
    fn search_reports_infeasible() {
        let mut cfg = Config::default();
        cfg.solver.tof_min = Some(0.001);
        cfg.solver.tof_max = Some(0.01);
        assert!(search_tof(&cfg).is_err());
    }
}
