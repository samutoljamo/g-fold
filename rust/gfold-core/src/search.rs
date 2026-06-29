//! Time-of-flight search: auto-bracket, coarse feasibility sweep, golden-section.
use crate::config::Config;
use crate::solve::{solve_fixed, Trajectory};

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

/// Search for the fuel-optimal time-of-flight and return (tof, trajectory).
pub fn search_tof(cfg: &Config) -> Result<(f64, Trajectory), String> {
    let (t_lo, t_hi) = match (cfg.solver.tof_min, cfg.solver.tof_max) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => auto_tof_bounds(cfg),
    };
    const SAMPLES: usize = 13;
    let step = (t_hi - t_lo) / ((SAMPLES - 1) as f64);
    // Sweep the whole grid, keeping the best *grid index* by final mass.
    // Infeasible probes are skipped but their grid slots still define the
    // bracket, so the refine can explore into a slot adjacent to the best
    // sample even if that neighbour itself was infeasible (the optimum often
    // sits between the lowest feasible ToF and the infeasible short-flight
    // regime).
    let mut best: Option<(usize, f64)> = None; // (grid index, final_mass)
    for i in 0..SAMPLES {
        let t = t_lo + step * (i as f64);
        if let Ok(tr) = solve_fixed(cfg, t) {
            if best.is_none_or(|(_, m)| tr.final_mass > m) {
                best = Some((i, tr.final_mass));
            }
        }
    }
    let bi = match best {
        Some((i, _)) => i,
        None => {
            return Err(format!(
                "infeasible: no feasible time-of-flight in [{t_lo:.3}, {t_hi:.3}]"
            ))
        }
    };
    let lo = t_lo + step * (bi.saturating_sub(1) as f64);
    let hi = t_lo + step * ((bi + 1).min(SAMPLES - 1) as f64);
    let (best_t, _) = golden_max(|t| solve_fixed(cfg, t).ok().map(|tr| tr.final_mass), lo, hi, 20);
    let traj = solve_fixed(cfg, best_t)?;
    Ok((best_t, traj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

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
