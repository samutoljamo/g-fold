//! Analytic engine-off coast: ballistic propagation, the admissible-coast
//! bound, and stitching the coast arc onto a burn trajectory.
use crate::config::Config;
use crate::solve::Trajectory;

/// Ballistically propagate the configured initial state by `t` seconds.
/// Mass is unchanged — the engine is off.
pub fn propagate(cfg: &Config, t: f64) -> ([f64; 3], [f64; 3]) {
    let r0 = cfg.spacecraft.initial_position;
    let v0 = cfg.spacecraft.initial_velocity;
    let g = cfg.environment.gravity;
    let mut r = [0.0; 3];
    let mut v = [0.0; 3];
    for c in 0..3 {
        r[c] = r0[c] + v0[c] * t + 0.5 * g[c] * t * t;
        v[c] = v0[c] + g[c] * t;
    }
    (r, v)
}

/// The burn problem that starts after `t_c` seconds of coast: the same config
/// with the initial state moved to the coast endpoint. Time-of-flight is left
/// unset for the ToF search; `allow_coast` is cleared so the inner problem is
/// a plain burn.
pub fn coast_config(cfg: &Config, t_c: f64) -> Config {
    let (r, v) = propagate(cfg, t_c);
    let mut out = cfg.clone();
    out.spacecraft.initial_position = r;
    out.spacecraft.initial_velocity = v;
    out.solver.time_of_flight = None;
    out.solver.allow_coast = false;
    out
}

/// Longest admissible coast: the ballistic arc must stay above the target
/// altitude, inside the glide-slope cone (same absolute-position form as
/// `validate`), and under `max_velocity`. Returns 0 when the arc never
/// descends to the target plane or violates a constraint immediately.
pub fn max_coast(cfg: &Config) -> f64 {
    let z0 = cfg.spacecraft.initial_position[2] - cfg.spacecraft.target_position[2];
    let vz = cfg.spacecraft.initial_velocity[2];
    let gz = cfg.environment.gravity[2];
    // earliest positive root of z0 + vz·t + gz·t²/2 = 0
    let t_impact = if gz.abs() > 1e-12 {
        let disc = vz * vz - 2.0 * gz * z0;
        if disc < 0.0 {
            return 0.0;
        }
        let sq = disc.sqrt();
        let (a, b) = ((-vz - sq) / gz, (-vz + sq) / gz);
        let (lo, hi) = (a.min(b), a.max(b));
        if lo > 0.0 {
            lo
        } else if hi > 0.0 {
            hi
        } else {
            return 0.0;
        }
    } else if vz < -1e-12 {
        z0 / -vz
    } else {
        return 0.0;
    };
    // walk the arc; return the last sample before a constraint violation
    const STEPS: usize = 256;
    let sin = cfg.sin_glide_slope();
    let vmax = cfg.spacecraft.max_velocity;
    for i in 1..=STEPS {
        let t = t_impact * (i as f64) / (STEPS as f64);
        let (r, v) = propagate(cfg, t);
        let gs = r[2] - sin * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if gs < 0.0 || speed > vmax {
            return t_impact * ((i - 1) as f64) / (STEPS as f64);
        }
    }
    t_impact
}

/// Stitch `t_c` seconds of engine-off coast in front of a burn trajectory
/// that was solved from the coast endpoint (`coast_config(cfg, t_c)`).
/// Coast samples are spaced at roughly the burn `dt`; the burn's time base
/// shifts by `t_c`.
pub fn prepend_coast(cfg: &Config, t_c: f64, burn: &Trajectory) -> Trajectory {
    let n_burn = burn.positions.len();
    let dt_burn = burn.time_of_flight / n_burn as f64;
    let k = ((t_c / dt_burn).ceil() as usize).max(1);
    let dt_c = t_c / k as f64;
    let z0 = cfg.log_wet_mass();
    let mut out = Trajectory {
        positions: Vec::with_capacity(k + n_burn),
        velocities: Vec::with_capacity(k + n_burn),
        thrusts: vec![0.0; k],
        normalized_thrusts: vec![0.0; k],
        z_values: vec![z0; k],
        u_values: vec![[0.0; 3]; k],
        s_values: vec![0.0; k],
        objective: burn.objective,
        final_mass: burn.final_mass,
        time_of_flight: t_c + burn.time_of_flight,
        time_points: Vec::with_capacity(k + n_burn),
        status: burn.status.clone(),
        coast_duration: t_c,
    };
    for i in 0..k {
        let t = dt_c * i as f64;
        let (r, v) = propagate(cfg, t);
        out.positions.push(r);
        out.velocities.push(v);
        out.time_points.push(t);
    }
    out.positions.extend_from_slice(&burn.positions);
    out.velocities.extend_from_slice(&burn.velocities);
    out.thrusts.extend_from_slice(&burn.thrusts);
    out.normalized_thrusts.extend_from_slice(&burn.normalized_thrusts);
    out.z_values.extend_from_slice(&burn.z_values);
    out.u_values.extend_from_slice(&burn.u_values);
    out.s_values.extend_from_slice(&burn.s_values);
    out.time_points.extend(burn.time_points.iter().map(|t| t + t_c));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    #[test]
    fn propagate_is_exact_ballistics() {
        let cfg = Config::default(); // r0=[450,-330,2400], v0=[-40,10,-10], g=[0,0,-3.71]
        let (r, v) = propagate(&cfg, 10.0);
        assert_relative_eq!(r[0], 450.0 - 40.0 * 10.0);
        assert_relative_eq!(r[1], -330.0 + 10.0 * 10.0);
        assert_relative_eq!(r[2], 2400.0 - 10.0 * 10.0 - 0.5 * 3.71 * 100.0);
        assert_relative_eq!(v[2], -10.0 - 3.71 * 10.0);
        assert_relative_eq!(v[0], -40.0);
    }

    #[test]
    fn coast_config_moves_the_initial_state() {
        let cfg = Config::default();
        let ccfg = coast_config(&cfg, 5.0);
        let (r, v) = propagate(&cfg, 5.0);
        assert_eq!(ccfg.spacecraft.initial_position, r);
        assert_eq!(ccfg.spacecraft.initial_velocity, v);
        assert!(ccfg.solver.time_of_flight.is_none());
        assert!(!ccfg.solver.allow_coast);
        // untouched: mass, thrust, target
        assert_eq!(ccfg.spacecraft.wet_mass, cfg.spacecraft.wet_mass);
    }

    #[test]
    fn max_coast_vertical_drop_hits_impact_time() {
        // pure vertical drop from 1000 m, no initial speed: t = sqrt(2h/g)
        let mut cfg = Config::default();
        cfg.spacecraft.initial_position = [0.0, 0.0, 1000.0];
        cfg.spacecraft.initial_velocity = [0.0, 0.0, 0.0];
        cfg.environment.gravity = [0.0, 0.0, -2.0];
        let expected = (2.0_f64 * 1000.0 / 2.0).sqrt();
        assert_relative_eq!(max_coast(&cfg), expected, epsilon = 1e-6);
    }

    #[test]
    fn max_coast_is_cut_by_the_speed_limit() {
        let mut cfg = Config::default();
        cfg.spacecraft.initial_position = [0.0, 0.0, 1000.0];
        cfg.spacecraft.initial_velocity = [0.0, 0.0, 0.0];
        cfg.environment.gravity = [0.0, 0.0, -2.0];
        cfg.spacecraft.max_velocity = 20.0; // reached at t = 10 s, impact ~31.6 s
        let t = max_coast(&cfg);
        assert!(t <= 10.0 + 0.2, "speed cap should cut the coast, got {t}");
        assert!(t > 8.0, "cut should be near the cap crossing, got {t}");
    }

    #[test]
    fn max_coast_never_descending_is_zero() {
        let mut cfg = Config::default();
        cfg.environment.gravity = [0.0, 0.0, 0.0];
        cfg.spacecraft.initial_velocity = [1.0, 0.0, 1.0];
        assert_eq!(max_coast(&cfg), 0.0);
    }

    #[test]
    fn prepend_coast_stitches_a_consistent_trajectory() {
        let cfg = Config::default();
        let ccfg = coast_config(&cfg, 8.0);
        let burn = crate::solve::solve_fixed(&ccfg, 40.0).expect("burn solve");
        let full = prepend_coast(&cfg, 8.0, &burn);
        let n_burn = burn.positions.len();
        let k = full.positions.len() - n_burn;
        assert!(k >= 1);
        assert_eq!(full.coast_duration, 8.0);
        assert_relative_eq!(full.time_of_flight, 8.0 + 40.0);
        // coast samples: zero thrust, constant wet mass, exact ballistics
        for i in 0..k {
            assert_eq!(full.thrusts[i], 0.0);
            assert_eq!(full.u_values[i], [0.0; 3]);
            assert_relative_eq!(full.z_values[i], cfg.log_wet_mass());
            let (r, _) = propagate(&cfg, full.time_points[i]);
            assert_relative_eq!(full.positions[i][2], r[2], epsilon = 1e-9);
        }
        // time base: strictly increasing, burn starts at t_c
        assert_relative_eq!(full.time_points[k], 8.0);
        for w in full.time_points.windows(2) {
            assert!(w[1] > w[0]);
        }
        // burn samples carried over unchanged (first burn node = coast endpoint)
        assert_eq!(full.positions[k], burn.positions[0]);
        assert_eq!(full.final_mass, burn.final_mass);
    }
}
