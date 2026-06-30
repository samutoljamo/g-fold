//! Independent physics/optimality checks.
use crate::config::Config;
use crate::derive::derive;
use crate::solve::Trajectory;

#[derive(Debug, Clone)]
pub struct Violation {
    pub name: String,
    pub index: usize,
    pub residual: f64,
}

fn norm3(v: &[f64; 3]) -> f64 { (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt() }

pub fn validate(cfg: &Config, traj: &Trajectory, tol: f64) -> Vec<Violation> {
    let n = cfg.solver.n;
    // Derive against the trajectory's actual time-of-flight: the config may
    // have `time_of_flight: None` (search path), but the per-step log-mass
    // linearization depends on the dt the trajectory was solved at.
    let mut derive_cfg = cfg.clone();
    derive_cfg.solver.time_of_flight = Some(traj.time_of_flight);
    let derived = derive(&derive_cfg);
    let dt = traj.time_of_flight / cfg.solver.n as f64;
    let dt2 = dt * dt;
    let g = cfg.environment.gravity;
    let a_dt = cfg.spacecraft.fuel_consumption * dt;
    let sin = cfg.sin_glide_slope();
    let mut v = Vec::new();

    let eq = |name: &str, i: usize, lhs: f64, rhs: f64, out: &mut Vec<Violation>| {
        let r = (lhs - rhs).abs();
        if r > tol { out.push(Violation { name: name.into(), index: i, residual: r }); }
    };

    // boundary
    for c in 0..3 {
        eq("init_pos", c, traj.positions[0][c], cfg.spacecraft.initial_position[c], &mut v);
        eq("init_vel", c, traj.velocities[0][c], cfg.spacecraft.initial_velocity[c], &mut v);
        eq("target_pos", c, traj.positions[n-1][c], cfg.spacecraft.target_position[c], &mut v);
        eq("target_vel", c, traj.velocities[n-1][c], cfg.spacecraft.target_velocity[c], &mut v);
    }

    // dynamics
    for i in 0..n-1 {
        for c in 0..3 {
            let acc = (traj.u_values[i+1][c] + traj.u_values[i][c]) / 2.0;
            // first-order-hold position integral (see assemble::equality_rows)
            let pos_pred = traj.positions[i][c]
                + traj.velocities[i][c] * dt
                + (2.0 * traj.u_values[i][c] + traj.u_values[i+1][c]) * dt2 / 6.0
                + g[c] * dt2 / 2.0;
            eq("pos_update", i, traj.positions[i+1][c], pos_pred, &mut v);
            let vel_pred = traj.velocities[i][c] + acc * dt + g[c] * dt;
            eq("vel_update", i, traj.velocities[i+1][c], vel_pred, &mut v);
        }
        let z_pred = traj.z_values[i] - (traj.s_values[i] + traj.s_values[i+1]) / 2.0 * a_dt;
        eq("mass_update", i, traj.z_values[i+1], z_pred, &mut v);
    }

    // inequalities
    let cos_max = cfg.cos_max_angle();
    // A full half-sphere (max_angle >= 180°) is already implied by the
    // thrust-slack cone (u_z >= -s), so only check pointing when it constrains.
    let enforce_pointing = cfg.environment.max_angle_deg < 180.0;
    for i in 0..n {
        let gs = traj.positions[i][2] - sin * norm3(&traj.positions[i]);
        if gs < -tol { v.push(Violation { name: "glide_slope".into(), index: i, residual: -gs }); }
        let vb = cfg.spacecraft.max_velocity - norm3(&traj.velocities[i]);
        if vb < -tol { v.push(Violation { name: "max_velocity".into(), index: i, residual: -vb }); }
        let ts = traj.s_values[i] - norm3(&traj.u_values[i]);
        if ts < -tol { v.push(Violation { name: "thrust_slack".into(), index: i, residual: -ts }); }
        // thrust pointing: n̂·u >= cos(max_angle)·s, with n̂ = +z (up).
        if enforce_pointing {
            let pa = traj.u_values[i][2] - cos_max * traj.s_values[i];
            if pa < -tol { v.push(Violation { name: "max_angle".into(), index: i, residual: -pa }); }
        }
    }
    let dry = traj.z_values[n-1] - cfg.log_dry_mass();
    if dry < -tol { v.push(Violation { name: "dry_mass".into(), index: n-1, residual: -dry }); }

    // thrust bounds
    for i in 0..n {
        let z = traj.z_values[i];
        let s = traj.s_values[i];
        let w = z - derived.z0[i];
        // upper: s*max_exp[i] <= 1 - w
        let upper_res = s * derived.max_exp[i] - (1.0 - w);
        if upper_res > tol { v.push(Violation { name: "max_thrust".into(), index: i, residual: upper_res }); }
        // lower: 1 - w + w^2/2 <= s*min_exp[i]  (skip if min_exp is +inf, i.e. min_thrust==0)
        if derived.min_exp[i].is_finite() {
            let lower_res = (1.0 - w + w * w / 2.0) - s * derived.min_exp[i];
            if lower_res > tol { v.push(Violation { name: "min_thrust".into(), index: i, residual: lower_res }); }
        }
    }

    v
}

pub fn assert_valid(cfg: &Config, traj: &Trajectory, tol: f64) {
    let v = validate(cfg, traj, tol);
    assert!(v.is_empty(), "physics violations: {:?}", v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::solve::solve;

    #[test]
    fn default_solution_is_physical() {
        let cfg = Config::default();
        let traj = solve(&cfg).unwrap();
        let v = validate(&cfg, &traj, 1e-4);
        assert!(v.is_empty(), "violations: {:?}", v);
    }

    #[test]
    fn thrust_respects_max_angle() {
        // Regression: the default max_angle (90°) confines thrust to the upper
        // hemisphere (u_z >= 0). Before the pointing constraint was assembled,
        // the fuel-optimal divert thrust ~44° below horizontal early on
        // (u_z < 0) — this guards against that returning.
        let cfg = Config::default();
        assert_eq!(cfg.environment.max_angle_deg, 90.0);
        let traj = solve(&cfg).unwrap();
        for (i, u) in traj.u_values.iter().enumerate() {
            assert!(u[2] >= -1e-4, "node {i}: thrust below horizon, u_z = {}", u[2]);
        }
    }

    #[test]
    fn pointing_can_make_a_descent_infeasible() {
        // Weak (lunar) gravity + a fixed 44.63 s horizon: the vehicle must drop
        // 2400 m, more than free-fall covers in time, so the unconstrained
        // optimum thrusts downward early. A 90° cone (u_z >= 0) forbids that,
        // so the problem becomes genuinely infeasible — the constraint bites.
        let mut cfg = Config::default();
        cfg.environment.gravity = [0.0, 0.0, -1.62];
        cfg.solver.time_of_flight = Some(44.63);
        cfg.environment.max_angle_deg = 90.0;
        assert!(solve(&cfg).is_err(), "expected pointing-induced infeasibility");
    }

    #[test]
    fn tight_max_angle_is_feasible_and_binding() {
        // A 60° cone is feasible for the default geometry and the constraint
        // binds (some node sits on u_z = cos(60°)·s). Non-degenerate cos value.
        let mut cfg = Config::default();
        cfg.environment.max_angle_deg = 60.0;
        let traj = solve(&cfg).unwrap();
        assert!(validate(&cfg, &traj, 1e-4).is_empty());
        let cos60 = 60f64.to_radians().cos();
        let min_ratio = traj
            .u_values
            .iter()
            .map(|u| u[2] / (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12))
            .fold(f64::INFINITY, f64::min);
        assert!(min_ratio >= cos60 - 1e-3, "respects 60° cone");
        assert!(min_ratio <= cos60 + 1e-2, "constraint is binding (active)");
    }
}
