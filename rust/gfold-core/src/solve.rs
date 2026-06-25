//! Clarabel driver + typed Trajectory.
use crate::assemble::{assemble, Layout};
use crate::config::Config;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Trajectory {
    pub positions: Vec<[f64; 3]>,
    pub velocities: Vec<[f64; 3]>,
    pub thrusts: Vec<f64>,
    pub normalized_thrusts: Vec<f64>,
    pub z_values: Vec<f64>,
    pub u_values: Vec<[f64; 3]>,
    pub s_values: Vec<f64>,
    pub objective: f64,
    pub final_mass: f64,
    pub time_points: Vec<f64>,
    pub status: String,
}

pub fn solve(cfg: &Config) -> Result<Trajectory, String> {
    let prob = assemble(cfg);
    let settings = DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    };
    // In Clarabel 0.11, DefaultSolver::new returns Result<Self, SolverError>
    let mut solver = DefaultSolver::new(
        &prob.p_mat, &prob.q, &prob.a_mat, &prob.b, &prob.cones, settings,
    )
    .map_err(|e| format!("solver construction error: {e}"))?;
    solver.solve();

    let status = solver.solution.status;
    if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        return Err(format!("solver status: {:?}", status));
    }

    let x = &solver.solution.x;
    let n = cfg.solver.n;
    let l = Layout { n };
    let dt = cfg.dt();
    let max_t = cfg.spacecraft.real_max_thrust;

    let mut positions = Vec::with_capacity(n);
    let mut velocities = Vec::with_capacity(n);
    let mut u_values = Vec::with_capacity(n);
    let mut s_values = Vec::with_capacity(n);
    let mut z_values = Vec::with_capacity(n);
    let mut thrusts = Vec::with_capacity(n);
    let mut normalized_thrusts = Vec::with_capacity(n);
    let mut time_points = Vec::with_capacity(n);

    for i in 0..n {
        positions.push([x[l.x(i, 0)], x[l.x(i, 1)], x[l.x(i, 2)]]);
        velocities.push([x[l.x(i, 3)], x[l.x(i, 4)], x[l.x(i, 5)]]);
        let u = [x[l.u(i, 0)], x[l.u(i, 1)], x[l.u(i, 2)]];
        u_values.push(u);
        s_values.push(x[l.s(i)]);
        let z = x[l.z(i)];
        z_values.push(z);
        let norm_u = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let thrust = norm_u * z.exp();
        thrusts.push(thrust);
        normalized_thrusts.push(thrust / max_t);
        time_points.push(i as f64 * dt);
    }

    let z_final = z_values[n - 1];
    Ok(Trajectory {
        positions,
        velocities,
        thrusts,
        normalized_thrusts,
        z_values,
        u_values,
        s_values,
        objective: z_final,
        final_mass: z_final.exp(),
        time_points,
        status: format!("{:?}", status),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    #[test]
    fn trajectory_serializes_to_json() {
        let traj = solve(&Config::default()).expect("solve");
        let json = serde_json::to_string(&traj).expect("serialize");
        assert!(json.contains("\"positions\""));
        assert!(json.contains("\"final_mass\""));
    }

    #[test]
    fn solves_default_problem() {
        let cfg = Config::default();
        let traj = solve(&cfg).expect("solve");
        let n = cfg.solver.n;
        assert_eq!(traj.positions.len(), n);
        // boundary conditions honored
        assert_relative_eq!(traj.positions[0][0], 450.0, epsilon = 1e-2);
        assert_relative_eq!(traj.positions[n - 1][0], 0.0, epsilon = 1e-2);
        assert_relative_eq!(traj.positions[n - 1][2], 0.0, epsilon = 1e-2);
        // final mass within dry/wet bounds
        assert!(traj.final_mass <= 2000.0 + 1.0);
        assert!(traj.final_mass >= 300.0 - 1.0);
    }
}
