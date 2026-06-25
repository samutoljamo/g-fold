use gfold_core::config::Config;
use gfold_core::solve::solve;
use gfold_core::validate::validate;
use approx::assert_relative_eq;

// Pointwise position agreement gate vs the CVXPY oracle. The binding
// correctness criteria are objective + final-mass (1e-3) + physics validity
// (1e-4) + velocity (0.5 m) below, all kept strict. Position is gated more
// loosely because min-fuel landing has a flat cost manifold: weak-gravity
// configs (e.g. moon) admit many equal-cost trajectories, so two independent
// optimizers settle ~1 m apart at interior nodes even at matching objective.
// 1.5 m tolerates that spread while still catching gross formulation errors.
const POS_GATE: f64 = 1.5;

#[derive(serde::Deserialize)]
struct Expected {
    objective: f64,
    final_mass: f64,
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    #[allow(dead_code)]
    thrusts: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    config: Config,
    expected: Expected,
}

fn check(path: &str) {
    let raw = std::fs::read_to_string(path).expect("read fixture");
    let fx: Fixture = serde_json::from_str(&raw).expect("parse fixture");
    let traj = solve(&fx.config).expect("solve");

    // leg 3: physics validity
    let viol = validate(&fx.config, &traj, 1e-4);
    assert!(viol.is_empty(), "{}: violations {:?}", fx.name, viol);

    // leg 2: agree with CVXPY oracle
    assert_relative_eq!(traj.objective, fx.expected.objective, epsilon = 1e-3);
    assert_relative_eq!(traj.final_mass, fx.expected.final_mass, max_relative = 1e-3);
    let n = traj.positions.len();
    for i in 0..n {
        for c in 0..3 {
            assert_relative_eq!(traj.positions[i][c], fx.expected.positions[i][c], epsilon = POS_GATE);
            assert_relative_eq!(traj.velocities[i][c], fx.expected.velocities[i][c], epsilon = 0.5);
        }
    }
}

#[test]
fn fixture_default() { check("../gfold-fixtures/data/default.json"); }
#[test]
fn fixture_moon() { check("../gfold-fixtures/data/moon.json"); }
#[test]
fn fixture_small_n() { check("../gfold-fixtures/data/small_n.json"); }
#[test]
fn fixture_glide() { check("../gfold-fixtures/data/glide.json"); }
