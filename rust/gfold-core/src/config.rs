//! Config types mirroring generator/gfold/config.py.

#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Spacecraft {
    pub wet_mass: f64,
    pub fuel: f64,
    pub real_max_thrust: f64,
    pub min_thrust_pct: f64,
    pub max_thrust_pct: f64,
    pub max_velocity: f64,
    pub initial_position: [f64; 3],
    pub initial_velocity: [f64; 3],
    pub target_velocity: [f64; 3],
    pub target_position: [f64; 3],
    pub fuel_consumption: f64,
}

impl Default for Spacecraft {
    fn default() -> Self {
        Self {
            wet_mass: 2000.0,
            fuel: 1700.0,
            real_max_thrust: 24000.0,
            min_thrust_pct: 0.2,
            max_thrust_pct: 0.8,
            max_velocity: 1000.0,
            initial_position: [450.0, -330.0, 2400.0],
            initial_velocity: [-40.0, 10.0, -10.0],
            target_velocity: [0.0, 0.0, 0.0],
            target_position: [0.0, 0.0, 0.0],
            fuel_consumption: 5e-4,
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl Spacecraft {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        wet_mass = 2000.0, fuel = 1700.0, real_max_thrust = 24000.0,
        min_thrust_pct = 0.2, max_thrust_pct = 0.8, max_velocity = 1000.0,
        initial_position = [450.0, -330.0, 2400.0], initial_velocity = [-40.0, 10.0, -10.0],
        target_velocity = [0.0, 0.0, 0.0], target_position = [0.0, 0.0, 0.0],
        fuel_consumption = 5e-4,
    ))]
    fn new(
        wet_mass: f64, fuel: f64, real_max_thrust: f64,
        min_thrust_pct: f64, max_thrust_pct: f64, max_velocity: f64,
        initial_position: [f64; 3], initial_velocity: [f64; 3],
        target_velocity: [f64; 3], target_position: [f64; 3],
        fuel_consumption: f64,
    ) -> Self {
        Self {
            wet_mass, fuel, real_max_thrust, min_thrust_pct, max_thrust_pct, max_velocity,
            initial_position, initial_velocity, target_velocity, target_position, fuel_consumption,
        }
    }
}

#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Environment {
    pub gravity: [f64; 3],
    pub glide_slope_angle_deg: f64,
    pub max_angle_deg: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Self { gravity: [0.0, 0.0, -3.71], glide_slope_angle_deg: 0.0, max_angle_deg: 90.0 }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl Environment {
    #[new]
    #[pyo3(signature = (gravity = [0.0, 0.0, -3.71], glide_slope_angle_deg = 0.0, max_angle_deg = 90.0))]
    fn new(gravity: [f64; 3], glide_slope_angle_deg: f64, max_angle_deg: f64) -> Self {
        Self { gravity, glide_slope_angle_deg, max_angle_deg }
    }
}

#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Solver {
    pub n: usize,
    /// Discretization horizon. `None` ⇒ `solve` searches for the fuel-optimal
    /// time-of-flight (see [`crate::search`]).
    pub time_of_flight: Option<f64>,
    /// Lower bound for the time-of-flight search. `None` ⇒ auto-bracket.
    pub tof_min: Option<f64>,
    /// Upper bound for the time-of-flight search. `None` ⇒ auto-bracket.
    pub tof_max: Option<f64>,
}

impl Default for Solver {
    fn default() -> Self {
        Self { n: 100, time_of_flight: None, tof_min: None, tof_max: None }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (n = 100, time_of_flight = None, tof_min = None, tof_max = None))]
    fn new(n: usize, time_of_flight: Option<f64>, tof_min: Option<f64>, tof_max: Option<f64>) -> Self {
        Self { n, time_of_flight, tof_min, tof_max }
    }
}

#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all, from_py_object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    pub spacecraft: Spacecraft,
    pub environment: Environment,
    pub solver: Solver,
}

impl Config {
    pub fn log_wet_mass(&self) -> f64 { self.spacecraft.wet_mass.ln() }
    pub fn log_dry_mass(&self) -> f64 { (self.spacecraft.wet_mass - self.spacecraft.fuel).ln() }
    pub fn min_thrust(&self) -> f64 { self.spacecraft.real_max_thrust * self.spacecraft.min_thrust_pct }
    pub fn max_thrust(&self) -> f64 { self.spacecraft.real_max_thrust * self.spacecraft.max_thrust_pct }
    pub fn sin_glide_slope(&self) -> f64 { self.environment.glide_slope_angle_deg.to_radians().sin() }
    pub fn cos_max_angle(&self) -> f64 { self.environment.max_angle_deg.to_radians().cos() }
    pub fn dt(&self) -> f64 {
        self.solver.time_of_flight.expect("time_of_flight must be set before dt()") / self.solver.n as f64
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl Config {
    #[new]
    #[pyo3(signature = (spacecraft = Spacecraft::default(), environment = Environment::default(), solver = Solver::default()))]
    fn new(spacecraft: Spacecraft, environment: Environment, solver: Solver) -> Self {
        Self { spacecraft, environment, solver }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn defaults_match_python() {
        let c = Config::default();
        assert_eq!(c.spacecraft.wet_mass, 2000.0);
        assert_eq!(c.spacecraft.fuel, 1700.0);
        assert_eq!(c.solver.n, 100);
        assert!(c.solver.time_of_flight.is_none());
        assert_eq!(c.environment.gravity, [0.0, 0.0, -3.71]);
    }

    #[test]
    fn partial_json_fills_defaults() {
        let cfg: Config = serde_json::from_str(r#"{"spacecraft":{"wet_mass":1500.0}}"#).unwrap();
        assert_eq!(cfg.spacecraft.wet_mass, 1500.0);
        // untouched fields fall back to Spacecraft::default()
        assert_eq!(cfg.spacecraft.fuel, 1700.0);
        assert_eq!(cfg.environment.gravity, [0.0, 0.0, -3.71]);
        assert_eq!(cfg.solver.n, 100);
    }

    #[test]
    fn derived_quantities() {
        let mut c = Config::default();
        c.solver.time_of_flight = Some(44.63);
        assert_relative_eq!(c.log_wet_mass(), (2000.0_f64).ln());
        assert_relative_eq!(c.log_dry_mass(), (2000.0_f64 - 1700.0).ln());
        assert_relative_eq!(c.min_thrust(), 24000.0 * 0.2);
        assert_relative_eq!(c.max_thrust(), 24000.0 * 0.8);
        assert_relative_eq!(c.sin_glide_slope(), 0.0); // angle 0
        assert_relative_eq!(c.dt(), 44.63 / 100.0);
    }
}
