//! Config types mirroring generator/gfold/config.py.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Solver {
    pub n: usize,
    pub time_of_flight: f64,
}

impl Default for Solver {
    fn default() -> Self {
        Self { n: 100, time_of_flight: 44.63 }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    pub fn dt(&self) -> f64 { self.solver.time_of_flight / self.solver.n as f64 }
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
        assert_relative_eq!(c.solver.time_of_flight, 44.63);
        assert_eq!(c.environment.gravity, [0.0, 0.0, -3.71]);
    }

    #[test]
    fn derived_quantities() {
        let c = Config::default();
        assert_relative_eq!(c.log_wet_mass(), (2000.0_f64).ln());
        assert_relative_eq!(c.log_dry_mass(), (2000.0_f64 - 1700.0).ln());
        assert_relative_eq!(c.min_thrust(), 24000.0 * 0.2);
        assert_relative_eq!(c.max_thrust(), 24000.0 * 0.8);
        assert_relative_eq!(c.sin_glide_slope(), 0.0); // angle 0
        assert_relative_eq!(c.dt(), 44.63 / 100.0);
    }
}
