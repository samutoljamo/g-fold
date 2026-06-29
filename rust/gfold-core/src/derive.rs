//! Per-step derived parameter arrays (mirrors solver.py lines 73-84).
use crate::config::Config;

#[derive(Debug, Clone)]
pub struct Derived {
    pub z0: Vec<f64>,
    pub exp_z0: Vec<f64>,
    pub max_exp: Vec<f64>,
    pub min_exp: Vec<f64>,
}

pub fn derive(cfg: &Config) -> Derived {
    let n = cfg.solver.n;
    let dt = cfg.dt();
    let a = cfg.spacecraft.fuel_consumption;
    let max_t = cfg.max_thrust();
    let min_t = cfg.min_thrust();
    let wet = cfg.spacecraft.wet_mass;

    let mut z0 = Vec::with_capacity(n);
    let mut exp_z0 = Vec::with_capacity(n);
    let mut max_exp = Vec::with_capacity(n);
    let mut min_exp = Vec::with_capacity(n);

    for i in 0..n {
        let z = (wet - a * dt * max_t * i as f64).ln();
        let e = (-z).exp();
        z0.push(z);
        exp_z0.push(e);
        max_exp.push(1.0 / (e * max_t));
        min_exp.push(if min_t == 0.0 { f64::INFINITY } else { 1.0 / (e * min_t) });
    }

    Derived { z0, exp_z0, max_exp, min_exp }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn derive_matches_formula() {
        let mut c = Config::default();
        c.solver.time_of_flight = Some(44.63);
        let d = derive(&c);
        let n = c.solver.n;
        assert_eq!(d.z0.len(), n);
        let dt = c.dt();
        let a = c.spacecraft.fuel_consumption;
        let max_t = c.max_thrust();
        // i = 0
        let z0_0 = (c.spacecraft.wet_mass).ln();
        assert_relative_eq!(d.z0[0], z0_0, max_relative = 1e-12);
        assert_relative_eq!(d.exp_z0[0], (-z0_0).exp(), max_relative = 1e-12);
        assert_relative_eq!(d.max_exp[0], 1.0 / ((-z0_0).exp() * max_t), max_relative = 1e-12);
        assert_relative_eq!(d.min_exp[0], 1.0 / ((-z0_0).exp() * c.min_thrust()), max_relative = 1e-12);
        // i = 5
        let z0_5 = (c.spacecraft.wet_mass - a * dt * max_t * 5.0).ln();
        assert_relative_eq!(d.z0[5], z0_5, max_relative = 1e-12);
    }
}
