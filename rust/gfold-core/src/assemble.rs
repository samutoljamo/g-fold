//! Problem assembly: variable index map and conic problem construction.

use crate::config::Config;
use crate::derive::Derived;

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub n: usize,
}

impl Layout {
    pub fn nvars(&self) -> usize { 11 * self.n }
    pub fn x(&self, i: usize, comp: usize) -> usize { 6 * i + comp }
    pub fn u(&self, i: usize, comp: usize) -> usize { 6 * self.n + 3 * i + comp }
    pub fn s(&self, i: usize) -> usize { 9 * self.n + i }
    pub fn z(&self, i: usize) -> usize { 10 * self.n + i }
}

#[derive(Debug, Clone)]
pub struct Row {
    pub coeffs: Vec<(usize, f64)>,
    pub b: f64,
}

pub fn eval_row(row: &Row, point: &[f64]) -> f64 {
    row.coeffs.iter().map(|&(idx, c)| c * point[idx]).sum()
}

#[derive(Debug)]
pub struct Builder {
    pub layout: Layout,
    pub rows: Vec<Row>,
}

impl Builder {
    pub fn new(layout: Layout) -> Self { Self { layout, rows: Vec::new() } }
    pub fn push(&mut self, row: Row) { self.rows.push(row); }
    pub fn nrows(&self) -> usize { self.rows.len() }
}

pub fn equality_rows(cfg: &Config, _der: &Derived) -> Vec<Row> {
    let n = cfg.solver.n;
    let l = Layout { n };
    let dt = cfg.dt();
    let dt2 = dt * dt;
    let g = cfg.environment.gravity;
    let a_dt = cfg.spacecraft.fuel_consumption * dt;
    let mut rows = Vec::new();

    // initial position / velocity
    for c in 0..3 {
        rows.push(Row { coeffs: vec![(l.x(0, c), 1.0)], b: cfg.spacecraft.initial_position[c] });
    }
    for c in 0..3 {
        rows.push(Row { coeffs: vec![(l.x(0, c + 3), 1.0)], b: cfg.spacecraft.initial_velocity[c] });
    }
    // z[0] = log_wet_mass
    rows.push(Row { coeffs: vec![(l.z(0), 1.0)], b: cfg.log_wet_mass() });

    // dynamics
    for i in 0..n - 1 {
        // position update, per component
        for c in 0..3 {
            rows.push(Row {
                coeffs: vec![
                    (l.x(i + 1, c), 1.0),
                    (l.x(i, c), -1.0),
                    (l.x(i, c + 3), -dt / 2.0),
                    (l.x(i + 1, c + 3), -dt / 2.0),
                    (l.u(i, c), -dt2 / 4.0),
                    (l.u(i + 1, c), -dt2 / 4.0),
                ],
                b: g[c] * dt2 / 2.0,
            });
        }
        // velocity update, per component
        for c in 0..3 {
            rows.push(Row {
                coeffs: vec![
                    (l.x(i + 1, c + 3), 1.0),
                    (l.x(i, c + 3), -1.0),
                    (l.u(i, c), -dt / 2.0),
                    (l.u(i + 1, c), -dt / 2.0),
                ],
                b: g[c] * dt,
            });
        }
        // mass update
        rows.push(Row {
            coeffs: vec![
                (l.z(i + 1), 1.0),
                (l.z(i), -1.0),
                (l.s(i), a_dt / 2.0),
                (l.s(i + 1), a_dt / 2.0),
            ],
            b: 0.0,
        });
    }

    // final position / velocity
    for c in 0..3 {
        rows.push(Row { coeffs: vec![(l.x(n - 1, c), 1.0)], b: cfg.spacecraft.target_position[c] });
    }
    for c in 0..3 {
        rows.push(Row { coeffs: vec![(l.x(n - 1, c + 3), 1.0)], b: cfg.spacecraft.target_velocity[c] });
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    #[test]
    fn layout_offsets() {
        let l = Layout { n: 100 };
        assert_eq!(l.nvars(), 1100);
        assert_eq!(l.x(0, 0), 0);
        assert_eq!(l.x(0, 5), 5);
        assert_eq!(l.x(1, 0), 6);
        assert_eq!(l.u(0, 0), 600);
        assert_eq!(l.u(1, 2), 605);
        assert_eq!(l.s(0), 900);
        assert_eq!(l.s(99), 999);
        assert_eq!(l.z(0), 1000);
        assert_eq!(l.z(99), 1099);
    }

    #[test]
    fn eval_row_dots_point() {
        let l = Layout { n: 2 };
        let row = Row { coeffs: vec![(l.x(0,0), 2.0), (l.z(1), -1.0)], b: 3.0 };
        let mut point = vec![0.0; l.nvars()];
        point[l.x(0,0)] = 5.0;
        point[l.z(1)] = 4.0;
        assert_eq!(eval_row(&row, &point), 2.0 * 5.0 + (-1.0) * 4.0);
    }

    #[test]
    fn velocity_update_row_residual_zero_for_consistent_point() {
        let cfg = Config::default();
        let der = crate::derive::derive(&cfg);
        let l = Layout { n: cfg.solver.n };
        let dt = cfg.dt();
        let g = cfg.environment.gravity;
        let rows = equality_rows(&cfg, &der);

        // Build a point that satisfies the velocity update for i=0, comp=0:
        // x[1,3] = x[0,3] + (u[0,0]+u[1,0])/2*dt + g[0]*dt
        let mut p = vec![0.0; l.nvars()];
        p[l.x(0,3)] = 2.0; p[l.u(0,0)] = 1.0; p[l.u(1,0)] = 3.0;
        p[l.x(1,3)] = 2.0 + (1.0 + 3.0)/2.0*dt + g[0]*dt;

        // find the velocity-update row for i=0, comp=0 by its coefficient signature
        let row = rows.iter().find(|r|
            r.coeffs.iter().any(|&(idx,c)| idx==l.x(1,3) && (c-1.0).abs()<1e-12)
            && r.coeffs.iter().any(|&(idx,_)| idx==l.u(1,0))
        ).expect("velocity update row");
        // Clarabel residual b - a^T x should be 0
        assert_relative_eq!(row.b - eval_row(row, &p), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn equality_row_count() {
        let cfg = Config::default();
        let der = crate::derive::derive(&cfg);
        let n = cfg.solver.n;
        let rows = equality_rows(&cfg, &der);
        // 3+3+1 boundary-initial + (n-1)*(3+3+1) dynamics + 3+3 final
        assert_eq!(rows.len(), 7 + (n-1)*7 + 6);
    }
}
