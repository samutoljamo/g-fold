//! Problem assembly: variable index map and conic problem construction.

use crate::config::Config;
use crate::derive::Derived;
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{self, NonnegativeConeT, SecondOrderConeT, ZeroConeT};

pub struct Problem {
    pub p_mat: CscMatrix<f64>,
    pub q: Vec<f64>,
    pub a_mat: CscMatrix<f64>,
    pub b: Vec<f64>,
    pub cones: Vec<SupportedConeT<f64>>,
    pub layout: Layout,
}

fn rows_to_csc(rows: &[Row], nrows: usize, ncols: usize) -> CscMatrix<f64> {
    // collect per-column entries: (row_index, value)
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ncols];
    for (r, row) in rows.iter().enumerate() {
        for &(c, v) in &row.coeffs {
            cols[c].push((r, v));
        }
    }
    let mut colptr = Vec::with_capacity(ncols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0usize);
    for col in cols.iter_mut() {
        col.sort_by_key(|&(r, _)| r);
        for &(r, v) in col.iter() {
            rowval.push(r);
            nzval.push(v);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

pub fn assemble(cfg: &Config) -> Problem {
    let n = cfg.solver.n;
    let layout = Layout { n };
    let nvars = layout.nvars();
    let der = crate::derive::derive(cfg);

    // objective
    let mut q = vec![0.0; nvars];
    q[layout.z(n - 1)] = -1.0;
    let p_mat = CscMatrix::zeros((nvars, nvars));

    // gather blocks
    let eq = equality_rows(cfg, &der);
    let (gs_soc, gs_nn) = glide_slope(cfg);
    let nn = nonneg_bounds(cfg, &der);
    let vel = velocity_soc(cfg);
    let tslack = thrust_slack_soc(cfg);
    let tlow = thrust_lower_soc(cfg, &der);

    let mut rows: Vec<Row> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    // 1. equalities
    cones.push(ZeroConeT(eq.len()));
    rows.extend(eq);

    // 2. nonnegatives (glide-slope-zero rows then thrust-upper+dry-mass)
    let nn_count = gs_nn.len() + nn.len();
    cones.push(NonnegativeConeT(nn_count));
    rows.extend(gs_nn);
    rows.extend(nn);

    // 3. SOC blocks
    for blk in vel.into_iter().chain(tslack).chain(gs_soc).chain(tlow) {
        cones.push(SecondOrderConeT(blk.dim));
        rows.extend(blk.rows);
    }

    let nrows = rows.len();
    let mut b = Vec::with_capacity(nrows);
    for row in &rows {
        b.push(row.b);
    }
    let a_mat = rows_to_csc(&rows, nrows, nvars);

    Problem { p_mat, q, a_mat, b, cones, layout }
}

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

#[derive(Debug, Clone)]
pub struct SocBlock {
    pub rows: Vec<Row>,
    pub dim: usize,
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

pub fn velocity_soc(cfg: &Config) -> Vec<SocBlock> {
    let n = cfg.solver.n;
    let l = Layout { n };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut rows = Vec::with_capacity(4);
        rows.push(Row { coeffs: vec![], b: cfg.spacecraft.max_velocity }); // t = max_vel
        for c in 0..3 {
            rows.push(Row { coeffs: vec![(l.x(i, c + 3), -1.0)], b: 0.0 }); // v_c = x[i,c+3]
        }
        out.push(SocBlock { rows, dim: 4 });
    }
    out
}

pub fn thrust_slack_soc(cfg: &Config) -> Vec<SocBlock> {
    let n = cfg.solver.n;
    let l = Layout { n };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut rows = Vec::with_capacity(4);
        rows.push(Row { coeffs: vec![(l.s(i), -1.0)], b: 0.0 }); // t = s[i]
        for c in 0..3 {
            rows.push(Row { coeffs: vec![(l.u(i, c), -1.0)], b: 0.0 }); // v_c = u[i,c]
        }
        out.push(SocBlock { rows, dim: 4 });
    }
    out
}

pub fn glide_slope(cfg: &Config) -> (Vec<SocBlock>, Vec<Row>) {
    let n = cfg.solver.n;
    let l = Layout { n };
    let sin = cfg.sin_glide_slope();
    if sin == 0.0 {
        let mut nn = Vec::with_capacity(n);
        for i in 0..n {
            nn.push(Row { coeffs: vec![(l.x(i, 2), -1.0)], b: 0.0 }); // x[i,2] >= 0
        }
        (Vec::new(), nn)
    } else {
        let mut soc = Vec::with_capacity(n);
        for i in 0..n {
            let mut rows = Vec::with_capacity(4);
            rows.push(Row { coeffs: vec![(l.x(i, 2), -1.0)], b: 0.0 }); // t = x[i,2]
            for c in 0..3 {
                rows.push(Row { coeffs: vec![(l.x(i, c), -sin)], b: 0.0 }); // v_c = sin * x[i,c]
            }
            soc.push(SocBlock { rows, dim: 4 });
        }
        (soc, Vec::new())
    }
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

pub fn thrust_lower_soc(cfg: &Config, der: &Derived) -> Vec<SocBlock> {
    let n = cfg.solver.n;
    let l = Layout { n };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let m = der.min_exp[i];
        let z0 = der.z0[i];
        let rows = vec![
            Row { coeffs: vec![(l.s(i), -m), (l.z(i), -1.0)], b: -z0 },        // t
            Row { coeffs: vec![(l.z(i), -2.0)], b: -2.0 * z0 },                // v0 = 2w
            Row { coeffs: vec![(l.s(i), -m), (l.z(i), -1.0)], b: -z0 - 2.0 },  // v1
        ];
        out.push(SocBlock { rows, dim: 3 });
    }
    out
}

pub fn nonneg_bounds(cfg: &Config, der: &Derived) -> Vec<Row> {
    let n = cfg.solver.n;
    let l = Layout { n };
    let mut rows = Vec::with_capacity(n + 1);
    for i in 0..n {
        rows.push(Row {
            coeffs: vec![(l.z(i), 1.0), (l.s(i), der.max_exp[i])],
            b: 1.0 + der.z0[i],
        });
    }
    rows.push(Row { coeffs: vec![(l.z(n - 1), -1.0)], b: -cfg.log_dry_mass() });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    fn soc_s(block: &SocBlock, p: &[f64]) -> Vec<f64> {
        block.rows.iter().map(|r| r.b - eval_row(r, p)).collect()
    }

    #[test]
    fn thrust_slack_cone_membership() {
        let cfg = Config::default();
        let l = Layout { n: cfg.solver.n };
        let blocks = thrust_slack_soc(&cfg);
        let mut p = vec![0.0; l.nvars()];
        // u[0] = (3,4,0) -> norm 5 ; s[0] = 6 >= 5 feasible
        p[l.u(0,0)] = 3.0; p[l.u(0,1)] = 4.0; p[l.s(0)] = 6.0;
        let s = soc_s(&blocks[0], &p);
        assert_eq!(s.len(), 4);
        assert_relative_eq!(s[0], 6.0, epsilon=1e-12); // t
        let norm = (s[1]*s[1]+s[2]*s[2]+s[3]*s[3]).sqrt();
        assert_relative_eq!(norm, 5.0, epsilon=1e-12);
        assert!(s[0] >= norm); // in cone
    }

    #[test]
    fn velocity_cone_t_is_maxvel() {
        let cfg = Config::default();
        let l = Layout { n: cfg.solver.n };
        let blocks = velocity_soc(&cfg);
        let p = vec![0.0; l.nvars()];
        let s = soc_s(&blocks[0], &p);
        assert_relative_eq!(s[0], cfg.spacecraft.max_velocity, epsilon=1e-12);
    }

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

    #[test]
    fn glide_slope_zero_angle_uses_nonneg() {
        let cfg = Config::default(); // angle 0
        let (soc, nn) = glide_slope(&cfg);
        assert!(soc.is_empty());
        assert_eq!(nn.len(), cfg.solver.n);
        let l = Layout { n: cfg.solver.n };
        let mut p = vec![0.0; l.nvars()];
        p[l.x(0,2)] = 7.0;
        assert_relative_eq!(nn[0].b - eval_row(&nn[0], &p), 7.0, epsilon=1e-12);
    }

    #[test]
    fn glide_slope_nonzero_angle_uses_soc() {
        let mut cfg = Config::default();
        cfg.environment.glide_slope_angle_deg = 30.0;
        let (soc, nn) = glide_slope(&cfg);
        assert!(nn.is_empty());
        assert_eq!(soc.len(), cfg.solver.n);
        assert_eq!(soc[0].dim, 4);
    }

    #[test]
    fn thrust_upper_bound_residual() {
        let cfg = Config::default();
        let der = crate::derive::derive(&cfg);
        let l = Layout { n: cfg.solver.n };
        let rows = nonneg_bounds(&cfg, &der);
        // first n rows are thrust-upper; row i=0
        let mut p = vec![0.0; l.nvars()];
        p[l.z(0)] = der.z0[0];  // z = z0 -> bracket = 1
        p[l.s(0)] = 0.0;
        // s_row = 1 + z0 - z - s*max_exp = 1 + z0[0] - z0[0] - 0 = 1
        assert_relative_eq!(rows[0].b - eval_row(&rows[0], &p), 1.0, epsilon=1e-9);
    }

    #[test]
    fn nonneg_row_count() {
        let cfg = Config::default();
        let der = crate::derive::derive(&cfg);
        let rows = nonneg_bounds(&cfg, &der);
        assert_eq!(rows.len(), cfg.solver.n + 1);
    }

    #[test]
    fn assemble_shapes_and_q() {
        let cfg = Config::default();
        let prob = assemble(&cfg);
        let n = cfg.solver.n;
        assert_eq!(prob.q.len(), 11 * n);
        assert_relative_eq!(prob.q[prob.layout.z(n-1)], -1.0, epsilon=1e-12);
        // A is (nrows x 11n)
        assert_eq!(prob.a_mat.n, 11 * n);
        assert_eq!(prob.b.len(), prob.a_mat.m);
    }

    #[test]
    fn thrust_lower_cone_membership_feasible() {
        let cfg = Config::default();
        let der = crate::derive::derive(&cfg);
        let l = Layout { n: cfg.solver.n };
        let m = der.min_exp[0];
        let z0 = der.z0[0];
        let blocks = thrust_lower_soc(&cfg, &der);
        assert_eq!(blocks[0].dim, 3);

        // choose w = 0 (z = z0). Then constraint: 1 - 0 + 0 <= s*m  => s*m >= 1.
        // pick s*m = 4 (well above 1) -> feasible, expect t >= ||v||.
        let s_val = 4.0 / m;
        let mut p = vec![0.0; l.nvars()];
        p[l.z(0)] = z0;
        p[l.s(0)] = s_val;
        let sv: Vec<f64> = blocks[0].rows.iter().map(|r| r.b - eval_row(r, &p)).collect();
        let t = sv[0];
        let norm = (sv[1]*sv[1] + sv[2]*sv[2]).sqrt();
        assert!(t >= norm - 1e-9, "t={t} norm={norm}");
        // boundary check: at s*m = 1 (w=0), constraint is tight -> t == norm
        p[l.s(0)] = 1.0 / m;
        let sv2: Vec<f64> = blocks[0].rows.iter().map(|r| r.b - eval_row(r, &p)).collect();
        let norm2 = (sv2[1]*sv2[1] + sv2[2]*sv2[2]).sqrt();
        assert_relative_eq!(sv2[0], norm2, epsilon=1e-9);
    }
}
