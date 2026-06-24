# G-FOLD Rust Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust crate (`gfold-core`) that assembles the basic G-FOLD SOCP directly and solves it with the Clarabel Rust crate, verified against the existing Python/CVXPY implementation and independent physics checks.

**Architecture:** A Cargo workspace under `rust/`. `gfold-core` has focused modules: `config` (serde structs + derived quantities), `derive` (per-step parameter arrays), `assemble` (variable index map + hand-built `P,q,A,b,cones`), `solve` (Clarabel driver + typed `Trajectory`), `validate` (physics checks). The Python package is retained as the test oracle and fixture generator. Correctness rests on three legs: matrix-diff vs CVXPYGen, solution fixtures vs CVXPY, and independent physics validation.

**Tech Stack:** Rust (edition 2021), `clarabel = "0.11"`, `serde` + `serde_json`, `approx` (float asserts), `criterion` (benches). Python side: existing `cvxpy`/`cvxpygen`.

## Global Constraints

- Decision vector ordering is fixed everywhere: `x` (6n) ‖ `u` (3n) ‖ `s` (n) ‖ `z` (n), total length `11n`. Column-index helpers are the single source of truth (defined in Task 4).
- All matrices passed to Clarabel are `clarabel::algebra::CscMatrix<f64>`; `P` is the zero matrix (objective is linear).
- Objective: maximize `z[n-1]` → Clarabel minimizes, so `q[idx_z(n-1)] = -1.0`, all other `q` entries `0.0`.
- Physics constants and defaults must match `generator/gfold/config.py` exactly (wet_mass 2000, fuel 1700, real_max_thrust 24000, min_thrust_pct 0.2, max_thrust_pct 0.8, max_velocity 1000, fuel_consumption 5e-4, Mars gravity [0,0,-3.71], default glide_slope_angle 0, n 100, time_of_flight 44.63).
- Float comparisons in tests use `approx::assert_relative_eq!` / `assert_abs_diff_eq!` — never `==`.
- Commit after every task. Conventional commit messages.
- Do not modify `generator/gfold/solver.py`'s formulation. Python changes are additive only (a fixture-dump command).
- Bindings (PyO3 / C ABI) are OUT OF SCOPE. Do not add them.

---

## File Structure

- `rust/Cargo.toml` — workspace manifest
- `rust/gfold-core/Cargo.toml` — core crate manifest
- `rust/gfold-core/src/lib.rs` — module wiring + public re-exports
- `rust/gfold-core/src/config.rs` — config structs, serde, derived quantities
- `rust/gfold-core/src/derive.rs` — per-step parameter arrays
- `rust/gfold-core/src/assemble.rs` — index map + `Problem` assembly
- `rust/gfold-core/src/solve.rs` — Clarabel driver + `Trajectory`
- `rust/gfold-core/src/validate.rs` — physics/optimality checks
- `rust/gfold-core/tests/fixtures.rs` — oracle comparison against committed JSON
- `rust/gfold-core/tests/matrix_diff.rs` — optional (ignored) cross-check vs CVXPYGen export
- `rust/gfold-core/benches/solve.rs` — criterion benchmarks
- `rust/gfold-fixtures/data/*.json` — committed fixtures
- `generator/gfold/fixtures.py` — fixture dump command (additive)

---

### Task 1: Workspace + crate skeleton

**Files:**
- Create: `rust/Cargo.toml`, `rust/gfold-core/Cargo.toml`, `rust/gfold-core/src/lib.rs`

**Interfaces:**
- Produces: a compiling `gfold-core` library crate; modules declared but empty.

- [ ] **Step 1: Create the workspace manifest**

`rust/Cargo.toml`:
```toml
[workspace]
members = ["gfold-core"]
resolver = "2"
```

- [ ] **Step 2: Create the core crate manifest**

`rust/gfold-core/Cargo.toml`:
```toml
[package]
name = "gfold-core"
version = "0.1.0"
edition = "2021"

[dependencies]
clarabel = "0.11"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
approx = "0.5"
```

- [ ] **Step 3: Create lib.rs with module declarations**

`rust/gfold-core/src/lib.rs`:
```rust
pub mod config;
pub mod derive;
pub mod assemble;
pub mod solve;
pub mod validate;
```

Create empty placeholder files `config.rs`, `derive.rs`, `assemble.rs`, `solve.rs`, `validate.rs` each containing only a doc comment line so the crate compiles, e.g. `//! Config types.`

- [ ] **Step 4: Verify it compiles**

Run: `cd rust && cargo build`
Expected: success, no errors.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "chore: scaffold rust workspace and gfold-core crate"
```

---

### Task 2: Config types and derived quantities

**Files:**
- Modify: `rust/gfold-core/src/config.rs`

**Interfaces:**
- Produces:
  - `struct Spacecraft { wet_mass, fuel, real_max_thrust, min_thrust_pct, max_thrust_pct, max_velocity: f64, initial_position, initial_velocity, target_velocity, target_position: [f64;3], fuel_consumption: f64 }`
  - `struct Environment { gravity: [f64;3], glide_slope_angle_deg: f64, max_angle_deg: f64 }`
  - `struct Solver { n: usize, time_of_flight: f64 }`
  - `struct Config { spacecraft: Spacecraft, environment: Environment, solver: Solver }`
  - Methods on `Config`: `log_wet_mass() -> f64`, `log_dry_mass() -> f64`, `min_thrust() -> f64`, `max_thrust() -> f64`, `sin_glide_slope() -> f64`, `cos_max_angle() -> f64`, `dt() -> f64`. `Config::default()` matching Python defaults.
- All structs `#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]`.

- [ ] **Step 1: Write failing tests for defaults and derived values**

```rust
// at bottom of config.rs
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core config`
Expected: FAIL — types not defined.

- [ ] **Step 3: Implement config.rs**

```rust
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
```

(Note: `Config::default()` requires each field's `Default`, which the above provides.)

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core config`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/config.rs
git commit -m "feat: gfold-core config types and derived quantities"
```

---

### Task 3: Derived per-step parameter arrays

**Files:**
- Modify: `rust/gfold-core/src/derive.rs`

**Interfaces:**
- Consumes: `config::Config`.
- Produces:
  - `struct Derived { z0: Vec<f64>, exp_z0: Vec<f64>, max_exp: Vec<f64>, min_exp: Vec<f64> }` (each length `n`).
  - `fn derive(cfg: &Config) -> Derived`.
- Mirrors `solver.py` lines 73–84 exactly:
  - `z0[i] = ln(wet_mass - fuel_consumption*dt*max_thrust*i)`
  - `exp_z0[i] = exp(-z0[i])`
  - `max_exp[i] = 1 / (exp(-z0[i]) * max_thrust)`
  - `min_exp[i] = 1 / (exp(-z0[i]) * min_thrust)` (min_thrust is nonzero for defaults; if min_thrust==0, set min_exp[i]=f64::INFINITY — this constraint then never binds. Default min_thrust_pct=0.2 so this branch is not exercised by defaults.)

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    #[test]
    fn derive_matches_formula() {
        let c = Config::default();
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core derive`
Expected: FAIL.

- [ ] **Step 3: Implement derive.rs**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core derive`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/derive.rs
git commit -m "feat: per-step derived parameter arrays"
```

---

### Task 4: Variable index map

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces (in `assemble.rs`, `pub`):
  - `struct Layout { n: usize }`
  - `impl Layout`: `fn nvars(&self) -> usize` (= `11*n`); `fn x(&self, i: usize, comp: usize) -> usize` (comp 0..6); `fn u(&self, i: usize, comp: usize) -> usize` (comp 0..3); `fn s(&self, i: usize) -> usize`; `fn z(&self, i: usize) -> usize`.
- Block offsets: x at `0`, u at `6n`, s at `9n`, z at `10n`.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

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
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core layout`
Expected: FAIL.

- [ ] **Step 3: Implement Layout**

```rust
//! Problem assembly: variable index map and conic problem construction.

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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core layout`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: variable index map (Layout)"
```

---

### Task 5: Affine-row builder and the `Problem` container

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces:
  - `struct Row { coeffs: Vec<(usize, f64)>, b: f64 }` — represents one row `sum(coeff*var) (+slack) = b` (the affine part `a^T x`, with `b` the RHS such that the Clarabel row is `b - a^T x = s`, i.e. Clarabel `A` row = `a`, `b` = `b`).
  - `struct Builder { layout: Layout, rows: Vec<Row> }` with `fn new(layout) -> Self`, `fn push(&mut self, row: Row)`, `fn nrows(&self) -> usize`.
  - `fn eval_row(row: &Row, point: &[f64]) -> f64` returning `sum(coeff*point[idx])` (the `a^T x` value, for testing).
- Clarabel convention used throughout: for a constraint `a^T x  (cone-relation)  b`, the row contributes `A` row = coefficients of `a`, and `b` entry = `b`, so that `b - A x = s ∈ cone`. We standardize: **every Row stores `a` (coeffs) and the scalar `b`** and the cone is decided by which group the row is pushed into. Equalities use ZeroCone (`s = 0` ⇒ `A x = b`).

- [ ] **Step 1: Write failing test for eval_row**

```rust
#[test]
fn eval_row_dots_point() {
    let l = Layout { n: 2 };
    let row = Row { coeffs: vec![(l.x(0,0), 2.0), (l.z(1), -1.0)], b: 3.0 };
    let mut point = vec![0.0; l.nvars()];
    point[l.x(0,0)] = 5.0;
    point[l.z(1)] = 4.0;
    assert_eq!(eval_row(&row, &point), 2.0 * 5.0 + (-1.0) * 4.0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core eval_row`
Expected: FAIL.

- [ ] **Step 3: Implement Row / Builder / eval_row**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core eval_row`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: affine Row builder and eval helper"
```

---

### Task 6: Equality (ZeroCone) rows — boundary + dynamics

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Consumes: `Layout`, `Builder`, `Config`, `Derived`.
- Produces: `fn equality_rows(cfg: &Config, der: &Derived) -> Vec<Row>` — all ZeroCone rows in this order:
  1. `x[0,c] = initial_position[c]` for c in 0..3
  2. `x[0,c+3] = initial_velocity[c]` for c in 0..3
  3. `z[0] = log_wet_mass`
  4. For i in 0..n-1: position update (3 rows), velocity update (3 rows), mass update (1 row)
  5. `x[n-1,c] = target_position[c]` for c in 0..3
  6. `x[n-1,c+3] = target_velocity[c]` for c in 0..3
- Dynamics, from `solver.py` lines 136–140 (with `acc = (u[i+1]+u[i])/2`, `dt2 = dt*dt`, `g_dt = g*dt`, `g_dt2 = g*dt*dt`):
  - Position (per comp c): `x[i+1,c] - x[i,c] - (x[i,c+3]+x[i+1,c+3])*dt/2 - (acc_c*dt2 + g[c]*dt2)/2 = 0`. As a Row with `b = (g[c]*dt2)/2`: coeffs = `{x(i+1,c):1, x(i,c):-1, x(i,c+3):-dt/2, x(i+1,c+3):-dt/2, u(i,c):-dt2/4, u(i+1,c):-dt2/4}`, `b = g[c]*dt2/2`.
  - Velocity (per comp c): `x[i+1,c+3] - x[i,c+3] - (u[i,c]+u[i+1,c])*dt/2 = g[c]*dt`. coeffs = `{x(i+1,c+3):1, x(i,c+3):-1, u(i,c):-dt/2, u(i+1,c):-dt/2}`, `b = g[c]*dt`.
  - Mass: `z[i+1] - z[i] + (s[i]+s[i+1])/2*a_dt = 0` where `a_dt = fuel_consumption*dt`. coeffs = `{z(i+1):1, z(i):-1, s(i):a_dt/2, s(i+1):a_dt/2}`, `b = 0`.

- [ ] **Step 1: Write failing tests verifying rows evaluate correctly on a constructed point**

```rust
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
```
(Add `use crate::config::Config; use approx::assert_relative_eq;` to the test module.)

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core equality`
Expected: FAIL — `equality_rows` not defined.

- [ ] **Step 3: Implement equality_rows**

```rust
use crate::config::Config;
use crate::derive::Derived;

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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core equality`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: equality (ZeroCone) rows for boundary and dynamics"
```

---

### Task 7: Second-order cone blocks — velocity bound and thrust-slack norms

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces:
  - `struct SocBlock { rows: Vec<Row>, dim: usize }` — `dim` consecutive rows forming one SOC (first row = the scalar bound `t`, remaining = the vector whose norm is bounded). Clarabel SOC requires `(t, v)` with `t >= ||v||`, encoded as `s = b - A x ∈ SOC` where `s[0]=t`, `s[1..]=v`.
  - `fn velocity_soc(cfg: &Config) -> Vec<SocBlock>` — for each i: `max_vel >= ||x[i,3:]||`. Block dim 4. Row 0: `t = max_vel - 0` → coeffs empty, `b = max_vel`. Rows 1..4: `v_c = x[i,c+3]` → for the cone we need `s[1..] = x[i,3:]`, i.e. `b=0`, coeffs `{x(i,c+3): -1.0}` so that `s = b - A x = x[i,c+3]`. (Wait: `s = b - A x`; to get `s = x`, set `A` coeff `= -1`, `b=0`.)
  - `fn thrust_slack_soc(cfg: &Config) -> Vec<SocBlock>` — for each i: `s[i] >= ||u[i,:]||`. Block dim 4. Row 0: `t = s[i]` → `s_row = b - A x = s[i]` ⇒ coeffs `{s(i): -1.0}`, `b=0`. Rows 1..4: `v_c = u[i,c]` ⇒ coeffs `{u(i,c): -1.0}`, `b=0`.

**Clarabel SOC sign rule (used here and in Tasks 8–10):** a Clarabel row produces `s = b - A x`. For `s[0] >= ||s[1..]||`:
- A term equal to `+expr` in the cone slot ⇒ `A` coeff `= -coeff_of_expr`, contribute `expr`'s constant to `b`.

- [ ] **Step 1: Write failing test (cone membership on a feasible point)**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core cone`
Expected: FAIL.

- [ ] **Step 3: Implement the two SOC builders**

```rust
#[derive(Debug, Clone)]
pub struct SocBlock {
    pub rows: Vec<Row>,
    pub dim: usize,
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core cone`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: SOC blocks for velocity bound and thrust-slack norm"
```

---

### Task 8: Glide-slope block (SOC or Nonnegative branch)

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces:
  - `fn glide_slope(cfg: &Config) -> (Vec<SocBlock>, Vec<Row>)` — returns `(soc_blocks, nonneg_rows)`. When `sin_glide_slope() > 0`: one SOC block per step of dim 4 for `x[i,2] >= sinγ * ||x[i,:3]||` ⇒ `t = x[i,2]` (coeffs `{x(i,2):-1}`,b=0), `v_c = sinγ * x[i,c]` (coeffs `{x(i,c):-sinγ}`, b=0). When `sinγ == 0`: empty SOC vec, and one nonneg row per step `x[i,2] >= 0` ⇒ Row coeffs `{x(i,2):-1.0}`, b=0 (so `s = x[i,2] >= 0`).

- [ ] **Step 1: Write failing tests for both branches**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core glide`
Expected: FAIL.

- [ ] **Step 3: Implement glide_slope**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core glide`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: glide-slope block with SOC and nonneg branches"
```

---

### Task 9: Nonnegative linear bounds — thrust upper bound and dry-mass

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces: `fn nonneg_bounds(cfg: &Config, der: &Derived) -> Vec<Row>` — nonnegative rows (each `s = b - A x >= 0`):
  - Thrust upper bound (`solver.py` line 134): `s[i]*max_exp[i] <= 1 - (z[i]-z0[i])` ⇒ `1 - z[i] + z0[i] - s[i]*max_exp[i] >= 0`. Row: coeffs `{z(i): 1.0, s(i): max_exp[i]}` with `b = 1.0 + z0[i]`. (Because `s_row = b - A x = 1 + z0[i] - z[i] - s[i]*max_exp[i]`.) One per step i in 0..n.
  - Dry-mass (`solver.py` line 147): `z[n-1] >= log_dry_mass` ⇒ `z[n-1] - log_dry_mass >= 0`. Row: coeffs `{z(n-1): -1.0}`, `b = -log_dry_mass`.

- [ ] **Step 1: Write failing test**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core nonneg`
Expected: FAIL.

- [ ] **Step 3: Implement nonneg_bounds**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core nonneg`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: nonnegative thrust-upper and dry-mass bounds"
```

---

### Task 10: Quadratic thrust lower bound → rotated SOC

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Produces: `fn thrust_lower_soc(cfg: &Config, der: &Derived) -> Vec<SocBlock>` — one SOC block (dim 3) per step encoding `solver.py` line 133: `1 - w + w²/2 <= s[i]*min_exp[i]`, where `w = z[i] - z0[i]`.

**Derivation (this is the analytically subtle block — implement exactly):**
Let `m = min_exp[i]`, `w = z[i] - z0[i]`, `p = s[i]*m` (the RHS). The constraint is `p >= 1 - w + w²/2`, i.e. `p - 1 + w >= w²/2`, i.e. `2(p - 1 + w) >= w²`. This is CVXPY's standard SOC reformulation of `square(w) <= 2*y` with `y = p - 1 + w`:
`square(w) <= y * 1` is the rotated form; CVXPY emits the equivalent standard SOC
`|| (2*w, y - 1) ||_2 <= y + 1` ... but to avoid sign mistakes we use the canonical CVXPY mapping it actually generates for `quad <= lin`: for `w² <= 2y`, the second-order cone is
`|| ( 2w , y - 1 ) ||_2 <= y + 1`.
Here `y = p - 1 + w = s[i]*m + w - 1 = s[i]*m + z[i] - z0[i] - 1`.

So the SOC (dim 3) is `t >= ||v||` with:
- `t = y + 1 = s[i]*m + z[i] - z0[i]`
- `v[0] = 2w = 2*(z[i] - z0[i])`
- `v[1] = y - 1 = s[i]*m + z[i] - z0[i] - 2`

As Clarabel rows (`s = b - A x`):
- Row t: `t = s[i]*m + z[i] - z0[i]` ⇒ coeffs `{s(i): -m, z(i): -1.0}`, `b = -z0[i]`.
- Row v0: `2*z[i] - 2*z0[i]` ⇒ coeffs `{z(i): -2.0}`, `b = -2.0*z0[i]`.
- Row v1: `s[i]*m + z[i] - z0[i] - 2` ⇒ coeffs `{s(i): -m, z(i): -1.0}`, `b = -z0[i] - 2.0`.

**This derivation MUST be cross-checked by the matrix-diff test (Task 15) against CVXPYGen.** If the SOC Clarabel emits differs, replace this block to match; the fixture test (Task 14) is the backstop.

- [ ] **Step 1: Write failing test (cone membership at a feasible point)**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core thrust_lower`
Expected: FAIL.

- [ ] **Step 3: Implement thrust_lower_soc**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core thrust_lower`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: quadratic thrust lower bound as rotated SOC"
```

---

### Task 11: Assemble full `Problem` (CSC + cones + objective)

**Files:**
- Modify: `rust/gfold-core/src/assemble.rs`

**Interfaces:**
- Consumes: all block builders above; `clarabel::algebra::CscMatrix`, `clarabel::solver::SupportedConeT`.
- Produces:
  - `struct Problem { p_mat: CscMatrix<f64>, q: Vec<f64>, a_mat: CscMatrix<f64>, b: Vec<f64>, cones: Vec<SupportedConeT<f64>>, layout: Layout }`
  - `fn assemble(cfg: &Config) -> Problem`.
- Row ordering (must be stable for matrix-diff): **all ZeroCone rows**, then **one NonnegativeConeT** covering all nonneg rows (glide-slope-zero rows, then thrust-upper + dry-mass), then **all SOC blocks** (velocity, then thrust-slack, then glide-slope-soc if any, then thrust-lower), each as its own `SecondOrderConeT(dim)`.
- `P` is the all-zero square matrix of size `nvars` (build an empty `CscMatrix::zeros((nvars, nvars))`).
- `q`: zeros except `q[layout.z(n-1)] = -1.0`.
- Build `A` from rows: convert the accumulated `Vec<Row>` (in cone order) into CSC. Use a triplet→CSC path: collect `(row, col, val)`, then build column-compressed arrays. (Helper `rows_to_csc(rows: &[Row], nrows, ncols) -> CscMatrix<f64>`.)

- [ ] **Step 1: Write failing test for shape, cones, q, and a known feasible solve**

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core assemble_shapes`
Expected: FAIL.

- [ ] **Step 3: Implement assemble + rows_to_csc**

```rust
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
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core assemble_shapes`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/assemble.rs
git commit -m "feat: assemble full Clarabel problem (CSC, cones, objective)"
```

---

### Task 12: Solve + typed `Trajectory`

**Files:**
- Modify: `rust/gfold-core/src/solve.rs`

**Interfaces:**
- Consumes: `assemble::{assemble, Problem, Layout}`, `config::Config`, Clarabel `DefaultSolver`, `DefaultSettings`, `DefaultSolverConfig` / builder, `SolverStatus`, `IPSolver` trait (for `.solve()`).
- Produces:
  - `struct Trajectory { positions: Vec<[f64;3]>, velocities: Vec<[f64;3]>, thrusts: Vec<f64>, normalized_thrusts: Vec<f64>, z_values: Vec<f64>, x_values: Vec<Vec<f64>>, u_values: Vec<[f64;3]>, s_values: Vec<f64>, objective: f64, final_mass: f64, time_points: Vec<f64>, status: String }`
  - `fn solve(cfg: &Config) -> Result<Trajectory, String>` — returns Err if status is not Solved/AlmostSolved.
- Thrust extraction mirrors `solver.py` lines 179–190: `thrusts[i] = ||u[i]|| * exp(z[i])`; `normalized = thrusts[i] / real_max_thrust`. `final_mass = exp(z[n-1])`. `objective` = solver objective value (`= -z[n-1]` minimized; report `z[n-1]` as `objective`). `time_points[i] = i*dt`.

- [ ] **Step 1: Write failing end-to-end test on default config**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use approx::assert_relative_eq;

    #[test]
    fn solves_default_problem() {
        let cfg = Config::default();
        let traj = solve(&cfg).expect("solve");
        let n = cfg.solver.n;
        assert_eq!(traj.positions.len(), n);
        // boundary conditions honored
        assert_relative_eq!(traj.positions[0][0], 450.0, epsilon=1e-2);
        assert_relative_eq!(traj.positions[n-1][0], 0.0, epsilon=1e-2);
        assert_relative_eq!(traj.positions[n-1][2], 0.0, epsilon=1e-2);
        // final mass within dry/wet bounds
        assert!(traj.final_mass <= 2000.0 + 1.0);
        assert!(traj.final_mass >= 300.0 - 1.0);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core solves_default`
Expected: FAIL.

- [ ] **Step 3: Implement solve.rs**

```rust
//! Clarabel driver + typed Trajectory.
use crate::assemble::{assemble, Layout};
use crate::config::Config;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};

#[derive(Debug, Clone)]
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
    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(
        &prob.p_mat, &prob.q, &prob.a_mat, &prob.b, &prob.cones, settings,
    );
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
        positions, velocities, thrusts, normalized_thrusts, z_values,
        u_values, s_values,
        objective: z_final,
        final_mass: z_final.exp(),
        time_points,
        status: format!("{:?}", status),
    })
}
```

(If the Clarabel 0.11 API surface differs — e.g. `solver.solution` field names — adjust to the version's docs; the example confirms `solver.solution.x` and `solver.solution.status`. `IPSolver` import provides `.solve()`.)

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core solves_default`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/solve.rs
git commit -m "feat: Clarabel solve and typed Trajectory extraction"
```

---

### Task 13: Independent physics validation

**Files:**
- Modify: `rust/gfold-core/src/validate.rs`

**Interfaces:**
- Consumes: `Config`, `Trajectory`, `derive`.
- Produces:
  - `struct Violation { name: String, index: usize, residual: f64 }`
  - `fn validate(cfg: &Config, traj: &Trajectory, tol: f64) -> Vec<Violation>` — checks, independent of Clarabel, returning all violations beyond `tol`:
    - boundary: `positions[0] == initial_position`, `velocities[0] == initial_velocity`, `positions[n-1] == target_position`, `velocities[n-1] == target_velocity`.
    - dynamics: for i in 0..n-1, recompute position/velocity/mass updates from `solver.py` formulas using `u`, `s`, `z` and assert equality with next step.
    - glide slope: `positions[i][2] >= sinγ*||positions[i][:3]|| - tol`.
    - velocity bound: `||velocities[i]|| <= max_vel + tol`.
    - thrust-slack: `s[i] >= ||u[i]|| - tol`.
    - dry mass: `z[n-1] >= log_dry_mass - tol`.
  - `fn assert_valid(cfg, traj, tol)` panicking with the first violation (test helper).

- [ ] **Step 1: Write failing test (default solution must be physically valid)**

```rust
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
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd rust && cargo test -p gfold-core default_solution_is_physical`
Expected: FAIL — `validate` not defined.

- [ ] **Step 3: Implement validate.rs**

```rust
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
    let dt = cfg.dt();
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
            let pos_pred = traj.positions[i][c]
                + (traj.velocities[i][c] + traj.velocities[i+1][c]) * dt / 2.0
                + (acc * dt2 + g[c] * dt2) / 2.0;
            eq("pos_update", i, traj.positions[i+1][c], pos_pred, &mut v);
            let vel_pred = traj.velocities[i][c] + acc * dt + g[c] * dt;
            eq("vel_update", i, traj.velocities[i+1][c], vel_pred, &mut v);
        }
        let z_pred = traj.z_values[i] - (traj.s_values[i] + traj.s_values[i+1]) / 2.0 * a_dt;
        eq("mass_update", i, traj.z_values[i+1], z_pred, &mut v);
    }

    // inequalities
    for i in 0..n {
        let gs = traj.positions[i][2] - sin * norm3(&traj.positions[i]);
        if gs < -tol { v.push(Violation { name: "glide_slope".into(), index: i, residual: -gs }); }
        let vb = cfg.spacecraft.max_velocity - norm3(&traj.velocities[i]);
        if vb < -tol { v.push(Violation { name: "max_velocity".into(), index: i, residual: -vb }); }
        let ts = traj.s_values[i] - norm3(&traj.u_values[i]);
        if ts < -tol { v.push(Violation { name: "thrust_slack".into(), index: i, residual: -ts }); }
    }
    let dry = traj.z_values[n-1] - cfg.log_dry_mass();
    if dry < -tol { v.push(Violation { name: "dry_mass".into(), index: n-1, residual: -dry }); }

    let _ = derive(cfg); // reserved for future thrust-bound checks vs z0 arrays
    v
}

pub fn assert_valid(cfg: &Config, traj: &Trajectory, tol: f64) {
    let v = validate(cfg, traj, tol);
    assert!(v.is_empty(), "physics violations: {:?}", v);
}
```

- [ ] **Step 4: Run tests**

Run: `cd rust && cargo test -p gfold-core default_solution_is_physical`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/gfold-core/src/validate.rs
git commit -m "feat: independent physics validation of trajectories"
```

---

### Task 14: Python fixture-dump command + committed fixtures

**Files:**
- Create: `generator/gfold/fixtures.py`
- Modify: `generator/gfold/cli.py` (add a `dump-fixtures` subcommand/flag)
- Create: `rust/gfold-fixtures/data/*.json` (generated, then committed)

**Prerequisite:** the Python env must have the package installed with cvxpy. From `generator/`: `pip install -e .`. If cvxpy is not importable, install it before this task.

**Interfaces:**
- Produces JSON files, one per named config, schema:
  ```json
  {
    "name": "default",
    "config": { "spacecraft": {...}, "environment": {...}, "solver": {...} },
    "expected": {
      "objective": <z[n-1]>,
      "final_mass": <float>,
      "positions": [[x,y,z], ...],
      "velocities": [[vx,vy,vz], ...],
      "thrusts": [<float>, ...]
    }
  }
  ```
- The `config` object must deserialize into the Rust `Config` (same field names: `spacecraft`, `environment` with `gravity`/`glide_slope_angle_deg`/`max_angle_deg`, `solver` with `n`/`time_of_flight`). Map Python config attributes to these names explicitly in `fixtures.py`.
- Configs to dump: `default` (Mars, n=100), `moon` (gravity [0,0,-1.62]), `earth` (gravity [0,0,-9.81]), `glide30` (glide_slope_angle 30, may need feasible geometry — if infeasible, skip and log), `small_n` (n=20).

- [ ] **Step 1: Write `fixtures.py`**

```python
"""Generate solver fixtures (oracle) for the Rust core tests."""
import json
import os
import numpy as np
from .config import GFoldConfig, EnvironmentConfig, SolverConfig
from .solver import GFoldSolver


def _config_json(cfg: GFoldConfig) -> dict:
    sc = cfg.spacecraft
    env = cfg.environment
    sol = cfg.solver
    return {
        "spacecraft": {
            "wet_mass": sc.wet_mass, "fuel": sc.fuel,
            "real_max_thrust": sc.real_max_thrust,
            "min_thrust_pct": sc.min_thrust_pct, "max_thrust_pct": sc.max_thrust_pct,
            "max_velocity": sc.max_velocity,
            "initial_position": list(map(float, sc.initial_position)),
            "initial_velocity": list(map(float, sc.initial_velocity)),
            "target_velocity": list(map(float, sc.target_velocity)),
            "target_position": list(map(float, sc.target_position)),
            "fuel_consumption": sc.fuel_consumption,
        },
        "environment": {
            "gravity": list(map(float, env.gravity)),
            "glide_slope_angle_deg": float(env.glide_slope_angle),
            "max_angle_deg": float(env.max_angle),
        },
        "solver": {"n": sol.n, "time_of_flight": sol.time_of_flight},
    }


def _dump_one(name: str, cfg: GFoldConfig, out_dir: str) -> None:
    solver = GFoldSolver(cfg)
    try:
        result = solver.solve()
    except Exception as e:  # infeasible / solver error
        print(f"skip {name}: {e}")
        return
    payload = {
        "name": name,
        "config": _config_json(cfg),
        "expected": {
            "objective": float(result["z_values"][-1]),
            "final_mass": float(result["final_mass"]),
            "positions": [list(map(float, p)) for p in result["positions"]],
            "velocities": [list(map(float, v)) for v in result["velocities"]],
            "thrusts": [float(t) for t in result["thrusts"]],
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {name}.json")


def dump_fixtures(out_dir: str) -> None:
    cases = {
        "default": GFoldConfig(),
        "moon": GFoldConfig(environment=EnvironmentConfig.moon()),
        "earth": GFoldConfig(environment=EnvironmentConfig.earth()),
        "small_n": GFoldConfig(solver=SolverConfig(n=20, time_of_flight=44.63)),
    }
    for name, cfg in cases.items():
        _dump_one(name, cfg, out_dir)
```

- [ ] **Step 2: Wire a CLI entry**

In `generator/gfold/cli.py`, add an argument `--dump-fixtures <dir>` that, when present, calls `from .fixtures import dump_fixtures; dump_fixtures(dir)` and returns. (Match the existing argparse style in that file; read it first and follow its pattern.)

- [ ] **Step 3: Generate the fixtures**

Run:
```bash
cd generator && pip install -e . && python -m gfold --dump-fixtures ../rust/gfold-fixtures/data
```
Expected: `wrote default.json`, `wrote moon.json`, `wrote earth.json`, `wrote small_n.json`.

- [ ] **Step 4: Sanity-check one fixture**

Run: `python -c "import json; d=json.load(open('rust/gfold-fixtures/data/default.json')); print(len(d['expected']['positions']), d['expected']['final_mass'])"`
Expected: `100 <some mass between 300 and 2000>`.

- [ ] **Step 5: Commit**

```bash
git add generator/gfold/fixtures.py generator/gfold/cli.py rust/gfold-fixtures/data/
git commit -m "feat: python fixture-dump command and committed oracle fixtures"
```

---

### Task 15: Rust fixture oracle test (leg 2)

**Files:**
- Create: `rust/gfold-core/tests/fixtures.rs`
- Modify: `rust/gfold-core/Cargo.toml` (add `[dev-dependencies]` already has serde_json via main dep; ensure `glob` not needed — list files explicitly)

**Interfaces:**
- Consumes: `gfold_core::config::Config`, `gfold_core::solve::solve`, `gfold_core::validate::validate`.
- Fixture struct mirrors the JSON: `struct Fixture { name: String, config: Config, expected: Expected }`, `struct Expected { objective: f64, final_mass: f64, positions: Vec<[f64;3]>, velocities: Vec<[f64;3]>, thrusts: Vec<f64> }` with `#[derive(serde::Deserialize)]`.

- [ ] **Step 1: Write the test that loads each fixture, solves, and compares**

```rust
use gfold_core::config::Config;
use gfold_core::solve::solve;
use gfold_core::validate::validate;
use approx::assert_relative_eq;

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
            assert_relative_eq!(traj.positions[i][c], fx.expected.positions[i][c], epsilon = 1.0);
            assert_relative_eq!(traj.velocities[i][c], fx.expected.velocities[i][c], epsilon = 0.5);
        }
    }
}

#[test]
fn fixture_default() { check("../gfold-fixtures/data/default.json"); }
#[test]
fn fixture_moon() { check("../gfold-fixtures/data/moon.json"); }
#[test]
fn fixture_earth() { check("../gfold-fixtures/data/earth.json"); }
#[test]
fn fixture_small_n() { check("../gfold-fixtures/data/small_n.json"); }
```

(Tolerances are generous because two solvers won't agree to machine precision; tighten once observed agreement is better. Position epsilon 1.0 m on a ~2.4 km descent is ~0.04%.)

- [ ] **Step 2: Run to verify (it should pass if assembly is correct; failures here mean an assembly bug)**

Run: `cd rust && cargo test -p gfold-core --test fixtures`
Expected: PASS for all four. If a test fails on `objective`/`positions`, the assembler has a bug (most likely Task 10) — debug before proceeding.

- [ ] **Step 3: If failing, add a focused diagnostic and fix**

If `fixture_default` fails: print `traj.objective` vs `expected.objective` and the first differing position index. Cross-reference the suspect block (Task 10 thrust-lower SOC is the prime suspect). Fix the block, re-run until green.

- [ ] **Step 4: Commit**

```bash
git add rust/gfold-core/tests/fixtures.rs
git commit -m "test: fixture oracle comparison against CVXPY solutions"
```

---

### Task 16: Matrix-diff cross-check vs CVXPYGen (leg 1, opt-in)

**Files:**
- Create: `generator/gfold/export_matrices.py` (dump CVXPYGen's canonical `P,q,A,b,cones` for a config to JSON)
- Create: `rust/gfold-core/tests/matrix_diff.rs` (marked `#[ignore]`, run on demand)

**Prerequisite:** cvxpy + cvxpygen installed (Task 14 prerequisite).

**Interfaces:**
- Python side: build the CVXPY problem for `default`, call `problem.get_problem_data(cp.CLARABEL)` to obtain the canonical `(P, q, A, b)` and cone dims, write to `rust/gfold-fixtures/data/matrices_default.json` with arrays in dense or COO form plus an ordered cone list `[{"type":"z","dim":...},{"type":"nn","dim":...},{"type":"soc","dim":...},...]`.
- Rust side: `assemble(&default)`, then compare to the imported matrices **up to row and column permutation**. Implement a permutation-tolerant comparison: match columns by the multiset of (cone-group-row-signature → value) and rows by content. Because exact permutation matching is involved, scope this test to verifying: (a) identical variable count and total row count, (b) identical cone-dimension multiset, (c) for each constraint expressed as `b - A x` evaluated at 20 random points, the *set* of residual vectors per cone type matches between Rust and the CVXPYGen export within 1e-9.

The random-point residual-set comparison sidesteps needing an explicit permutation: if the two formulations are equivalent, then for any point the multiset of per-cone residuals is identical.

- [ ] **Step 1: Write `export_matrices.py`**

```python
"""Export CVXPYGen canonical problem data for matrix-diff testing."""
import json
import numpy as np
import cvxpy as cp
from .config import GFoldConfig
from .solver import GFoldSolver


def export(out_path: str) -> None:
    solver = GFoldSolver(GFoldConfig())
    prob = solver.problem
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    A = data["A"]  # scipy sparse
    P = data.get("P")
    payload = {
        "n_vars": int(A.shape[1]),
        "n_rows": int(A.shape[0]),
        "A": {
            "rows": A.tocoo().row.tolist(),
            "cols": A.tocoo().col.tolist(),
            "vals": A.tocoo().data.tolist(),
            "shape": list(A.shape),
        },
        "b": np.asarray(data["b"]).ravel().tolist(),
        "q": np.asarray(data["c"]).ravel().tolist(),
        "cones": _cone_list(data),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {out_path}")


def _cone_list(data: dict) -> list:
    dims = data["dims"]
    out = []
    if getattr(dims, "zero", 0):
        out.append({"type": "z", "dim": int(dims.zero)})
    if getattr(dims, "nonneg", 0):
        out.append({"type": "nn", "dim": int(dims.nonneg)})
    for soc in getattr(dims, "soc", []):
        out.append({"type": "soc", "dim": int(soc)})
    return out
```

(Note: `get_problem_data` returns objective vector under key `"c"` for conic solvers; cone dims object attributes may differ by cvxpy version — inspect `data["dims"]` and adapt the attribute names. Verify by printing `data.keys()` and `vars(data["dims"])` first.)

- [ ] **Step 2: Generate the export**

Run: `cd generator && python -m gfold --export-matrices ../rust/gfold-fixtures/data/matrices_default.json`
(Wire `--export-matrices` in `cli.py` like `--dump-fixtures`.)
Expected: `wrote .../matrices_default.json`. Confirm `n_vars == 1100`.

- [ ] **Step 3: Write the ignored Rust matrix-diff test**

```rust
use gfold_core::config::Config;
use gfold_core::assemble::assemble;

// compares cone-dimension multiset and per-cone residual sets at random points
#[test]
#[ignore = "requires CVXPYGen export; run with --ignored"]
fn matrix_equivalence_default() {
    let export = std::fs::read_to_string("../gfold-fixtures/data/matrices_default.json")
        .expect("export present");
    let v: serde_json::Value = serde_json::from_str(&export).unwrap();
    let prob = assemble(&Config::default());

    assert_eq!(prob.layout.nvars(), v["n_vars"].as_u64().unwrap() as usize);
    assert_eq!(prob.b.len(), v["n_rows"].as_u64().unwrap() as usize);
    // cone dimension multiset equality
    // (build sorted Vec<(kind,dim)> from prob.cones and from v["cones"], assert_eq!)
    // ... see Step 4 for the residual-set comparison
}
```

- [ ] **Step 4: Implement the residual-set comparison**

Build, in the test: a deterministic set of 20 pseudo-random points (use a fixed LCG seeded constant — no `rand` dep). For each point compute Rust residual vector `b - A x` grouped by cone, and the export's `b - A x` grouped by cone (reconstruct A from COO). Sort each cone group's residual values and `assert_relative_eq!` element-wise within 1e-9. Equivalent formulations yield identical residual multisets at every point.

(Full code: implement an LCG `fn lcg(state: &mut u64) -> f64 { *state = state.wrapping_mul(6364136223846793005).wrapping_add(1); ((*state >> 33) as f64) / (1u64<<31) as f64 }`, reconstruct the export A as dense `Vec<Vec<f64>>` from COO, and compare grouped sorted residuals.)

- [ ] **Step 5: Run the ignored test**

Run: `cd rust && cargo test -p gfold-core --test matrix_diff -- --ignored`
Expected: PASS. A failure localizes an assembly mismatch by cone group (e.g. SOC group residuals differ ⇒ revisit Task 10).

- [ ] **Step 6: Commit**

```bash
git add generator/gfold/export_matrices.py generator/gfold/cli.py rust/gfold-core/tests/matrix_diff.rs rust/gfold-fixtures/data/matrices_default.json
git commit -m "test: opt-in matrix-diff cross-check vs CVXPYGen"
```

---

### Task 17: Criterion benchmarks

**Files:**
- Modify: `rust/gfold-core/Cargo.toml` (add criterion dev-dep + bench target)
- Create: `rust/gfold-core/benches/solve.rs`

**Interfaces:**
- Consumes: `gfold_core::assemble::assemble`, `gfold_core::solve::solve`, `Config`.

- [ ] **Step 1: Add criterion to Cargo.toml**

```toml
[dev-dependencies]
approx = "0.5"
criterion = "0.5"

[[bench]]
name = "solve"
harness = false
```

- [ ] **Step 2: Write the benchmark**

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use gfold_core::config::Config;
use gfold_core::assemble::assemble;
use gfold_core::solve::solve;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gfold");
    for &n in &[20usize, 50, 100, 200] {
        let mut cfg = Config::default();
        cfg.solver.n = n;
        group.bench_with_input(BenchmarkId::new("assemble", n), &cfg, |b, cfg| {
            b.iter(|| assemble(cfg));
        });
        group.bench_with_input(BenchmarkId::new("solve", n), &cfg, |b, cfg| {
            b.iter(|| solve(cfg).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 3: Run the benchmarks**

Run: `cd rust && cargo bench -p gfold-core`
Expected: criterion reports timings for `assemble`/`solve` at each n. Record the n=100 solve time.

- [ ] **Step 4: Commit**

```bash
git add rust/gfold-core/Cargo.toml rust/gfold-core/benches/solve.rs
git commit -m "bench: criterion benchmarks for assemble and solve"
```

---

## Self-Review Notes

- **Spec coverage:** workspace layout (Task 1); config + derived (Tasks 2–3); hand-derived structured assembler with index map and per-cone blocks (Tasks 4–11); Clarabel solve + Trajectory (Task 12); three correctness legs — physics validation (Task 13), solution fixtures (Tasks 14–15), matrix-diff (Task 16); benchmarks (Task 17). Python retained as oracle (Tasks 14, 16). Bindings excluded (stated in Global Constraints).
- **Clarabel API risk:** exact field/trait names (`solver.solution.x`, `SolverStatus`, `IPSolver`, `CscMatrix::new`/`::zeros`) are taken from the 0.11 example; the first task that touches them (Task 12) should verify against `cargo doc --open` for the pinned version and adapt if a name differs. This is the one external-API uncertainty; everything else is self-contained.
- **The quadratic thrust block (Task 10) is the highest-risk derivation.** It has three independent backstops: its own cone-membership unit test, the fixture oracle (Task 15), and the matrix-diff (Task 16). If Task 15 fails, Task 10 is the first suspect.
- **CVXPY dependency:** Tasks 14 and 16 require a Python env with cvxpy/cvxpygen; the prerequisite is called out in each. Tasks 1–13, 15, 17 do not need Python (15 only reads committed JSON).
