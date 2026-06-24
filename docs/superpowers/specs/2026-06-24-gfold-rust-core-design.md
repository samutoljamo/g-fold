# G-FOLD Rust Core — Design

**Date:** 2026-06-24
**Status:** Approved for planning
**Scope:** Rust core that builds and solves the basic G-FOLD SOCP directly against Clarabel (Rust crate), replacing the CVXPY → CVXPYGen path for the core solve. Bindings and extended formulations are out of scope for this spec.

## Motivation

The current implementation describes the problem in CVXPY (`generator/gfold/solver.py`) and uses CVXPYGen to emit C/C++ for downstream bindings. This path is awkward (Python model → C++ generation → bindings on top) and leaves performance on the table — CVXPY cannot exploit the known block/banded structure of the problem the way we can by hand.

This rewrite builds the conic problem (`P, q, A, b`, cones) directly in Rust from the known structure, solves with the Clarabel Rust crate, and keeps strong correctness guarantees by testing against the existing Python implementation as an oracle plus independent physics checks.

**Philosophy:** make it correct first, optimize later. The hand-derived *structure* is the target, but the first assembler may build via triplets → CSC for readability; structural fill optimization comes after correctness is established.

## Goals

- A clean, fast Rust crate (`gfold-core`) that takes a config and returns a landing trajectory.
- Correctness backed by three independent legs (see Correctness Strategy).
- Performance comparison machinery (benchmarks).
- Seams left in place (serde config, typed trajectory) for later language bindings.

## Non-Goals (this spec)

- Language bindings (PyO3 / C ABI) — a later spec; do not build now.
- Time-of-flight search loop, minimum-landing-error (Blackmore Problem 3), planetary-pit variant — future formulations.
- Micro-optimization of matrix assembly — correctness first.
- Removing or changing the Python package beyond adding a fixture-dump command. The Python package is retained as the test oracle.

## Reference: the problem (from `solver.py`)

Decision variables (per `n` timesteps):
- `x` ∈ ℝ^(n×6): position (3) + velocity (3)
- `u` ∈ ℝ^(n×3): thrust acceleration
- `s` ∈ ℝ^n: slack, `s ≥ |u|`
- `z` ∈ ℝ^n: `ln(mass)`

Objective: **maximize `z[n-1]`** (final log-mass) → linear.

Constraints (see `solver.py` lines 122–148):
- Boundary: `x[0]` = initial pos/vel, `z[0] = log_mass`, `x[n-1]` = target pos/vel, `z[n-1] ≥ log_dry_mass`.
- Per step: `‖x[i,3:]‖ ≤ max_vel`; glide slope `x[i,2] ≥ ‖x[i,:3]‖·sinγ`; `s[i] ≥ ‖u[i]‖`; thrust lower bound `1-(z-z0)+(z-z0)²/2 ≤ s·min_exp`; thrust upper bound `s·max_exp ≤ 1-(z-z0)`.
- Dynamics (i = 0..n-2): trapezoidal position update, velocity update, mass update — all equalities.

Precomputed per-step parameter arrays (from config): `z0`, `exp_z0`, `max_exp`, `min_exp`.

## Architecture

Cargo workspace; Python package retained.

```
g-fold/
  generator/              # existing Python — KEPT. Becomes oracle + fixture generator.
  rust/
    gfold-core/           # the product
    gfold-fixtures/       # test-only: load committed JSON fixtures, compare
    benches/ (or xtask)   # criterion benchmarks
  docs/superpowers/specs/ # this spec
```

### `gfold-core` modules

Each module has one job and a clean interface.

- **`config.rs`** — plain structs mirroring `GFoldConfig`: `SpacecraftConfig`, `EnvironmentConfig`, `SolverConfig`, `GFoldConfig`. Serde-derived (shared by fixtures and future bindings). Derived quantities (`log_wet_mass`, `log_dry_mass`, `min_thrust`, `max_thrust`, `sin_glide_slope`, `cos_max_angle`) as pure functions/methods.
- **`derive.rs`** (may live in `config.rs` if small) — computes per-step arrays `z0`, `exp_z0`, `max_exp`, `min_exp` from config, matching the Python loop (`solver.py` lines 73–84). Pure function returning a `Derived` struct.
- **`assemble.rs`** — the hand-derived structured assembler. Produces Clarabel inputs: `P` (zero), `q`, `A` (CSC), `b`, and the ordered cone list, from the known structure. Contains the single source of truth for the variable index mapping.
- **`solve.rs`** — drives Clarabel; maps the raw primal solution vector back into a typed `Trajectory`.
- **`validate.rs`** — independent physics/optimality checks; usable in tests and as `Trajectory::check()`.

### Decision vector ordering

Fixed layout, length `11n`: `x` (6n) ‖ `u` (3n) ‖ `s` (n) ‖ `z` (n).

Column index from `(block, step, component)` is computed from const offsets in one place (`assemble.rs`), with exhaustive unit tests. Everything depends on this mapping, so it is isolated and tested first.

### Cone composition

| Source | Conic form |
|---|---|
| boundary equalities, dynamics updates (pos/vel/mass) | ZeroCone |
| `‖x[i,3:]‖ ≤ max_vel` | SecondOrderCone dim 4, ×n |
| `s[i] ≥ ‖u[i]‖` | SecondOrderCone dim 4, ×n |
| glide slope `x[i,2] ≥ ‖x[i,:3]‖·sinγ` | SOC dim 4 ×n when `sinγ>0`; degenerates to `x[i,2] ≥ 0` (Nonnegative) when `sinγ=0` (the default). Assembler chooses per-config. |
| `s·max_exp ≤ 1-(z-z0)`, `z[n-1] ≥ log_dry_mass` | Nonnegative |
| `1-(z-z0)+(z-z0)²/2 ≤ s·min_exp` (quadratic thrust lower bound) | SecondOrderCone (rotated-cone derivation), ×n |

The quadratic thrust lower bound is the only analytically subtle block and the highest-risk part of the hand assembler. Its SOC form must be derived carefully and cross-checked (see Correctness Strategy, leg 1).

## Data flow

`Config` → `derive()` → `assemble()` → Clarabel solve → `Trajectory` → optional `validate()`.

`Trajectory` (typed): `positions` (n×3), `velocities` (n×3), `thrusts` (n, mass-adjusted as in `solver.py` lines 179–183), `z_values`, `objective`, `time_points`, plus raw `x/u/s` for validation.

## Correctness Strategy

> **Update (implementation):** The originally planned leg 1 — a direct matrix-level diff of the assembled `(P, q, A, b, cones)` against CVXPY's canonical form — proved **not viable** and was dropped. CVXPY canonicalizes the problem into a *lifted* form with auxiliary epigraph variables (1500 vars / 400 SOC blocks vs the Rust direct form's 1100 vars / 300 SOC blocks), and it represents the angle-0 glide-slope constraint as a parametric SOC where the Rust assembler emits a nonnegative row. The two formulations are equivalent at the solution but are not the same matrices, so a residual/permutation comparison cannot match. Notably, the Zero cone (706/706 rows) and Nonnegative cone (501/501 rows) *do* match exactly, confirming the equalities and linear bounds are identical. Leg 1's intent — an independent structural check — is folded into the oracle leg (leg 2), which compares full solutions against CVXPY. The remaining two legs below are what the implementation relies on.

1. **Committed solution fixtures vs CVXPY oracle.** End-to-end: same config → Rust solution agrees with the Python/CVXPY solution within tolerance. (The default-config final mass matches CVXPY to the kilogram: 1801.05 kg.)
2. **Independent physics/optimality checks (`validate.rs`).** Verify the returned trajectory satisfies the actual G-FOLD constraints (dynamics residuals, glide slope, thrust bounds, velocity bound, boundary conditions) and basic optimality, independent of any reference implementation.

### Fixtures

- Location: `rust/gfold-fixtures/data/*.json`.
- Each fixture: `{ config, expected: { objective, z_final, positions, velocities, thrusts } }`.
- Generated offline by a new command in the Python package (e.g. `gfold dump-fixtures`). Committed to the repo. No Python required in CI. Regenerated when the formulation changes.
- Comparison: tolerance-based — positions/velocities tight (~1e-4 relative), objective tight. Solver-noise-tolerant.
- Config coverage: default Mars; Moon and Earth gravity; nonzero glide slope (exercises the SOC branch vs the Nonnegative degenerate branch); small `n` (fast); large `n` (default 100).

## Testing tiers

1. **Unit** — index mapping; each constraint block assembled in isolation.
2. **Fixture/oracle** — end-to-end solution comparison vs CVXPY (leg 1).
3. **Physics validation** — `validate.rs` run on every solved trajectory (leg 2).

## Benchmarks

- Criterion benches on `assemble` and `solve` separately, across a range of `n`.
- Optional Python wall-clock comparison reported in bench output (not part of CI).

## Milestones

1. `config.rs` + `derive.rs` + index mapping, with unit tests.
2. `assemble.rs` producing `(P, q, A, b, cones)` + matrix-diff cross-check vs CVXPYGen.
3. `solve.rs` + `Trajectory` extraction + fixture/oracle tests passing.
4. `validate.rs` physics checks.
5. Criterion benches (+ optional Python comparison).

## Open questions / risks

- Exact rotated-SOC derivation of the quadratic thrust lower bound — mitigated by the matrix-diff cross-check.
- CVXPYGen output permutation: matrix-diff needs a permutation-aware comparison (match rows/cols by content), not a raw equality.
- Clarabel Rust crate version pinning and settings parity with the Python solve (tolerances) so fixture comparison is fair.
