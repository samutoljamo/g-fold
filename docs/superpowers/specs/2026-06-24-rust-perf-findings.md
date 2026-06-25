# rust-perf: Solve-Performance Profiling Findings

**Date:** 2026-06-24
**Branch:** `rust-perf` (from `rust` @ 1167fe5)
**Goal:** Decide how (and whether) to speed up the G-FOLD solve via a custom structured solver, before building anything.

## TL;DR

- The solve dominates; assembly is negligible (<1% even at n=800).
- **Clarabel is already O(n) per iteration** on our block-banded KKT — per-iteration cost ÷ n is ~constant (~0.0052 ms/timestep) from n=20 to n=800. A structured (Riccati/block-tridiagonal) solver is *also* O(n), so there is **no asymptotic win available** — only a constant-factor one.
- **84% of each iteration is `kkt update` (43%) + `kkt solve` (41%)**, and both route through Clarabel's pluggable `DirectLDLSolver` trait. So a custom structured LDL could in principle target that 84% while leaving Clarabel's IPM and cone math (the proven-correct part) untouched.
- **But the constant-factor ceiling looks low:** swapping Clarabel's LDL backend from QDLDL to `faer` (a heavily optimized sparse solver) changed total solve time by **1–3%**. Two mature sparse backends are effectively tied, i.e. the sparse-LDL constant factor on this KKT is near its floor.
- **Recommendation:** A custom structured solver is **not** worth a from-scratch IPM, and likely not worth a Clarabel fork either, unless a quick prototype shows the dense-block factorization beats QDLDL/faer by a wide margin. The honest expected payoff is ~1.5–2.5× on the solve, not 10×. Cheaper wins (warm-starting across re-solves, reducing iteration count, fewer timesteps) should be weighed first.

## Method

All measurements on this machine, `--release`, default Mars config, n swept. Two harnesses (kept in `rust/gfold-core/examples/profile.rs`; the backend comparison was a one-off, numbers captured below):
1. `profile.rs` — solves a fresh problem per n, reports assemble time, iteration count, total solve, per-iteration, and Clarabel's internal section timers (Clarabel instruments `kkt update`, `kkt solve`, `scale cones`, etc. — always on).
2. backend comparison — same problems solved with `direct_solve_method = "qdldl"` vs `"faer"`.

## Data

### Scaling across n

| n | vars | rows | iters | assemble | solve | per-iter | per-iter ÷ n |
|----:|----:|----:|----:|----:|----:|----:|----:|
| 20 | 220 | 407 | 17 | 0.020 ms | 2.19 ms | 0.129 ms | 0.00645 |
| 50 | 550 | 1007 | 20 | 0.052 ms | 5.42 ms | 0.271 ms | 0.00542 |
| 100 | 1100 | 2007 | 21 | 0.103 ms | 11.32 ms | 0.539 ms | 0.00539 |
| 200 | 2200 | 4007 | 23 | 0.207 ms | 23.96 ms | 1.042 ms | 0.00521 |
| 400 | 4400 | 8007 | 25 | 0.428 ms | 51.74 ms | 2.070 ms | 0.00518 |
| 800 | 8800 | 16007 | 32 | 0.926 ms | 130.57 ms | 4.080 ms | 0.00510 |

Per-iteration cost is linear in n (the last column is flat). Iteration count grows slowly (17→32), so total is mildly superlinear — driven by iteration growth, not per-iteration complexity.

### Where each iteration goes (n=100, 21 iterations, 10.64 ms solve)

```
solve : 10.64 ms
  default start : 0.41 ms
  IP iteration  : 10.23 ms
      scale cones : 0.24 ms   (2.3%)   <- NT scaling; keep
      kkt solve   : 4.35 ms   (41%)    <- backsolves + iterative refinement (2 solves/iter)
      kkt update  : 4.58 ms   (43%)    <- write scaling values into KKT + refactor
setup : 0.67 ms   (one-time: equilibration 0.20, kktinit 0.44)
```

`kkt update` and `kkt solve` both delegate to the `DirectLDLSolver` trait: `update`→`update_values`/`scale_values`+`refactor` (factorization lives here), `solve`→`solve` (triangular solves + iterative refinement). The remaining ~10% of the iteration is residuals, step-length search, and centering.

### Backend swap (QDLDL vs faer), per-solve

| n | qdldl | faer | speedup |
|----:|----:|----:|----:|
| 100 | 11.62 ms | 11.35 ms | 1.02× |
| 400 | 53.03 ms | 52.63 ms | 1.01× |
| 800 | 132.29 ms | 128.01 ms | 1.03× |

## Interpretation

- The trajectory KKT is block-tridiagonal (dynamics couple only adjacent timesteps; P=0; cone scalings are block-diagonal). Clarabel's AMD fill-reducing ordering already recovers near-optimal O(n) factorization on this structure — confirmed by the flat per-iter÷n column. A hand-rolled Riccati factorization would match that order, not beat it.
- The constant factor is the only lever, and the QDLDL≈faer tie shows the sparse approach is near its floor. A custom **dense small-block** factorization (per-timestep blocks are fixed-size — 11 primal vars/step plus fixed cone blocks, independent of n) could still beat sparse by avoiding indirection and using cache-friendly dense kernels. Literature for structured trajectory-opt solvers vs general sparse typically reports ~2–5× on the linear algebra — which here, applied to 84% of the solve, would be roughly **1.6–2.5× overall**.
- `kkt update` being as expensive as `kkt solve` is notable: ~half the "linear algebra" cost is *writing the updated scaling values into the sparse KKT and refactoring*, not the solve. A structured solver that keeps the KKT in block form would cut the update bookkeeping too — both are behind the trait, so both are addressable.

## Options, re-scored against the data

| Option | Expected solve speedup | Effort | Correctness risk |
|---|---|---|---|
| Vendor Clarabel + custom structured `DirectLDLSolver` | ~1.6–2.5× (if dense-block beats sparse) | Medium (one trait + fork maintenance) | Low (keeps IPM/cones) |
| Standalone structured SOCP IPM | similar ceiling, maybe a bit more | High (NT scaling from scratch) | High (re-opens what we proved) |
| Stay on Clarabel, cut iterations / warm-start | bounded (<1.5×; IPMs warm-start poorly) | Low | None |
| Fixed (const-generic N) variant | constant-factor on top (stack alloc, unroll); block sizes are already compile-time fixed | Low–Medium | Low |

On the **fixed vs dynamic** idea: the per-timestep block dimensions are compile-time constants *regardless of N* (state 6, control 3, slack 1, log-mass 1, plus fixed cone blocks). So the structured factorization already operates on fixed-size dense blocks in a length-n loop. A const-generic fixed-N variant adds stack allocation, no heap, and loop unrolling — a real but secondary constant-factor gain. The dynamic variant is the same code with runtime n and heap buffers.

## Recommendation

1. **Do not** build a from-scratch IPM — the ceiling doesn't justify the correctness risk.
2. **Gate the structured-solver effort on a small prototype** (the one unmeasured unknown): implement a block-tridiagonal LDL/Riccati factorization of a *representative* KKT (correct sparsity + a plausible SPD cone-scaling block) and benchmark factor+solve against QDLDL/faer on the identical matrix.
   - **Go threshold:** ≥3× on factor+solve in isolation (→ ~2× overall after the untouched 16%). Then pursue the vendor-Clarabel `DirectLDLSolver` path, fixed and dynamic variants.
   - **No-go:** <2× in isolation → not worth the fork; revisit warm-starting and iteration-count reduction instead.
3. Keep `examples/profile.rs` as the regression/benchmark harness for whatever path is chosen.

## Artifacts

- `rust/gfold-core/examples/profile.rs` — n-sweep profiler with Clarabel section timers.
- Correctness baseline unchanged: 21 tests green on `rust-perf`.

---

## Experiment 1 — Clarabel settings (branch `perf/settings`, merged)

Swept tolerances, equilibration, presolve, and iterative refinement against time + accuracy vs the oracle.

**Finding:** iterative refinement is the only meaningful settings lever. Default targets reltol 1e-13 with up to 10 passes — overkill here and a large share of "kkt solve".

| reltol | max pos dev (default n=100) | solve | speedup |
|---:|---:|---:|---:|
| 1e-13 (default) | 0.039 m | 11.45 ms | — |
| 1e-12 | 0.137 m | 10.85 ms | ~5% |
| 5e-12 | 0.386 m | 10.41 ms | ~9% |
| 1e-11 | 0.804 m | 10.49 ms | ~8% |
| 1e-10 | 1.971 m | 9.88 ms | ~14% |

Caveats from the evidence:
- The cost optimum is flat: looser refinement gives a near-identical-*cost* trajectory that sits a small distance away in path space. Drift is **config-dependent** — the glide-slope fixture drifts ~1 m already at 5e-12, while default is at 0.39 m.
- Loosening to 1e-10 reaches ~14% but (a) breaks the 1.0 m oracle fixture gate and (b) **destabilizes IP iteration count at large n** (n=800: 32→58–73 iters, net slower, final mass drifts 1801→1794). Not safe.

**Shipped:** `iterative_refinement_reltol = 1e-12` in `solve()` — every committed fixture stays within ~0.14 m, ~5% solve speedup, no iteration-count regression. Conservative on purpose: respects the existing oracle contract rather than relaxing it.

**Note for later:** the binding limit here is the oracle *position* tolerance, not optimality or feasibility — a flat-cost problem has many ~equal-cost trajectories. If we ever decide position-to-1m-vs-one-oracle is too strict a gate (objective-match + physics-validation being the true correctness criteria), the larger ~14% settings win becomes available. Not pursued unilaterally.
