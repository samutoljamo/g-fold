# gfold bindings layer + Rust CLI — design

**Date:** 2026-06-25
**Branch:** `feat/bindings`
**Status:** design (approved, pending spec review)

## Motivation

`gfold-core` (Rust, direct Clarabel) is the production solver. We want to expose
it to many languages — Python first, WebAssembly next, others later — and to
provide a native CLI, so the legacy Python package (`generator/gfold`) can
eventually be retired. The priority is an architecture where **adding a language
is cheap and uniform**, not a pile of bespoke per-language wrappers.

## Goals

- A single, language-agnostic boundary over `gfold-core` that every language
  binds to with a thin shim.
- Python binding (solver) as the first instantiation; WASM as the documented,
  worked second example proving the pattern extends.
- A native `gfold-cli` Rust binary that uses `gfold-core` directly.
- Adding a new language ≈ generate types from a schema + a ~30-line shim.

## Non-goals

- Per-language idiomatic hand-written bindings (PyO3 classes, wasm-bindgen
  structs per type). Rejected: best ergonomics but bespoke machinery and
  recurring maintenance per language × API change.
- UniFFI. Rejected: excludes WASM (a primary target).
- Visualization. The CLI emits data; plotting is a separate effort.
- `cvxpygen` C-codegen. Obsolete — the Rust core *is* the solver.
- Building the WASM/JS package in this plan (it is documented and is the next
  effort; only the `wasm` build feature is wired up).

## Strategy (decided)

**Serialized core + generated types.** The core's public surface is one pure
function — `solve(&Config) -> Result<Trajectory, String>` — with `Config` and
`Trajectory` both serde-serializable. So the boundary is a serialized
request/response, exported via a C ABI and WASM; per-language *types* are
generated from the serde schema; each language adds a thin idiomatic shim. This
is the cheapest path to many languages **including WASM**, while keeping typed
ergonomics for end users.

## Architecture & layout

```
rust/
  gfold-core/      # solver lib (exists)
  gfold-ffi/       # the serialized boundary: solve_json + C-ABI + wasm + pyo3 exports
  gfold-cli/       # Rust binary on gfold-core (replaces cli.py)
  gfold-fixtures/  # CVXPY oracle (exists)
bindings/
  python/          # maturin wheel: gfold-ffi pyo3 ext + thin shim + generated types
  # js/  (later)   # wasm-pack package + generated .d.ts — same recipe, next effort
schemas/
  gfold.schema.json   # emitted from gfold-core's serde types; source for generated types
```

Rust crates live in the existing `rust/` workspace; language wrappers live under
`bindings/<lang>/`. `gfold-ffi` and `gfold-cli` are new workspace members;
`gfold-fixtures` remains a non-Cargo dir.

### Component: `gfold-ffi` (the boundary)

One implementation, three exports:

```rust
// the single hand-written boundary, shared by every language
pub fn solve_json(input: &str) -> String {
    let out = serde_json::from_str::<Config>(input)
        .map_err(|e| format!("bad config: {e}"))
        .and_then(|cfg| solve(&cfg));
    match out {
        Ok(traj) => serde_json::json!({ "ok": traj }).to_string(),
        Err(e)   => serde_json::json!({ "err": e }).to_string(),
    }
}
// validate_json(&str) -> String  — same envelope shape
```

- **C ABI** (native languages): `gfold_solve(ptr, len, *out_len) -> *u8` and
  `gfold_free(ptr, len)`. The caller passes UTF-8 JSON bytes and frees the
  returned buffer.
- **WASM** (`feature = "wasm"`): `#[wasm_bindgen] pub fn solve(input: &str) -> String`.
- **Python** (`feature = "python"`): `#[pyfunction] solve_json`, so maturin
  bundles a wheel.

**Error model.** Every failure — malformed JSON, infeasible problem, solver
error — is returned as `{"err": message}`. The FFI entry points wrap the call in
`std::panic::catch_unwind` so a Rust panic becomes an `{"err"}` rather than
unwinding across the FFI boundary.

### Component: types & schema (the "generated" part)

- Derive `schemars::JsonSchema` on `Config` and `Trajectory` in `gfold-core`,
  behind a `schema` feature (keeps the default build lean).
- A small binary target in `gfold-ffi` gated by the `schema` feature
  (`cargo run -p gfold-ffi --features schema --bin emit-schema`) writes
  `schemas/gfold.schema.json`.
- Per-language types are generated from that schema with a standard generator
  (e.g. `datamodel-codegen` for Python dataclasses, `quicktype` for TS) and
  **committed** under each wrapper. They change only when `Config`/`Trajectory`
  change, so a `make schema` script regenerates them and a CI check fails if the
  committed types/schema are stale.
- The hand-written `gfold_oracle/config.py` is a ready cross-check for the
  generated Python `Config`.

### Component: `bindings/python` (first instantiation)

- Built with maturin; bundles the `gfold-ffi` Python ext (the `solve_json`
  passthrough) into a wheel.
- Thin shim:

```python
def solve(config: Config) -> Trajectory:
    resp = json.loads(_native.solve_json(json.dumps(config.to_dict())))
    if "err" in resp:
        raise GFoldError(resp["err"])
    return Trajectory.from_dict(resp["ok"])
```

- `Config`/`Trajectory` are the generated dataclasses. End-user API:
  `gfold.solve(cfg) -> traj`, `traj.positions[-1]`, etc.

### Component: `gfold-cli` (Rust binary)

- Links `gfold-core` directly (no FFI). Commands:
  - `gfold solve <config.json> [--out <file>] [--format json|csv]` — read config,
    solve, emit the trajectory; non-zero exit + stderr message on infeasible/error.
  - `gfold validate <config.json> <trajectory.json>` — run the physics validator.
- Replaces `generator/gfold/cli.py`. Visualization is out of scope.

## Adding a language (the payoff)

Documented recipe, with WASM/JS as the worked example:
1. Build the `wasm` feature (`wasm-pack build`), or link the C ABI for a native
   language.
2. Generate the language's types from `schemas/gfold.schema.json`.
3. Write a ~30-line shim: serialize config → call the one entry point →
   deserialize the `{ok|err}` envelope → idiomatic result/error.

## Testing

- **`gfold-ffi`:** round-trip unit test (default config JSON → `ok` envelope →
  parses to a Trajectory landing at the origin); a test that a forced panic
  returns `{"err"}` instead of aborting.
- **`bindings/python`:** pytest smoke test (`solve(default)` lands at origin,
  raises `GFoldError` on an infeasible config); a test that the generated types
  match the schema.
- **`gfold-cli`:** integration test — run the built binary on a config fixture,
  assert the emitted trajectory and exit code.
- **Schema freshness:** CI regenerates the schema and generated types and fails
  if they differ from committed.

## Decisions to flag

- **Generated types are committed** (unlike the oracle fixtures): they change
  only with the type definitions, so committing them is reviewable and avoids a
  build-time codegen dependency; a CI freshness check guards staleness.
- **WASM is documented, not built** in this plan — it is the immediate next
  effort and the `wasm` feature is wired so it is a small follow-up.

## Retiring `generator/gfold`

Once `bindings/python` (solver) and `gfold-cli` cover the needed surface,
`generator/gfold` is removed in a separate cleanup. The visualization decision
(Rust plotting in the CLI vs. data-out + external plotting) rides with that.
