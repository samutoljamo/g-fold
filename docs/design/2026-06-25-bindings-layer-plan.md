# gfold Bindings Layer + Rust CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `gfold-core` to other languages through one serialized, language-agnostic boundary (`gfold-ffi`) with schema-generated types and a Python wrapper, plus a native `gfold-cli` Rust binary.

**Architecture:** `gfold-ffi` wraps the single pure `solve(&Config) -> Result<Trajectory, String>` as `solve_json(&str) -> String` returning a `{"ok"|"err"}` envelope, re-exported over a C ABI, a WASM binding, and a PyO3 function. Per-language types are generated from a JSON Schema emitted from the serde types. `gfold-cli` links `gfold-core` directly.

**Tech Stack:** Rust (clarabel, serde, schemars, pyo3, wasm-bindgen, clap), maturin, Python 3.11+, uv, datamodel-codegen.

## Global Constraints

- The boundary is a serialized envelope: success `{"ok": <Trajectory>}`, failure `{"err": "<message>"}`. Never panic across FFI — wrap entry points in `std::panic::catch_unwind`.
- One shared implementation (`solve_json`/`validate_json`); C-ABI, WASM, and PyO3 are thin re-exports of it.
- Per-language types are GENERATED from `schemas/gfold.schema.json` and committed; a CI check fails if they are stale.
- Rust crates are workspace members under `rust/`; language wrappers live under `bindings/<lang>/`.
- WASM is wired (the `wasm` feature compiles) but the JS package is NOT built here.
- `gfold-cli` links `gfold-core` directly (no FFI). CLI commands: `solve`, `validate`. JSON/CSV output. No visualization.
- Do not modify `generator/gfold/`. Do not break `gfold-core`'s existing API or tests.
- Work on branch `feat/bindings` (already created off `rust`).

---

## File Structure

```
rust/
  Cargo.toml                       # workspace: add gfold-ffi, gfold-cli members
  gfold-core/
    Cargo.toml                     # add optional schemars dep + `schema` feature
    src/config.rs                  # JsonSchema derive (feature-gated) on Config + sub-structs
    src/solve.rs                   # add Serialize + feature-gated JsonSchema on Trajectory
  gfold-ffi/
    Cargo.toml                     # gfold-core, serde_json; features: python, wasm, schema
    src/lib.rs                     # solve_json/validate_json + C-ABI + wasm + pyo3 exports
    src/bin/emit_schema.rs         # writes schemas/gfold.schema.json (feature `schema`)
  gfold-cli/
    Cargo.toml                     # gfold-core, clap, serde_json
    src/main.rs                    # `solve` / `validate` subcommands
bindings/
  python/
    pyproject.toml                 # maturin; module name `gfold`
    Cargo.toml                     # depends gfold-ffi (feature python)
    src/lib.rs                     # re-export the pyo3 module
    python/gfold/__init__.py       # solve() shim + GFoldError
    python/gfold/_types.py         # GENERATED dataclasses (Config, Trajectory)
    tests/test_solve.py
schemas/
  gfold.schema.json                # emitted, committed
.github/workflows/bindings.yml     # build/test ffi, cli, python, schema freshness
```

---

## Task 1: Core — serializable Trajectory + schema support

**Files:**
- Modify: `rust/gfold-core/Cargo.toml`
- Modify: `rust/gfold-core/src/solve.rs` (Trajectory derives)
- Modify: `rust/gfold-core/src/config.rs` (sub-struct + Config derives)

**Interfaces:**
- Produces: `gfold_core::solve::Trajectory: serde::Serialize`; a `schema` cargo feature on `gfold-core` that adds `schemars::JsonSchema` to `Config`, `Spacecraft`, `Environment`, `Solver`, and `Trajectory`.

- [ ] **Step 1: Write the failing test** — append to `rust/gfold-core/src/solve.rs` `mod tests`:

```rust
    #[test]
    fn trajectory_serializes_to_json() {
        let traj = solve(&Config::default()).expect("solve");
        let json = serde_json::to_string(&traj).expect("serialize");
        assert!(json.contains("\"positions\""));
        assert!(json.contains("\"final_mass\""));
    }
```

- [ ] **Step 2: Run it; verify it fails to compile**

Run: `cd rust && cargo test -p gfold-core trajectory_serializes_to_json`
Expected: compile error — `Trajectory` does not implement `Serialize`.

- [ ] **Step 3: Add `Serialize` to Trajectory** — `rust/gfold-core/src/solve.rs`, change the derive:

```rust
#[derive(Debug, Clone, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Trajectory {
```

- [ ] **Step 4: Add the `schema` feature + optional schemars** — `rust/gfold-core/Cargo.toml`:

```toml
[dependencies]
clarabel = "0.11"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = { version = "0.8", optional = true }

[features]
schema = ["dep:schemars"]
```

- [ ] **Step 5: Gate JsonSchema on the config types** — `rust/gfold-core/src/config.rs`, add to EACH of `Spacecraft`, `Environment`, `Solver`, `Config` the attr line directly under their existing `#[derive(...)]`:

```rust
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
```

- [ ] **Step 6: Run tests, both feature settings**

Run: `cd rust && cargo test -p gfold-core && cargo build -p gfold-core --features schema`
Expected: all tests pass; `--features schema` builds clean.

- [ ] **Step 7: Commit**

```bash
git add rust/gfold-core/Cargo.toml rust/gfold-core/src/solve.rs rust/gfold-core/src/config.rs
git commit -m "feat(core): serializable Trajectory + optional JsonSchema (schema feature)"
```

---

## Task 2: `gfold-ffi` — the serialized boundary

**Files:**
- Modify: `rust/Cargo.toml` (add member)
- Create: `rust/gfold-ffi/Cargo.toml`, `rust/gfold-ffi/src/lib.rs`

**Interfaces:**
- Consumes: `gfold_core::{config::Config, solve::solve, solve::Trajectory, validate::validate}`.
- Produces: `gfold_ffi::solve_json(&str) -> String` and `gfold_ffi::validate_json(&str) -> String`, each returning `{"ok": ...}` or `{"err": "..."}`. `validate_json` input is `{"config": <Config>, "trajectory": <Trajectory>, "tol": <f64>}` and returns `{"ok": <[Violation]>}`.

- [ ] **Step 1: Add the workspace member** — `rust/Cargo.toml`:

```toml
[workspace]
members = ["gfold-core", "gfold-ffi"]
resolver = "2"
```

- [ ] **Step 2: Create `rust/gfold-ffi/Cargo.toml`**

```toml
[package]
name = "gfold-ffi"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["rlib", "cdylib", "staticlib"]

[dependencies]
gfold-core = { path = "../gfold-core" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[features]
schema = ["gfold-core/schema", "dep:schemars"]

[dependencies.schemars]
version = "0.8"
optional = true
```

- [ ] **Step 3: Write the failing test** — `rust/gfold-ffi/src/lib.rs`:

```rust
//! Serialized, language-agnostic boundary over gfold-core.

use gfold_core::{config::Config, solve::solve, solve::Trajectory, validate::validate};
use serde::Deserialize;

/// Solve from a JSON `Config`; return `{"ok": Trajectory}` or `{"err": msg}`.
pub fn solve_json(input: &str) -> String {
    let result = serde_json::from_str::<Config>(input)
        .map_err(|e| format!("bad config json: {e}"))
        .and_then(|cfg| solve(&cfg));
    envelope(result)
}

#[derive(Deserialize)]
struct ValidateReq {
    config: Config,
    trajectory: Trajectory,
    tol: f64,
}

/// Validate a trajectory; input `{"config","trajectory","tol"}`.
pub fn validate_json(input: &str) -> String {
    let parsed = serde_json::from_str::<ValidateReq>(input)
        .map_err(|e| format!("bad validate json: {e}"));
    match parsed {
        Ok(req) => {
            let v = validate(&req.config, &req.trajectory, req.tol);
            serde_json::json!({ "ok": v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>() })
                .to_string()
        }
        Err(e) => serde_json::json!({ "err": e }).to_string(),
    }
}

fn envelope<T: serde::Serialize>(r: Result<T, String>) -> String {
    match r {
        Ok(v) => serde_json::json!({ "ok": v }).to_string(),
        Err(e) => serde_json::json!({ "err": e }).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_json_default_ok() {
        let cfg = serde_json::to_string(&Config::default()).unwrap();
        let resp: serde_json::Value = serde_json::from_str(&solve_json(&cfg)).unwrap();
        let traj = &resp["ok"];
        assert!(traj["positions"].is_array());
        // lands near origin
        let last = traj["positions"].as_array().unwrap().last().unwrap();
        assert!(last[0].as_f64().unwrap().abs() < 1.0);
    }

    #[test]
    fn solve_json_bad_input_err() {
        let resp: serde_json::Value = serde_json::from_str(&solve_json("not json")).unwrap();
        assert!(resp["err"].as_str().unwrap().contains("bad config"));
    }
}
```

`Trajectory` must `Deserialize` for `ValidateReq`. Add it in Task 1? It is only needed here, so add it now: in `rust/gfold-core/src/solve.rs` change the derive to `#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]` (it was set to `Serialize` in Task 1 — add `Deserialize`). If reading tasks out of order: the Trajectory derive line must read `#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]`.

- [ ] **Step 4: Run tests, verify pass**

Run: `cd rust && cargo test -p gfold-ffi`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add rust/Cargo.toml rust/gfold-ffi/Cargo.toml rust/gfold-ffi/src/lib.rs rust/gfold-core/src/solve.rs
git commit -m "feat(ffi): serialized solve_json/validate_json boundary"
```

---

## Task 3: `gfold-ffi` — C-ABI + WASM exports, panic-safe

**Files:**
- Modify: `rust/gfold-ffi/Cargo.toml` (wasm feature deps)
- Modify: `rust/gfold-ffi/src/lib.rs` (extern "C" + wasm exports)

**Interfaces:**
- Produces: `extern "C" fn gfold_solve(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8`, `extern "C" fn gfold_validate(...)` (same shape), `unsafe extern "C" fn gfold_free(ptr: *mut u8, len: usize)`; and (feature `wasm`) `wasm` `solve`/`validate` taking/returning `String`.

- [ ] **Step 1: Add wasm deps** — `rust/gfold-ffi/Cargo.toml`, extend:

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }

[features]
schema = ["gfold-core/schema", "dep:schemars"]
wasm = ["dep:wasm-bindgen"]
```

- [ ] **Step 2: Write the failing test** — append to `gfold-ffi/src/lib.rs` `mod tests`:

```rust
    #[test]
    fn c_abi_roundtrip_and_free() {
        let cfg = serde_json::to_string(&Config::default()).unwrap();
        let mut out_len: usize = 0;
        let p = gfold_solve(cfg.as_ptr(), cfg.len(), &mut out_len as *mut usize);
        assert!(!p.is_null() && out_len > 0);
        let bytes = unsafe { std::slice::from_raw_parts(p, out_len) };
        let resp: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        assert!(resp["ok"]["positions"].is_array());
        unsafe { gfold_free(p, out_len) };
    }

    #[test]
    fn panic_becomes_err() {
        // an over-long len would slice out of bounds inside the call; catch_unwind -> err
        let resp = call_catching(|| panic!("boom"));
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(v["err"].as_str().unwrap().contains("panic"));
    }
```

- [ ] **Step 3: Implement C-ABI + wasm + panic guard** — append to `gfold-ffi/src/lib.rs`:

```rust
/// Run `f`, converting a panic into an `{"err"}` envelope string.
fn call_catching(f: impl FnOnce() -> String + std::panic::UnwindSafe) -> String {
    std::panic::catch_unwind(f)
        .unwrap_or_else(|_| r#"{"err":"internal panic"}"#.to_string())
}

fn to_buf(s: String, out_len: *mut usize) -> *mut u8 {
    let mut boxed = s.into_bytes().into_boxed_slice();
    unsafe { *out_len = boxed.len(); }
    let p = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    p
}

unsafe fn read_str<'a>(ptr: *const u8, len: usize) -> &'a str {
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len)).unwrap_or("")
}

/// # Safety: `ptr`/`len` must describe valid UTF-8 JSON; free the result with `gfold_free`.
#[no_mangle]
pub unsafe extern "C" fn gfold_solve(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let input = read_str(ptr, len).to_owned();
    to_buf(call_catching(move || solve_json(&input)), out_len)
}

/// # Safety: see `gfold_solve`.
#[no_mangle]
pub unsafe extern "C" fn gfold_validate(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let input = read_str(ptr, len).to_owned();
    to_buf(call_catching(move || validate_json(&input)), out_len)
}

/// # Safety: `ptr`/`len` must come from a `gfold_*` call.
#[no_mangle]
pub unsafe extern "C" fn gfold_free(ptr: *mut u8, len: usize) {
    drop(Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)));
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    pub fn solve(input: &str) -> String { super::call_catching(|| super::solve_json(input)) }
    #[wasm_bindgen]
    pub fn validate(input: &str) -> String { super::call_catching(|| super::validate_json(input)) }
}
```

- [ ] **Step 4: Run native tests**

Run: `cd rust && cargo test -p gfold-ffi`
Expected: 4 passed.

- [ ] **Step 5: Verify the wasm target compiles**

Run: `rustup target add wasm32-unknown-unknown && cd rust && cargo build -p gfold-ffi --target wasm32-unknown-unknown --features wasm`
Expected: builds clean (no JS package produced — that is a later effort).

- [ ] **Step 6: Commit**

```bash
git add rust/gfold-ffi/Cargo.toml rust/gfold-ffi/src/lib.rs
git commit -m "feat(ffi): C-ABI + wasm exports, panic-safe"
```

---

## Task 4: `gfold-ffi` — schema emitter + committed schema

**Files:**
- Create: `rust/gfold-ffi/src/bin/emit_schema.rs`
- Create: `schemas/gfold.schema.json` (generated output, committed)

**Interfaces:**
- Produces: a binary `emit-schema` (built with `--features schema`) that prints the JSON Schema for `Config` and `Trajectory` to stdout; the committed `schemas/gfold.schema.json`.

- [ ] **Step 1: Write the emitter** — `rust/gfold-ffi/src/bin/emit_schema.rs`:

```rust
//! Emit the JSON Schema for the binding types. Run with: --features schema
//! cargo run -p gfold-ffi --features schema --bin emit-schema > ../schemas/gfold.schema.json
fn main() {
    let schema = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "gfold",
        "definitions": {
            "Config": schemars::schema_for!(gfold_core::config::Config).schema,
            "Trajectory": schemars::schema_for!(gfold_core::solve::Trajectory).schema,
        }
    });
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());
}
```

- [ ] **Step 2: Generate the committed schema**

Run: `cd rust && cargo run -p gfold-ffi --features schema --bin emit-schema > ../schemas/gfold.schema.json`
Expected: writes `schemas/gfold.schema.json` containing `Config` and `Trajectory` definitions.

- [ ] **Step 3: Sanity-check the schema**

Run: `python3 -c "import json; d=json.load(open('schemas/gfold.schema.json')); assert 'Config' in d['definitions'] and 'Trajectory' in d['definitions']; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add rust/gfold-ffi/src/bin/emit_schema.rs schemas/gfold.schema.json
git commit -m "feat(ffi): JSON Schema emitter + committed schema"
```

---

## Task 5: `gfold-ffi` — PyO3 module

**Files:**
- Modify: `rust/gfold-ffi/Cargo.toml` (python feature)
- Modify: `rust/gfold-ffi/src/lib.rs` (pyo3 module)

**Interfaces:**
- Produces: (feature `python`) a `#[pymodule] fn _gfold` exposing `solve_json(str) -> str` and `validate_json(str) -> str`.

- [ ] **Step 1: Add the python feature** — `rust/gfold-ffi/Cargo.toml`:

```toml
[dependencies.pyo3]
version = "0.23"
features = ["extension-module"]
optional = true

[features]
schema = ["gfold-core/schema", "dep:schemars"]
wasm = ["dep:wasm-bindgen"]
python = ["dep:pyo3"]
```

- [ ] **Step 2: Implement the module** — append to `gfold-ffi/src/lib.rs`:

```rust
#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    #[pyfunction]
    fn solve_json(input: &str) -> String { super::call_catching(|| super::solve_json(input)) }
    #[pyfunction]
    fn validate_json(input: &str) -> String { super::call_catching(|| super::validate_json(input)) }

    #[pymodule]
    fn _gfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(solve_json, m)?)?;
        m.add_function(wrap_pyfunction!(validate_json, m)?)?;
        Ok(())
    }
}
```

- [ ] **Step 3: Verify it builds with the python feature**

Run: `cd rust && cargo build -p gfold-ffi --features python`
Expected: builds clean (native tests unaffected: `cargo test -p gfold-ffi` still 4 passed).

- [ ] **Step 4: Commit**

```bash
git add rust/gfold-ffi/Cargo.toml rust/gfold-ffi/src/lib.rs
git commit -m "feat(ffi): PyO3 module exposing solve_json/validate_json"
```

---

## Task 6: `bindings/python` — maturin package, generated types, shim, tests

**Files:**
- Create: `bindings/python/pyproject.toml`, `bindings/python/Cargo.toml`, `bindings/python/src/lib.rs`
- Create: `bindings/python/python/gfold/__init__.py`, `bindings/python/python/gfold/_types.py` (generated)
- Create: `bindings/python/tests/test_solve.py`

**Interfaces:**
- Consumes: the `_gfold` PyO3 module (Task 5).
- Produces: Python `gfold.solve(config: Config) -> Trajectory`, `gfold.Config`, `gfold.Trajectory`, `gfold.GFoldError`.

- [ ] **Step 1: maturin project files** — `bindings/python/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "gfold"
version = "0.1.0"
requires-python = ">=3.11"

[tool.maturin]
features = ["pyo3/extension-module"]
manifest-path = "Cargo.toml"
module-name = "gfold._gfold"
python-source = "python"
```

`bindings/python/Cargo.toml`:

```toml
[package]
name = "gfold-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "_gfold"
crate-type = ["cdylib"]

[dependencies]
gfold-ffi = { path = "../../rust/gfold-ffi", features = ["python"] }
pyo3 = { version = "0.23", features = ["extension-module"] }
```

`bindings/python/src/lib.rs` — re-export the module built in `gfold-ffi`:

```rust
// The #[pymodule] lives in gfold-ffi behind the `python` feature; re-exporting
// its symbol here lets maturin build this crate as the `gfold._gfold` extension.
pub use gfold_ffi::python::*;
```

> The `python` module in `gfold-ffi` must be `pub` for this re-export. In `gfold-ffi/src/lib.rs` change `mod python` to `pub mod python` (Task 5).

- [ ] **Step 2: Generate the Python types from the schema**

Run:
```bash
cd bindings/python
uvx datamodel-codegen --input ../../schemas/gfold.schema.json --input-file-type jsonschema \
  --output python/gfold/_types.py --output-model-type dataclasses.dataclass
```
Expected: writes `python/gfold/_types.py` with `Config`, `Trajectory` dataclasses. Verify it imports: `uvx python -c "import importlib.util"` (or inspect the file has `class Config` and `class Trajectory`).

If the generated `Config` lacks `to_dict`/`from_dict`, add a thin helper in `__init__.py` instead (Step 3) using `dataclasses.asdict`.

- [ ] **Step 3: The shim** — `bindings/python/python/gfold/__init__.py`:

```python
"""Python bindings for the gfold Rust solver."""
import dataclasses
import json

from ._gfold import solve_json as _solve_json
from ._types import Config, Trajectory


class GFoldError(Exception):
    """Raised when the solver reports an error."""


def solve(config: Config) -> Trajectory:
    resp = json.loads(_solve_json(json.dumps(dataclasses.asdict(config))))
    if "err" in resp:
        raise GFoldError(resp["err"])
    return Trajectory(**resp["ok"])


__all__ = ["solve", "Config", "Trajectory", "GFoldError"]
```

> If the serde `Config` JSON is nested (`spacecraft`/`environment`/`solver`) but the generated dataclass is flat, mirror the nesting in `dataclasses.asdict` handling, or generate the nested dataclasses (datamodel-codegen produces nested models from the schema's nested objects — prefer that). The `Trajectory(**resp["ok"])` construction assumes field names match the JSON keys.

- [ ] **Step 4: Write the test** — `bindings/python/tests/test_solve.py`:

```python
import math
import pytest
import gfold


def _default_config() -> gfold.Config:
    # Construct from the same defaults as gfold-core's Config::default()
    raw = {
        "spacecraft": {"wet_mass": 2000.0, "fuel": 1700.0, "real_max_thrust": 24000.0,
                       "min_thrust_pct": 0.2, "max_thrust_pct": 0.8, "max_velocity": 1000.0,
                       "initial_position": [450.0, -330.0, 2400.0],
                       "initial_velocity": [-40.0, 10.0, -10.0],
                       "target_velocity": [0.0, 0.0, 0.0], "target_position": [0.0, 0.0, 0.0],
                       "fuel_consumption": 5e-4},
        "environment": {"gravity": [0.0, 0.0, -3.71], "glide_slope_angle_deg": 0.0, "max_angle_deg": 90.0},
        "solver": {"n": 100, "time_of_flight": 44.63},
    }
    return gfold.Config(**raw)  # adjust if dataclass is nested vs flat


def test_solve_lands_at_origin():
    traj = gfold.solve(_default_config())
    assert len(traj.positions) == 100
    assert all(abs(c) < 1.0 for c in traj.positions[-1])
    assert math.isclose(traj.final_mass, math.exp(traj.objective), rel_tol=1e-6)


def test_solver_error_raises():
    cfg = _default_config()
    cfg.solver.time_of_flight = 0.001  # infeasible
    with pytest.raises(gfold.GFoldError):
        gfold.solve(cfg)
```

- [ ] **Step 5: Build + test**

Run:
```bash
cd bindings/python
uv venv && uv pip install maturin pytest
uv run maturin develop
uv run pytest -q
```
Expected: wheel builds, `test_solve_lands_at_origin` passes, `test_solver_error_raises` raises `GFoldError`.

> If `Config(**raw)` shape mismatches the generated dataclass, adjust `_default_config` and the `solve` (de)serialization to match the generated model exactly. The contract is: `dataclasses.asdict(config)` must produce JSON that deserializes to gfold-core's `Config` (nested `spacecraft`/`environment`/`solver`).

- [ ] **Step 6: Commit**

```bash
git add bindings/python rust/gfold-ffi/src/lib.rs
git commit -m "feat(python): maturin bindings (solve), generated types, tests"
```

---

## Task 7: `gfold-cli` — native CLI on gfold-core

**Files:**
- Modify: `rust/Cargo.toml` (add member)
- Create: `rust/gfold-cli/Cargo.toml`, `rust/gfold-cli/src/main.rs`
- Create: `rust/gfold-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `gfold_core::{config::Config, solve::solve, validate::validate, solve::Trajectory}`.
- Produces: a `gfold` binary with `solve <config.json> [--out <file>] [--format json|csv]` and `validate <config.json> <trajectory.json> [--tol <f64>]`.

- [ ] **Step 1: Workspace member + Cargo.toml** — `rust/Cargo.toml` members: `["gfold-core", "gfold-ffi", "gfold-cli"]`. `rust/gfold-cli/Cargo.toml`:

```toml
[package]
name = "gfold-cli"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "gfold"
path = "src/main.rs"

[dependencies]
gfold-core = { path = "../gfold-core" }
clap = { version = "4", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
assert_cmd = "2"
```

- [ ] **Step 2: Write the failing integration test** — `rust/gfold-cli/tests/cli.rs`:

```rust
use assert_cmd::Command;
use std::io::Write;

fn default_config_json() -> String {
    serde_json::to_string(&gfold_core::config::Config::default()).unwrap()
}

#[test]
fn solve_emits_trajectory_json() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(default_config_json().as_bytes()).unwrap();
    let out = Command::cargo_bin("gfold").unwrap()
        .args(["solve", f.path().to_str().unwrap()])
        .assert().success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["positions"].is_array());
}

#[test]
fn solve_bad_config_fails() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(b"not json").unwrap();
    Command::cargo_bin("gfold").unwrap()
        .args(["solve", f.path().to_str().unwrap()])
        .assert().failure();
}
```

Add `tempfile = "3"` to `[dev-dependencies]`.

- [ ] **Step 3: Run it; verify it fails**

Run: `cd rust && cargo test -p gfold-cli`
Expected: fails to compile / no binary — `gfold` not implemented.

- [ ] **Step 4: Implement the CLI** — `rust/gfold-cli/src/main.rs`:

```rust
use clap::{Parser, Subcommand};
use gfold_core::{config::Config, solve::solve, validate::validate, solve::Trajectory};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "gfold", about = "G-FOLD powered-descent guidance")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Solve a config and print the trajectory.
    Solve {
        config: std::path::PathBuf,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        #[arg(long, default_value = "json")]
        format: String,
    },
    /// Validate a trajectory against a config.
    Validate {
        config: std::path::PathBuf,
        trajectory: std::path::PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,
    },
}

fn read<T: serde::de::DeserializeOwned>(p: &std::path::Path) -> Result<T, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("parse {}: {e}", p.display()))
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Solve { config, out, format } => {
            let cfg: Config = read(&config)?;
            let traj = solve(&cfg)?;
            let rendered = match format.as_str() {
                "json" => serde_json::to_string_pretty(&traj).unwrap(),
                "csv" => render_csv(&traj),
                other => return Err(format!("unknown format: {other}")),
            };
            match out {
                Some(p) => std::fs::write(&p, rendered).map_err(|e| format!("write: {e}"))?,
                None => println!("{rendered}"),
            }
            Ok(())
        }
        Cmd::Validate { config, trajectory, tol } => {
            let cfg: Config = read(&config)?;
            let traj: Trajectory = read(&trajectory)?;
            let v = validate(&cfg, &traj, tol);
            if v.is_empty() { println!("valid"); Ok(()) }
            else { Err(format!("{} violation(s): {:?}", v.len(), v)) }
        }
    }
}

fn render_csv(t: &Trajectory) -> String {
    let mut s = String::from("t,px,py,pz,vx,vy,vz,thrust\n");
    for i in 0..t.positions.len() {
        let p = t.positions[i];
        let v = t.velocities[i];
        s += &format!("{},{},{},{},{},{},{},{}\n",
            t.time_points[i], p[0], p[1], p[2], v[0], v[1], v[2], t.thrusts[i]);
    }
    s
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd rust && cargo test -p gfold-cli`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add rust/Cargo.toml rust/gfold-cli
git commit -m "feat(cli): gfold solve/validate binary on gfold-core"
```

---

## Task 8: CI — build/test bindings, CLI, schema freshness

**Files:**
- Create: `.github/workflows/bindings.yml`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Write the workflow** — `.github/workflows/bindings.yml`:

```yaml
name: Bindings + CLI
on:
  push:
  pull_request:

jobs:
  rust:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: rust
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
          components: clippy
      - run: cargo test -p gfold-ffi -p gfold-cli
      - run: cargo build -p gfold-ffi --target wasm32-unknown-unknown --features wasm
      - run: cargo clippy -p gfold-ffi -p gfold-cli --all-targets -- -D warnings

  schema-fresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Regenerate schema
        working-directory: rust
        run: cargo run -p gfold-ffi --features schema --bin emit-schema > ../schemas/gfold.schema.json
      - name: Fail if schema is stale
        run: git diff --exit-code schemas/gfold.schema.json

  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: astral-sh/setup-uv@v5
      - name: Build + test the wheel
        working-directory: bindings/python
        run: |
          uv venv
          uv pip install maturin pytest
          uv run maturin develop
          uv run pytest -q
```

> The `schema-fresh` byte-diff assumes deterministic schema output (it is — `schemars` emits a stable structure); unlike solver fixtures, the schema has no float noise.

- [ ] **Step 2: Validate workflow steps locally (dry equivalents)**

Run:
```bash
cd rust && cargo test -p gfold-ffi -p gfold-cli && cargo clippy -p gfold-ffi -p gfold-cli --all-targets -- -D warnings
cargo run -p gfold-ffi --features schema --bin emit-schema > ../schemas/gfold.schema.json
cd .. && git diff --exit-code schemas/gfold.schema.json && echo "schema fresh"
```
Expected: tests + clippy pass; schema diff clean.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bindings.yml
git commit -m "ci: build/test ffi + cli + python bindings + schema freshness"
```

---

## Self-Review

**1. Spec coverage:**
- Serialized boundary (`solve_json`/`validate_json`, `{ok|err}`, catch_unwind) → Tasks 2, 3. ✓
- C-ABI + WASM + PyO3 exports of one impl → Tasks 3, 5. ✓
- Schema emitted from serde types; per-language types generated + committed → Tasks 1, 4, 6. ✓
- Python wrapper (solver), maturin, thin shim → Task 6. ✓
- `gfold-cli` (solve/validate, JSON/CSV, no viz) → Task 7. ✓
- `bindings/<lang>/` layout, crates in `rust/` workspace → Tasks 2, 6, 7. ✓
- WASM wired (feature compiles) but JS not built → Task 3 Step 5; CI builds the target. ✓
- Schema freshness CI check → Task 8. ✓
- Don't touch generator/, don't break gfold-core → only additive core change (Task 1) with tests. ✓

**2. Placeholder scan:** No TBD/TODO. The "adjust if nested vs flat" notes in Task 6 are explicit fallback instructions with the exact contract (`asdict(config)` → nested serde JSON), not vague placeholders — the generator output shape is the one thing that can't be known until run, so the contract is pinned instead.

**3. Type consistency:** `solve_json(&str)->String`, `validate_json(&str)->String`, `gfold_solve/gfold_validate/gfold_free` C-ABI, `_gfold` pymodule, `gfold.solve(Config)->Trajectory`, CLI `solve`/`validate` — names match across Tasks 2–8. `Trajectory` gains `Serialize` (Task 1) + `Deserialize` (Task 2, needed by `validate_json` and the CLI's trajectory read). `pub mod python` required by Task 6's re-export, noted in Tasks 5/6.
