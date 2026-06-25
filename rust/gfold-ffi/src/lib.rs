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

    #[test]
    fn c_abi_roundtrip_and_free() {
        let cfg = serde_json::to_string(&Config::default()).unwrap();
        let mut out_len: usize = 0;
        let p = unsafe { gfold_solve(cfg.as_ptr(), cfg.len(), &mut out_len as *mut usize) };
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
}

/// Run `f`, converting a panic into an `{"err"}` envelope string.
pub fn call_catching(f: impl FnOnce() -> String + std::panic::UnwindSafe) -> String {
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

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    #[pyfunction]
    fn solve_json(input: &str) -> String { super::call_catching(|| super::solve_json(input)) }
    #[pyfunction]
    fn validate_json(input: &str) -> String { super::call_catching(|| super::validate_json(input)) }

    #[pymodule]
    pub fn _gfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(solve_json, m)?)?;
        m.add_function(wrap_pyfunction!(validate_json, m)?)?;
        Ok(())
    }
}
