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
