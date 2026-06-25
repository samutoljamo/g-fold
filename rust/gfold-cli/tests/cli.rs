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
