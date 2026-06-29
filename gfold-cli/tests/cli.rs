use assert_cmd::Command;
use std::io::Write;

fn default_config_json() -> String {
    serde_json::to_string(&gfold_core::config::Config::default()).unwrap()
}

fn write_default_config() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(default_config_json().as_bytes()).unwrap();
    f
}

#[test]
fn solve_emits_trajectory_json() {
    let f = write_default_config();
    let out = Command::cargo_bin("gfold").unwrap()
        .args(["solve", f.path().to_str().unwrap()])
        .assert().success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["positions"].is_array());
}

#[test]
fn solve_plot_writes_png() {
    let cfg = write_default_config();
    let png = tempfile::NamedTempFile::new().unwrap();
    let png_path = png.path().with_extension("png");

    Command::cargo_bin("gfold").unwrap()
        .args(["solve", cfg.path().to_str().unwrap(),
               "--plot", png_path.to_str().unwrap()])
        .assert().success();

    // File must exist and be non-empty with a valid PNG magic header.
    let bytes = std::fs::read(&png_path).expect("PNG file should exist");
    assert!(!bytes.is_empty(), "PNG file should not be empty");
    assert_eq!(&bytes[..4], b"\x89PNG", "file should start with PNG magic bytes");
}

#[test]
fn solve_bad_config_fails() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(b"not json").unwrap();
    Command::cargo_bin("gfold").unwrap()
        .args(["solve", f.path().to_str().unwrap()])
        .assert().failure();
}
