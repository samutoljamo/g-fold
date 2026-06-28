use std::process::Command;

#[test]
fn init_emits_default_config_json() {
    let out = Command::new(env!("CARGO_BIN_EXE_gfold"))
        .arg("init")
        .output()
        .expect("run gfold init");
    assert!(out.status.success());
    let cfg: gfold_core::config::Config = serde_json::from_slice(&out.stdout).expect("parse stdout");
    assert_eq!(cfg.spacecraft.wet_mass, 2000.0);
    assert!(cfg.solver.time_of_flight.is_none());
}
