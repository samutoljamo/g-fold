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
