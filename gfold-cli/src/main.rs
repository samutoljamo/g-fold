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
        /// Write a 2x2 trajectory overview PNG to this path.
        #[arg(long)]
        plot: Option<std::path::PathBuf>,
    },
    /// Validate a trajectory against a config.
    Validate {
        config: std::path::PathBuf,
        trajectory: std::path::PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,
    },
    /// Write a fully-populated default config (JSON) to edit.
    Init {
        /// Output path; omit to print to stdout.
        #[arg(long, short)]
        out: Option<std::path::PathBuf>,
    },
}

fn read<T: serde::de::DeserializeOwned>(p: &std::path::Path) -> Result<T, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("parse {}: {e}", p.display()))
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Solve { config, out, format, plot } => {
            let cfg: Config = read(&config)?;
            let traj = solve(&cfg)?;
            if let Some(ref p) = plot {
                render_plot(&traj, p)?;
            }
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
        Cmd::Init { out } => {
            let json = serde_json::to_string_pretty(&Config::default()).unwrap();
            match out {
                Some(p) => std::fs::write(&p, json).map_err(|e| format!("write: {e}"))?,
                None => println!("{json}"),
            }
            Ok(())
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

/// Render a 2x2 PNG overview of the solved trajectory.
///
/// Panels (mirroring `generator/gfold/visualization.py`):
///   top-left    — side view: downrange (x) vs altitude (z)
///   top-right   — ground track: x vs y
///   bottom-left — velocity profile: total speed & vertical speed vs time
///   bottom-right — thrust profile: normalized thrust % vs time
fn render_plot(traj: &Trajectory, path: &std::path::Path) -> Result<(), String> {
    use plotters::prelude::*;

    // Register a bundled font so the bitmap backend works without system font libs.
    static FONT_BYTES: &[u8] = include_bytes!("../assets/Cantarell-Regular.otf");
    plotters::style::register_font("sans-serif", plotters::style::FontStyle::Normal, FONT_BYTES)
        .map_err(|_| "failed to register bundled font".to_string())?;

    let root = BitMapBackend::new(path, (1200, 1000))
        .into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("plot fill: {e}"))?;

    let areas = root.split_evenly((2, 2));

    // ── helpers ──────────────────────────────────────────────────────────────
    let n = traj.positions.len();
    if n == 0 {
        return Err("trajectory is empty".into());
    }

    let xs: Vec<f64> = traj.positions.iter().map(|p| p[0]).collect();
    let ys: Vec<f64> = traj.positions.iter().map(|p| p[1]).collect();
    let zs: Vec<f64> = traj.positions.iter().map(|p| p[2]).collect();
    let vz: Vec<f64> = traj.velocities.iter().map(|v| v[2]).collect();
    let speed: Vec<f64> = traj.velocities.iter()
        .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
        .collect();
    let times = &traj.time_points;
    let thrust_pct: Vec<f64> = traj.normalized_thrusts.iter().map(|t| t * 100.0).collect();

    fn range_of(vals: &[f64]) -> (f64, f64) {
        let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pad = (hi - lo).abs() * 0.05 + 1.0;
        (lo - pad, hi + pad)
    }

    // ── panel 0: side view (downrange x vs altitude z) ───────────────────────
    {
        let (x_lo, x_hi) = range_of(&xs);
        let (z_lo, z_hi) = range_of(&zs);
        let mut chart = ChartBuilder::on(&areas[0])
            .caption("Side View (downrange vs altitude)", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(x_lo..x_hi, z_lo..z_hi)
            .map_err(|e| format!("chart build: {e}"))?;
        chart.configure_mesh()
            .x_desc("X / m")
            .y_desc("Z / m")
            .draw()
            .map_err(|e| format!("mesh: {e}"))?;
        chart.draw_series(LineSeries::new(
            xs.iter().zip(zs.iter()).map(|(&x, &z)| (x, z)),
            RED,
        )).map_err(|e| format!("series: {e}"))?
            .label("path")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        chart.configure_series_labels().border_style(BLACK).draw()
            .map_err(|e| format!("legend: {e}"))?;
    }

    // ── panel 1: ground track (x vs y) ───────────────────────────────────────
    {
        let (x_lo, x_hi) = range_of(&xs);
        let (y_lo, y_hi) = range_of(&ys);
        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Ground Track", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)
            .map_err(|e| format!("chart build: {e}"))?;
        chart.configure_mesh()
            .x_desc("X / m")
            .y_desc("Y / m")
            .draw()
            .map_err(|e| format!("mesh: {e}"))?;
        chart.draw_series(LineSeries::new(
            xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
            BLUE,
        )).map_err(|e| format!("series: {e}"))?
            .label("ground track")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
        chart.configure_series_labels().border_style(BLACK).draw()
            .map_err(|e| format!("legend: {e}"))?;
    }

    // ── panel 2: velocity profile ─────────────────────────────────────────────
    {
        let t_lo = times.first().copied().unwrap_or(0.0) - 1.0;
        let t_hi = times.last().copied().unwrap_or(1.0) + 1.0;
        let all_v: Vec<f64> = speed.iter().chain(vz.iter()).cloned().collect();
        let (v_lo, v_hi) = range_of(&all_v);
        let mut chart = ChartBuilder::on(&areas[2])
            .caption("Velocity Profile", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(55)
            .build_cartesian_2d(t_lo..t_hi, v_lo..v_hi)
            .map_err(|e| format!("chart build: {e}"))?;
        chart.configure_mesh()
            .x_desc("Time / s")
            .y_desc("Velocity / (m/s)")
            .draw()
            .map_err(|e| format!("mesh: {e}"))?;
        chart.draw_series(LineSeries::new(
            times.iter().zip(speed.iter()).map(|(&t, &s)| (t, s)),
            RED,
        )).map_err(|e| format!("series: {e}"))?
            .label("total speed")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
        chart.draw_series(LineSeries::new(
            times.iter().zip(vz.iter()).map(|(&t, &v)| (t, v)),
            BLUE,
        )).map_err(|e| format!("series: {e}"))?
            .label("Z velocity")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
        chart.configure_series_labels().border_style(BLACK).draw()
            .map_err(|e| format!("legend: {e}"))?;
    }

    // ── panel 3: thrust profile ───────────────────────────────────────────────
    {
        let t_lo = times.first().copied().unwrap_or(0.0) - 1.0;
        let t_hi = times.last().copied().unwrap_or(1.0) + 1.0;
        let mut chart = ChartBuilder::on(&areas[3])
            .caption("Thrust Profile", ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(t_lo..t_hi, 0f64..105f64)
            .map_err(|e| format!("chart build: {e}"))?;
        chart.configure_mesh()
            .x_desc("Time / s")
            .y_desc("Thrust / %")
            .draw()
            .map_err(|e| format!("mesh: {e}"))?;
        chart.draw_series(LineSeries::new(
            times.iter().zip(thrust_pct.iter()).map(|(&t, &th)| (t, th)),
            GREEN,
        )).map_err(|e| format!("series: {e}"))?
            .label("thrust %")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));
        chart.configure_series_labels().border_style(BLACK).draw()
            .map_err(|e| format!("legend: {e}"))?;
    }

    root.present().map_err(|e| format!("present: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::FAILURE }
    }
}
