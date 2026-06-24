"""Generate solver fixtures (oracle) for the Rust core tests."""
import json
import os
from .config import GFoldConfig, EnvironmentConfig, SolverConfig
from .solver import GFoldSolver


def _config_json(cfg: GFoldConfig) -> dict:
    sc = cfg.spacecraft
    env = cfg.environment
    sol = cfg.solver
    return {
        "spacecraft": {
            "wet_mass": sc.wet_mass, "fuel": sc.fuel,
            "real_max_thrust": sc.real_max_thrust,
            "min_thrust_pct": sc.min_thrust_pct, "max_thrust_pct": sc.max_thrust_pct,
            "max_velocity": sc.max_velocity,
            "initial_position": list(map(float, sc.initial_position)),
            "initial_velocity": list(map(float, sc.initial_velocity)),
            "target_velocity": list(map(float, sc.target_velocity)),
            "target_position": list(map(float, sc.target_position)),
            "fuel_consumption": sc.fuel_consumption,
        },
        "environment": {
            "gravity": list(map(float, env.gravity)),
            "glide_slope_angle_deg": float(env.glide_slope_angle),
            "max_angle_deg": float(env.max_angle),
        },
        "solver": {"n": sol.n, "time_of_flight": sol.time_of_flight},
    }


def _dump_one(name: str, cfg: GFoldConfig, out_dir: str) -> None:
    solver = GFoldSolver(cfg)
    try:
        result = solver.solve()
    except Exception as e:  # infeasible / solver error
        print(f"skip {name}: {e}")
        return
    payload = {
        "name": name,
        "config": _config_json(cfg),
        "expected": {
            "objective": float(result["z_values"][-1]),
            "final_mass": float(result["final_mass"]),
            "positions": [list(map(float, p)) for p in result["positions"]],
            "velocities": [list(map(float, v)) for v in result["velocities"]],
            "thrusts": [float(t) for t in result["thrusts"]],
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {name}.json")


def dump_fixtures(out_dir: str) -> None:
    cases = {
        "default": GFoldConfig(),
        "moon": GFoldConfig(environment=EnvironmentConfig.moon()),
        "earth": GFoldConfig(environment=EnvironmentConfig.earth()),
        "small_n": GFoldConfig(solver=SolverConfig(n=20, time_of_flight=44.63)),
        "glide": GFoldConfig(environment=EnvironmentConfig(glide_slope_angle=10)),
    }
    for name, cfg in cases.items():
        _dump_one(name, cfg, out_dir)
