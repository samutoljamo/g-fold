import math
import pytest
import gfold
from gfold._types import Spacecraft, Environment, Solver


def _default_config() -> gfold.Config:
    return gfold.Config(
        spacecraft=Spacecraft(
            wet_mass=2000.0, fuel=1700.0, real_max_thrust=24000.0,
            min_thrust_pct=0.2, max_thrust_pct=0.8, max_velocity=1000.0,
            initial_position=[450.0, -330.0, 2400.0],
            initial_velocity=[-40.0, 10.0, -10.0],
            target_velocity=[0.0, 0.0, 0.0], target_position=[0.0, 0.0, 0.0],
            fuel_consumption=5e-4,
        ),
        environment=Environment(gravity=[0.0, 0.0, -3.71], glide_slope_angle_deg=0.0, max_angle_deg=90.0),
        solver=Solver(n=100, time_of_flight=44.63),
    )


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
