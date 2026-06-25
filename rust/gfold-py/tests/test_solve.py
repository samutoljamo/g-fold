import math
import numpy as np
import pytest
import gfold


def make_cfg():
    return gfold.Config(
        spacecraft=gfold.Spacecraft(
            wet_mass=2000.0, fuel=1700.0, real_max_thrust=24000.0,
            min_thrust_pct=0.2, max_thrust_pct=0.8, max_velocity=1000.0,
            initial_position=[450.0, -330.0, 2400.0], initial_velocity=[-40.0, 10.0, -10.0],
            target_velocity=[0.0, 0.0, 0.0], target_position=[0.0, 0.0, 0.0], fuel_consumption=5e-4),
        environment=gfold.Environment(gravity=[0.0, 0.0, -3.71], glide_slope_angle_deg=0.0, max_angle_deg=90.0),
        solver=gfold.Solver(n=100, time_of_flight=44.63),
    )


def test_solve_returns_numpy_and_lands():
    traj = gfold.solve(make_cfg())
    assert isinstance(traj.positions, np.ndarray)
    assert traj.positions.shape == (100, 3)
    assert traj.thrusts.shape == (100,)
    assert np.all(np.abs(traj.positions[-1]) < 1.0)
    assert math.isclose(traj.final_mass, math.exp(traj.objective), rel_tol=1e-9)


def test_infeasible_raises():
    cfg = make_cfg()
    cfg.solver = gfold.Solver(n=100, time_of_flight=0.001)
    with pytest.raises(ValueError):
        gfold.solve(cfg)
