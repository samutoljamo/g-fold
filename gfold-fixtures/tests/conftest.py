import pytest

DEFAULT = {
    "spacecraft": {
        "wet_mass": 2000.0, "fuel": 1700.0, "real_max_thrust": 24000.0,
        "min_thrust_pct": 0.2, "max_thrust_pct": 0.8, "max_velocity": 1000.0,
        "initial_position": [450.0, -330.0, 2400.0],
        "initial_velocity": [-40.0, 10.0, -10.0],
        "target_velocity": [0.0, 0.0, 0.0],
        "target_position": [0.0, 0.0, 0.0],
        "fuel_consumption": 5e-4,
    },
    "environment": {"gravity": [0.0, 0.0, -3.71], "glide_slope_angle_deg": 0.0, "max_angle_deg": 90.0},
    "solver": {"n": 100, "time_of_flight": 44.63},
}

@pytest.fixture
def default_dict():
    return DEFAULT
