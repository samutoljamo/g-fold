import math
from gfold_oracle.config import Config
from gfold_oracle.model import solve

def test_default_solves_and_lands(default_dict):
    cfg = Config.from_dict(default_dict)
    sol = solve(cfg)
    assert len(sol["positions"]) == 100
    assert len(sol["velocities"]) == 100
    assert len(sol["thrusts"]) == 100
    # boundary: start at initial, end at target
    assert sol["positions"][0] == [450.0, -330.0, 2400.0] or \
        all(math.isclose(a, b, abs_tol=1e-3) for a, b in zip(sol["positions"][0], [450.0, -330.0, 2400.0]))
    assert all(abs(c) < 1e-2 for c in sol["positions"][-1])      # lands at origin
    assert all(abs(c) < 1e-2 for c in sol["velocities"][-1])     # soft landing
    # objective ~ ln(final mass); known-good value for this config is ~7.496
    assert math.isclose(sol["objective"], 7.496, abs_tol=5e-2)
    assert math.isclose(sol["final_mass"], math.exp(sol["objective"]), rel_tol=1e-9)
