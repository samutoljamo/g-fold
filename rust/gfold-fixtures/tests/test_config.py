import math
from gfold_oracle.config import Config

def test_from_dict_and_derived(default_dict):
    c = Config.from_dict(default_dict)
    assert c.n == 100
    assert c.gravity == [0.0, 0.0, -3.71]
    assert math.isclose(c.log_wet_mass, math.log(2000.0))
    assert math.isclose(c.log_dry_mass, math.log(300.0))
    assert math.isclose(c.min_thrust, 4800.0)
    assert math.isclose(c.max_thrust, 19200.0)
    assert math.isclose(c.sin_glide_slope, 0.0)
    assert math.isclose(c.dt, 44.63 / 100)

def test_roundtrip_to_dict(default_dict):
    c = Config.from_dict(default_dict)
    assert c.to_dict() == default_dict
