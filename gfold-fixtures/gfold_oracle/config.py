"""Problem config mirroring gfold-core's config.rs serde shape."""
import math
from dataclasses import dataclass


@dataclass
class Config:
    # spacecraft
    wet_mass: float
    fuel: float
    real_max_thrust: float
    min_thrust_pct: float
    max_thrust_pct: float
    max_velocity: float
    initial_position: list
    initial_velocity: list
    target_velocity: list
    target_position: list
    fuel_consumption: float
    # environment
    gravity: list
    glide_slope_angle_deg: float
    max_angle_deg: float
    # solver
    n: int
    time_of_flight: float

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**d["spacecraft"], **d["environment"], **d["solver"])

    def to_dict(self) -> dict:
        return {
            "spacecraft": {
                "wet_mass": self.wet_mass, "fuel": self.fuel,
                "real_max_thrust": self.real_max_thrust,
                "min_thrust_pct": self.min_thrust_pct, "max_thrust_pct": self.max_thrust_pct,
                "max_velocity": self.max_velocity,
                "initial_position": self.initial_position,
                "initial_velocity": self.initial_velocity,
                "target_velocity": self.target_velocity,
                "target_position": self.target_position,
                "fuel_consumption": self.fuel_consumption,
            },
            "environment": {
                "gravity": self.gravity,
                "glide_slope_angle_deg": self.glide_slope_angle_deg,
                "max_angle_deg": self.max_angle_deg,
            },
            "solver": {"n": self.n, "time_of_flight": self.time_of_flight},
        }

    @property
    def log_wet_mass(self) -> float:
        return math.log(self.wet_mass)

    @property
    def log_dry_mass(self) -> float:
        return math.log(self.wet_mass - self.fuel)

    @property
    def min_thrust(self) -> float:
        return self.real_max_thrust * self.min_thrust_pct

    @property
    def max_thrust(self) -> float:
        return self.real_max_thrust * self.max_thrust_pct

    @property
    def sin_glide_slope(self) -> float:
        return math.sin(math.radians(self.glide_slope_angle_deg))

    @property
    def cos_max_angle(self) -> float:
        return math.cos(math.radians(self.max_angle_deg))

    @property
    def dt(self) -> float:
        return self.time_of_flight / self.n
