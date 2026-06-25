import dataclasses
from typing import List


@dataclasses.dataclass
class Spacecraft:
    wet_mass: float = 2000.0
    fuel: float = 1700.0
    real_max_thrust: float = 24000.0
    min_thrust_pct: float = 0.2
    max_thrust_pct: float = 0.8
    max_velocity: float = 1000.0
    initial_position: List[float] = dataclasses.field(default_factory=lambda: [450.0, -330.0, 2400.0])
    initial_velocity: List[float] = dataclasses.field(default_factory=lambda: [-40.0, 10.0, -10.0])
    target_velocity: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_position: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    fuel_consumption: float = 5e-4


@dataclasses.dataclass
class Environment:
    gravity: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, -3.71])
    glide_slope_angle_deg: float = 0.0
    max_angle_deg: float = 90.0


@dataclasses.dataclass
class Solver:
    n: int = 100
    time_of_flight: float = 44.63


@dataclasses.dataclass
class Config:
    spacecraft: Spacecraft = dataclasses.field(default_factory=Spacecraft)
    environment: Environment = dataclasses.field(default_factory=Environment)
    solver: Solver = dataclasses.field(default_factory=Solver)


@dataclasses.dataclass
class Trajectory:
    positions: List[List[float]] = dataclasses.field(default_factory=list)
    velocities: List[List[float]] = dataclasses.field(default_factory=list)
    thrusts: List[float] = dataclasses.field(default_factory=list)
    normalized_thrusts: List[float] = dataclasses.field(default_factory=list)
    z_values: List[float] = dataclasses.field(default_factory=list)
    u_values: List[List[float]] = dataclasses.field(default_factory=list)
    s_values: List[float] = dataclasses.field(default_factory=list)
    objective: float = 0.0
    final_mass: float = 0.0
    time_points: List[float] = dataclasses.field(default_factory=list)
    status: str = ""
