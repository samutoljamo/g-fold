"""The G-FOLD SOCP in CVXPY. The readable spec; mirrors gfold-core/src/assemble.rs.

min-fuel powered descent, n nodes, first-order hold on the thrust acceleration u.
Velocity uses the trapezoidal rule (exact for FOH); position uses the matching
exact integral. Lossless convexification: ||u|| <= s with the log-mass thrust
bounds. Maximizes terminal log-mass.
"""
import cvxpy as cp
import numpy as np

from .config import Config


def solve(cfg: Config) -> dict:
    n = cfg.n
    dt = cfg.dt
    g = np.array(cfg.gravity, dtype=float)
    a_dt = cfg.fuel_consumption * dt

    x = cp.Variable((n, 6))   # 0:3 position, 3:6 velocity
    u = cp.Variable((n, 3))   # thrust acceleration
    s = cp.Variable(n)        # slack: ||u|| <= s
    z = cp.Variable(n)        # log mass

    # log-mass linearization of the thrust bounds (mirror derive.rs)
    z0 = np.array([np.log(cfg.wet_mass - cfg.fuel_consumption * dt * cfg.max_thrust * i)
                   for i in range(n)])
    max_exp = np.exp(z0) / cfg.max_thrust
    has_min = cfg.min_thrust > 0.0
    min_exp = (np.exp(z0) / cfg.min_thrust) if has_min else None

    cons = [
        x[0, :3] == cfg.initial_position,
        x[0, 3:] == cfg.initial_velocity,
        z[0] == cfg.log_wet_mass,
    ]
    sin_gs = cfg.sin_glide_slope
    cos_max = cfg.cos_max_angle
    enforce_pointing = cfg.max_angle_deg < 180.0           # full sphere is vacuous
    for i in range(n):
        cons.append(cp.norm(x[i, 3:]) <= cfg.max_velocity)
        cons.append(x[i, 2] >= sin_gs * cp.norm(x[i, :3]))      # glide slope (>=0 when sin_gs==0)
        cons.append(cp.norm(u[i, :]) <= s[i])                   # thrust slack
        if enforce_pointing:                                    # thrust pointing: <= max_angle from vertical
            cons.append(u[i, 2] >= cos_max * s[i])
        cons.append(s[i] * max_exp[i] <= 1 - (z[i] - z0[i]))    # upper thrust bound
        if has_min:                                             # lower thrust bound
            cons.append(1 - (z[i] - z0[i]) + cp.square(z[i] - z0[i]) / 2 <= s[i] * min_exp[i])
        if i != n - 1:
            cons += [
                # first-order-hold position integral (matches assemble.rs)
                x[i + 1, :3] == x[i, :3] + x[i, 3:] * dt
                + (2 * u[i, :] + u[i + 1, :]) * dt**2 / 6 + g * dt**2 / 2,
                x[i + 1, 3:] == x[i, 3:] + (u[i, :] + u[i + 1, :]) * dt / 2 + g * dt,
                z[i + 1] == z[i] - (s[i] + s[i + 1]) / 2 * a_dt,
            ]
    cons += [
        x[n - 1, :3] == cfg.target_position,
        x[n - 1, 3:] == cfg.target_velocity,
        z[n - 1] >= cfg.log_dry_mass,
    ]

    prob = cp.Problem(cp.Maximize(z[n - 1]), cons)
    # Match the Rust core's accuracy so the oracle isn't the looser solver when
    # the pointing cone binds (the min-fuel optimum is flat, so an
    # under-converged reference drifts from the implementation it gates).
    prob.solve(solver=cp.CLARABEL, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=400)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"oracle solve failed: {prob.status}")

    xv = x.value
    zv = z.value
    uv = u.value
    positions = [[float(v) for v in xv[i, :3]] for i in range(n)]
    velocities = [[float(v) for v in xv[i, 3:]] for i in range(n)]
    thrusts = [float(np.linalg.norm(uv[i]) * np.exp(zv[i])) for i in range(n)]
    return {
        "objective": float(zv[-1]),
        "final_mass": float(np.exp(zv[-1])),
        "positions": positions,
        "velocities": velocities,
        "thrusts": thrusts,
    }
