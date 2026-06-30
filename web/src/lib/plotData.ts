import type { Trajectory } from "../wasm/init";

export interface ProfileRow {
  t: number;
  altitude: number;
  speed: number;
  thrustPct: number;
  mass: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

function norm(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function buildProfiles(traj: Trajectory): ProfileRow[] {
  return traj.time_points.map((t, i) => {
    const z = traj.z_values[i];
    // Real solver output keeps every array the same length, so z is always
    // present; the guard is defensive against truncated/synthetic trajectories.
    const mass = z !== undefined ? Math.exp(z) : traj.final_mass;
    return {
      t,
      altitude: traj.positions[i]?.[2] ?? 0,
      speed: traj.velocities[i] ? norm(traj.velocities[i]) : 0,
      thrustPct: (traj.normalized_thrusts[i] ?? 0) * 100,
      mass,
    };
  });
}

export function buildPath3D(traj: Trajectory): Point3D[] {
  return traj.positions.map(([x, y, z]) => ({ x, y, z }));
}

/**
 * Linearly interpolate a profile row at time `t` (seconds), clamped to the
 * trajectory's time span. Used to place the playback cursor's dot exactly on
 * each curve. Returns null for an empty profile.
 */
export function interpolateProfile(rows: ProfileRow[], t: number): ProfileRow | null {
  if (rows.length === 0) return null;
  const first = rows[0];
  const last = rows[rows.length - 1];
  if (t <= first.t) return first;
  if (t >= last.t) return last;
  let i = 0;
  while (i < rows.length - 1 && rows[i + 1].t < t) i++;
  const a = rows[i];
  const b = rows[i + 1];
  const f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
  const lerp = (x: number, y: number) => x + (y - x) * f;
  return {
    t,
    altitude: lerp(a.altitude, b.altitude),
    speed: lerp(a.speed, b.speed),
    thrustPct: lerp(a.thrustPct, b.thrustPct),
    mass: lerp(a.mass, b.mass),
  };
}
