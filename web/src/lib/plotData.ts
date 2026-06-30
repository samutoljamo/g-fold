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
