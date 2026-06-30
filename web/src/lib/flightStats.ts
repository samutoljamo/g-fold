import type { Trajectory } from "../wasm/init";
import type { AppConfig } from "./config";

export interface KinematicState {
  t: number;
  position: [number, number, number];
  velocity: [number, number, number];
  thrustDir: [number, number, number]; // unit vector
  speed: number;
  throttlePct: number;
  thrustN: number;
  mass: number;
}

type V3 = [number, number, number];
const lerp = (a: number, b: number, f: number) => a + (b - a) * f;
const lerpV = (a: V3, b: V3, f: number): V3 => [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f)];
function normalize(v: V3): V3 {
  const m = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / m, v[1] / m, v[2] / m];
}

/** Linearly interpolate the solver samples at time `t` (clamped to [0, ToF]). */
export function interpolateState(traj: Trajectory, t: number): KinematicState {
  const tp = traj.time_points;
  const last = tp.length - 1;
  const clamped = Math.min(Math.max(t, tp[0]), tp[last]);
  let i = 0;
  while (i < last && tp[i + 1] < clamped) i++;
  const segSpan = tp[i + 1] !== undefined ? tp[i + 1] - tp[i] : 1;
  const f = segSpan > 0 ? (clamped - tp[i]) / segSpan : 0;
  const j = Math.min(i + 1, last);

  const position = lerpV(traj.positions[i], traj.positions[j], f);
  const velocity = lerpV(traj.velocities[i], traj.velocities[j], f);
  const thrustDir = normalize(lerpV(traj.u_values[i], traj.u_values[j], f));
  const throttlePct = lerp(traj.normalized_thrusts[i], traj.normalized_thrusts[j], f) * 100;
  const thrustN = lerp(traj.thrusts[i], traj.thrusts[j], f);
  const mass = Math.exp(lerp(traj.z_values[i], traj.z_values[j], f));
  return {
    t: clamped,
    position, velocity, thrustDir,
    speed: Math.hypot(velocity[0], velocity[1], velocity[2]),
    throttlePct, thrustN, mass,
  };
}

export interface FlightStats extends KinematicState {
  altitude: number;
  descentRate: number;
  downrange: number;
  thrustKN: number;
  gimbalDeg: number;
  fuelRemaining: number;
}

/** Full derived readout at time `t`, including config-relative quantities. */
export function sampleFlightStats(traj: Trajectory, cfg: AppConfig, t: number): FlightStats {
  const k = interpolateState(traj, t);
  const target = cfg.spacecraft.target_position;
  const dryMass = cfg.spacecraft.wet_mass - cfg.spacecraft.fuel;
  return {
    ...k,
    altitude: k.position[2],
    descentRate: -k.velocity[2],
    downrange: Math.hypot(k.position[0] - target[0], k.position[1] - target[1]),
    thrustKN: k.thrustN / 1000,
    gimbalDeg: Math.acos(Math.min(1, Math.max(-1, k.thrustDir[2]))) * (180 / Math.PI),
    fuelRemaining: k.mass - dryMass,
  };
}
