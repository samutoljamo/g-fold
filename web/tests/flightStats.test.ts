import { describe, it, expect } from "vitest";
import { interpolateState } from "../src/lib/flightStats";
import { sampleFlightStats } from "../src/lib/flightStats";
import type { Trajectory } from "../src/wasm/init";
import type { AppConfig } from "../src/lib/config";

const traj: Trajectory = {
  positions: [[0, 0, 100], [0, 0, 50], [0, 0, 0]],
  velocities: [[0, 0, -10], [0, 0, -10], [0, 0, 0]],
  thrusts: [10000, 12000, 8000],
  normalized_thrusts: [0.4, 0.6, 0.2],
  z_values: [Math.log(2000), Math.log(1900), Math.log(1800)],
  u_values: [[0, 0, 5], [1, 0, 5], [0, 0, 5]],
  s_values: [5, 5, 5],
  objective: 1800, final_mass: 1800, time_of_flight: 20,
  time_points: [0, 10, 20], status: "Solved",
};

describe("interpolateState", () => {
  it("returns the exact sample at a grid time", () => {
    const s = interpolateState(traj, 10);
    expect(s.position[2]).toBeCloseTo(50);
    expect(s.throttlePct).toBeCloseTo(60);
  });
  it("interpolates linearly between grid times", () => {
    const s = interpolateState(traj, 5); // halfway between idx 0 and 1
    expect(s.position[2]).toBeCloseTo(75);
    expect(s.throttlePct).toBeCloseTo(50);
    expect(s.speed).toBeCloseTo(10);
  });
  it("clamps below 0 and above duration", () => {
    expect(interpolateState(traj, -5).position[2]).toBeCloseTo(100);
    expect(interpolateState(traj, 999).position[2]).toBeCloseTo(0);
  });
  it("normalizes the thrust direction", () => {
    const s = interpolateState(traj, 0);
    const m = Math.hypot(...s.thrustDir);
    expect(m).toBeCloseTo(1);
  });
});

const cfg = {
  spacecraft: {
    wet_mass: 2000, fuel: 1700, real_max_thrust: 24000, min_thrust_pct: 0.2,
    max_thrust_pct: 0.8, max_velocity: 1000, initial_position: [0, 0, 100],
    initial_velocity: [0, 0, -10], target_velocity: [0, 0, 0],
    target_position: [30, 40, 0], fuel_consumption: 5e-4,
  },
  environment: { gravity: [0, 0, -3.71], glide_slope_angle_deg: 0, max_angle_deg: 90 },
  solver: { n: 100 },
} satisfies AppConfig;

describe("sampleFlightStats", () => {
  it("computes downrange to the target (horizontal)", () => {
    const s = sampleFlightStats(traj, cfg, 20); // pos [0,0,0], target [30,40,0]
    expect(s.downrange).toBeCloseTo(50); // hypot(30,40)
  });
  it("computes descent rate as downward vertical speed", () => {
    const s = sampleFlightStats(traj, cfg, 0); // v=[0,0,-10]
    expect(s.descentRate).toBeCloseTo(10);
  });
  it("computes fuel remaining = mass - dry_mass (dry = wet - fuel)", () => {
    const s = sampleFlightStats(traj, cfg, 20); // mass = 1800; dry = 300
    expect(s.fuelRemaining).toBeCloseTo(1500);
  });
  it("gimbal angle is 0 when thrust is straight up", () => {
    const s = sampleFlightStats(traj, cfg, 0); // u=[0,0,5] -> vertical
    expect(s.gimbalDeg).toBeCloseTo(0);
  });
  it("altitude equals position z, thrustKN = thrustN/1000", () => {
    const s = sampleFlightStats(traj, cfg, 0);
    expect(s.altitude).toBeCloseTo(100);
    expect(s.thrustKN).toBeCloseTo(10); // thrusts[0]=10000 N
  });
});
