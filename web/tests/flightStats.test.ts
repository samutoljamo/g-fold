import { describe, it, expect } from "vitest";
import { interpolateState } from "../src/lib/flightStats";
import type { Trajectory } from "../src/wasm/init";

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
