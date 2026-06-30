import { describe, it, expect } from "vitest";
import { buildProfiles, buildPath3D } from "../src/lib/plotData";
import type { Trajectory } from "../src/wasm/init";

const traj: Trajectory = {
  positions: [[0, 0, 100], [1, 1, 50], [2, 2, 0]],
  velocities: [[0, 0, -10], [0, 0, -10], [0, 0, 0]],
  thrusts: [12000, 14000, 10000],
  normalized_thrusts: [0.5, 0.58, 0.42],
  z_values: [7.6, 7.5, 7.4],
  u_values: [[0, 0, 6], [0, 0, 7], [0, 0, 5]],
  s_values: [6, 7, 5],
  objective: 1693,
  final_mass: 1693,
  time_of_flight: 20,
  time_points: [0, 10, 20],
  status: "Solved",
};

describe("buildProfiles", () => {
  it("produces one row per time point with altitude/speed/thrust", () => {
    const rows = buildProfiles(traj);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toMatchObject({ t: 0, altitude: 100, thrustPct: 50 });
    expect(rows[0].speed).toBeCloseTo(10);
  });

  it("derives mass from z_values via exp(z)", () => {
    const rows = buildProfiles(traj);
    expect(rows[rows.length - 1].mass).toBeCloseTo(Math.exp(7.4), 0);
  });
});

describe("buildPath3D", () => {
  it("maps positions to {x,y,z} triples", () => {
    const pts = buildPath3D(traj);
    expect(pts).toHaveLength(3);
    expect(pts[2]).toEqual({ x: 2, y: 2, z: 0 });
  });
});
