// Typed smoke test: type-checks against the tsify-generated pkg/gfold_wasm.d.ts
// (the real verification that the TS interface is correct) AND runs the solve at
// runtime. `tsc --noEmit` checks types; `tsx` runs this file. See package.json.
import { solve, type Config } from "../pkg/gfold_wasm.js";

const config: Config = {
  spacecraft: {
    wet_mass: 2000,
    fuel: 1700,
    real_max_thrust: 24000,
    min_thrust_pct: 0.2,
    max_thrust_pct: 0.8,
    max_velocity: 1000,
    initial_position: [450, -330, 2400],
    initial_velocity: [-40, 10, -10],
    target_velocity: [0, 0, 0],
    target_position: [0, 0, 0],
    fuel_consumption: 5e-4,
  },
  environment: { gravity: [0, 0, -3.71], glide_slope_angle_deg: 0, max_angle_deg: 90 },
  solver: { n: 100, time_of_flight: 44.63 },
};

const traj = solve(config); // typed: Trajectory
const last = traj.positions[traj.positions.length - 1]; // number[] (an [x,y,z])
if (Math.hypot(...last) > 1.0) {
  console.error("did not land:", last);
  process.exit(1);
}
console.log("ok: landed", last);
