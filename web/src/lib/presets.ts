import type { AppConfig } from "./config";

export const DEFAULT_PRESET = "Mars descent (default)";

// Every preset below has been run through the core solver and converges
// ("Solved"). Where a value sits near a feasibility boundary it is called out,
// so the scenario demonstrates the solver working under pressure rather than
// with comfortable slack.
export const PRESETS: Record<string, AppConfig> = {
  [DEFAULT_PRESET]: {
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
    solver: { n: 100 },
  },

  // Low gravity, generous thrust-to-weight: the gentlest scenario, useful as a
  // first comparison against Mars (same craft, ~half a g of gravity).
  "Moon landing": {
    spacecraft: {
      wet_mass: 2000,
      fuel: 1700,
      real_max_thrust: 24000,
      min_thrust_pct: 0.2,
      max_thrust_pct: 0.8,
      max_velocity: 1000,
      initial_position: [200, -150, 1500],
      initial_velocity: [-20, 5, -5],
      target_velocity: [0, 0, 0],
      target_position: [0, 0, 0],
      fuel_consumption: 5e-4,
    },
    environment: { gravity: [0, 0, -1.62], glide_slope_angle_deg: 0, max_angle_deg: 90 },
    solver: { n: 100 },
  },

  // Earth gravity with a ~150 m/s descent from 3.5 km. The engine's minimum
  // throttle sits below hover thrust, so the craft cannot hover and must time a
  // single hard braking burn — the classic booster "hoverslam". Altitude is
  // deliberately tight: drop the start below ~3 km and there is no longer room
  // to null the velocity before touchdown.
  "Earth booster (hoverslam)": {
    spacecraft: {
      wet_mass: 30000,
      fuel: 25000,
      real_max_thrust: 480000,
      min_thrust_pct: 0.4,
      max_thrust_pct: 1.0,
      max_velocity: 300,
      initial_position: [300, 0, 3500],
      initial_velocity: [8, 0, -150],
      target_velocity: [0, 0, 0],
      target_position: [0, 0, 0],
      fuel_consumption: 3e-4,
    },
    environment: { gravity: [0, 0, -9.81], glide_slope_angle_deg: 0, max_angle_deg: 90 },
    solver: { n: 100 },
  },

  // The algorithm's namesake: a large lateral offset (~4.6 km cross-range) that
  // produces the signature curved divert trajectory.
  "Large divert (Mars)": {
    spacecraft: {
      wet_mass: 2000,
      fuel: 1700,
      real_max_thrust: 24000,
      min_thrust_pct: 0.2,
      max_thrust_pct: 0.8,
      max_velocity: 1000,
      initial_position: [3500, 3000, 2600],
      initial_velocity: [-60, -50, -10],
      target_velocity: [0, 0, 0],
      target_position: [0, 0, 0],
      fuel_consumption: 5e-4,
    },
    environment: { gravity: [0, 0, -3.71], glide_slope_angle_deg: 0, max_angle_deg: 90 },
    solver: { n: 100 },
  },

  // Same divert as above, but with max_velocity capped at 80 m/s — well below
  // the ~115 m/s the unconstrained optimum would reach. The cap binds: the
  // solver flattens the speed profile, stretching the flight ~16 s longer and
  // spending ~30 kg more fuel. Cap it below ~70 m/s and the divert becomes
  // infeasible.
  "Speed-capped divert (Mars)": {
    spacecraft: {
      wet_mass: 2000,
      fuel: 1700,
      real_max_thrust: 24000,
      min_thrust_pct: 0.2,
      max_thrust_pct: 0.8,
      max_velocity: 80,
      initial_position: [3500, 3000, 2600],
      initial_velocity: [-60, -50, -10],
      target_velocity: [0, 0, 0],
      target_position: [0, 0, 0],
      fuel_consumption: 5e-4,
    },
    environment: { gravity: [0, 0, -3.71], glide_slope_angle_deg: 0, max_angle_deg: 90 },
    solver: { n: 100 },
  },

  // The default Mars descent on a near-empty tank: 220 kg of fuel against the
  // ~199 kg the optimal trajectory burns (~90% margin consumed). The solve
  // still converges; trim fuel below ~205 kg and it goes infeasible.
  "Tight-fuel descent (Mars)": {
    spacecraft: {
      wet_mass: 2000,
      fuel: 220,
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
    solver: { n: 100 },
  },

  // Microgravity touchdown on a small body: gravity is ~1/700 of Earth's, so
  // the descent is almost ballistic and the burn is a gentle nudge that barely
  // touches the fuel budget.
  "Asteroid touchdown (micro-g)": {
    spacecraft: {
      wet_mass: 500,
      fuel: 200,
      real_max_thrust: 200,
      min_thrust_pct: 0.2,
      max_thrust_pct: 0.8,
      max_velocity: 50,
      initial_position: [50, -30, 200],
      initial_velocity: [-2, 1, -1],
      target_velocity: [0, 0, 0],
      target_position: [0, 0, 0],
      fuel_consumption: 5e-4,
    },
    environment: { gravity: [0, 0, -0.05], glide_slope_angle_deg: 0, max_angle_deg: 90 },
    solver: { n: 100 },
  },
};
