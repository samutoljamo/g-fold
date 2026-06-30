import { describe, it, expect } from "vitest";
import { PRESETS, DEFAULT_PRESET } from "../src/lib/presets";

describe("presets", () => {
  it("exposes the default Mars preset with valid structure", () => {
    const cfg = PRESETS[DEFAULT_PRESET];
    expect(cfg.spacecraft.wet_mass).toBe(2000);
    expect(cfg.spacecraft.initial_position).toEqual([450, -330, 2400]);
    expect(cfg.environment.gravity).toEqual([0, 0, -3.71]);
    expect(cfg.solver.n).toBe(100);
  });

  it("every preset has fuel strictly less than wet_mass", () => {
    for (const cfg of Object.values(PRESETS)) {
      expect(cfg.spacecraft.fuel).toBeLessThan(cfg.spacecraft.wet_mass);
    }
  });
});
