import { useCallback, useState } from "react";
import { solve, type Trajectory } from "../wasm/init";
import type { AppConfig } from "../lib/config";

export interface SolveState {
  trajectory: Trajectory | null;
  error: string | null;
  solving: boolean;
  run: (config: AppConfig) => void;
}

export function useSolve(): SolveState {
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [solving, setSolving] = useState(false);

  const run = useCallback((config: AppConfig) => {
    setSolving(true);
    setError(null);
    // Defer so the "solving" state paints before the (sync) solve blocks.
    setTimeout(() => {
      try {
        setTrajectory(solve(config));
      } catch (e) {
        setTrajectory(null);
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSolving(false);
      }
    }, 0);
  }, []);

  return { trajectory, error, solving, run };
}
