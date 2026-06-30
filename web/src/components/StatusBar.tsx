import type { Trajectory } from "../wasm/init";

interface Props {
  trajectory: Trajectory | null;
  error: string | null;
}

export default function StatusBar({ trajectory, error }: Props) {
  if (error) {
    return <div className="status status-error">Solve failed: {error}</div>;
  }
  if (!trajectory) {
    return <div className="status status-idle">Edit a config and press Solve.</div>;
  }
  return (
    <div className="status status-ok">
      <span><strong>Status:</strong> {trajectory.status}</span>
      <span><strong>Final mass:</strong> {trajectory.final_mass.toFixed(1)} kg</span>
      <span><strong>Time of flight:</strong> {trajectory.time_of_flight.toFixed(2)} s</span>
      <span><strong>Objective:</strong> {trajectory.objective.toFixed(2)}</span>
    </div>
  );
}
