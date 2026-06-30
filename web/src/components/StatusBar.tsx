import type { Trajectory } from "../wasm/init";

interface Props {
  trajectory: Trajectory | null;
  error: string | null;
}

export default function StatusBar({ trajectory, error }: Props) {
  if (error) {
    return (
      <div className="px-3 py-2.5 rounded-md flex gap-4 flex-wrap text-sm bg-red-950 border border-red-700 text-red-300">
        <span className="font-mono text-[10px] tracking-widest uppercase text-red-500 self-center">Error</span>
        <span>{error}</span>
      </div>
    );
  }
  if (!trajectory) {
    return (
      <div className="px-3 py-2.5 rounded-md flex gap-4 flex-wrap text-sm bg-slate-800 text-slate-400">
        <span className="font-mono text-[10px] tracking-widest uppercase text-slate-600 self-center">Idle</span>
        <span>Edit a config and press Solve.</span>
      </div>
    );
  }
  return (
    <div className="px-3 py-2.5 rounded-md flex gap-4 flex-wrap text-sm bg-emerald-950 border border-emerald-700 text-emerald-100">
      <span className="font-mono text-[10px] tracking-widest uppercase text-emerald-500 self-center">Solved</span>
      <span><span className="text-emerald-400">Status</span> {trajectory.status}</span>
      <span><span className="text-emerald-400">Final mass</span> {trajectory.final_mass.toFixed(1)} kg</span>
      <span><span className="text-emerald-400">ToF</span> {trajectory.time_of_flight.toFixed(2)} s</span>
      <span><span className="text-emerald-400">Objective</span> {trajectory.objective.toFixed(2)}</span>
    </div>
  );
}
