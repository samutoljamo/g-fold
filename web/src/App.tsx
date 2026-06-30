import { useState } from "react";
import type { AppConfig } from "./lib/config";
import { PRESETS, DEFAULT_PRESET } from "./lib/presets";
import { useSolve } from "./hooks/useSolve";
import { usePlayback } from "./hooks/usePlayback";
import ConfigForm from "./components/ConfigForm";
import StatusBar from "./components/StatusBar";
import ProfilePlots from "./components/ProfilePlots";
import TrajectoryView3D from "./components/TrajectoryView3D";
import TransportControls from "./components/TransportControls";
import StatsReadout from "./components/StatsReadout";

export default function App() {
  const [config, setConfig] = useState<AppConfig>(() => structuredClone(PRESETS[DEFAULT_PRESET]));
  const { trajectory, error, solving, run } = useSolve();
  const playback = usePlayback(trajectory?.time_of_flight ?? 0);

  return (
    <div className="grid grid-cols-1 md:grid-cols-[360px_1fr] md:h-screen">
      <aside className="overflow-y-auto p-4 bg-slate-50 border-r border-slate-200">
        <h1 className="font-mono text-[11px] tracking-[0.2em] uppercase text-slate-400 mb-0.5">G-FOLD</h1>
        <p className="text-[13px] font-semibold text-slate-800 mb-4">Powered Descent Playground</p>
        <ConfigForm config={config} onChange={setConfig} onSolve={() => run(config)} solving={solving} />
      </aside>
      <main className="overflow-y-auto p-4 flex flex-col gap-4 bg-slate-900 text-slate-100">
        <StatusBar trajectory={trajectory} error={error} />
        {trajectory && (
          <>
            <TrajectoryView3D trajectory={trajectory} tRef={playback.tRef} />
            <TransportControls playback={playback} />
            <StatsReadout trajectory={trajectory} config={config} t={playback.t} />
            <ProfilePlots trajectory={trajectory} />
          </>
        )}
      </main>
    </div>
  );
}
