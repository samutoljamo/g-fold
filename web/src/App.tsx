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
  const [showConfig, setShowConfig] = useState(true);
  const { trajectory, error, solving, run } = useSolve();
  const playback = usePlayback(trajectory?.time_of_flight ?? 0);

  return (
    <div className={`grid ${showConfig ? "md:grid-cols-[340px_1fr]" : "grid-cols-1"} md:h-screen`}>
      {showConfig && (
        <aside className="overflow-y-auto p-4 bg-slate-50 border-r border-slate-200">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="font-mono text-[11px] tracking-[0.2em] uppercase text-slate-400 mb-0.5">G-FOLD</h1>
              <p className="text-[13px] font-semibold text-slate-800">Powered Descent Playground</p>
            </div>
            <button
              type="button"
              onClick={() => setShowConfig(false)}
              title="Hide config — expand the results to full width"
              className="font-mono text-xs text-slate-500 hover:text-slate-800 border border-slate-300 rounded px-1.5 py-0.5"
            >
              ‹ hide
            </button>
          </div>
          <ConfigForm config={config} onChange={setConfig} onSolve={() => run(config)} solving={solving} />
        </aside>
      )}
      <main className="overflow-y-auto p-4 flex flex-col gap-4 bg-slate-900 text-slate-100">
        <div className="flex items-center gap-3">
          {!showConfig && (
            <button
              type="button"
              onClick={() => setShowConfig(true)}
              title="Show config"
              className="font-mono text-xs text-slate-300 hover:text-white border border-slate-600 rounded px-2 py-1 shrink-0"
            >
              ☰ config
            </button>
          )}
          <div className="flex-1 min-w-0">
            <StatusBar trajectory={trajectory} error={error} />
          </div>
        </div>
        {trajectory && (
          <>
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="lg:flex-1 min-w-0 flex flex-col gap-3">
                <TrajectoryView3D trajectory={trajectory} tRef={playback.tRef} />
                <TransportControls playback={playback} />
              </div>
              <div className="lg:w-56 shrink-0">
                <StatsReadout trajectory={trajectory} config={config} t={playback.t} />
              </div>
            </div>
            <ProfilePlots trajectory={trajectory} t={playback.t} />
          </>
        )}
      </main>
    </div>
  );
}
