import { useState } from "react";
import "./App.css";
import type { AppConfig } from "./lib/config";
import { PRESETS, DEFAULT_PRESET } from "./lib/presets";
import { useSolve } from "./hooks/useSolve";
import ConfigForm from "./components/ConfigForm";
import StatusBar from "./components/StatusBar";
import ProfilePlots from "./components/ProfilePlots";
import TrajectoryView3D from "./components/TrajectoryView3D";

export default function App() {
  const [config, setConfig] = useState<AppConfig>(() => structuredClone(PRESETS[DEFAULT_PRESET]));
  const { trajectory, error, solving, run } = useSolve();

  return (
    <div className="layout">
      <aside className="pane-left">
        <h1>G-FOLD Playground</h1>
        <ConfigForm
          config={config}
          onChange={setConfig}
          onSolve={() => run(config)}
          solving={solving}
        />
      </aside>
      <main className="pane-right">
        <StatusBar trajectory={trajectory} error={error} />
        {trajectory && (
          <>
            <TrajectoryView3D trajectory={trajectory} />
            <ProfilePlots trajectory={trajectory} />
          </>
        )}
      </main>
    </div>
  );
}
