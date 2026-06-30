import type { Playback } from "../hooks/usePlayback";

interface Props {
  playback: Playback;
}

const SPEEDS = [0.5, 1, 2];

export default function TransportControls({ playback }: Props) {
  const { t, duration, playing, speed, toggle, setSpeed, seek } = playback;
  return (
    <div className="flex items-center gap-3 text-slate-100">
      <button
        type="button"
        onClick={toggle}
        className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 w-16"
      >
        {playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        min={0}
        max={duration}
        step={duration / 500 || 0.01}
        value={t}
        onChange={(e) => seek(Number(e.target.value))}
        className="flex-1 accent-amber-400"
      />
      <span className="font-mono text-sm tabular-nums w-20 text-right">
        {t.toFixed(1)}/{duration.toFixed(1)}s
      </span>
      <div className="flex gap-1">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSpeed(s)}
            className={`px-2 py-1 rounded text-sm ${speed === s ? "bg-amber-400 text-slate-900" : "bg-slate-700"}`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
