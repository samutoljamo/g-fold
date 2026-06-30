import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, ReferenceDot } from "recharts";
import type { Trajectory } from "../wasm/init";
import { buildProfiles, interpolateProfile } from "../lib/plotData";

interface Props {
  trajectory: Trajectory;
  /** Current playback time (seconds); drives the cursor across all charts. */
  t: number;
}

const CHARTS: { key: "altitude" | "speed" | "thrustPct" | "mass"; label: string; color: string }[] = [
  { key: "altitude", label: "Altitude (m)", color: "#f59e0b" },
  { key: "speed", label: "Speed (m/s)", color: "#38bdf8" },
  { key: "thrustPct", label: "Thrust (%)", color: "#a78bfa" },
  { key: "mass", label: "Mass (kg)", color: "#34d399" },
];

export default function ProfilePlots({ trajectory, t }: Props) {
  const data = buildProfiles(trajectory);
  const cursor = interpolateProfile(data, t);
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {CHARTS.map((c) => (
        <div key={c.key} className="border border-slate-700 rounded-lg p-2 bg-slate-800">
          <h4 className="m-0 mb-1 text-[13px] font-mono tracking-widest uppercase text-slate-400">{c.label}</h4>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(0)} unit="s" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis width={48} stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", fontSize: "12px" }}
                labelStyle={{ color: "#94a3b8" }}
                itemStyle={{ color: c.color }}
              />
              <Line type="monotone" dataKey={c.key} stroke={c.color} dot={false} isAnimationActive={false} strokeWidth={1.5} />
              {/* recharts detects these by component type, so they must be
                  direct children of LineChart — not wrapped in a Fragment. */}
              {cursor && <ReferenceLine x={cursor.t} stroke="#e2e8f0" strokeOpacity={0.6} strokeDasharray="2 2" />}
              {cursor && <ReferenceDot x={cursor.t} y={cursor[c.key]} r={3.5} fill={c.color} stroke="#0f172a" strokeWidth={1.5} isFront />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
