import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { Trajectory } from "../wasm/init";
import { buildProfiles } from "../lib/plotData";

interface Props {
  trajectory: Trajectory;
}

const CHARTS: { key: "altitude" | "speed" | "thrustPct" | "mass"; label: string; color: string }[] = [
  { key: "altitude", label: "Altitude (m)", color: "#3b82f6" },
  { key: "speed", label: "Speed (m/s)", color: "#10b981" },
  { key: "thrustPct", label: "Thrust (%)", color: "#f59e0b" },
  { key: "mass", label: "Mass (kg)", color: "#ef4444" },
];

export default function ProfilePlots({ trajectory }: Props) {
  const data = buildProfiles(trajectory);
  return (
    <div className="profile-plots">
      {CHARTS.map((c) => (
        <div key={c.key} className="plot">
          <h4>{c.label}</h4>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" tickFormatter={(v) => v.toFixed(0)} unit="s" />
              <YAxis width={48} />
              <Tooltip />
              <Line type="monotone" dataKey={c.key} stroke={c.color} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
