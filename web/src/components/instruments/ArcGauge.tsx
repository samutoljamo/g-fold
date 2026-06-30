interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  unit?: string;
  color?: string;
}

const START = 150; // degrees
const SWEEP = 240;
const polar = (cx: number, cy: number, r: number, deg: number) => {
  const a = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
};
const arcPath = (cx: number, cy: number, r: number, from: number, to: number) => {
  const [x1, y1] = polar(cx, cy, r, from);
  const [x2, y2] = polar(cx, cy, r, to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
};

export default function ArcGauge({ label, value, min, max, unit, color = "#34d399" }: Props) {
  const frac = Math.min(1, Math.max(0, (value - min) / (max - min || 1)));
  const end = START + SWEEP * frac;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 80" className="w-24 h-20">
        <path d={arcPath(50, 50, 38, START, START + SWEEP)} fill="none" stroke="#334155" strokeWidth={8} strokeLinecap="round" />
        <path d={arcPath(50, 50, 38, START, end)} fill="none" stroke={color} strokeWidth={8} strokeLinecap="round" />
        <text x={50} y={52} textAnchor="middle" className="fill-slate-100 font-mono" fontSize={14}>
          {value.toFixed(value >= 100 ? 0 : 1)}
        </text>
      </svg>
      <span className="text-[10px] uppercase tracking-wide text-slate-400">
        {label}{unit ? ` (${unit})` : ""}
      </span>
    </div>
  );
}
