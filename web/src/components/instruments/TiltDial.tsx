interface Props {
  angleDeg: number; // 0 = thrust straight up
}

export default function TiltDial({ angleDeg }: Props) {
  const clamped = Math.min(45, Math.max(0, angleDeg));
  const needle = -90 + clamped; // rotate from vertical
  const a = (needle * Math.PI) / 180;
  const x = 50 + 32 * Math.cos(a);
  const y = 50 + 32 * Math.sin(a);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 80" className="w-24 h-20">
        <circle cx={50} cy={50} r={36} fill="#0f172a" stroke="#334155" strokeWidth={2} />
        <line x1={50} y1={14} x2={50} y2={20} stroke="#475569" strokeWidth={2} />
        <line x1={50} y1={50} x2={x} y2={y} stroke="#f59e0b" strokeWidth={3} strokeLinecap="round" />
        <circle cx={50} cy={50} r={3} fill="#f59e0b" />
        <text x={50} y={70} textAnchor="middle" className="fill-slate-100 font-mono" fontSize={12}>
          {angleDeg.toFixed(0)}°
        </text>
      </svg>
      <span className="text-[10px] uppercase tracking-wide text-slate-400">Gimbal</span>
    </div>
  );
}
