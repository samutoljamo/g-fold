interface Props {
  angleDeg: number; // 0 = thrust straight up
}

export default function TiltDial({ angleDeg }: Props) {
  // Thrust tilt from vertical spans 0..180° (0 = straight up, 90 = horizontal,
  // 180 = straight down). Map across the right half-dial: up → right → down.
  const clamped = Math.min(180, Math.max(0, angleDeg));
  const needle = -90 + clamped;
  const a = (needle * Math.PI) / 180;
  const x = 50 + 30 * Math.cos(a);
  const y = 50 + 30 * Math.sin(a);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 80" className="w-24 h-20">
        <circle cx={50} cy={50} r={34} fill="#0f172a" stroke="#334155" strokeWidth={2} />
        {/* reference ticks: up=0°, right=90° */}
        <line x1={50} y1={16} x2={50} y2={22} stroke="#475569" strokeWidth={2} />
        <line x1={78} y1={50} x2={84} y2={50} stroke="#475569" strokeWidth={2} />
        <line x1={50} y1={50} x2={x} y2={y} stroke="#f59e0b" strokeWidth={3} strokeLinecap="round" />
        <circle cx={50} cy={50} r={3} fill="#f59e0b" />
      </svg>
      <span className="text-[10px] uppercase tracking-wide text-slate-400">
        Gimbal <span className="font-mono text-slate-200">{angleDeg.toFixed(0)}°</span>
      </span>
    </div>
  );
}
