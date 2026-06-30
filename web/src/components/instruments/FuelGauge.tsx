interface Props {
  remaining: number;
  capacity: number;
}

export default function FuelGauge({ remaining, capacity }: Props) {
  const frac = Math.min(1, Math.max(0, capacity > 0 ? remaining / capacity : 0));
  const color = frac > 0.5 ? "bg-emerald-400" : frac > 0.2 ? "bg-amber-400" : "bg-red-500";
  return (
    <div className="flex flex-col items-center">
      <div className="w-6 h-20 bg-slate-800 rounded border border-slate-600 flex items-end overflow-hidden">
        <div className={`w-full ${color}`} style={{ height: `${frac * 100}%` }} />
      </div>
      <span className="text-[10px] uppercase tracking-wide text-slate-400 mt-1">Fuel</span>
      <span className="font-mono text-emerald-300 text-xs">{remaining.toFixed(0)}</span>
    </div>
  );
}
