interface Props {
  label: string;
  value: string;
  unit?: string;
  big?: boolean;
}

export default function DigitalReadout({ label, value, unit, big }: Props) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-[10px] uppercase tracking-wide text-slate-400">{label}</span>
      <span className={`font-mono tabular-nums text-emerald-300 ${big ? "text-2xl" : "text-base"}`}>
        {value}
        {unit && <span className="text-slate-400 text-xs ml-1">{unit}</span>}
      </span>
    </div>
  );
}
