import type { AppConfig } from "../lib/config";
import { PRESETS, DEFAULT_PRESET } from "../lib/presets";

interface Props {
  config: AppConfig;
  onChange: (next: AppConfig) => void;
  onSolve: () => void;
  solving: boolean;
}

function NumberField({
  label, value, onChange,
}: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="flex flex-col gap-0.5 text-[13px] mb-2">
      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-500">
        {label}
      </span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="px-1.5 py-1 border border-slate-300 rounded bg-white text-slate-800 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/40"
      />
    </label>
  );
}

// Standard gravity in the specific-impulse definition. By convention Isp(s) is
// always referenced to this constant, NOT the local/planetary gravity.
const G0 = 9.80665;

// Edits specific impulse Isp (s) but stores the solver's mass-flow constant
// α = fuel_consumption = 1/(Isp·g₀). Isp is the conventional, intuitive input.
function IspField({
  value, onChange,
}: { value: number; onChange: (alpha: number) => void }) {
  const isp = value > 0 ? 1 / (value * G0) : 0;
  return (
    <label className="flex flex-col gap-0.5 text-[13px] mb-2">
      <span
        title="Specific impulse. Sets the mass-flow constant α = 1/(Isp·g₀) using standard g₀ = 9.80665 m/s² (not the local gravity)."
        className="font-mono text-[10px] tracking-widest uppercase text-slate-500 cursor-help decoration-dotted underline underline-offset-2"
      >
        Specific impulse (s)
      </span>
      <input
        type="number"
        value={Math.round(isp)}
        onChange={(e) => {
          const i = Number(e.target.value);
          onChange(i > 0 ? 1 / (i * G0) : 0);
        }}
        className="px-1.5 py-1 border border-slate-300 rounded bg-white text-slate-800 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/40"
      />
    </label>
  );
}

// Edits a fraction (0..1) but shows/accepts whole percent (0..100), so the label
// "(%)" matches what the user types. The stored config value stays a fraction.
function PercentField({
  label, value, onChange,
}: { label: string; value: number; onChange: (fraction: number) => void }) {
  return (
    <label className="flex flex-col gap-0.5 text-[13px] mb-2">
      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-500">{label} (%)</span>
      <input
        type="number"
        value={Math.round(value * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="px-1.5 py-1 border border-slate-300 rounded bg-white text-slate-800 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/40"
      />
    </label>
  );
}

function VectorField({
  label, value, onChange,
}: { label: string; value: [number, number, number]; onChange: (v: [number, number, number]) => void }) {
  return (
    <div className="flex flex-col gap-0.5 text-[13px] mb-2">
      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-500">{label}</span>
      <div className="grid grid-cols-3 gap-1">
        {value.map((c, i) => (
          <input
            key={i}
            type="number"
            value={c}
            onChange={(e) => {
              const next = [...value] as [number, number, number];
              next[i] = Number(e.target.value);
              onChange(next);
            }}
            className="px-1.5 py-1 border border-slate-300 rounded bg-white text-slate-800 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/40"
          />
        ))}
      </div>
    </div>
  );
}

export default function ConfigForm({ config, onChange, onSolve, solving }: Props) {
  const sc = config.spacecraft;
  const env = config.environment;
  const sv = config.solver;
  const setSc = (patch: Partial<typeof sc>) =>
    onChange({ ...config, spacecraft: { ...sc, ...patch } });
  const setEnv = (patch: Partial<typeof env>) =>
    onChange({ ...config, environment: { ...env, ...patch } });
  const setSv = (patch: Partial<typeof sv>) =>
    onChange({ ...config, solver: { ...sv, ...patch } });

  return (
    <form className="flex flex-col gap-3" onSubmit={(e) => { e.preventDefault(); onSolve(); }}>
      <label className="flex flex-col gap-0.5 text-[13px] mb-2">
        <span className="font-mono text-[10px] tracking-widest uppercase text-slate-500">Preset</span>
        <select
          onChange={(e) => onChange(structuredClone(PRESETS[e.target.value]))}
          defaultValue={DEFAULT_PRESET}
          className="px-1.5 py-1 border border-slate-300 rounded bg-white text-slate-800 focus:outline-none focus:border-amber-400 focus:ring-1 focus:ring-amber-400/40"
        >
          {Object.keys(PRESETS).map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </label>

      <fieldset className="border border-slate-200 rounded-lg p-2">
        <legend className="font-mono text-[10px] tracking-widest uppercase text-slate-500 px-1.5">Spacecraft</legend>
        <NumberField label="Wet mass (kg)" value={sc.wet_mass} onChange={(v) => setSc({ wet_mass: v })} />
        <NumberField label="Fuel (kg)" value={sc.fuel} onChange={(v) => setSc({ fuel: v })} />
        <NumberField label="Max thrust (N)" value={sc.real_max_thrust} onChange={(v) => setSc({ real_max_thrust: v })} />
        <PercentField label="Min thrust" value={sc.min_thrust_pct} onChange={(v) => setSc({ min_thrust_pct: v })} />
        <PercentField label="Max thrust" value={sc.max_thrust_pct} onChange={(v) => setSc({ max_thrust_pct: v })} />
        <NumberField label="Max velocity (m/s)" value={sc.max_velocity} onChange={(v) => setSc({ max_velocity: v })} />
        <IspField value={sc.fuel_consumption} onChange={(a) => setSc({ fuel_consumption: a })} />
        <VectorField label="Initial position" value={sc.initial_position} onChange={(v) => setSc({ initial_position: v })} />
        <VectorField label="Initial velocity" value={sc.initial_velocity} onChange={(v) => setSc({ initial_velocity: v })} />
        <VectorField label="Target position" value={sc.target_position} onChange={(v) => setSc({ target_position: v })} />
        <VectorField label="Target velocity" value={sc.target_velocity} onChange={(v) => setSc({ target_velocity: v })} />
      </fieldset>

      <fieldset className="border border-slate-200 rounded-lg p-2">
        <legend className="font-mono text-[10px] tracking-widest uppercase text-slate-500 px-1.5">Environment</legend>
        <VectorField label="Gravity (m/s²)" value={env.gravity} onChange={(v) => setEnv({ gravity: v })} />
        <NumberField label="Glide slope (deg)" value={env.glide_slope_angle_deg} onChange={(v) => setEnv({ glide_slope_angle_deg: v })} />
        <NumberField label="Max angle (deg)" value={env.max_angle_deg} onChange={(v) => setEnv({ max_angle_deg: v })} />
      </fieldset>

      <fieldset className="border border-slate-200 rounded-lg p-2">
        <legend className="font-mono text-[10px] tracking-widest uppercase text-slate-500 px-1.5">Solver</legend>
        <NumberField label="Steps (n)" value={sv.n} onChange={(v) => setSv({ n: v })} />
        <NumberField
          label="Time of flight (0 = auto)"
          value={sv.time_of_flight ?? 0}
          onChange={(v) => setSv({ time_of_flight: v > 0 ? v : undefined })}
        />
      </fieldset>

      <button
        type="submit"
        disabled={solving}
        className="w-full py-2.5 rounded-md text-white text-[15px] font-semibold bg-amber-500 hover:bg-amber-400 disabled:bg-slate-400 disabled:cursor-default cursor-pointer transition-colors"
      >
        {solving ? "Solving…" : "Solve"}
      </button>
    </form>
  );
}
