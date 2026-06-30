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
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function VectorField({
  label, value, onChange,
}: { label: string; value: [number, number, number]; onChange: (v: [number, number, number]) => void }) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="vec">
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
    <form className="config-form" onSubmit={(e) => { e.preventDefault(); onSolve(); }}>
      <label className="field">
        <span>Preset</span>
        <select
          onChange={(e) => onChange(structuredClone(PRESETS[e.target.value]))}
          defaultValue={DEFAULT_PRESET}
        >
          {Object.keys(PRESETS).map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </label>

      <fieldset>
        <legend>Spacecraft</legend>
        <NumberField label="Wet mass (kg)" value={sc.wet_mass} onChange={(v) => setSc({ wet_mass: v })} />
        <NumberField label="Fuel (kg)" value={sc.fuel} onChange={(v) => setSc({ fuel: v })} />
        <NumberField label="Max thrust (N)" value={sc.real_max_thrust} onChange={(v) => setSc({ real_max_thrust: v })} />
        <NumberField label="Min thrust %" value={sc.min_thrust_pct} onChange={(v) => setSc({ min_thrust_pct: v })} />
        <NumberField label="Max thrust %" value={sc.max_thrust_pct} onChange={(v) => setSc({ max_thrust_pct: v })} />
        <NumberField label="Max velocity (m/s)" value={sc.max_velocity} onChange={(v) => setSc({ max_velocity: v })} />
        <NumberField label="Fuel consumption" value={sc.fuel_consumption} onChange={(v) => setSc({ fuel_consumption: v })} />
        <VectorField label="Initial position" value={sc.initial_position} onChange={(v) => setSc({ initial_position: v })} />
        <VectorField label="Initial velocity" value={sc.initial_velocity} onChange={(v) => setSc({ initial_velocity: v })} />
        <VectorField label="Target position" value={sc.target_position} onChange={(v) => setSc({ target_position: v })} />
        <VectorField label="Target velocity" value={sc.target_velocity} onChange={(v) => setSc({ target_velocity: v })} />
      </fieldset>

      <fieldset>
        <legend>Environment</legend>
        <VectorField label="Gravity (m/s²)" value={env.gravity} onChange={(v) => setEnv({ gravity: v })} />
        <NumberField label="Glide slope (deg)" value={env.glide_slope_angle_deg} onChange={(v) => setEnv({ glide_slope_angle_deg: v })} />
        <NumberField label="Max angle (deg)" value={env.max_angle_deg} onChange={(v) => setEnv({ max_angle_deg: v })} />
      </fieldset>

      <fieldset>
        <legend>Solver</legend>
        <NumberField label="Steps (n)" value={sv.n} onChange={(v) => setSv({ n: v })} />
        <NumberField
          label="Time of flight (0 = auto)"
          value={sv.time_of_flight ?? 0}
          onChange={(v) => setSv({ time_of_flight: v > 0 ? v : undefined })}
        />
      </fieldset>

      <button type="submit" disabled={solving}>
        {solving ? "Solving…" : "Solve"}
      </button>
    </form>
  );
}
