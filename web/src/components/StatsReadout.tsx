import type { Trajectory } from "../wasm/init";
import type { AppConfig } from "../lib/config";
import { sampleFlightStats } from "../lib/flightStats";
import ArcGauge from "./instruments/ArcGauge";
import TiltDial from "./instruments/TiltDial";
import FuelGauge from "./instruments/FuelGauge";
import DigitalReadout from "./instruments/DigitalReadout";

interface Props {
  trajectory: Trajectory;
  config: AppConfig;
  t: number;
}

export default function StatsReadout({ trajectory, config, t }: Props) {
  const s = sampleFlightStats(trajectory, config, t);
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800 p-3 h-full flex flex-wrap content-start items-center justify-center gap-x-3 gap-y-4">
      <DigitalReadout label="Altitude" value={s.altitude.toFixed(0)} unit="m" big />
      <ArcGauge label="Speed" value={s.speed} min={0} max={Math.max(100, s.speed)} unit="m/s" />
      <DigitalReadout label="Descent" value={s.descentRate.toFixed(1)} unit="m/s" />
      <DigitalReadout label="Downrange" value={s.downrange.toFixed(0)} unit="m" />
      <ArcGauge label="Throttle" value={s.throttlePct} min={0} max={100} unit="%" color="#f59e0b" />
      <ArcGauge label="Thrust" value={s.thrustKN} min={0} max={Math.max(1, config.spacecraft.real_max_thrust / 1000)} unit="kN" />
      <TiltDial angleDeg={s.gimbalDeg} />
      <FuelGauge remaining={s.fuelRemaining} capacity={config.spacecraft.fuel} />
    </div>
  );
}
