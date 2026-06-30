import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, Grid } from "@react-three/drei";
import { useMemo, useRef } from "react";
import type { Trajectory } from "../wasm/init";
import { buildPath3D } from "../lib/plotData";
import Rocket from "./Rocket";

interface Props {
  trajectory: Trajectory;
  tRef: React.MutableRefObject<number>;
}

type Vec3 = [number, number, number];

export default function TrajectoryView3D({ trajectory, tRef }: Props) {
  // Largest absolute coordinate sets the scene scale. Normalize so the whole
  // trajectory lives in ~[-1,1]; this is the real flicker fix — a tight depth
  // range (near 0.01 / far 100) instead of a 2000:1 range over kilometres.
  const { points, scale } = useMemo(() => {
    const raw = buildPath3D(trajectory);
    const span = Math.max(...raw.flatMap((p) => [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)]), 1);
    const s = 1 / span;
    // Solver Z is up; three.js Y is up — swap z<->y, then scale to unit box.
    const pts: Vec3[] = raw.map((p) => [p.x * s, p.z * s, p.y * s]);
    return { points: pts, scale: s };
  }, [trajectory]);

  const controlsRef = useRef<React.ElementRef<typeof OrbitControls>>(null);

  const start = points[0];
  const end = points[points.length - 1];
  if (!start || !end) return null;

  return (
    <div className="h-[58vh] min-h-[340px] rounded-lg overflow-hidden border border-slate-700 bg-slate-950">
      <Canvas
        camera={{ position: [1.6, 1.4, 1.6], fov: 50, near: 0.01, far: 100 }}
        gl={{ antialias: true, logarithmicDepthBuffer: true }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 10, 5]} />
        <Grid args={[4, 4]} cellSize={0.2} sectionSize={1} fadeDistance={8} cellColor="#334155" sectionColor="#475569" />
        <Line points={points} color="#3b82f6" lineWidth={2} />
        <Rocket trajectory={trajectory} scale={scale} tRef={tRef} controlsRef={controlsRef} />
        <mesh position={start}>
          <sphereGeometry args={[scale * 30, 16, 16]} />
          <meshStandardMaterial color="#10b981" />
        </mesh>
        <mesh position={end}>
          <sphereGeometry args={[scale * 30, 16, 16]} />
          <meshStandardMaterial color="#ef4444" />
        </mesh>
        <OrbitControls ref={controlsRef} makeDefault />
      </Canvas>
    </div>
  );
}
