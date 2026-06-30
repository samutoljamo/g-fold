import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, Grid } from "@react-three/drei";
import { useMemo } from "react";
import type { Trajectory } from "../wasm/init";
import { buildPath3D } from "../lib/plotData";

interface Props {
  trajectory: Trajectory;
}

type Vec3 = [number, number, number];

export default function TrajectoryView3D({ trajectory }: Props) {
  const points = useMemo<Vec3[]>(
    () => buildPath3D(trajectory).map((p) => [p.x, p.z, p.y]),
    [trajectory],
  );
  const start = points[0];
  const end = points[points.length - 1];
  // Scale camera distance to trajectory extent.
  const span = Math.max(...points.flatMap((p) => p.map(Math.abs)), 1);

  if (!start || !end) return null;

  return (
    <div className="view3d">
      {/* near/far must scale with the trajectory: the camera sits ~1.7*span from
          the origin and the grid reaches several span out, so the R3F default
          far of 1000 would clip the entire scene for km-scale descents. */}
      <Canvas camera={{ position: [span, span, span], fov: 50, near: span / 100, far: span * 20 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[10, 20, 10]} />
        <Grid args={[span * 4, span * 4]} cellSize={span / 5} infiniteGrid fadeDistance={span * 6} />
        <Line points={points} color="#3b82f6" lineWidth={2} />
        <mesh position={start}>
          <sphereGeometry args={[span / 40, 16, 16]} />
          <meshStandardMaterial color="#10b981" />
        </mesh>
        <mesh position={end}>
          <sphereGeometry args={[span / 40, 16, 16]} />
          <meshStandardMaterial color="#ef4444" />
        </mesh>
        <OrbitControls />
      </Canvas>
    </div>
  );
}
