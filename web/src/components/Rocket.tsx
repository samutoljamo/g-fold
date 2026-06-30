import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import { Group, Vector3, Quaternion } from "three";
import type { Trajectory } from "../wasm/init";
import { interpolateState } from "../lib/flightStats";

// Minimal structural shape of an OrbitControls instance we drive for follow-cam.
interface FollowControls {
  target: Vector3;
  update: () => void;
}

interface Props {
  trajectory: Trajectory;
  scale: number;            // scene normalization factor (1/span)
  tRef: React.MutableRefObject<number>;
  controlsRef?: React.RefObject<FollowControls | null>;
}

const UP = new Vector3(0, 1, 0);

export default function Rocket({ trajectory, scale, tRef, controlsRef }: Props) {
  const group = useRef<Group>(null);
  const plume = useRef<Group>(null);
  const tmpDir = useRef(new Vector3());
  const tmpQuat = useRef(new Quaternion());

  useFrame(() => {
    const g = group.current;
    if (!g) return;
    const s = interpolateState(trajectory, tRef.current);
    // position: solver Z-up -> three Y-up, scaled to unit scene
    g.position.set(s.position[0] * scale, s.position[2] * scale, s.position[1] * scale);
    // orient body +Y along thrust direction (also remapped z<->y). Skip on a
    // (near-)zero thrust vector — setFromUnitVectors on a zero vector yields a
    // NaN quaternion; keep the last good orientation instead.
    tmpDir.current.set(s.thrustDir[0], s.thrustDir[2], s.thrustDir[1]);
    if (tmpDir.current.lengthSq() > 1e-10) {
      tmpDir.current.normalize();
      tmpQuat.current.setFromUnitVectors(UP, tmpDir.current);
      g.quaternion.copy(tmpQuat.current);
    }
    // plume length/visibility from throttle
    const p = plume.current;
    if (p) {
      const throttle = s.throttlePct / 100;
      p.visible = throttle > 0.02;
      p.scale.setScalar(0.5 + throttle); // grows with throttle
    }
    // follow-cam: ease the orbit target toward the rocket so it stays centered
    // while the user can still orbit/zoom freely around it.
    const c = controlsRef?.current;
    if (c) {
      c.target.lerp(g.position, 0.1);
      c.update();
    }
  });

  // Sizes are in normalized scene units (scene ~[-1,1]); constants chosen so the
  // rocket reads as a small craft regardless of trajectory scale.
  const R = 0.02;
  return (
    <group ref={group}>
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[R, R, R * 5, 16]} />
        <meshStandardMaterial color="#e2e8f0" metalness={0.3} roughness={0.4} />
      </mesh>
      <mesh position={[0, R * 3.5, 0]}>
        <coneGeometry args={[R, R * 2, 16]} />
        <meshStandardMaterial color="#cbd5e1" metalness={0.3} roughness={0.4} />
      </mesh>
      <mesh position={[0, -R * 3, 0]}>
        <coneGeometry args={[R * 1.2, R * 1.5, 16]} />
        <meshStandardMaterial color="#475569" metalness={0.5} roughness={0.3} />
      </mesh>
      <group ref={plume} position={[0, -R * 4, 0]}>
        <mesh position={[0, -R * 2.5, 0]} rotation={[Math.PI, 0, 0]}>
          <coneGeometry args={[R * 1.1, R * 5, 16, 1, true]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.7} />
        </mesh>
      </group>
    </group>
  );
}
