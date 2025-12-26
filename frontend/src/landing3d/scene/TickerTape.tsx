import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { Color, Matrix4, Object3D, Vector3 } from 'three';
import type { QualitySettings } from '../lib/perf/quality';

interface TickerTapeProps {
  count: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
}

export function TickerTape({ count, visibilityRef, quality }: TickerTapeProps) {
  const baseRef = useRef<THREE.InstancedMesh>(null);
  const accentRef = useRef<THREE.InstancedMesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempMatrix = useMemo(() => new Matrix4(), []);
  const tempColor = useMemo(() => new Color(), []);

  const tickerData = useMemo(() => {
    const rows = 6;
    const items: { base: Vector3; speed: number; scale: number; color: string }[] = [];
    for (let i = 0; i < count; i += 1) {
      const row = i % rows;
      const offset = Math.floor(i / rows);
      const x = -18 + (offset % 12) * 3.2;
      const y = 1.6 - row * 0.52;
      const z = 2 + (offset % 3) * -1.6;
      const speed = 0.6 + (i % 7) * 0.08;
      const scale = 0.9 + (i % 5) * 0.08;
      const color = i % 4 === 0 ? '#4d8cff' : '#18304f';
      items.push({ base: new Vector3(x, y, z), speed, scale, color });
    }
    return items;
  }, [count]);

  useEffect(() => {
    if (!baseRef.current || !accentRef.current) return;
    tickerData.forEach((item, index) => {
      tempColor.set(item.color);
      baseRef.current?.setColorAt(index, tempColor);
      tempColor.set('#2953a8');
      accentRef.current?.setColorAt(index, tempColor);
    });
    if (baseRef.current.instanceColor) {
      baseRef.current.instanceColor.needsUpdate = true;
    }
    if (accentRef.current.instanceColor) {
      accentRef.current.instanceColor.needsUpdate = true;
    }
  }, [tickerData, tempColor]);

  useFrame((state) => {
    if (!visibilityRef.current) return;
    const motion = quality.motionScale;

    tickerData.forEach((item, index) => {
      const move = ((state.clock.elapsedTime * item.speed * motion) % 24) - 12;
      tempObject.position.set(item.base.x + move, item.base.y, item.base.z);
      tempObject.scale.set(item.scale, 0.12, 1);
      tempObject.updateMatrix();
      tempMatrix.copy(tempObject.matrix);
      baseRef.current?.setMatrixAt(index, tempMatrix);

      tempObject.scale.set(item.scale * 0.6, 0.04, 1);
      tempObject.position.set(item.base.x + move + 0.4, item.base.y + 0.12, item.base.z + 0.01);
      tempObject.updateMatrix();
      tempMatrix.copy(tempObject.matrix);
      accentRef.current?.setMatrixAt(index, tempMatrix);
    });

    if (baseRef.current) {
      baseRef.current.instanceMatrix.needsUpdate = true;
    }
    if (accentRef.current) {
      accentRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <group>
      <instancedMesh ref={baseRef} args={[undefined, undefined, tickerData.length]}>
        <planeGeometry args={[2.4, 0.18]} />
        <meshStandardMaterial
          emissive="#1c3d7a"
          color="#0b1626"
          roughness={0.4}
          metalness={0.2}
          vertexColors
        />
      </instancedMesh>

      <instancedMesh ref={accentRef} args={[undefined, undefined, tickerData.length]}>
        <planeGeometry args={[1.1, 0.06]} />
        <meshStandardMaterial
          emissive="#3b6cff"
          color="#16253b"
          roughness={0.2}
          metalness={0.4}
          vertexColors
        />
      </instancedMesh>

      <group position={[-3.5, 2.2, 0]}>
        <Text fontSize={0.4} color="#d6e4ff" anchorX="left" anchorY="middle">
          AURX +1.24%
        </Text>
        <Text fontSize={0.32} color="#8ab6ff" anchorX="left" anchorY="middle" position={[0, -0.6, 0]}>
          NVLX 182.32
        </Text>
        <Text fontSize={0.32} color="#ff8c8c" anchorX="left" anchorY="middle" position={[0, -1.1, 0]}>
          SYNR -0.92%
        </Text>
      </group>

      <mesh position={[0, 0.1, -1]}>
        <planeGeometry args={[22, 4]} />
        <meshStandardMaterial color="#060b14" emissive="#0a1220" opacity={0.65} transparent />
      </mesh>
    </group>
  );
}
