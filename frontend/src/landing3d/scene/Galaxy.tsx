import { useEffect, useMemo, useRef, useState } from 'react';
import { Html } from '@react-three/drei';
import { Color, Object3D, Vector3 } from 'three';
import type { AssetPoint } from '../lib/data/types';

interface GalaxyProps {
  assets: AssetPoint[];
  snapshot: Record<string, { ret1d: number; vol: number; price: number }>;
  count: number;
}

export function Galaxy({ assets, snapshot, count }: GalaxyProps) {
  const instancedRef = useRef<THREE.InstancedMesh>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const highlightPosition = useMemo(() => new Vector3(), []);
  const baseColor = useMemo(() => new Color(), []);

  const points = useMemo(() => assets.slice(0, count), [assets, count]);

  useEffect(() => {
    if (!instancedRef.current || points.length === 0) return;
    points.forEach((point, index) => {
      tempObject.position.set(point.x, point.y, point.z);
      const size = 0.12 + Math.min(point.marketCap / 900, 0.25);
      tempObject.scale.set(size, size, size);
      tempObject.updateMatrix();
      instancedRef.current?.setMatrixAt(index, tempObject.matrix);

      const ret = snapshot[point.symbol]?.ret1d ?? point.ret1d;
      const color = ret >= 0 ? '#5fd1ff' : '#ff7b7b';
      baseColor.set(color).multiplyScalar(0.9 + Math.min(Math.abs(ret) * 8, 0.4));
      instancedRef.current?.setColorAt(index, baseColor);
    });
    instancedRef.current.instanceMatrix.needsUpdate = true;
    if (instancedRef.current.instanceColor) {
      instancedRef.current.instanceColor.needsUpdate = true;
    }
  }, [points, snapshot, tempObject, baseColor]);

  const hoveredPoint = hovered !== null ? points[hovered] : null;

  useEffect(() => {
    if (hoveredPoint) {
      highlightPosition.set(hoveredPoint.x, hoveredPoint.y, hoveredPoint.z);
    }
  }, [hoveredPoint, highlightPosition]);

  return (
    <group>
      <instancedMesh
        ref={instancedRef}
        args={[undefined, undefined, points.length]}
        onPointerMove={(event) => {
          if (event.instanceId === undefined) return;
          event.stopPropagation();
          setHovered(event.instanceId);
        }}
        onPointerOut={() => setHovered(null)}
      >
        <sphereGeometry args={[0.22, 18, 18]} />
        <meshStandardMaterial emissive="#1a4666" color="#0d1c2a" roughness={0.35} metalness={0.3} />
      </instancedMesh>

      {hoveredPoint && (
        <group>
          <mesh position={highlightPosition}>
            <sphereGeometry args={[0.38, 24, 24]} />
            <meshStandardMaterial
              emissive="#7efcff"
              color="#0b2333"
              roughness={0.1}
              metalness={0.2}
              transparent
              opacity={0.9}
            />
          </mesh>
          <Html
            position={[highlightPosition.x + 0.6, highlightPosition.y + 0.35, highlightPosition.z]}
            distanceFactor={10}
          >
            <div className="landing3d-tooltip">
              <div className="landing3d-tooltip-title">{hoveredPoint.symbol}</div>
              <div className="landing3d-tooltip-row">{hoveredPoint.sector}</div>
              <div className="landing3d-tooltip-row">
                ${hoveredPoint.marketCap.toFixed(0)}B cap
              </div>
              <div className="landing3d-tooltip-row">
                {(snapshot[hoveredPoint.symbol]?.ret1d ?? hoveredPoint.ret1d) >= 0 ? '+' : ''}
                {((snapshot[hoveredPoint.symbol]?.ret1d ?? hoveredPoint.ret1d) * 100).toFixed(2)}% 1D
              </div>
            </div>
          </Html>
        </group>
      )}

      <mesh rotation={[0, 0, 0]} position={[0, 0, 0]}>
        <sphereGeometry args={[20, 24, 24]} />
        <meshBasicMaterial color="#0b1020" transparent opacity={0.08} />
      </mesh>
    </group>
  );
}
