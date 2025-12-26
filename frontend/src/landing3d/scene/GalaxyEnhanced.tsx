import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Color, Object3D, Vector3, AdditiveBlending, Group, InstancedMesh } from 'three';
import type { AssetPoint } from '../lib/data/types';

interface GalaxyProps {
  assets: AssetPoint[];
  snapshot: Record<string, { ret1d: number; vol: number; price: number }>;
  count: number;
  position?: [number, number, number];
}

// Seeded random for deterministic galaxy shape
function seededRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export function Galaxy({ assets, snapshot, count, position = [0, 0, 0] }: GalaxyProps) {
  const groupRef = useRef<Group>(null);
  const instancedRef = useRef<InstancedMesh>(null);
  const glowRef = useRef<InstancedMesh>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const highlightPosition = useMemo(() => new Vector3(), []);
  const baseColor = useMemo(() => new Color(), []);
  const glowColor = useMemo(() => new Color(), []);

  // Generate spiral galaxy positions for synthetic fallback
  const galaxyPositions = useMemo(() => {
    const positions: { x: number; y: number; z: number; arm: number }[] = [];
    const arms = 5;
    const pointsPerArm = Math.ceil(count / arms);

    for (let arm = 0; arm < arms; arm++) {
      const armAngle = (arm / arms) * Math.PI * 2;

      for (let i = 0; i < pointsPerArm; i++) {
        const seed = arm * 10000 + i;
        const t = i / pointsPerArm;
        const spiralAngle = armAngle + t * Math.PI * 3;
        const radius = 2 + t * 14;

        // Add variation
        const rx = (seededRandom(seed * 1) - 0.5) * (1 + t * 3);
        const ry = (seededRandom(seed * 2) - 0.5) * 0.8;
        const rz = (seededRandom(seed * 3) - 0.5) * (1 + t * 3);

        positions.push({
          x: Math.cos(spiralAngle) * radius + rx,
          y: ry + Math.sin(t * Math.PI) * 0.5,
          z: Math.sin(spiralAngle) * radius + rz,
          arm,
        });
      }
    }
    return positions.slice(0, count);
  }, [count]);

  // Merge with asset data
  const points = useMemo(() => {
    return galaxyPositions.map((gp, index) => {
      const asset = assets[index];
      if (asset) {
        return {
          ...asset,
          x: gp.x,
          y: gp.y,
          z: gp.z,
          arm: gp.arm,
        };
      }
      return {
        symbol: `SYN${index}`,
        sector: 'Synthetic',
        marketCap: 50 + seededRandom(index * 100) * 450,
        ret1d: (seededRandom(index * 200) - 0.5) * 0.1,
        vol: 0.2 + seededRandom(index * 300) * 0.3,
        x: gp.x,
        y: gp.y,
        z: gp.z,
        arm: gp.arm,
      };
    });
  }, [galaxyPositions, assets]);

  // Update instances
  useEffect(() => {
    if (!instancedRef.current || !glowRef.current || points.length === 0) return;

    points.forEach((point, index) => {
      tempObject.position.set(point.x, point.y, point.z);
      const size = 0.08 + Math.min(point.marketCap / 1200, 0.28);
      tempObject.scale.set(size, size, size);
      tempObject.updateMatrix();
      instancedRef.current?.setMatrixAt(index, tempObject.matrix);

      // Glow - slightly larger
      tempObject.scale.multiplyScalar(2.2);
      tempObject.updateMatrix();
      glowRef.current?.setMatrixAt(index, tempObject.matrix);

      const ret = snapshot[point.symbol]?.ret1d ?? point.ret1d;
      const isUp = ret >= 0;
      const intensity = Math.min(Math.abs(ret) * 12, 1);

      // Main color
      if (isUp) {
        baseColor.setHSL(0.52 + intensity * 0.08, 0.8, 0.5 + intensity * 0.2);
      } else {
        baseColor.setHSL(0.0 - intensity * 0.02, 0.85, 0.5 + intensity * 0.15);
      }
      instancedRef.current?.setColorAt(index, baseColor);

      // Glow color - more saturated, dimmer
      glowColor.copy(baseColor).multiplyScalar(0.4);
      glowRef.current?.setColorAt(index, glowColor);
    });

    instancedRef.current.instanceMatrix.needsUpdate = true;
    glowRef.current.instanceMatrix.needsUpdate = true;
    if (instancedRef.current.instanceColor) {
      instancedRef.current.instanceColor.needsUpdate = true;
    }
    if (glowRef.current.instanceColor) {
      glowRef.current.instanceColor.needsUpdate = true;
    }
  }, [points, snapshot, tempObject, baseColor, glowColor]);

  const hoveredPoint = hovered !== null ? points[hovered] : null;

  useEffect(() => {
    if (hoveredPoint) {
      highlightPosition.set(hoveredPoint.x, hoveredPoint.y, hoveredPoint.z);
    }
  }, [hoveredPoint, highlightPosition]);

  // Slow rotation
  useFrame((_, delta) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y += delta * 0.015;
  });

  return (
    <group ref={groupRef} position={position}>
      {/* Glow layer behind */}
      <instancedMesh
        ref={glowRef}
        args={[undefined, undefined, points.length]}
      >
        <sphereGeometry args={[0.15, 8, 8]} />
        <meshBasicMaterial
          transparent
          opacity={0.15}
          blending={AdditiveBlending}
          depthWrite={false}
          vertexColors
        />
      </instancedMesh>

      {/* Main orbs */}
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
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          emissive="#1a4666"
          emissiveIntensity={1.2}
          color="#0d1c2a"
          roughness={0.2}
          metalness={0.6}
          vertexColors
        />
      </instancedMesh>

      {/* Highlight on hover */}
      {hoveredPoint && (
        <group>
          <mesh position={highlightPosition}>
            <sphereGeometry args={[0.4, 24, 24]} />
            <meshStandardMaterial
              emissive="#7efcff"
              emissiveIntensity={2}
              color="#0b2333"
              roughness={0.1}
              metalness={0.3}
              transparent
              opacity={0.95}
            />
          </mesh>
          {/* Outer glow ring */}
          <mesh position={highlightPosition}>
            <sphereGeometry args={[0.55, 16, 16]} />
            <meshBasicMaterial
              color="#5ef8ff"
              transparent
              opacity={0.2}
              blending={AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
          <Html
            position={[highlightPosition.x + 0.8, highlightPosition.y + 0.5, highlightPosition.z]}
            distanceFactor={8}
            portal={{ current: document.body }}
            style={{ pointerEvents: 'none' }}
          >
            <div className="landing3d-tooltip landing3d-tooltip-glow">
              <div className="landing3d-tooltip-title">{hoveredPoint.symbol}</div>
              <div className="landing3d-tooltip-row">{hoveredPoint.sector}</div>
              <div className="landing3d-tooltip-row">
                ${hoveredPoint.marketCap.toFixed(0)}B market cap
              </div>
              <div
                className={`landing3d-tooltip-row ${
                  (snapshot[hoveredPoint.symbol]?.ret1d ?? hoveredPoint.ret1d) >= 0
                    ? 'landing3d-positive'
                    : 'landing3d-negative'
                }`}
              >
                {(snapshot[hoveredPoint.symbol]?.ret1d ?? hoveredPoint.ret1d) >= 0 ? '▲' : '▼'}
                {' '}
                {Math.abs((snapshot[hoveredPoint.symbol]?.ret1d ?? hoveredPoint.ret1d) * 100).toFixed(2)}% today
              </div>
            </div>
          </Html>
        </group>
      )}

      {/* Central core glow */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[2.5, 32, 32]} />
        <meshBasicMaterial
          color="#1a3a6a"
          transparent
          opacity={0.12}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[1.2, 24, 24]} />
        <meshBasicMaterial
          color="#4488cc"
          transparent
          opacity={0.18}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Ambient background sphere */}
      <mesh rotation={[0, 0, 0]} position={[0, 0, 0]}>
        <sphereGeometry args={[22, 24, 24]} />
        <meshBasicMaterial
          color="#0b1020"
          transparent
          opacity={0.05}
          side={1}
        />
      </mesh>
    </group>
  );
}
