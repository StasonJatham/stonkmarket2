import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { Color, Object3D, InstancedMesh, MathUtils } from 'three';
import type { QualitySettings } from '../lib/perf/quality';
import { getThemeColors } from '../lib/useThemeColors';

interface MarketCityProps {
  buildingCount?: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
  colorblindMode?: boolean;
  customColors?: { up: string; down: string };
}

// Seeded random for deterministic placement
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + seed * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

// Sector data for building clusters
const SECTORS = [
  { name: 'Tech', x: -8, z: 0, count: 12, avgHeight: 1.2 },
  { name: 'Finance', x: 0, z: -3, count: 15, avgHeight: 1.0 },
  { name: 'Healthcare', x: 8, z: 1, count: 10, avgHeight: 0.9 },
  { name: 'Energy', x: -6, z: 4, count: 8, avgHeight: 0.7 },
  { name: 'Consumer', x: 5, z: 5, count: 10, avgHeight: 0.8 },
];

export function MarketCity({
  buildingCount = 60,
  visibilityRef,
  quality,
  colorblindMode = false,
  customColors,
}: MarketCityProps) {
  const meshRef = useRef<InstancedMesh>(null);
  const glowRef = useRef<InstancedMesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempColor = useMemo(() => new Color(), []);
  
  const colors = useMemo(
    () => getThemeColors(colorblindMode, customColors),
    [colorblindMode, customColors]
  );

  // Generate building data
  const buildings = useMemo(() => {
    const result: Array<{
      x: number;
      z: number;
      width: number;
      depth: number;
      height: number;
      targetHeight: number;
      sector: string;
      performance: number; // -1 to 1, affects color
    }> = [];

    let idx = 0;
    for (const sector of SECTORS) {
      const sectorBuildings = Math.min(sector.count, Math.floor(buildingCount / SECTORS.length) + 2);
      
      for (let i = 0; i < sectorBuildings; i++) {
        const seed = idx * 123 + i * 456;
        const angle = seededRandom(seed) * Math.PI * 2;
        const radius = seededRandom(seed + 1) * 3 + 0.5;
        
        const performance = (seededRandom(seed + 2) - 0.5) * 2; // -1 to 1
        const heightVariance = 0.5 + seededRandom(seed + 3) * 1.5;
        
        result.push({
          x: sector.x + Math.cos(angle) * radius,
          z: sector.z + Math.sin(angle) * radius,
          width: 0.3 + seededRandom(seed + 4) * 0.4,
          depth: 0.3 + seededRandom(seed + 5) * 0.4,
          height: 0, // Start at 0, animate to target
          targetHeight: sector.avgHeight * heightVariance * 3,
          sector: sector.name,
          performance,
        });
        idx++;
      }
    }
    
    return result;
  }, [buildingCount]);

  // Set colors once
  useEffect(() => {
    if (!meshRef.current || !glowRef.current) return;

    buildings.forEach((building, index) => {
      // Main building color - grayscale with slight tint based on performance
      const baseGray = 0.15 + building.performance * 0.05;
      if (building.performance >= 0) {
        tempColor.setRGB(
          baseGray,
          baseGray + building.performance * 0.1,
          baseGray
        );
      } else {
        tempColor.setRGB(
          baseGray - building.performance * 0.1,
          baseGray,
          baseGray
        );
      }
      meshRef.current?.setColorAt(index, tempColor);

      // Glow color based on performance
      if (building.performance >= 0) {
        tempColor.copy(colors.positiveColor).multiplyScalar(0.5 + building.performance * 0.5);
      } else {
        tempColor.copy(colors.negativeColor).multiplyScalar(0.5 - building.performance * 0.5);
      }
      glowRef.current?.setColorAt(index, tempColor);
    });

    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
    if (glowRef.current.instanceColor) glowRef.current.instanceColor.needsUpdate = true;
  }, [buildings, colors, tempColor]);

  useFrame((state) => {
    if (!visibilityRef.current || !meshRef.current || !glowRef.current) return;

    const time = state.clock.elapsedTime;

    buildings.forEach((building, index) => {
      // Animate height growth
      building.height = MathUtils.lerp(
        building.height,
        building.targetHeight,
        0.02 * quality.motionScale
      );

      // Subtle height pulsing based on performance
      const pulse = Math.sin(time * 0.5 + index * 0.3) * 0.05 * Math.abs(building.performance);
      const currentHeight = building.height * (1 + pulse);

      // Main building
      tempObject.position.set(building.x, currentHeight / 2, building.z);
      tempObject.scale.set(building.width, currentHeight, building.depth);
      tempObject.updateMatrix();
      meshRef.current?.setMatrixAt(index, tempObject.matrix);

      // Glow strip at top
      tempObject.position.set(building.x, currentHeight - 0.1, building.z);
      tempObject.scale.set(building.width * 1.1, 0.15, building.depth * 1.1);
      tempObject.updateMatrix();
      glowRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    glowRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <group>
      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
        <planeGeometry args={[30, 20]} />
        <meshStandardMaterial
          color="#080a0f"
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Grid lines on ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.04, 0]}>
        <planeGeometry args={[30, 20, 30, 20]} />
        <meshBasicMaterial
          color="#1a2030"
          wireframe
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Buildings */}
      <instancedMesh ref={meshRef} args={[undefined, undefined, buildings.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          metalness={0.7}
          roughness={0.3}
          vertexColors
        />
      </instancedMesh>

      {/* Building top glow strips */}
      <instancedMesh ref={glowRef} args={[undefined, undefined, buildings.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          transparent
          opacity={0.8}
          vertexColors
        />
      </instancedMesh>

      {/* Ambient city lights */}
      <pointLight position={[-8, 5, 0]} intensity={0.3} color={colors.positive} distance={15} />
      <pointLight position={[8, 5, 1]} intensity={0.3} color={colors.negative} distance={15} />
      <pointLight position={[0, 6, -3]} intensity={0.4} color="#ffffff" distance={20} />
    </group>
  );
}
