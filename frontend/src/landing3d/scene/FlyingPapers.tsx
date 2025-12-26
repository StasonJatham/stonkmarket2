import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { Object3D, InstancedMesh, Color, Euler } from 'three';
import type { QualitySettings } from '../lib/perf/quality';

interface FlyingPapersProps {
  count?: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
  spread?: number;
  windDirection?: number; // radians
}

// Seeded random for deterministic behavior
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + seed * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

interface Paper {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  rotX: number;
  rotY: number;
  rotZ: number;
  rotSpeedX: number;
  rotSpeedY: number;
  rotSpeedZ: number;
  scale: number;
  startDelay: number;
}

export function FlyingPapers({
  count = 80,
  visibilityRef,
  quality,
  spread = 25,
  windDirection = Math.PI / 4,
}: FlyingPapersProps) {
  const meshRef = useRef<InstancedMesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempEuler = useMemo(() => new Euler(), []);
  const tempColor = useMemo(() => new Color(), []);

  // Adjust count based on quality
  const actualCount = quality.tier === 'low' ? Math.floor(count * 0.5) : count;

  // Generate paper data
  const papers = useMemo<Paper[]>(() => {
    const result: Paper[] = [];
    
    for (let i = 0; i < actualCount; i++) {
      const seed = i * 789;
      
      result.push({
        // Start position - scattered around
        x: (seededRandom(seed) - 0.5) * spread,
        y: seededRandom(seed + 1) * 8 - 2,
        z: (seededRandom(seed + 2) - 0.5) * spread,
        
        // Velocity - general upward/wind direction
        vx: Math.cos(windDirection) * 0.02 + (seededRandom(seed + 3) - 0.5) * 0.01,
        vy: 0.01 + seededRandom(seed + 4) * 0.02,
        vz: Math.sin(windDirection) * 0.02 + (seededRandom(seed + 5) - 0.5) * 0.01,
        
        // Rotation
        rotX: seededRandom(seed + 6) * Math.PI * 2,
        rotY: seededRandom(seed + 7) * Math.PI * 2,
        rotZ: seededRandom(seed + 8) * Math.PI * 2,
        
        // Rotation speed
        rotSpeedX: (seededRandom(seed + 9) - 0.5) * 0.05,
        rotSpeedY: (seededRandom(seed + 10) - 0.5) * 0.05,
        rotSpeedZ: (seededRandom(seed + 11) - 0.5) * 0.08,
        
        // Scale
        scale: 0.15 + seededRandom(seed + 12) * 0.2,
        
        // Stagger start
        startDelay: seededRandom(seed + 13) * 3,
      });
    }
    
    return result;
  }, [actualCount, spread, windDirection]);

  // Set initial colors (slight variation in white/cream)
  useEffect(() => {
    if (!meshRef.current) return;

    papers.forEach((_, index) => {
      const seed = index * 321;
      const brightness = 0.85 + seededRandom(seed) * 0.15;
      tempColor.setRGB(brightness, brightness * 0.98, brightness * 0.95);
      meshRef.current?.setColorAt(index, tempColor);
    });

    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [papers, tempColor]);

  useFrame((state) => {
    if (!visibilityRef.current || !meshRef.current) return;

    const time = state.clock.elapsedTime;
    const speed = quality.motionScale;

    papers.forEach((paper, index) => {
      // Wait for start delay
      if (time < paper.startDelay) {
        tempObject.scale.set(0, 0, 0);
        tempObject.updateMatrix();
        meshRef.current?.setMatrixAt(index, tempObject.matrix);
        return;
      }

      const activeTime = time - paper.startDelay;

      // Turbulent wind effect
      const turbulenceX = Math.sin(activeTime * 0.7 + index * 0.5) * 0.015;
      const turbulenceY = Math.sin(activeTime * 0.5 + index * 0.3) * 0.01;
      const turbulenceZ = Math.cos(activeTime * 0.6 + index * 0.4) * 0.015;

      // Update position
      paper.x += (paper.vx + turbulenceX) * speed;
      paper.y += (paper.vy + turbulenceY) * speed;
      paper.z += (paper.vz + turbulenceZ) * speed;

      // Wrap around when out of bounds
      if (paper.y > 10) {
        paper.y = -3;
        paper.x = (seededRandom(index * time * 100) - 0.5) * spread;
        paper.z = (seededRandom(index * time * 200) - 0.5) * spread;
      }
      if (Math.abs(paper.x) > spread / 2) {
        paper.x = -Math.sign(paper.x) * spread / 2;
      }
      if (Math.abs(paper.z) > spread / 2) {
        paper.z = -Math.sign(paper.z) * spread / 2;
      }

      // Update rotation (fluttering effect)
      paper.rotX += paper.rotSpeedX * speed;
      paper.rotY += paper.rotSpeedY * speed;
      paper.rotZ += paper.rotSpeedZ * speed + Math.sin(activeTime * 2 + index) * 0.01;

      // Apply transform
      tempEuler.set(paper.rotX, paper.rotY, paper.rotZ);
      tempObject.position.set(paper.x, paper.y, paper.z);
      tempObject.rotation.copy(tempEuler);
      
      // Slight scale variation for flutter
      const scaleFlutter = 1 + Math.sin(activeTime * 3 + index) * 0.05;
      tempObject.scale.set(paper.scale * scaleFlutter, paper.scale * 0.7, 0.002);
      
      tempObject.updateMatrix();
      meshRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, actualCount]}>
      <planeGeometry args={[1, 1]} />
      <meshStandardMaterial
        color="#ffffff"
        side={2} // DoubleSide
        transparent
        opacity={0.9}
        roughness={0.8}
        metalness={0}
        vertexColors
      />
    </instancedMesh>
  );
}

/**
 * Ticker strips flying through the scene (like stock ticker tape)
 */
export function FlyingTickers({
  count = 40,
  visibilityRef,
  quality,
}: {
  count?: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
}) {
  const meshRef = useRef<InstancedMesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempColor = useMemo(() => new Color(), []);

  const actualCount = quality.tier === 'low' ? Math.floor(count * 0.5) : count;

  // Generate ticker strip data
  const strips = useMemo(() => {
    const result: Array<{
      x: number;
      y: number;
      z: number;
      speed: number;
      width: number;
      isPositive: boolean;
    }> = [];

    for (let i = 0; i < actualCount; i++) {
      const seed = i * 567;
      result.push({
        x: (seededRandom(seed) - 0.5) * 40,
        y: seededRandom(seed + 1) * 6 - 1,
        z: (seededRandom(seed + 2) - 0.5) * 20,
        speed: 0.05 + seededRandom(seed + 3) * 0.1,
        width: 1 + seededRandom(seed + 4) * 2,
        isPositive: seededRandom(seed + 5) > 0.45,
      });
    }

    return result;
  }, [actualCount]);

  // Set colors
  useEffect(() => {
    if (!meshRef.current) return;

    strips.forEach((strip, index) => {
      if (strip.isPositive) {
        tempColor.set('#22c55e');
      } else {
        tempColor.set('#ef4444');
      }
      meshRef.current?.setColorAt(index, tempColor);
    });

    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [strips, tempColor]);

  useFrame(() => {
    if (!visibilityRef.current || !meshRef.current) return;

    strips.forEach((strip, index) => {
      // Move across screen
      strip.x -= strip.speed * quality.motionScale;

      // Wrap around
      if (strip.x < -25) {
        strip.x = 25;
      }

      tempObject.position.set(strip.x, strip.y, strip.z);
      tempObject.scale.set(strip.width, 0.08, 0.01);
      tempObject.updateMatrix();
      meshRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, actualCount]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial
        transparent
        opacity={0.6}
        vertexColors
      />
    </instancedMesh>
  );
}
