import { useFrame } from '@react-three/fiber';
import { useScroll } from '@react-three/drei';
import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import {
  CatmullRomCurve3,
  Object3D,
  Vector3,
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  InstancedMesh,
  Mesh,
  MeshBasicMaterial,
  Points,
  PointsMaterial,
} from 'three';
import type { Series } from '../lib/data/types';

// Type helpers for material casting

interface ChartRevealProps {
  series: Series | null;
  candleCount: number;
  visibilityRef: MutableRefObject<boolean>;
}

// Seeded random
function seededRandom(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export function ChartReveal({ series, candleCount, visibilityRef }: ChartRevealProps) {
  const scroll = useScroll();
  const bodyRef = useRef<InstancedMesh>(null);
  const wickRef = useRef<InstancedMesh>(null);
  const lineRef = useRef<Mesh>(null);
  const glowRef = useRef<Mesh>(null);
  const particlesRef = useRef<Points>(null);
  const particlesGeomRef = useRef<BufferGeometry>(null);
  const gridRef = useRef<Mesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const bodyColor = useMemo(() => new Color(), []);

  const candles = useMemo(() => {
    if (!series || series.candles.length === 0) return [];
    const slice = series.candles.slice(-candleCount);
    if (slice.length === 0) return [];
    const min = Math.min(...slice.map((c) => c.l));
    const max = Math.max(...slice.map((c) => c.h));
    const range = max - min || 1;
    return slice.map((c, index) => {
      const x = (index - slice.length / 2) * 0.22;
      const open = (c.o - min) / range;
      const close = (c.c - min) / range;
      const high = (c.h - min) / range;
      const low = (c.l - min) / range;
      const up = c.c >= c.o;
      const volume = c.v / 1000000;
      return { x, open, close, high, low, up, volume };
    });
  }, [series, candleCount]);

  // Line curve
  const { lineGeometry, lineCurve } = useMemo(() => {
    if (candles.length === 0) return { lineGeometry: null, lineCurve: null };
    const points = candles.map((c) => new Vector3(c.x, c.close * 5 - 2, 0.1));
    const curve = new CatmullRomCurve3(points, false, 'catmullrom', 0.3);
    const geometry = curve.getPoints(candles.length * 8);
    return { lineGeometry: geometry, lineCurve: new CatmullRomCurve3(geometry) };
  }, [candles]);

  // Particle positions along the line for sparkle effect
  const particlePositions = useMemo(() => {
    if (!lineGeometry) return new Float32Array(0);
    const count = 150;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const idx = Math.floor(seededRandom(i * 123) * lineGeometry.length);
      const point = lineGeometry[idx] || new Vector3(0, 0, 0);
      positions[i * 3] = point.x + (seededRandom(i * 234) - 0.5) * 0.3;
      positions[i * 3 + 1] = point.y + (seededRandom(i * 345) - 0.5) * 0.3;
      positions[i * 3 + 2] = point.z + seededRandom(i * 456) * 0.2;
    }
    return positions;
  }, [lineGeometry]);

  // Set particle geometry imperatively
  useEffect(() => {
    if (!particlesGeomRef.current || particlePositions.length === 0) return;
    particlesGeomRef.current.setAttribute('position', new BufferAttribute(particlePositions, 3));
  }, [particlePositions]);

  // Set candle colors once
  useEffect(() => {
    if (!bodyRef.current || !wickRef.current || candles.length === 0) return;

    candles.forEach((candle, index) => {
      bodyColor.set(candle.up ? '#00dd77' : '#ff3355');
      bodyRef.current?.setColorAt(index, bodyColor);

      bodyColor.set(candle.up ? '#00aa55' : '#cc2244');
      wickRef.current?.setColorAt(index, bodyColor);
    });

    if (bodyRef.current.instanceColor) bodyRef.current.instanceColor.needsUpdate = true;
    if (wickRef.current.instanceColor) wickRef.current.instanceColor.needsUpdate = true;
  }, [candles, bodyColor]);

  useFrame((state) => {
    if (!visibilityRef.current) return;
    if (!bodyRef.current || !wickRef.current || !lineRef.current || candles.length === 0) return;

    const time = state.clock.elapsedTime;

    // Calculate reveal progress based on scroll position
    // Section 3 is the chart reveal section (page 2-3, roughly 0.5-0.75 in scroll range)
    const sectionSize = 1 / 4;
    const sectionStart = sectionSize * 2;
    const sectionLength = sectionSize;
    const progress = scroll.range(sectionStart, sectionLength);
    const reveal = Math.min(progress * 1.6, 1);

    // Easing function for smoother reveal
    const eased = 1 - Math.pow(1 - reveal, 3);

    // Update candles with reveal animation
    candles.forEach((candle, index) => {
      const candleDelay = index / candles.length;
      const candleProgress = Math.max(0, Math.min(1, (eased - candleDelay * 0.3) / 0.7));

      // Body
      const baseHeight = Math.max(Math.abs(candle.close - candle.open) * 5, 0.08);
      const height = baseHeight * candleProgress;
      const center = (candle.open + candle.close) * 2.5 - 2;
      const yOffset = (1 - candleProgress) * -1; // Rise from below

      tempObject.position.set(candle.x, center + yOffset, 0);
      tempObject.scale.set(0.12, Math.max(height, 0.001), 0.12);
      tempObject.updateMatrix();
      bodyRef.current?.setMatrixAt(index, tempObject.matrix);

      // Wick
      const wickHeight = Math.max(candle.high - candle.low, 0.04) * 5 * candleProgress;
      tempObject.position.set(candle.x, (candle.high + candle.low) * 2.5 - 2 + yOffset, 0);
      tempObject.scale.set(0.03, Math.max(wickHeight, 0.001), 0.03);
      tempObject.updateMatrix();
      wickRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    bodyRef.current.instanceMatrix.needsUpdate = true;
    wickRef.current.instanceMatrix.needsUpdate = true;

    // Line reveal
    if (lineRef.current.geometry) {
      const segments = lineRef.current.geometry.attributes.position.count;
      const drawCount = Math.floor(segments * eased);
      lineRef.current.geometry.setDrawRange(0, drawCount);
    }

    // Glow pulse
    if (glowRef.current) {
      const pulse = 0.8 + Math.sin(time * 2) * 0.2;
      (glowRef.current.material as MeshBasicMaterial).opacity = 0.12 * eased * pulse;
    }

    // Grid fade in
    if (gridRef.current) {
      (gridRef.current.material as MeshBasicMaterial).opacity = 0.08 * eased;
    }

    // Particles shimmer
    if (particlesRef.current && particlesRef.current.geometry.attributes.position) {
      const posAttr = particlesRef.current.geometry.attributes.position as BufferAttribute;
      for (let i = 0; i < posAttr.count; i++) {
        const originalY = particlePositions[i * 3 + 1] || 0;
        posAttr.setY(i, originalY + Math.sin(time * 3 + i * 0.5) * 0.02);
      }
      posAttr.needsUpdate = true;
      (particlesRef.current.material as PointsMaterial).opacity = 0.6 * eased;
    }
  });

  if (candles.length === 0 || !lineGeometry || !lineCurve) return null;

  return (
    <group>
      {/* Background panel */}
      <mesh position={[0, -0.5, -0.5]}>
        <planeGeometry args={[18, 8]} />
        <meshBasicMaterial
          color="#050a15"
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Grid lines */}
      <mesh ref={gridRef} position={[0, -0.5, -0.3]}>
        <planeGeometry args={[16, 6]} />
        <meshBasicMaterial
          color="#1a2a4a"
          transparent
          opacity={0}
          wireframe
        />
      </mesh>

      {/* Horizontal grid lines */}
      {[-2, -1, 0, 1, 2].map((y, i) => (
        <mesh key={`hgrid-${i}`} position={[0, y, -0.2]}>
          <planeGeometry args={[16, 0.005]} />
          <meshBasicMaterial color="#1a2a4a" transparent opacity={0.15} />
        </mesh>
      ))}

      {/* Candle bodies */}
      <instancedMesh ref={bodyRef} args={[undefined, undefined, candles.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive="#5bd1ff"
          emissiveIntensity={1.2}
          color="#14243a"
          roughness={0.25}
          metalness={0.3}
          vertexColors
        />
      </instancedMesh>

      {/* Candle wicks */}
      <instancedMesh ref={wickRef} args={[undefined, undefined, candles.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive="#3b7cff"
          emissiveIntensity={0.8}
          color="#0b1b2a"
          roughness={0.3}
          metalness={0.2}
          vertexColors
        />
      </instancedMesh>

      {/* Price line */}
      <mesh ref={lineRef} position={[0, 0, 0.15]}>
        <tubeGeometry args={[lineCurve, 250, 0.025, 8, false]} />
        <meshStandardMaterial
          emissive="#7cf4ff"
          emissiveIntensity={2}
          color="#0f1f2b"
          roughness={0.1}
          metalness={0.5}
        />
      </mesh>

      {/* Line glow */}
      <mesh ref={glowRef} position={[0, 0, 0.1]}>
        <tubeGeometry args={[lineCurve, 100, 0.08, 8, false]} />
        <meshBasicMaterial
          color="#5ef8ff"
          transparent
          opacity={0}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Sparkle particles along the line */}
      <points ref={particlesRef}>
        <bufferGeometry ref={particlesGeomRef} />
        <pointsMaterial
          size={0.04}
          color="#aaffff"
          transparent
          opacity={0}
          blending={AdditiveBlending}
          sizeAttenuation
          depthWrite={false}
        />
      </points>

      {/* Axis labels area */}
      <mesh position={[-8.5, -0.5, 0.1]}>
        <planeGeometry args={[0.8, 5]} />
        <meshBasicMaterial color="#05080f" transparent opacity={0.5} />
      </mesh>
      <mesh position={[0, -3.3, 0.1]}>
        <planeGeometry args={[16, 0.5]} />
        <meshBasicMaterial color="#05080f" transparent opacity={0.5} />
      </mesh>
    </group>
  );
}
