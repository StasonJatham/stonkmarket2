import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useEffect } from 'react';
import type { MutableRefObject } from 'react';
import { Color, AdditiveBlending, BufferAttribute, BufferGeometry } from 'three';
import type { Points } from 'three';
import type { QualitySettings } from '../lib/perf/quality';

interface ParticlesProps {
  count: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
  spread?: number;
  baseSpeed?: number;
  color?: string;
}

export function Particles({
  count,
  visibilityRef,
  quality,
  spread = 30,
  baseSpeed = 0.15,
  color = '#4d8cff',
}: ParticlesProps) {
  const pointsRef = useRef<Points>(null);
  const geometryRef = useRef<BufferGeometry>(null);
  const particleColor = useMemo(() => new Color(color), [color]);

  const { positions, velocities } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const siz = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      // Spread particles in a sphere with more density toward center
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.pow(Math.random(), 0.5) * spread;

      pos[i3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i3 + 2] = radius * Math.cos(phi);

      // Velocity - gentle drift
      vel[i3] = (Math.random() - 0.5) * 0.02;
      vel[i3 + 1] = (Math.random() - 0.5) * 0.02;
      vel[i3 + 2] = (Math.random() - 0.5) * 0.02;

      // Size variation
      siz[i] = 0.02 + Math.random() * 0.05;
    }

    return { positions: pos, velocities: vel, sizes: siz };
  }, [count, spread]);

  useEffect(() => {
    if (!geometryRef.current) return;
    geometryRef.current.setAttribute('position', new BufferAttribute(positions, 3));
  }, [positions]);

  useFrame((state) => {
    if (!visibilityRef.current || !pointsRef.current || !geometryRef.current) return;

    const positionAttr = geometryRef.current.attributes.position as BufferAttribute;
    const time = state.clock.elapsedTime;
    const speed = baseSpeed * quality.motionScale;

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Gentle floating motion
      positionAttr.array[i3] += Math.sin(time * 0.3 + i * 0.1) * 0.001 * speed;
      positionAttr.array[i3 + 1] += velocities[i3 + 1] * 0.016 * speed * 10;
      positionAttr.array[i3 + 2] += Math.cos(time * 0.2 + i * 0.15) * 0.001 * speed;

      // Wrap around
      if (positionAttr.array[i3 + 1] > spread) {
        positionAttr.array[i3 + 1] = -spread;
      }
    }

    positionAttr.needsUpdate = true;

    // Gentle rotation
    pointsRef.current.rotation.y += 0.016 * 0.02 * speed;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry ref={geometryRef} />
      <pointsMaterial
        size={0.08}
        color={particleColor}
        transparent
        opacity={0.7}
        blending={AdditiveBlending}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}

export function FloatingDust({
  count = 200,
  visibilityRef,
  quality,
}: {
  count?: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
}) {
  const pointsRef = useRef<Points>(null);
  const geometryRef = useRef<BufferGeometry>(null);

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      pos[i3] = (Math.random() - 0.5) * 40;
      pos[i3 + 1] = (Math.random() - 0.5) * 20;
      pos[i3 + 2] = (Math.random() - 0.5) * 80 - 40;
    }
    return pos;
  }, [count]);

  useEffect(() => {
    if (!geometryRef.current) return;
    geometryRef.current.setAttribute('position', new BufferAttribute(positions, 3));
  }, [positions]);

  useFrame((state) => {
    if (!visibilityRef.current || !pointsRef.current) return;
    const time = state.clock.elapsedTime;

    pointsRef.current.rotation.y = time * 0.005 * quality.motionScale;
    pointsRef.current.rotation.x = Math.sin(time * 0.1) * 0.01;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry ref={geometryRef} />
      <pointsMaterial
        size={0.03}
        color="#8bb8ff"
        transparent
        opacity={0.4}
        blending={AdditiveBlending}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}
