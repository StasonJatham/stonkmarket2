import { useFrame } from '@react-three/fiber';
import { useScroll } from '@react-three/drei';
import { useEffect, useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { CatmullRomCurve3, Matrix4, Object3D, Vector3 } from 'three';
import type { Series } from '../lib/data/types';

interface ChartRevealProps {
  series: Series | null;
  candleCount: number;
  visibilityRef: MutableRefObject<boolean>;
}

export function ChartReveal({ series, candleCount, visibilityRef }: ChartRevealProps) {
  const scroll = useScroll();
  const bodyRef = useRef<THREE.InstancedMesh>(null);
  const wickRef = useRef<THREE.InstancedMesh>(null);
  const lineRef = useRef<THREE.Mesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempMatrix = useMemo(() => new Matrix4(), []);

  const candles = useMemo(() => {
    if (!series || series.candles.length === 0) return [];
    const slice = series.candles.slice(-candleCount);
    if (slice.length === 0) return [];
    const min = Math.min(...slice.map((c) => c.l));
    const max = Math.max(...slice.map((c) => c.h));
    const range = max - min || 1;
    return slice.map((c, index) => {
      const x = (index - slice.length / 2) * 0.28;
      const open = (c.o - min) / range;
      const close = (c.c - min) / range;
      const high = (c.h - min) / range;
      const low = (c.l - min) / range;
      return { x, open, close, high, low, up: c.c >= c.o };
    });
  }, [series, candleCount]);

  const lineGeometry = useMemo(() => {
    if (candles.length === 0) return null;
    const points = candles.map((c) => new Vector3(c.x, c.close * 6 - 2.5, 0));
    const curve = new CatmullRomCurve3(points);
    return curve.getPoints(points.length * 6);
  }, [candles]);
  const lineCurve = useMemo(() => {
    if (!lineGeometry) return null;
    return new CatmullRomCurve3(lineGeometry);
  }, [lineGeometry]);

  useEffect(() => {
    if (!bodyRef.current || !wickRef.current || candles.length === 0) return;

    candles.forEach((candle, index) => {
      const center = (candle.open + candle.close) * 3 - 2.5;
      const height = Math.max(Math.abs(candle.close - candle.open) * 6, 0.12);
      tempObject.position.set(candle.x, center, 0);
      tempObject.scale.set(0.18, height, 0.18);
      tempObject.updateMatrix();
      bodyRef.current?.setMatrixAt(index, tempObject.matrix);

      const wickHeight = Math.max(candle.high - candle.low, 0.06) * 6;
      tempObject.position.set(candle.x, (candle.high + candle.low) * 3 - 2.5, 0);
      tempObject.scale.set(0.05, wickHeight, 0.05);
      tempObject.updateMatrix();
      wickRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    bodyRef.current.instanceMatrix.needsUpdate = true;
    wickRef.current.instanceMatrix.needsUpdate = true;
  }, [candles, tempObject]);

  useFrame(() => {
    if (!visibilityRef.current) return;
    if (!bodyRef.current || !wickRef.current || !lineRef.current || candles.length === 0) return;

    const sectionSize = 1 / 3;
    const sectionStart = sectionSize * 2;
    const sectionLength = sectionSize;
    const progress = scroll.range(sectionStart, sectionLength);
    const reveal = Math.min(progress * 1.4, 1);

    candles.forEach((candle, index) => {
      const baseHeight = Math.max(Math.abs(candle.close - candle.open) * 6, 0.12);
      const height = baseHeight * reveal;
      const center = (candle.open + candle.close) * 3 - 2.5;
      tempObject.position.set(candle.x, center, 0);
      tempObject.scale.set(0.18, height, 0.18);
      tempObject.updateMatrix();
      tempMatrix.copy(tempObject.matrix);
      bodyRef.current?.setMatrixAt(index, tempMatrix);

      const wickHeight = Math.max(candle.high - candle.low, 0.06) * 6 * reveal;
      tempObject.position.set(candle.x, (candle.high + candle.low) * 3 - 2.5, 0);
      tempObject.scale.set(0.05, wickHeight, 0.05);
      tempObject.updateMatrix();
      tempMatrix.copy(tempObject.matrix);
      wickRef.current?.setMatrixAt(index, tempMatrix);
    });

    bodyRef.current.instanceMatrix.needsUpdate = true;
    wickRef.current.instanceMatrix.needsUpdate = true;

    if (lineRef.current.geometry) {
      const segments = lineRef.current.geometry.attributes.position.count;
      const drawCount = Math.floor(segments * reveal);
      lineRef.current.geometry.setDrawRange(0, drawCount);
    }
  });

  if (candles.length === 0 || !lineGeometry || !lineCurve) return null;

  return (
    <group>
      <instancedMesh ref={bodyRef} args={[undefined, undefined, candles.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial emissive="#5bd1ff" color="#14243a" roughness={0.35} metalness={0.2} />
      </instancedMesh>
      <instancedMesh ref={wickRef} args={[undefined, undefined, candles.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial emissive="#3b7cff" color="#0b1b2a" roughness={0.3} metalness={0.2} />
      </instancedMesh>

      <mesh ref={lineRef} position={[0, 0, 0.2]}>
        <tubeGeometry args={[lineCurve, 200, 0.04, 8, false]} />
        <meshStandardMaterial emissive="#7cf4ff" color="#0f1f2b" roughness={0.2} metalness={0.4} />
      </mesh>

      <mesh position={[0, -2.8, -0.3]}>
        <planeGeometry args={[10, 6]} />
        <meshStandardMaterial color="#060b14" emissive="#0a1220" opacity={0.7} transparent />
      </mesh>
    </group>
  );
}
