import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { useMemo, useRef } from 'react';
import type { MutableRefObject } from 'react';
import { Color, Object3D, AdditiveBlending, InstancedMesh } from 'three';
import type { QualitySettings } from '../lib/perf/quality';

interface TickerTapeProps {
  count: number;
  visibilityRef: MutableRefObject<boolean>;
  quality: QualitySettings;
}

// Sample ticker symbols for visual effect
const TICKER_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
  'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
  'KO', 'COST', 'TMO', 'WMT', 'BAC', 'CSCO', 'ACN', 'MCD', 'ABT', 'DHR',
];

// Seeded random for consistent placement
function seededRandom(seed: number) {
  const x = Math.sin(seed * 9.8765) * 10000;
  return x - Math.floor(x);
}

export function TickerTape({ count, visibilityRef, quality }: TickerTapeProps) {
  const baseRef = useRef<InstancedMesh>(null);
  const accentRef = useRef<InstancedMesh>(null);
  const glowRef = useRef<InstancedMesh>(null);
  const tempObject = useMemo(() => new Object3D(), []);
  const tempColor = useMemo(() => new Color(), []);

  const tickerData = useMemo(() => {
    const rows = 8;
    const layers = 4;
    const items: {
      x: number;
      y: number;
      z: number;
      speed: number;
      width: number;
      isHighlight: boolean;
      symbol: string;
      change: number;
      seed: number;
    }[] = [];

    for (let i = 0; i < count; i++) {
      const seed = i * 12.345;
      const row = i % rows;
      const layer = Math.floor(i / rows) % layers;
      const offset = Math.floor(i / (rows * layers));

      const x = -20 + (offset * 2.8) % 40;
      const y = 3.5 - row * 0.55;
      const z = 4 - layer * 2.5;

      const speed = 0.4 + seededRandom(seed) * 0.5;
      const width = 1.8 + seededRandom(seed + 1) * 1.2;
      const isHighlight = seededRandom(seed + 2) > 0.85;
      const symbol = TICKER_SYMBOLS[Math.floor(seededRandom(seed + 3) * TICKER_SYMBOLS.length)];
      const change = (seededRandom(seed + 4) - 0.45) * 8; // -4% to +4%

      items.push({ x, y, z, speed, width, isHighlight, symbol, change, seed });
    }
    return items;
  }, [count]);

  // Hero tickers - the larger, more prominent ones
  const heroTickers = useMemo(() => {
    return tickerData.filter(t => t.isHighlight).slice(0, 6);
  }, [tickerData]);

  // Set initial colors
  useMemo(() => {
    if (!baseRef.current || !accentRef.current || !glowRef.current) return;

    tickerData.forEach((item, index) => {
      const isUp = item.change >= 0;

      // Base strip color
      if (item.isHighlight) {
        tempColor.set(isUp ? '#1a5a3a' : '#5a1a1a');
      } else {
        tempColor.set('#0a1830');
      }
      baseRef.current?.setColorAt(index, tempColor);

      // Accent strip
      tempColor.set(isUp ? '#00ff88' : '#ff4466');
      tempColor.multiplyScalar(item.isHighlight ? 0.8 : 0.4);
      accentRef.current?.setColorAt(index, tempColor);

      // Glow
      tempColor.set(isUp ? '#00ff88' : '#ff4466');
      tempColor.multiplyScalar(0.3);
      glowRef.current?.setColorAt(index, tempColor);
    });

    if (baseRef.current.instanceColor) baseRef.current.instanceColor.needsUpdate = true;
    if (accentRef.current.instanceColor) accentRef.current.instanceColor.needsUpdate = true;
    if (glowRef.current.instanceColor) glowRef.current.instanceColor.needsUpdate = true;
  }, [tickerData, tempColor]);

  useFrame((state) => {
    if (!visibilityRef.current) return;
    if (!baseRef.current || !accentRef.current || !glowRef.current) return;

    const time = state.clock.elapsedTime;
    const motion = quality.motionScale;

    tickerData.forEach((item, index) => {
      // Calculate scrolling position
      const moveRange = 45;
      const move = ((time * item.speed * motion + item.seed) % moveRange) - moveRange / 2;

      // Base strip
      tempObject.position.set(item.x + move, item.y, item.z);
      tempObject.scale.set(item.width, item.isHighlight ? 0.18 : 0.12, 1);
      tempObject.updateMatrix();
      baseRef.current?.setMatrixAt(index, tempObject.matrix);

      // Accent strip on top
      tempObject.position.set(item.x + move + item.width * 0.2, item.y + 0.08, item.z + 0.02);
      tempObject.scale.set(item.width * 0.4, 0.04, 1);
      tempObject.updateMatrix();
      accentRef.current?.setMatrixAt(index, tempObject.matrix);

      // Glow behind
      tempObject.position.set(item.x + move, item.y, item.z - 0.05);
      tempObject.scale.set(item.width * 1.3, item.isHighlight ? 0.35 : 0.22, 1);
      tempObject.updateMatrix();
      glowRef.current?.setMatrixAt(index, tempObject.matrix);
    });

    baseRef.current.instanceMatrix.needsUpdate = true;
    accentRef.current.instanceMatrix.needsUpdate = true;
    glowRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <group>
      {/* Glow layer behind */}
      <instancedMesh ref={glowRef} args={[undefined, undefined, tickerData.length]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial
          transparent
          opacity={0.15}
          blending={AdditiveBlending}
          depthWrite={false}
          vertexColors
        />
      </instancedMesh>

      {/* Base strips */}
      <instancedMesh ref={baseRef} args={[undefined, undefined, tickerData.length]}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial
          emissive="#1c3d6a"
          emissiveIntensity={0.8}
          color="#0b1626"
          roughness={0.3}
          metalness={0.4}
          vertexColors
        />
      </instancedMesh>

      {/* Accent strips */}
      <instancedMesh ref={accentRef} args={[undefined, undefined, tickerData.length]}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial
          emissive="#3b6cff"
          emissiveIntensity={1.5}
          color="#16253b"
          roughness={0.2}
          metalness={0.5}
          vertexColors
        />
      </instancedMesh>

      {/* Hero ticker labels - actual text for prominent ones */}
      {heroTickers.map((ticker, i) => {
        const isUp = ticker.change >= 0;
        return (
          <group key={`hero-${i}`}>
            <Text
              position={[ticker.x - 5 + i * 3.5, ticker.y, ticker.z + 0.1]}
              fontSize={0.18}
              color={isUp ? '#00ff88' : '#ff4466'}
              anchorX="left"
              anchorY="middle"
              renderOrder={10}
            >
              {ticker.symbol}
            </Text>
            <Text
              position={[ticker.x - 3.5 + i * 3.5, ticker.y, ticker.z + 0.1]}
              fontSize={0.14}
              color={isUp ? '#00ff88' : '#ff4466'}
              anchorX="left"
              anchorY="middle"
              renderOrder={10}
            >
              {isUp ? '+' : ''}{ticker.change.toFixed(2)}%
            </Text>
          </group>
        );
      })}

      {/* Gradient fade at edges */}
      <mesh position={[-22, 1.5, 2]}>
        <planeGeometry args={[6, 6]} />
        <meshBasicMaterial
          color="#05060a"
          transparent
          opacity={0.95}
        />
      </mesh>
      <mesh position={[22, 1.5, 2]}>
        <planeGeometry args={[6, 6]} />
        <meshBasicMaterial
          color="#05060a"
          transparent
          opacity={0.95}
        />
      </mesh>
    </group>
  );
}
