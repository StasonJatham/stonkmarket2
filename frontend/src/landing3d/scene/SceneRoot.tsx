import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Color, Fog, Vector3 } from 'three';
import { ScrollControls, Scroll } from '@react-three/drei';
import { LANDING3D_CONFIG } from '../config';
import type { AssetPoint, Series } from '../lib/data/types';
import type { QualitySettings } from '../lib/perf/quality';
import { CameraRig } from './CameraRig';
import { Galaxy } from './Galaxy';
import { TickerTape } from './TickerTape';
import { ChartReveal } from './ChartReveal';
import { Overlay } from '../ui/Overlay';

interface SceneRootProps {
  quality: QualitySettings;
}

export function SceneRoot({ quality }: SceneRootProps) {
  const [assets, setAssets] = useState<AssetPoint[]>([]);
  const [snapshot, setSnapshot] = useState<Record<string, { ret1d: number; vol: number; price: number }>>({});
  const [series, setSeries] = useState<Series | null>(null);
  const visibilityRef = useRef(true);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const onVisibility = () => {
      visibilityRef.current = document.visibilityState === 'visible';
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const provider = LANDING3D_CONFIG.dataProvider;

    Promise.all([
      provider.getUniverse(),
      provider.getLatestSnapshot(),
      provider.getCandles(LANDING3D_CONFIG.chartSymbol),
    ]).then(([universe, latest, candles]) => {
      if (!isMounted) return;
      setAssets(universe);
      setSnapshot(latest);
      setSeries(candles);
    });

    return () => {
      isMounted = false;
    };
  }, []);

  const fog = useMemo(() => new Fog(new Color('#05070f'), 18, 60), []);
  const sectionPositions = useMemo(
    () => [
      new Vector3(0, 2, 12),
      new Vector3(-1, 1, -12),
      new Vector3(0, 1, -34),
      new Vector3(0, 1, -55),
    ],
    []
  );
  const lookTargets = useMemo(
    () => [
      new Vector3(0, 0, 0),
      new Vector3(0, 0, -20),
      new Vector3(0, 0, -38),
      new Vector3(0, 0, -58),
    ],
    []
  );

  return (
    <Suspense fallback={null}>
      <color attach="background" args={['#05060a']} />
      <primitive attach="fog" object={fog} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[6, 10, 6]} intensity={1.2} color="#c8d6ff" />
      <pointLight position={[-8, 6, -8]} intensity={0.7} color="#77c7ff" />

      <ScrollControls pages={4} damping={quality.reducedMotion ? 0.2 : 0.35}>
        <CameraRig
          positions={sectionPositions}
          targets={lookTargets}
          quality={quality}
          visibilityRef={visibilityRef}
        />
        <Scroll>
          <group position={[0, 0, 0]}>
            <TickerTape count={quality.tickerCount} visibilityRef={visibilityRef} quality={quality} />
          </group>

          <group position={[0, 0, -20]}>
            <Galaxy
              assets={assets}
              snapshot={snapshot}
              count={quality.galaxyCount}
            />
          </group>

          <group position={[0, -1.4, -40]}>
            <ChartReveal
              series={series}
              candleCount={quality.candleCount}
              visibilityRef={visibilityRef}
              quality={quality}
            />
          </group>

          <group position={[0, 0, -60]}>
            <mesh>
              <torusGeometry args={[4, 0.18, 24, 120]} />
              <meshStandardMaterial emissive="#6db7ff" color="#0c1b2e" metalness={0.6} roughness={0.2} />
            </mesh>
            <mesh position={[0, 0, 0]}>
              <sphereGeometry args={[1.4, 32, 32]} />
              <meshStandardMaterial emissive="#1e6fff" color="#0b0f1b" metalness={0.5} roughness={0.3} />
            </mesh>
          </group>
        </Scroll>
        <Scroll html>
          <Overlay />
        </Scroll>
      </ScrollControls>
    </Suspense>
  );
}
