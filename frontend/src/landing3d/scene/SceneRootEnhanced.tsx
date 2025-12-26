import { Suspense, useEffect, useMemo, useRef, useState, type ComponentProps } from 'react';
import { Color, Fog, Vector3 } from 'three';
import { ScrollControls, Scroll, Stars } from '@react-three/drei';
import type { AssetPoint, DataProvider, Series } from '../lib/data/types';
import type { QualitySettings } from '../lib/perf/quality';
import { CameraRig } from './CameraRig';
import { Galaxy } from './GalaxyEnhanced';
import { TickerTape } from './TickerTapeEnhanced';
import { ChartReveal } from './ChartRevealEnhanced';
import { Particles, FloatingDust } from './Particles';
import { Overlay } from '../ui/Overlay';

interface SceneRootProps {
  quality: QualitySettings;
  provider: DataProvider;
  chartSymbol: string;
  overlayProps: ComponentProps<typeof Overlay>;
}

export function SceneRoot({ quality, provider, chartSymbol, overlayProps }: SceneRootProps) {
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

    setAssets([]);
    setSnapshot({});
    setSeries(null);

    Promise.all([
      provider.getUniverse(),
      provider.getLatestSnapshot(),
      provider.getCandles(chartSymbol),
    ])
      .then(([universe, latest, candles]) => {
        if (!isMounted) return;
        setAssets(universe);
        setSnapshot(latest);
        setSeries(candles);
      })
      .catch(() => {});

    return () => {
      isMounted = false;
    };
  }, [provider, chartSymbol]);

  const fog = useMemo(() => new Fog(new Color('#030508'), 25, 80), []);

  // Camera positions for each section - more dramatic movement
  const sectionPositions = useMemo(
    () => [
      new Vector3(0, 2.5, 14),    // S1: Looking at ticker tape
      new Vector3(-2, 3, -10),    // S2: Orbiting around galaxy
      new Vector3(0, 1.5, -32),   // S3: Looking at chart
      new Vector3(0, 2, -54),     // S4: Final CTA view
    ],
    []
  );

  const lookTargets = useMemo(
    () => [
      new Vector3(0, 1, 0),
      new Vector3(0, 0, -18),
      new Vector3(0, 0, -38),
      new Vector3(0, 0, -58),
    ],
    []
  );

  return (
    <Suspense fallback={null}>
      {/* Dark space background */}
      <color attach="background" args={['#020305']} />
      <primitive attach="fog" object={fog} />

      {/* Ambient lighting */}
      <ambientLight intensity={0.25} color="#8899bb" />

      {/* Key light - main illumination */}
      <directionalLight
        position={[8, 12, 8]}
        intensity={1.0}
        color="#c8d6ff"
        castShadow={false}
      />

      {/* Rim lights for depth */}
      <pointLight position={[-12, 8, -15]} intensity={0.6} color="#4488ff" distance={50} />
      <pointLight position={[15, 5, -30]} intensity={0.5} color="#ff8844" distance={40} />
      <pointLight position={[0, -5, -45]} intensity={0.4} color="#44ffaa" distance={35} />

      {/* Starfield background */}
      <Stars
        radius={100}
        depth={60}
        count={quality.tier === 'low' ? 2000 : 4000}
        factor={3}
        saturation={0.1}
        fade
        speed={0.3}
      />

      <ScrollControls pages={4} damping={quality.reducedMotion ? 0.15 : 0.25}>
        <CameraRig
          positions={sectionPositions}
          targets={lookTargets}
          quality={quality}
          visibilityRef={visibilityRef}
        />

        <Scroll>
          {/* Floating ambient dust throughout */}
          <FloatingDust
            count={quality.tier === 'low' ? 100 : 250}
            visibilityRef={visibilityRef}
            quality={quality}
          />

          {/* Section 1: Trading Floor Energy - Ticker Tape */}
          <group position={[0, 0, 0]}>
            <TickerTape
              count={quality.tickerCount}
              visibilityRef={visibilityRef}
              quality={quality}
            />

            {/* Atmospheric particles for section 1 */}
            <Particles
              count={quality.tier === 'low' ? 150 : 400}
              visibilityRef={visibilityRef}
              quality={quality}
              spread={20}
              baseSpeed={0.1}
              color="#3366ff"
            />

            {/* Floor reflection hint */}
            <mesh position={[0, -2, 2]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[50, 30]} />
              <meshStandardMaterial
                color="#050810"
                metalness={0.9}
                roughness={0.1}
                transparent
                opacity={0.4}
              />
            </mesh>
          </group>

          {/* Section 2: Market Galaxy */}
          <group position={[0, 0, -18]}>
            <Galaxy
              assets={assets}
              snapshot={snapshot}
              count={quality.galaxyCount}
            />

            {/* Nebula-like ambient glow */}
            <mesh position={[5, 2, -3]}>
              <sphereGeometry args={[8, 16, 16]} />
              <meshBasicMaterial
                color="#1a3366"
                transparent
                opacity={0.04}
              />
            </mesh>
            <mesh position={[-6, -1, 2]}>
              <sphereGeometry args={[6, 16, 16]} />
              <meshBasicMaterial
                color="#662244"
                transparent
                opacity={0.03}
              />
            </mesh>
          </group>

          {/* Section 3: Chart Reveal */}
          <group position={[0, -1, -38]}>
            <ChartReveal
              series={series}
              candleCount={quality.candleCount}
              visibilityRef={visibilityRef}
            />

            {/* Accent particles around chart */}
            <Particles
              count={quality.tier === 'low' ? 80 : 180}
              visibilityRef={visibilityRef}
              quality={quality}
              spread={12}
              baseSpeed={0.05}
              color="#00ffaa"
            />
          </group>

          {/* Section 4: CTA - Final composition */}
          <group position={[0, 0, -58]}>
            {/* Central orb - the "command center" */}
            <mesh position={[0, 0.5, 0]}>
              <sphereGeometry args={[2, 48, 48]} />
              <meshStandardMaterial
                emissive="#1e4fff"
                emissiveIntensity={0.8}
                color="#0a1530"
                metalness={0.8}
                roughness={0.1}
              />
            </mesh>

            {/* Glowing ring */}
            <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.5, 0]}>
              <torusGeometry args={[3.5, 0.08, 16, 100]} />
              <meshStandardMaterial
                emissive="#5eb8ff"
                emissiveIntensity={2}
                color="#0c2040"
                metalness={0.7}
                roughness={0.2}
              />
            </mesh>

            {/* Outer decorative ring */}
            <mesh rotation={[Math.PI / 2, 0, Math.PI / 6]} position={[0, 0.5, 0]}>
              <torusGeometry args={[5, 0.03, 12, 80]} />
              <meshStandardMaterial
                emissive="#3388cc"
                emissiveIntensity={1}
                color="#081828"
                metalness={0.6}
                roughness={0.3}
              />
            </mesh>

            {/* Particles around CTA */}
            <Particles
              count={quality.tier === 'low' ? 100 : 300}
              visibilityRef={visibilityRef}
              quality={quality}
              spread={15}
              baseSpeed={0.08}
              color="#5588ff"
            />

            {/* Floor */}
            <mesh position={[0, -3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <circleGeometry args={[20, 64]} />
              <meshStandardMaterial
                color="#050a15"
                metalness={0.95}
                roughness={0.05}
                transparent
                opacity={0.6}
              />
            </mesh>
          </group>
        </Scroll>

        {/* HTML Overlay */}
        <Scroll html>
          <Overlay {...overlayProps} />
        </Scroll>
      </ScrollControls>
    </Suspense>
  );
}
