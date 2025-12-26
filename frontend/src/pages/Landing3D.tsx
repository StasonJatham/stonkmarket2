import { useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { useNavigate } from 'react-router-dom';
import { SceneRoot } from '@/landing3d/scene/SceneRoot';
import { useQualitySettings } from '@/landing3d/lib/perf/quality';
import { LANDING3D_CONFIG } from '@/landing3d/config';
import { SyntheticProvider } from '@/landing3d/lib/data/syntheticProvider';
import { RealDataProvider } from '@/landing3d/lib/data/realProvider.stub';
import { useAuth } from '@/context/AuthContext';
import '@/landing3d/ui/styles.css';

type DataMode = 'synthetic' | 'real';

export function Landing3D() {
  const quality = useQualitySettings();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const [dataMode, setDataMode] = useState<DataMode>(() => {
    if (typeof window === 'undefined') return 'synthetic';
    const saved = window.localStorage.getItem('landing3d-data');
    return saved === 'real' ? 'real' : 'synthetic';
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem('landing3d-data', dataMode);
  }, [dataMode]);

  const provider = useMemo(() => {
    if (dataMode === 'real') {
      return new RealDataProvider();
    }
    return new SyntheticProvider(LANDING3D_CONFIG.seed, LANDING3D_CONFIG.universeSize);
  }, [dataMode]);

  const handleCta = () => {
    navigate(isAuthenticated ? LANDING3D_CONFIG.ctaHrefAuthenticated : LANDING3D_CONFIG.ctaHrefGuest);
  };

  return (
    <div className="landing3d-root">
      <Canvas
        className="landing3d-canvas"
        dpr={quality.dpr}
        camera={{ position: [0, 2, 12], fov: 45, near: 0.1, far: 200 }}
        gl={{ antialias: quality.tier !== 'low', powerPreference: 'high-performance' }}
      >
        <SceneRoot
          quality={quality}
          provider={provider}
          chartSymbol={LANDING3D_CONFIG.chartSymbol}
          overlayProps={{
            dataMode,
            onToggleMode: () => setDataMode((prev) => (prev === 'synthetic' ? 'real' : 'synthetic')),
            onCta: handleCta,
          }}
        />
      </Canvas>
    </div>
  );
}
