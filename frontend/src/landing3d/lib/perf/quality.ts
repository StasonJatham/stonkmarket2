import { useEffect, useState } from 'react';

export type QualityTier = 'low' | 'mid' | 'high';

export interface QualitySettings {
  tier: QualityTier;
  dpr: number;
  galaxyCount: number;
  tickerCount: number;
  candleCount: number;
  enableMotion: boolean;
  motionScale: number;
  reducedMotion: boolean;
}

function getDeviceHints() {
  if (typeof window === 'undefined') {
    return { isMobile: false, prefersReducedMotion: false };
  }

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isMobile = window.matchMedia('(max-width: 768px)').matches || navigator.maxTouchPoints > 1;

  return { isMobile, prefersReducedMotion };
}

export function computeQualitySettings(): QualitySettings {
  const { isMobile, prefersReducedMotion } = getDeviceHints();
  const dprBase = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const tier: QualityTier = isMobile ? 'low' : dprBase > 1.5 ? 'high' : 'mid';

  const base = {
    high: { dpr: 1.5, galaxy: 2200, ticker: 280, candles: 140 },
    mid: { dpr: 1.25, galaxy: 1700, ticker: 220, candles: 120 },
    low: { dpr: 1.0, galaxy: 900, ticker: 120, candles: 90 },
  }[tier];

  const motionScale = prefersReducedMotion ? 0.35 : 1;
  const densityScale = prefersReducedMotion ? 0.7 : 1;

  return {
    tier,
    dpr: Math.min(base.dpr, dprBase),
    galaxyCount: Math.round(base.galaxy * densityScale),
    tickerCount: Math.round(base.ticker * densityScale),
    candleCount: Math.round(base.candles * densityScale),
    enableMotion: !prefersReducedMotion,
    motionScale,
    reducedMotion: prefersReducedMotion,
  };
}

export function useQualitySettings(): QualitySettings {
  const [quality, setQuality] = useState<QualitySettings>(() => computeQualitySettings());

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleChange = () => setQuality(computeQualitySettings());
    const reduceMotionMedia = window.matchMedia('(prefers-reduced-motion: reduce)');

    window.addEventListener('resize', handleChange);
    reduceMotionMedia.addEventListener('change', handleChange);

    return () => {
      window.removeEventListener('resize', handleChange);
      reduceMotionMedia.removeEventListener('change', handleChange);
    };
  }, []);

  return quality;
}
