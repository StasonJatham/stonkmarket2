import type { AssetPoint, Candle, DataProvider, Series } from './types';

const SECTORS = [
  'Tech',
  'Finance',
  'Energy',
  'Health',
  'Industrial',
  'Consumer',
  'Utilities',
  'Materials',
];

const SYMBOLS = [
  'AURX', 'NVLX', 'SYNR', 'VELT', 'ORCA', 'PULSE', 'VANTA', 'KITE',
  'LUMO', 'AXIS', 'RIFT', 'IONA', 'QORA', 'HIVE', 'NOVA', 'CYGN',
  'DRFT', 'CORE', 'ZENI', 'NEXO', 'OMNI', 'LITE', 'APEX', 'LYRA',
];

function mulberry32(seed: number): () => number {
  return () => {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) || 1;
}

function pickSymbol(index: number): string {
  const base = SYMBOLS[index % SYMBOLS.length];
  if (index < SYMBOLS.length) return base;
  return `${base}${index % 10}`;
}

export class SyntheticProvider implements DataProvider {
  private seed: number;
  private universeSize: number;
  private cachedUniverse: AssetPoint[] | null = null;
  private cachedSnapshot: Record<string, { ret1d: number; vol: number; price: number }> | null = null;

  constructor(seed: number, universeSize: number) {
    this.seed = seed;
    this.universeSize = universeSize;
  }

  async getUniverse(): Promise<AssetPoint[]> {
    if (this.cachedUniverse) return this.cachedUniverse;

    const rand = mulberry32(this.seed);
    const points: AssetPoint[] = [];
    const spread = 18;
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));

    for (let i = 0; i < this.universeSize; i += 1) {
      const t = i / Math.max(1, this.universeSize - 1);
      const y = 1 - 2 * t;
      const radius = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = i * goldenAngle;
      const jitter = (rand() - 0.5) * 0.4;

      const x = (Math.cos(theta) * radius + jitter) * spread;
      const z = (Math.sin(theta) * radius + jitter) * spread;
      const symbol = pickSymbol(i);
      const sector = SECTORS[Math.floor(rand() * SECTORS.length)];
      const marketCap = 20 + rand() * 780;
      const ret1d = (rand() - 0.5) * 0.08;
      const vol = 0.15 + rand() * 0.6;

      points.push({
        symbol,
        sector,
        marketCap,
        ret1d,
        vol,
        x,
        y: y * spread * 0.7,
        z,
      });
    }

    this.cachedUniverse = points;
    return points;
  }

  async getLatestSnapshot(): Promise<Record<string, { ret1d: number; vol: number; price: number }>> {
    if (this.cachedSnapshot) return this.cachedSnapshot;

    const universe = await this.getUniverse();
    const snapshot: Record<string, { ret1d: number; vol: number; price: number }> = {};

    universe.forEach((asset, index) => {
      const rand = mulberry32(this.seed + index * 13);
      const price = 20 + rand() * 280;
      snapshot[asset.symbol] = {
        ret1d: asset.ret1d,
        vol: asset.vol,
        price,
      };
    });

    this.cachedSnapshot = snapshot;
    return snapshot;
  }

  async getCandles(symbol: string): Promise<Series> {
    const seed = this.seed + hashString(symbol);
    const rand = mulberry32(seed);
    const candles: Candle[] = [];
    const total = 120;
    let price = 120 + rand() * 80;

    for (let i = 0; i < total; i += 1) {
      const drift = (rand() - 0.5) * 2.1;
      const volatility = 1.5 + rand() * 2.5;
      const open = price + drift;
      const close = open + (rand() - 0.5) * volatility;
      const high = Math.max(open, close) + rand() * 2.2;
      const low = Math.min(open, close) - rand() * 2.0;
      const volume = 600 + rand() * 1800;

      candles.push({
        t: i,
        o: open,
        h: high,
        l: low,
        c: close,
        v: volume,
      });

      price = close;
    }

    return { symbol, candles };
  }
}
