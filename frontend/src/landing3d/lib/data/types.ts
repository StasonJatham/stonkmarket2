export interface AssetPoint {
  symbol: string;
  sector: string;
  marketCap: number;
  ret1d: number;
  vol: number;
  x: number;
  y: number;
  z: number;
}

export interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface Series {
  symbol: string;
  candles: Candle[];
}

export interface DataProvider {
  getUniverse(): Promise<AssetPoint[]>;
  getLatestSnapshot(): Promise<Record<string, { ret1d: number; vol: number; price: number }>>;
  getCandles(symbol: string): Promise<Series>;
}
