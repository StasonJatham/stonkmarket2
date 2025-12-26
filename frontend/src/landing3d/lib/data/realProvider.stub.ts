import type { AssetPoint, DataProvider, Series } from './types';

export class RealDataProvider implements DataProvider {
  async getUniverse(): Promise<AssetPoint[]> {
    throw new Error('Not implemented: connect to your real data source here.');
  }

  async getLatestSnapshot(): Promise<Record<string, { ret1d: number; vol: number; price: number }>> {
    throw new Error('Not implemented: connect to your real data source here.');
  }

  async getCandles(_symbol: string): Promise<Series> {
    throw new Error('Not implemented: connect to your real data source here.');
  }
}
