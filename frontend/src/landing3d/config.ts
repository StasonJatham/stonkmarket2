import { SyntheticProvider } from './lib/data/syntheticProvider';

export const LANDING3D_CONFIG = {
  seed: 42,
  universeSize: 2800,
  chartSymbol: 'AURX',
  dataProvider: new SyntheticProvider(42, 2800),
  ctaHref: '/suggest',
};
