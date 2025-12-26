type DataMode = 'synthetic' | 'real';

interface OverlayProps {
  dataMode: DataMode;
  onToggleMode: () => void;
  onCta: () => void;
}

export function Overlay({ dataMode, onToggleMode, onCta }: OverlayProps) {
  return (
    <div className="landing3d-overlay">
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Trading Floor Energy</p>
          <h1>Markets, but make it cinematic.</h1>
          <p>
            A scroll-driven journey through momentum, liquidity, and price discovery.
            Built for real-time data, rendered with surgical performance.
          </p>
        </div>
      </section>

      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Market Galaxy</p>
          <h2>Every asset, mapped as a living constellation.</h2>
          <p>
            Hover any orb to surface its synthetic snapshot: sector, cap, and daily heat.
          </p>
        </div>
      </section>

      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Chart Reveal</p>
          <h2>Candles rise. Flow emerges.</h2>
          <p>
            Watch the price structure assemble as liquidity paints the line. Designed to
            swap in real ticks without changing the scene.
          </p>
        </div>
      </section>

      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Portfolio Command</p>
          <h2>Build conviction with signal-grade clarity.</h2>
          <p>
            {dataMode === 'synthetic'
              ? 'Synthetic demo data is powering this view. Swap in the real provider to pipe live universe stats and candles into the same narrative.'
              : 'Real provider mode is selected. Wire your API calls to unlock live universe stats and candles.'}
          </p>
          <div className="landing3d-badges">
            <span className="landing3d-badge">
              {dataMode === 'synthetic' ? 'Synthetic demo data' : 'Real provider selected'}
            </span>
            <span className="landing3d-note">Hook in real data via realProvider.stub.ts</span>
          </div>
          <div className="landing3d-cta-row">
            <button className="landing3d-cta" type="button" onClick={onCta}>
            Launch the platform
            </button>
            <button className="landing3d-toggle" type="button" onClick={onToggleMode}>
              {dataMode === 'synthetic' ? 'Use real provider' : 'Use synthetic demo'}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
