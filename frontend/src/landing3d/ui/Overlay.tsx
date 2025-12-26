type DataMode = 'synthetic' | 'real';

interface OverlayProps {
  dataMode: DataMode;
  onToggleMode: () => void;
  onCta: () => void;
}

export function Overlay({ dataMode, onToggleMode, onCta }: OverlayProps) {
  return (
    <div className="landing3d-overlay">
      {/* Section 1: Trading Floor */}
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Trading Floor</p>
          <h1>Markets, but make it cinematic.</h1>
          <p>
            A scroll-driven journey through momentum, liquidity, and price discovery.
            Thousands of data points rendered in real-time.
          </p>
        </div>
      </section>

      {/* Section 2: Market Galaxy */}
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Market Galaxy</p>
          <h2>Every asset, mapped as a living constellation.</h2>
          <p>
            Thousands of stocks visualized as orbs. Size encodes market cap,
            color reveals momentum. Hover to explore.
          </p>
        </div>
      </section>

      {/* Section 3: Market City */}
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Market Skyline</p>
          <h2>Sector performance, built into architecture.</h2>
          <p>
            Buildings rise with earnings. Colors shift with sentiment.
            A city that breathes with the market.
          </p>
        </div>
      </section>

      {/* Section 4: Charts in Space */}
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Chart Reveal</p>
          <h2>Candles rise. Flow emerges.</h2>
          <p>
            Watch price structure assemble as liquidity paints the line.
            Heat surfaces pulse with sector momentum.
          </p>
        </div>
      </section>

      {/* Section 5: CTA */}
      <section className="landing3d-section">
        <div className="landing3d-copy">
          <p className="landing3d-kicker">Command Center</p>
          <h2>Build conviction with signal-grade clarity.</h2>
          <p>
            {dataMode === 'synthetic'
              ? 'Synthetic demo data is powering this view. The same engine drives live market data.'
              : 'Real provider mode is selected. Wire your API calls to unlock live data.'}
          </p>
          <div className="landing3d-badges">
            <span className="landing3d-badge">
              {dataMode === 'synthetic' ? 'Demo Mode' : 'Live Mode'}
            </span>
          </div>
          <div className="landing3d-cta-row">
            <button className="landing3d-cta" type="button" onClick={onCta}>
              Launch Dashboard
            </button>
            <button className="landing3d-toggle" type="button" onClick={onToggleMode}>
              {dataMode === 'synthetic' ? 'Switch to live' : 'Switch to demo'}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
