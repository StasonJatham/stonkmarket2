import { Outlet, Link, useNavigate } from 'react-router-dom';
import { Header } from './Header';
import { DipTicker } from '@/components/DipTicker';
import { useDips } from '@/context/DipContext';
import { useTheme } from '@/context/ThemeContext';
import { useObfuscatedContact } from '@/lib/obfuscate';
import { ColorPickerInline } from '@/components/ui/color-picker';
import { Heart, Mail, Eye, EyeOff } from 'lucide-react';

export function Layout() {
  const { decoded, decode, getPayPalLink } = useObfuscatedContact();
  const { tickerStocks, isLoadingTicker } = useDips();
  const { colorblindMode, setColorblindMode, customColors, setCustomColors, resetColors } = useTheme();
  const navigate = useNavigate();

  const handleSelectStock = (symbol: string) => {
    // Navigate to dashboard with the stock selected
    // Add showAll=true so if the stock isn't in dips, we show all stocks
    navigate(`/?stock=${symbol}&showAll=true`);
  };

  return (
    <div className="relative flex min-h-screen flex-col bg-background">
      <DipTicker 
        stocks={tickerStocks} 
        isLoading={isLoadingTicker} 
        onSelectStock={handleSelectStock} 
      />
      <Header />
      <main className="flex-1">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          <Outlet />
        </div>
      </main>
      {/* Footer */}
      <footer className="border-t border-border/40 bg-background/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
          {/* Mobile: Stack vertically, Desktop: Single row */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between text-sm text-muted-foreground">
            
            {/* Row 1: Branding + Legal links */}
            <div className="flex items-center justify-between sm:justify-start gap-4">
              <span className="shrink-0">
                StonkMarket • <span className="text-warning/80">Not financial advice</span>
              </span>
              <div className="flex items-center gap-3 sm:hidden">
                <Link to="/privacy" className="hover:text-foreground transition-colors">Privacy</Link>
                <Link to="/imprint" className="hover:text-foreground transition-colors">Imprint</Link>
              </div>
            </div>
            
            {/* Row 2 (Mobile) / Right side (Desktop): Controls + Links */}
            <div className="flex items-center justify-between sm:justify-end gap-4">
              {/* Color controls group */}
              <div className="flex items-center gap-3">
                <div className={colorblindMode ? 'opacity-40 pointer-events-none' : ''}>
                  <ColorPickerInline
                    upColor={customColors.up}
                    downColor={customColors.down}
                    onUpChange={(color) => setCustomColors({ ...customColors, up: color })}
                    onDownChange={(color) => setCustomColors({ ...customColors, down: color })}
                    onReset={resetColors}
                  />
                </div>
                <button 
                  onClick={() => setColorblindMode(!colorblindMode)}
                  className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
                  title={colorblindMode ? 'Disable colorblind mode' : 'Enable colorblind mode'}
                >
                  {colorblindMode ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
                  <span className="text-xs">Colorblind</span>
                </button>
              </div>
              
              {/* Links group - Desktop only (mobile shows above) */}
              <div className="hidden sm:flex items-center gap-3">
                <Link to="/about" className="hover:text-foreground transition-colors">About</Link>
                <Link to="/privacy" className="hover:text-foreground transition-colors">Privacy</Link>
                <Link to="/imprint" className="hover:text-foreground transition-colors">Imprint</Link>
                <Link to="/contact" className="inline-flex items-center gap-1 hover:text-foreground transition-colors">
                  <Mail className="h-3 w-3" />
                  Contact
                </Link>
              </div>
              
              {/* Donate - Always visible */}
              {decoded ? (
                <a 
                  href={getPayPalLink() || '#'} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-primary hover:text-primary/80 transition-colors shrink-0"
                >
                  <Heart className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Donate</span>
                </a>
              ) : (
                <button 
                  onClick={decode}
                  className="inline-flex items-center gap-1 text-primary hover:text-primary/80 transition-colors shrink-0"
                >
                  <Heart className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Donate</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
