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
    navigate(`/?stock=${symbol}`);
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
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-muted-foreground">
            {/* Main text */}
            <p>
              StonkMarket • <span className="text-warning/80">Not financial advice</span>
            </p>
            
            {/* Links and Color Controls */}
            <div className="flex items-center gap-4 flex-wrap justify-center">
              {/* Color pickers - always visible, but dimmed when colorblind mode is on */}
              <div className={colorblindMode ? 'opacity-40 pointer-events-none' : ''}>
                <ColorPickerInline
                  upColor={customColors.up}
                  downColor={customColors.down}
                  onUpChange={(color) => setCustomColors({ ...customColors, up: color })}
                  onDownChange={(color) => setCustomColors({ ...customColors, down: color })}
                  onReset={resetColors}
                />
              </div>
              <span className="text-border">•</span>
              <button 
                onClick={() => setColorblindMode(!colorblindMode)}
                className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
                title={colorblindMode ? 'Disable colorblind mode' : 'Enable colorblind mode (uses blue/orange instead of green/red)'}
              >
                {colorblindMode ? (
                  <Eye className="h-3 w-3" />
                ) : (
                  <EyeOff className="h-3 w-3" />
                )}
                <span className="hidden sm:inline">Colorblind</span>
              </button>
              <span className="text-border">•</span>
              <Link to="/privacy" className="hover:text-foreground transition-colors">
                Privacy
              </Link>
              <Link to="/imprint" className="hover:text-foreground transition-colors">
                Imprint
              </Link>
              <Link to="/contact" className="inline-flex items-center gap-1 hover:text-foreground transition-colors">
                <Mail className="h-3 w-3" />
                Contact
              </Link>
              <span className="text-border">•</span>
              {decoded ? (
                <a 
                  href={getPayPalLink() || '#'} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-primary hover:text-primary/80 transition-colors"
                >
                  <Heart className="h-3 w-3" />
                  Donate
                </a>
              ) : (
                <button 
                  onClick={decode}
                  className="inline-flex items-center gap-1 text-primary hover:text-primary/80 transition-colors"
                >
                  <Heart className="h-3 w-3" />
                  Donate
                </button>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
