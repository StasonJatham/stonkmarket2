import { Outlet, Link } from 'react-router-dom';
import { Header } from './Header';
import { useObfuscatedContact } from '@/lib/obfuscate';
import { Heart, Mail } from 'lucide-react';

export function Layout() {
  const { decoded, decode, getPayPalLink } = useObfuscatedContact();

  return (
    <div className="relative flex min-h-screen flex-col bg-background">
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
            
            {/* Links */}
            <div className="flex items-center gap-4 flex-wrap justify-center">
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
