import { useState } from 'react';
import { Building2 } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';
import { cn } from '@/lib/utils';

interface StockLogoProps {
  symbol: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  showFallbackIcon?: boolean;
}

// Size configurations
const sizeConfig = {
  xs: { container: 'w-6 h-6', icon: 'w-3 h-3', padding: 'p-0.5' },
  sm: { container: 'w-8 h-8', icon: 'w-4 h-4', padding: 'p-0.5' },
  md: { container: 'w-10 h-10', icon: 'w-5 h-5', padding: 'p-0.5' },
  lg: { container: 'w-12 h-12', icon: 'w-6 h-6', padding: 'p-1' },
  xl: { container: 'w-14 h-14', icon: 'w-7 h-7', padding: 'p-1.5' },
};

// Get stock logo URL from backend logo API (handles Logo.dev + favicon fallback)
function getStockLogoUrl(symbol: string, theme: 'light' | 'dark' = 'light'): string {
  return `/api/logos/${symbol.toUpperCase()}?theme=${theme}`;
}

/**
 * StockLogo - Displays a company logo with automatic fallback
 * 
 * Fetches logos from the backend Logo.dev API which handles:
 * - Logo.dev API fetch
 * - Google Favicon fallback
 * - Caching in database
 * Falls back to Building icon if all else fails.
 */
export function StockLogo({ 
  symbol, 
  size = 'md',
  className,
  showFallbackIcon = true,
}: StockLogoProps) {
  const { resolvedTheme } = useTheme();
  const [logoError, setLogoError] = useState(false);
  
  const config = sizeConfig[size];
  
  // Get logo URL from backend (backend handles all fallback logic)
  const logoUrl = (() => {
    return getStockLogoUrl(symbol, resolvedTheme);
  })();

  return (
    <div 
      className={cn(
        config.container,
        'rounded-lg bg-muted flex items-center justify-center overflow-hidden border border-border/50',
        className
      )}
    >
      {!logoError ? (
        <img 
          src={logoUrl}
          alt={`${symbol} logo`}
          className={cn('w-full h-full object-contain rounded-lg', config.padding)}
          loading="lazy"
          decoding="async"
          onError={() => setLogoError(true)}
        />
      ) : showFallbackIcon ? (
        <Building2 className={cn(config.icon, 'text-muted-foreground')} />
      ) : null}
    </div>
  );
}
