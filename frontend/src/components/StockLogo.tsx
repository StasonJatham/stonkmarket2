import { useState, useMemo } from 'react';
import { Building2 } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';
import { cn } from '@/lib/utils';

interface StockLogoProps {
  symbol: string;
  website?: string | null;
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

// Get stock logo URL from backend logo API
function getStockLogoUrl(symbol: string, theme: 'light' | 'dark' = 'light'): string {
  return `/api/logos/${symbol.toUpperCase()}?theme=${theme}`;
}

// Fallback to Google Favicon API if logo API fails
function getFaviconFallbackUrl(symbol: string, website?: string | null): string {
  // If website provided, extract domain
  if (website) {
    try {
      const url = new URL(website);
      return `https://www.google.com/s2/favicons?domain=${url.hostname}&sz=128`;
    } catch {
      // Invalid URL, fall through to domain map
    }
  }
  
  // Fallback: Map common symbols to company domains
  const cleanSymbol = symbol.replace('.', '-').toLowerCase();
  const domainMap: Record<string, string> = {
    'aapl': 'apple.com',
    'msft': 'microsoft.com',
    'googl': 'google.com',
    'goog': 'google.com',
    'amzn': 'amazon.com',
    'meta': 'meta.com',
    'tsla': 'tesla.com',
    'nvda': 'nvidia.com',
    'amd': 'amd.com',
    'intc': 'intel.com',
    'nflx': 'netflix.com',
    'crm': 'salesforce.com',
    'orcl': 'oracle.com',
    'adbe': 'adobe.com',
    'csco': 'cisco.com',
    'ibm': 'ibm.com',
    'pypl': 'paypal.com',
    'dis': 'disney.com',
    'nke': 'nike.com',
    'ko': 'coca-cola.com',
    'pep': 'pepsico.com',
    'wmt': 'walmart.com',
    'jpm': 'jpmorgan.com',
    'v': 'visa.com',
    'ma': 'mastercard.com',
    'bac': 'bankofamerica.com',
    'gs': 'goldmansachs.com',
    'ms': 'morganstanley.com',
    'pltr': 'palantir.com',
    'snow': 'snowflake.com',
    'coin': 'coinbase.com',
    'uber': 'uber.com',
    'lyft': 'lyft.com',
    'sq': 'squareup.com',
    'shop': 'shopify.com',
    'twlo': 'twilio.com',
    'zm': 'zoom.us',
    'docu': 'docusign.com',
  };
  const domain = domainMap[cleanSymbol] || `${cleanSymbol}.com`;
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=128`;
}

/**
 * StockLogo - Displays a company logo with automatic fallback
 * 
 * Fetches logos from the backend Logo.dev API with automatic fallback to
 * Google Favicon and finally a Building icon if all else fails.
 */
export function StockLogo({ 
  symbol, 
  website, 
  size = 'md',
  className,
  showFallbackIcon = true,
}: StockLogoProps) {
  const { resolvedTheme } = useTheme();
  const [logoError, setLogoError] = useState(false);
  const [useFallbackLogo, setUseFallbackLogo] = useState(false);
  
  const config = sizeConfig[size];
  
  // Get logo URL - use backend API first, fallback to Google Favicon on error
  const logoUrl = useMemo(() => {
    if (useFallbackLogo) {
      return getFaviconFallbackUrl(symbol, website);
    }
    return getStockLogoUrl(symbol, resolvedTheme);
  }, [symbol, website, resolvedTheme, useFallbackLogo]);
  
  const handleLogoError = () => {
    if (!useFallbackLogo) {
      // First error: try favicon fallback
      setUseFallbackLogo(true);
    } else {
      // Fallback also failed: show icon
      setLogoError(true);
    }
  };

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
          className={cn('w-full h-full object-contain', config.padding)}
          onError={handleLogoError}
        />
      ) : showFallbackIcon ? (
        <Building2 className={cn(config.icon, 'text-muted-foreground')} />
      ) : null}
    </div>
  );
}
