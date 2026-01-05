import { useState, useRef, useEffect } from 'react';
import type { DipStock } from '@/services/api';
import { TrendingDown, TrendingUp, CircleDot } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';

// Extended stock type with optional action field
export interface TickerStock extends DipStock {
  action?: 'BUY' | 'SELL' | 'HOLD';
}

interface DipTickerProps {
  stocks: TickerStock[];
  onSelectStock: (symbol: string) => void;
  isLoading?: boolean;
}

export function DipTicker({ stocks, onSelectStock, isLoading }: DipTickerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const pausedOffsetRef = useRef<number>(0);
  const isPausedRef = useRef(isPaused);
  const { colorblindMode, customColors } = useTheme();
  
  // Keep ref in sync with state (in effect to avoid render-time ref access)
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);
  
  // Get colors - use colorblind colors if enabled, otherwise custom colors
  const upColor = colorblindMode ? '#3b82f6' : customColors.up;
  const downColor = colorblindMode ? '#f97316' : customColors.down;
  
  // Show all available stocks up to 40
  const tickerStocks = stocks.slice(0, 40);
  
  // Speed in pixels per second (slower for readability)
  const speed = 20;

  useEffect(() => {
    if (tickerStocks.length === 0) return;
    
    const animate = (timestamp: number) => {
      if (!lastTimeRef.current) {
        lastTimeRef.current = timestamp;
      }
      
      const delta = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;
      
      if (!isPausedRef.current && contentRef.current) {
        const contentWidth = contentRef.current.scrollWidth / 2;
        
        setOffset(prev => {
          const newOffset = prev + (speed * delta) / 1000;
          // Reset when we've scrolled through one full set
          return newOffset >= contentWidth ? 0 : newOffset;
        });
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [tickerStocks.length]);

  const handleMouseEnter = () => {
    pausedOffsetRef.current = offset;
    setIsPaused(true);
  };

  const handleMouseLeave = () => {
    // Resume from where we left off, no jump
    lastTimeRef.current = 0;
    setIsPaused(false);
  };

  if (isLoading || tickerStocks.length === 0) {
    return null;
  }

  // Duplicate stocks for seamless loop
  const duplicatedStocks = [...tickerStocks, ...tickerStocks];

  return (
    <div 
      ref={containerRef}
      className="relative overflow-hidden py-1.5 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-border/40"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div
        ref={contentRef}
        className="flex gap-6 whitespace-nowrap"
        style={{ transform: `translateX(-${offset}px)` }}
      >
        {duplicatedStocks.map((stock, i) => {
          const changePercent = stock.change_percent ?? 0;
          const isPositiveChange = changePercent >= 0;
          const action = stock.action;
          
          // Action colors
          const actionColor = action === 'BUY' ? upColor 
            : action === 'SELL' ? downColor 
            : 'hsl(var(--muted-foreground))';
          
          return (
            <button
              key={`${stock.symbol}-${i}`}
              onClick={() => onSelectStock(stock.symbol)}
              className="inline-flex items-center gap-2 text-xs hover:bg-muted/50 px-2 py-0.5 rounded transition-colors cursor-pointer shrink-0"
            >
              {/* Action indicator */}
              {action && (
                <span 
                  className="inline-flex items-center gap-0.5 font-semibold text-[10px] px-1 py-0.5 rounded"
                  style={{ 
                    color: actionColor,
                    backgroundColor: action === 'BUY' ? `${upColor}15` 
                      : action === 'SELL' ? `${downColor}15`
                      : 'hsl(var(--muted)/0.3)'
                  }}
                >
                  <CircleDot className="h-2.5 w-2.5" />
                  {action}
                </span>
              )}
              <span className="font-semibold text-foreground">{stock.symbol}</span>
              <span className="text-muted-foreground">${stock.last_price.toFixed(2)}</span>
              {/* Daily change percent - primary metric */}
              {stock.change_percent !== null && (
                <span 
                  className="inline-flex items-center gap-0.5 font-mono"
                  style={{ color: isPositiveChange ? upColor : downColor }}
                >
                  {isPositiveChange ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  {isPositiveChange ? '+' : ''}{changePercent.toFixed(1)}%
                </span>
              )}
            </button>
          );
        })}
      </div>
      
      {/* Fade edges */}
      <div className="absolute inset-y-0 left-0 w-12 bg-gradient-to-r from-background to-transparent pointer-events-none" />
      <div className="absolute inset-y-0 right-0 w-12 bg-gradient-to-l from-background to-transparent pointer-events-none" />
    </div>
  );
}
