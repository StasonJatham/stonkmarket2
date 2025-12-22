import { useState } from 'react';
import { motion } from 'framer-motion';
import type { DipStock } from '@/services/api';
import { Badge } from '@/components/ui/badge';
import { TrendingDown } from 'lucide-react';

interface DipTickerProps {
  stocks: DipStock[];
  onSelectStock: (symbol: string) => void;
  isLoading?: boolean;
}

export function DipTicker({ stocks, onSelectStock, isLoading }: DipTickerProps) {
  const [isPaused, setIsPaused] = useState(false);
  
  // Take top 10 dips for ticker
  const tickerStocks = stocks.slice(0, 10);
  
  // Duplicate for seamless loop
  const duplicatedStocks = [...tickerStocks, ...tickerStocks];

  if (isLoading || tickerStocks.length === 0) {
    return null;
  }

  return (
    <div 
      className="relative overflow-hidden py-2 border-y border-border/50 bg-muted/20"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <motion.div
        className="flex gap-3 whitespace-nowrap"
        animate={{
          x: isPaused ? undefined : [0, -50 * tickerStocks.length],
        }}
        transition={{
          x: {
            duration: tickerStocks.length * 4,
            repeat: Infinity,
            ease: "linear",
          },
        }}
      >
        {duplicatedStocks.map((stock, i) => (
          <button
            key={`${stock.symbol}-${i}`}
            onClick={() => onSelectStock(stock.symbol)}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border/50 bg-background hover:bg-muted/50 transition-colors cursor-pointer shrink-0"
          >
            <TrendingDown className="h-3 w-3 text-danger" />
            <span className="font-medium text-sm">{stock.symbol}</span>
            <Badge variant="outline" className="text-xs px-1.5 py-0 h-5 text-danger border-danger/30">
              {(stock.depth * 100).toFixed(1)}%
            </Badge>
            <span className="text-xs text-muted-foreground">
              ${stock.last_price.toFixed(2)}
            </span>
          </button>
        ))}
      </motion.div>
      
      {/* Fade edges */}
      <div className="absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-background to-transparent pointer-events-none" />
      <div className="absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background to-transparent pointer-events-none" />
    </div>
  );
}
