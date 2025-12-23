import { useMemo, memo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import type { DipStock, ChartDataPoint } from '@/services/api';
import { prefetchStockChart, prefetchStockInfo } from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { TrendingUp, TrendingDown, ChevronRight } from 'lucide-react';

interface StockCardProps {
  stock: DipStock;
  chartData?: ChartDataPoint[];
  isLoading?: boolean;
  isSelected?: boolean;
  onClick?: () => void;
}

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toFixed(0)}`;
}

function formatPercent(value: number | null | undefined, showSign = true): string {
  if (value === null || value === undefined) return '—';
  const sign = showSign ? (value >= 0 ? '+' : '') : '';
  return `${sign}${value.toFixed(2)}%`;
}

function formatDipPercent(depth: number): string {
  // Depth is a positive fraction (0.15 = 15% dip from peak)
  // Display as negative percentage since it's a dip down
  const pct = depth * 100;
  return `-${pct.toFixed(1)}%`;
}

// Memoize the component to prevent unnecessary re-renders
export const StockCard = memo(function StockCard({ 
  stock, 
  chartData, 
  isLoading, 
  isSelected, 
  onClick 
}: StockCardProps) {
  const isPositive = (stock.change_percent ?? 0) >= 0;
  const chartColor = isPositive ? 'var(--success)' : 'var(--danger)';

  const miniChartData = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    // Take last 30 points for mini chart
    return chartData.slice(-30).map((p, i) => ({ x: i, y: p.close }));
  }, [chartData]);

  if (isLoading) {
    return (
      <Card className="overflow-hidden">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <Skeleton className="h-5 w-16" />
              <Skeleton className="h-4 w-32" />
            </div>
            <Skeleton className="h-12 w-24" />
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prefetch data when user hovers over card
  const handleMouseEnter = useCallback(() => {
    prefetchStockChart(stock.symbol, 365);
    prefetchStockInfo(stock.symbol);
  }, [stock.symbol]);

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      transition={{ duration: 0.15 }}
      className="h-full"
      onMouseEnter={handleMouseEnter}
    >
      <Card 
        className={`overflow-hidden cursor-pointer transition-all duration-200 hover:shadow-md h-full ${
          isSelected ? 'ring-2 ring-foreground' : ''
        }`}
        onClick={onClick}
      >
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            {/* Stock Info */}
            <div className="flex-1 min-w-0 overflow-hidden">
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="font-semibold text-lg shrink-0">{stock.symbol}</span>
                  {stock.sector && (
                    <Badge variant="secondary" className="text-[10px] font-normal px-1.5 py-0 h-4 truncate max-w-[80px] sm:max-w-[120px]">
                      {stock.sector}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground truncate">
                  {stock.name || stock.symbol}
                </p>
              </div>
            </div>

            {/* Mini Chart */}
            {miniChartData.length > 0 && (
              <div className="w-20 h-10 hidden sm:block">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={miniChartData}>
                    <defs>
                      <linearGradient id={`mini-gradient-${stock.symbol}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="y"
                      stroke={chartColor}
                      strokeWidth={1.5}
                      fill={`url(#mini-gradient-${stock.symbol})`}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Price & Change */}
            <div className="text-right shrink-0">
              <div className="font-semibold text-lg font-mono">
                ${stock.last_price.toFixed(2)}
              </div>
              {stock.change_percent !== null && stock.change_percent !== undefined ? (
                <div className={`flex items-center justify-end gap-1 text-sm ${
                  stock.change_percent >= 0 ? 'text-success' : 'text-danger'
                }`}>
                  {stock.change_percent >= 0 ? (
                    <TrendingUp className="h-3.5 w-3.5" />
                  ) : (
                    <TrendingDown className="h-3.5 w-3.5" />
                  )}
                  <span className="font-medium">
                    {formatPercent(stock.change_percent)}
                  </span>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">—</div>
              )}
            </div>

            {/* Arrow */}
            <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0" />
          </div>

          {/* Expanded Details Row */}
          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/50 text-xs text-muted-foreground">
            <div>
              <span className="text-foreground/60">Dip:</span>{' '}
              <span className="font-medium text-danger">
                {formatDipPercent(stock.depth)}
              </span>
            </div>
            <div>
              <span className="text-foreground/60">MCap:</span>{' '}
              <span className="font-medium">{formatMarketCap(stock.market_cap)}</span>
            </div>
            {stock.pe_ratio && (
              <div>
                <span className="text-foreground/60">P/E:</span>{' '}
                <span className="font-medium">{stock.pe_ratio.toFixed(1)}</span>
              </div>
            )}
            {stock.days_since_dip && (
              <div className="hidden sm:block">
                <span className="text-foreground/60">Days in dip:</span>{' '}
                <span className="font-medium">{stock.days_since_dip}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
});

export function StockCardSkeleton() {
  return (
    <Card className="overflow-hidden h-full">
      <CardContent className="p-4">
        <div className="flex items-center gap-4">
          <div className="flex-1 space-y-2">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-4 w-32" />
          </div>
          <Skeleton className="h-10 w-20 hidden sm:block" />
          <div className="text-right space-y-1">
            <Skeleton className="h-5 w-16 ml-auto" />
            <Skeleton className="h-4 w-12 ml-auto" />
          </div>
          <Skeleton className="h-5 w-5" />
        </div>
        <div className="flex gap-4 mt-3 pt-3 border-t border-border/50">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-16" />
        </div>
      </CardContent>
    </Card>
  );
}
