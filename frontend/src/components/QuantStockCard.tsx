import { useMemo, memo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import { CHART_MINI_ANIMATION } from '@/lib/chartConfig';
import type { QuantRecommendation, ChartDataPoint } from '@/services/api';
import { prefetchStockChart, prefetchStockInfo } from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ChevronRight, Minus, ArrowUp, ArrowDown, Activity } from 'lucide-react';

interface QuantStockCardProps {
  stock: QuantRecommendation;
  chartData?: ChartDataPoint[];
  isLoading?: boolean;
  isSelected?: boolean;
  onClick?: () => void;
}

function formatPercent(value: number | null | undefined, showSign = true): string {
  if (value === null || value === undefined) return '—';
  const pct = value * 100;
  const sign = showSign ? (pct >= 0 ? '+' : '') : '';
  return `${sign}${pct.toFixed(2)}%`;
}

function formatMarginalUtility(value: number): string {
  // Marginal utility is a small number, show with more precision
  if (Math.abs(value) < 0.0001) return '0.00';
  return value.toFixed(4);
}

function getActionColor(action: 'BUY' | 'SELL' | 'HOLD'): string {
  switch (action) {
    case 'BUY':
      return 'text-success bg-success/10 border-success/20';
    case 'SELL':
      return 'text-danger bg-danger/10 border-danger/20';
    case 'HOLD':
      return 'text-muted-foreground bg-muted/10 border-muted/20';
  }
}

function getActionIcon(action: 'BUY' | 'SELL' | 'HOLD') {
  switch (action) {
    case 'BUY':
      return <ArrowUp className="h-3.5 w-3.5" />;
    case 'SELL':
      return <ArrowDown className="h-3.5 w-3.5" />;
    case 'HOLD':
      return <Minus className="h-3.5 w-3.5" />;
  }
}

function getDipBucketVariant(bucket: string | null): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (!bucket) return 'outline';
  switch (bucket.toLowerCase()) {
    case 'deep':
      return 'default'; // Green - good dip opportunity
    case 'moderate':
      return 'secondary'; // Neutral
    case 'shallow':
      return 'outline'; // Light - minor dip
    default:
      return 'outline';
  }
}

// Memoize the component to prevent unnecessary re-renders
export const QuantStockCard = memo(function QuantStockCard({ 
  stock, 
  chartData, 
  isLoading, 
  isSelected, 
  onClick 
}: QuantStockCardProps) {
  // Use mu_hat for chart color (positive = green, negative = red)
  const isPositive = stock.mu_hat >= 0;
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
    prefetchStockChart(stock.ticker, 90);
    prefetchStockInfo(stock.ticker);
  }, [stock.ticker]);

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
                  <span className="font-semibold text-lg shrink-0">{stock.ticker}</span>
                  <Badge 
                    variant="outline" 
                    className={`text-[10px] font-medium px-1.5 py-0 h-4 ${getActionColor(stock.action)}`}
                  >
                    <span className="flex items-center gap-0.5">
                      {getActionIcon(stock.action)}
                      {stock.action}
                    </span>
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground truncate">
                  {stock.name || stock.ticker}
                </p>
              </div>
            </div>

            {/* Mini Chart */}
            {miniChartData.length > 0 && (
              <div className="w-20 h-10 hidden sm:block">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={miniChartData}>
                    <defs>
                      <linearGradient id={`mini-gradient-${stock.ticker}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="y"
                      stroke={chartColor}
                      strokeWidth={1.5}
                      fill={`url(#mini-gradient-${stock.ticker})`}
                      {...CHART_MINI_ANIMATION}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Expected Return & Utility */}
            <div className="text-right shrink-0">
              <div className={`font-semibold text-lg font-mono ${stock.mu_hat >= 0 ? 'text-success' : 'text-danger'}`}>
                {formatPercent(stock.mu_hat)}
              </div>
              <div className="flex items-center justify-end gap-1 text-sm text-muted-foreground">
                <Activity className="h-3.5 w-3.5" />
                <span className="font-medium font-mono">
                  {formatMarginalUtility(stock.marginal_utility)}
                </span>
              </div>
            </div>

            {/* Arrow */}
            <ChevronRight className="h-5 w-5 text-muted-foreground shrink-0" />
          </div>

          {/* Expanded Details Row */}
          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/50 text-xs text-muted-foreground">
            <div>
              <span className="text-foreground/60">Target Wt:</span>{' '}
              <span className="font-medium">
                {(stock.target_weight * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-foreground/60">Risk:</span>{' '}
              <span className="font-medium">{(stock.risk_contribution * 100).toFixed(1)}%</span>
            </div>
            {stock.dip_bucket && (
              <Badge variant={getDipBucketVariant(stock.dip_bucket)} className="text-xs h-5">
                {stock.dip_bucket.charAt(0).toUpperCase() + stock.dip_bucket.slice(1)} Dip
              </Badge>
            )}
            {stock.dip_score !== null && (
              <div>
                <span className="text-foreground/60">Quality:</span>{' '}
                <span className={`font-medium ${stock.dip_score > 0.5 ? 'text-success' : stock.dip_score < 0.3 ? 'text-danger' : ''}`}>
                  {(stock.dip_score * 100).toFixed(0)}
                </span>
              </div>
            )}
            {stock.legacy_days_in_dip !== null && (
              <div className="hidden sm:block">
                <span className="text-foreground/60">Days:</span>{' '}
                <span className="font-medium">{stock.legacy_days_in_dip}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
});

export function QuantStockCardSkeleton() {
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
