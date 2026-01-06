/**
 * StockDetailPanel - Clean, consolidated stock details view.
 * 
 * SINGLE SOURCE OF TRUTH: All data comes from QuantRecommendation.
 * No additional API calls - uses pre-loaded data from dashboard.
 * 
 * Sections:
 * 1. Header - Logo, ticker, price, change
 * 2. Chart - Price chart with dip markers
 * 3. Action Card - BUY/HOLD/SELL with reasoning
 * 4. Strategy - Performance metrics (if strategy beats B&H)
 * 5. Fundamentals - Key metrics summary
 */

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ComposedChart,
  ReferenceLine,
  Line,
} from 'recharts';
import { ChartTooltip, SimpleChartTooltipContent } from '@/components/ui/chart';
import { CHART_LINE_ANIMATION, CHART_ANIMATION } from '@/lib/chartConfig';
import type { ChartDataPoint, QuantRecommendation, BenchmarkType, ComparisonChartData } from '@/services/api';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { StockLogo } from '@/components/StockLogo';
import { ComparisonChart } from '@/components/ComparisonChart';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { 
  Building2, 
  BarChart3, 
  Calendar,
  ExternalLink,
  Zap,
  AlertTriangle,
  CheckCircle2,
  Shield,
  ChevronDown,
  MinusCircle,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface StockDetailPanelProps {
  recommendation: QuantRecommendation | null;
  chartData: ChartDataPoint[];
  chartPeriod: number;
  onPeriodChange: (days: number) => void;
  isLoadingChart: boolean;
  onClose: () => void;
  // Optional benchmark comparison
  benchmark?: BenchmarkType;
  comparisonData?: ComparisonChartData[];
  isLoadingBenchmark?: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const PERIODS = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: '5Y', days: 1825 },
];

const NORMALIZED_CHART_POINTS = 180;

// ============================================================================
// Helper Functions
// ============================================================================

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toFixed(0)}`;
}

function formatPrice(value: number | null): string {
  if (value == null) return '—';
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
}

function normalizeChartData<T extends { close: number; date: string }>(
  data: T[], 
  targetPoints: number = NORMALIZED_CHART_POINTS,
  preserveDates?: Set<string>  // Dates that MUST be included (e.g., trade dates)
): T[] {
  if (data.length === 0) return data;
  if (data.length <= targetPoints) return data;
  
  // If we have dates to preserve, first extract them
  const preservedPoints: T[] = [];
  const remainingData: T[] = [];
  
  if (preserveDates && preserveDates.size > 0) {
    for (const point of data) {
      const dateKey = point.date.split('T')[0];
      if (preserveDates.has(dateKey)) {
        preservedPoints.push(point);
      } else {
        remainingData.push(point);
      }
    }
    
    // Adjust target to account for preserved points
    const adjustedTarget = Math.max(10, targetPoints - preservedPoints.length);
    
    // Downsample the remaining data
    const downsampled = downsampleLTTB(remainingData, adjustedTarget);
    
    // Merge and sort by date
    const merged = [...preservedPoints, ...downsampled];
    merged.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    return merged;
  }
  
  return downsampleLTTB(data, targetPoints);
}

// LTTB (Largest Triangle Three Buckets) downsampling
function downsampleLTTB<T extends { close: number; date: string }>(
  data: T[],
  targetPoints: number
): T[] {
  if (data.length === 0) return data;
  if (data.length <= targetPoints) return data;
  
  const result: T[] = [data[0]];
  const bucketSize = (data.length - 2) / (targetPoints - 2);
  
  for (let i = 1; i < targetPoints - 1; i++) {
    const start = Math.floor((i - 1) * bucketSize) + 1;
    const end = Math.floor(i * bucketSize) + 1;
    const bucket = data.slice(start, end + 1);
    
    if (bucket.length === 0) {
      result.push(data[Math.floor(i * bucketSize) + 1]);
      continue;
    }
    
    // Pick point with largest deviation (LTTB-style)
    const lastPoint = result[result.length - 1];
    const nextBucketMid = Math.min(data.length - 1, Math.floor((i + 0.5) * bucketSize) + 1);
    const nextY = data[nextBucketMid].close;
    
    let maxArea = 0;
    let selectedPoint = bucket[0];
    
    for (const point of bucket) {
      const area = Math.abs(
        (lastPoint.close - nextY) * (new Date(point.date).getTime() - new Date(lastPoint.date).getTime())
      );
      if (area > maxArea) {
        maxArea = area;
        selectedPoint = point;
      }
    }
    
    result.push(selectedPoint);
  }
  
  result.push(data[data.length - 1]);
  return result;
}

// ============================================================================
// Main Component
// ============================================================================

export function StockDetailPanel({
  recommendation: rec,
  chartData,
  chartPeriod,
  onPeriodChange,
  isLoadingChart,
  onClose: _onClose,
  benchmark,
  comparisonData = [],
  isLoadingBenchmark = false,
}: StockDetailPanelProps) {
  const [dotsVisible, setDotsVisible] = useState(false);
  
  // Animate dots after chart loads - use a key to track chart changes
  const chartKey = `${chartPeriod}-${chartData.length}`;
  
  useEffect(() => {
    // Reset dots visibility when chart changes via cleanup + delayed set
    const timer = setTimeout(() => setDotsVisible(true), CHART_ANIMATION.animationDuration + 50);
    return () => {
      setDotsVisible(false);
      clearTimeout(timer);
    };
  }, [chartKey]);

  // Format chart data with trade markers embedded
  const { formattedChartData } = (() => {
    // First, collect all trade dates so we can preserve them during normalization
    const tradeDates = new Set<string>();
    if (rec?.strategy_recent_trades) {
      for (const trade of rec.strategy_recent_trades) {
        tradeDates.add(trade.entry_date.split('T')[0]);
        if (trade.exit_date) {
          tradeDates.add(trade.exit_date.split('T')[0]);
        }
      }
    }
    
    // Normalize chart data, preserving trade dates
    const normalized = normalizeChartData(chartData, NORMALIZED_CHART_POINTS, tradeDates);
    
    // Build a map of trade dates to marker info
    const tradeMap = new Map<string, { type: 'entry' | 'exit'; tradePrice: number; pnl_pct?: number }>();
    
    // Get chart date range to filter trades
    const chartStartDate = normalized.length > 0 ? new Date(normalized[0].date) : null;
    const chartEndDate = normalized.length > 0 ? new Date(normalized[normalized.length - 1].date) : null;
    
    if (rec?.strategy_recent_trades && chartStartDate && chartEndDate) {
      for (const trade of rec.strategy_recent_trades) {
        const entryDate = new Date(trade.entry_date);
        
        // Only include trades within the chart date range
        if (entryDate >= chartStartDate && entryDate <= chartEndDate) {
          // Find exact match or closest chart point for entry
          const entryDateKey = trade.entry_date.split('T')[0];
          const exactMatch = normalized.find(p => p.date.split('T')[0] === entryDateKey);
          
          if (exactMatch) {
            tradeMap.set(exactMatch.date, {
              type: 'entry',
              tradePrice: trade.entry_price,
            });
          }
        }
        
        // Find chart point for exit (if exists and within range)
        if (trade.exit_date && trade.exit_price) {
          const exitDate = new Date(trade.exit_date);
          if (exitDate >= chartStartDate && exitDate <= chartEndDate) {
            const exitDateKey = trade.exit_date.split('T')[0];
            const exactMatch = normalized.find(p => p.date.split('T')[0] === exitDateKey);
            
            if (exactMatch) {
              tradeMap.set(exactMatch.date, {
                type: 'exit',
                tradePrice: trade.exit_price,
                pnl_pct: trade.pnl_pct,
              });
            }
          }
        }
      }
    }
    
    // Build chart points with marker info embedded
    const chartPoints = normalized.map((point, index) => {
      const dateKey = point.date.split('T')[0];
      // Look up marker by date key (handles both full date and date-only formats)
      let marker = tradeMap.get(point.date);
      if (!marker) {
        // Try finding by date prefix
        for (const [key, val] of tradeMap.entries()) {
          if (key.split('T')[0] === dateKey) {
            marker = val;
            break;
          }
        }
      }
      return {
        ...point,
        displayDate: new Date(point.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        index, // Use index for ReferenceDot x position
        markerType: marker?.type ?? null,
        markerPrice: marker ? point.close : null, // Use chart close for alignment
        tradePrice: marker?.tradePrice ?? null,
        pnl_pct: marker?.pnl_pct ?? null,
      };
    });
    
    return { formattedChartData: chartPoints };
  })();

  // Calculate price change
  const priceChange = (() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  })();

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'var(--success)' : 'var(--danger)';

  // Empty state
  if (!rec) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent className="text-center text-muted-foreground py-12">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p>Select a stock to view details</p>
        </CardContent>
      </Card>
    );
  }

  const displayPrice = rec.last_price ?? (chartData.length > 0 ? chartData[chartData.length - 1].close : 0);
  const hasStrategy = rec.strategy_beats_bh && rec.strategy_name && rec.strategy_name !== 'switch_to_spy';
  
  // Use chart-derived change if rec.change_percent is null
  const displayChangePercent = rec.change_percent ?? priceChange;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.2 }}
      className="h-full"
    >
      <Card className="h-full flex flex-col">
        {/* Header */}
        <CardHeader className="py-3 px-4 flex-shrink-0 border-b border-border/50">
          <div className="flex items-start gap-3">
            <StockLogo symbol={rec.ticker} size="lg" className="shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold">{rec.ticker}</h2>
                <a 
                  href={`https://finance.yahoo.com/quote/${rec.ticker}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
              <p className="text-sm text-muted-foreground truncate">{rec.name || rec.ticker}</p>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-xl font-bold">{formatPrice(displayPrice)}</span>
                <span className={cn(
                  "text-sm font-medium",
                  displayChangePercent >= 0 ? "text-success" : "text-danger"
                )}>
                  {formatPercent(displayChangePercent)}
                </span>
              </div>
            </div>
          </div>
        </CardHeader>

        <ScrollArea className="flex-1 min-h-0">
          <div className="p-4 space-y-4 pb-8">
            {/* Period Selector */}
            <div className="flex gap-1">
              {PERIODS.map(({ label, days }) => (
                <Button
                  key={days}
                  variant={chartPeriod === days ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => onPeriodChange(days)}
                  className="px-3 h-7 text-xs"
                >
                  {label}
                </Button>
              ))}
            </div>

            {/* Chart */}
            <div className="h-48 w-full">
              {isLoadingChart ? (
                <Skeleton className="h-full w-full" />
              ) : formattedChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={formattedChartData}>
                    <defs>
                      <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis 
                      dataKey="displayDate" 
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis 
                      domain={['auto', 'auto']}
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `$${v}`}
                      width={50}
                    />
                    <ChartTooltip content={<SimpleChartTooltipContent />} />
                    <Area
                      type="monotone"
                      dataKey="close"
                      stroke={chartColor}
                      strokeWidth={2}
                      fill="url(#chartGradient)"
                      {...CHART_LINE_ANIMATION}
                    />
                    {/* Current price reference line */}
                    {dotsVisible && displayPrice && (
                      <ReferenceLine 
                        y={displayPrice} 
                        stroke="var(--muted-foreground)"
                        strokeDasharray="3 3"
                        strokeOpacity={0.5}
                      />
                    )}
                    {/* Historical trade entry markers (green dots) - Line with dots only */}
                    {dotsVisible && formattedChartData.some(p => p.markerType === 'entry') && (
                      <Line
                        type="monotone"
                        dataKey="markerPrice"
                        stroke="transparent"
                        strokeWidth={0}
                        dot={(props: { cx?: number; cy?: number; payload?: { markerType?: string | null } }) => {
                          if (props.payload?.markerType !== 'entry' || !props.cx || !props.cy) return <g />;
                          return (
                            <circle
                              cx={props.cx}
                              cy={props.cy}
                              r={6}
                              fill="hsl(142.1 76.2% 36.3%)"
                              stroke="white"
                              strokeWidth={2}
                            />
                          );
                        }}
                        activeDot={false}
                        isAnimationActive={false}
                      />
                    )}
                    {/* Historical trade exit markers (red dots) */}
                    {dotsVisible && formattedChartData.some(p => p.markerType === 'exit') && (
                      <Line
                        type="monotone"
                        dataKey="markerPrice"
                        stroke="transparent"
                        strokeWidth={0}
                        dot={(props: { cx?: number; cy?: number; payload?: { markerType?: string | null } }) => {
                          if (props.payload?.markerType !== 'exit' || !props.cx || !props.cy) return <g />;
                          return (
                            <circle
                              cx={props.cx}
                              cy={props.cy}
                              r={6}
                              fill="hsl(0 84.2% 60.2%)"
                              stroke="white"
                              strokeWidth={2}
                            />
                          );
                        }}
                        activeDot={false}
                        isAnimationActive={false}
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                  No chart data
                </div>
              )}
            </div>

            {/* Benchmark Comparison */}
            {benchmark && comparisonData.length > 0 && (
              <ComparisonChart 
                data={comparisonData}
                stockSymbol={rec.ticker}
                stockName={rec.name}
                benchmark={benchmark}
                isLoading={isLoadingBenchmark}
                height={120}
                compact
              />
            )}

            <Separator />

            {/* Timing Signal - Is NOW a good time to buy? */}
            {(() => {
              // Match Dashboard logic: Active Buy Signal = action is BUY or quant_mode is CERTIFIED_BUY/DIP_ENTRY
              const isActiveBuySignal = rec.action === 'BUY' || rec.quant_mode === 'CERTIFIED_BUY' || rec.quant_mode === 'DIP_ENTRY';
              // Additional: is there an active dip opportunity (even better timing)?
              const hasDipOpportunity = rec.dip_entry_is_buy_now || rec.is_unusual_dip || rec.opportunity_type === 'OUTLIER' || rec.opportunity_type === 'BOUNCE' || rec.opportunity_type === 'BOTH';
              
              return (
                <div className="space-y-3">
                  {/* Primary Signal: Buy Timing */}
                  <div className={cn(
                    "p-4 rounded-lg border",
                    isActiveBuySignal 
                      ? "bg-success/10 border-success/30" 
                      : "bg-muted/30 border-border"
                  )}>
                    <div className="flex items-start gap-3">
                      {isActiveBuySignal ? (
                        <div className="p-2 rounded-full bg-success/20">
                          {hasDipOpportunity ? (
                            <Zap className="h-5 w-5 text-success" />
                          ) : (
                            <CheckCircle2 className="h-5 w-5 text-success" />
                          )}
                        </div>
                      ) : (
                        <div className="p-2 rounded-full bg-muted">
                          <MinusCircle className="h-5 w-5 text-muted-foreground" />
                        </div>
                      )}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={cn(
                            "text-lg font-bold",
                            isActiveBuySignal ? "text-success" : ""
                          )}>
                            {isActiveBuySignal ? 'BUY' : 'HOLD'}
                          </span>
                          {hasDipOpportunity && (
                            <Badge variant="outline" className="text-[10px] border-amber-500/50 text-amber-600 bg-amber-500/10">
                              DIP ENTRY
                            </Badge>
                          )}
                          {rec.quant_mode && rec.quant_mode !== 'HOLD' && (
                            <Badge variant="outline" className="text-[10px]">
                              {rec.quant_mode.replace('_', ' ')}
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {isActiveBuySignal 
                            ? hasDipOpportunity 
                              ? 'Active buy signal with dip entry opportunity'
                              : 'Active buy signal - good quality stock'
                            : 'Not a buy signal at this time'}
                        </p>
                      </div>
                    </div>

                    {/* Key Metrics */}
                    {hasDipOpportunity && (
                      <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-current/10">
                        {rec.dip_vs_typical != null && (
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">Dip Size</p>
                            <p className={cn(
                              "text-sm font-bold",
                              rec.dip_vs_typical >= 1.5 ? "text-amber-600" : "text-success"
                            )}>
                              {rec.dip_vs_typical.toFixed(1)}x typical
                            </p>
                          </div>
                        )}
                        {rec.expected_recovery_days != null && (
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">Recovery</p>
                            <p className="text-sm font-bold">
                              ~{rec.expected_recovery_days.toFixed(0)}d
                            </p>
                          </div>
                        )}
                        {rec.win_rate != null && (
                          <div className="text-center">
                            <p className="text-xs text-muted-foreground">Win Rate</p>
                            <p className={cn(
                              "text-sm font-bold",
                              rec.win_rate >= 60 ? "text-success" : rec.win_rate < 50 ? "text-danger" : ""
                            )}>
                              {rec.win_rate.toFixed(0)}%
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Tail Event Alert */}
                    {rec.is_tail_event && rec.return_period_years != null && (
                      <div className="mt-3 p-2 rounded bg-amber-500/10 border border-amber-500/20">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                          <span className="text-xs font-medium text-amber-700 dark:text-amber-500">
                            Rare event: ~1 in {rec.return_period_years.toFixed(1)} years
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Secondary: Quant Scores */}
                  {(rec.quant_score_a != null || rec.quant_score_b != null) && (
                    <div className="flex gap-2">
                      {rec.quant_score_a != null && (
                        <div className="flex-1 p-2 rounded-lg bg-muted/30 text-center">
                          <p className="text-xs text-muted-foreground">Quality</p>
                          <p className="text-lg font-bold">{rec.quant_score_a.toFixed(0)}</p>
                        </div>
                      )}
                      {rec.quant_score_b != null && (
                        <div className="flex-1 p-2 rounded-lg bg-muted/30 text-center">
                          <p className="text-xs text-muted-foreground">Timing</p>
                          <p className="text-lg font-bold">{rec.quant_score_b.toFixed(0)}</p>
                        </div>
                      )}
                      {rec.opportunity_score != null && (
                        <div className="flex-1 p-2 rounded-lg bg-muted/30 text-center">
                          <p className="text-xs text-muted-foreground">Opportunity</p>
                          <p className="text-lg font-bold">{rec.opportunity_score.toFixed(0)}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Reason */}
                  {rec.best_chance_reason && (
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {rec.best_chance_reason}
                    </p>
                  )}
                </div>
              );
            })()}

            {/* Strategy Performance - Only if beats B&H */}
            {hasStrategy && (
              <details className="group" open>
                <summary className="flex items-center justify-between cursor-pointer py-2">
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-primary" />
                    <span className="text-sm font-semibold">Strategy Performance</span>
                    <Badge variant="outline" className="text-[10px] border-success/50 text-success bg-success/5">
                      ✓ Beats B&H
                    </Badge>
                  </div>
                  <ChevronDown className="h-4 w-4 text-muted-foreground group-open:rotate-180 transition-transform" />
                </summary>
                
                <div className="pt-3 space-y-3">
                  {/* Strategy Name */}
                  {rec.strategy_name && (
                    <p className="text-sm font-medium text-muted-foreground">
                      Strategy: <span className="text-foreground">{rec.strategy_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    </p>
                  )}
                  {/* Metrics Grid */}
                  <div className="grid grid-cols-4 gap-2">
                    <div className="text-center p-2 rounded-lg bg-muted/30">
                      <p className="text-xs text-muted-foreground">Win Rate</p>
                      <p className={cn(
                        "text-sm font-bold",
                        (rec.strategy_win_rate ?? 0) >= 60 ? "text-success" : 
                        (rec.strategy_win_rate ?? 0) < 50 ? "text-danger" : ""
                      )}>
                        {rec.strategy_win_rate?.toFixed(0) ?? '—'}%
                      </p>
                    </div>
                    <div className="text-center p-2 rounded-lg bg-muted/30">
                      <p className="text-xs text-muted-foreground">Total Return</p>
                      <p className={cn(
                        "text-sm font-bold",
                        (rec.strategy_total_return_pct ?? 0) >= 0 ? "text-success" : "text-danger"
                      )}>
                        {formatPercent(rec.strategy_total_return_pct)}
                      </p>
                    </div>
                    <div className="text-center p-2 rounded-lg bg-muted/30">
                      <p className="text-xs text-muted-foreground">vs B&H</p>
                      <p className={cn(
                        "text-sm font-bold",
                        (rec.strategy_vs_bh_pct ?? 0) >= 0 ? "text-success" : "text-danger"
                      )}>
                        {formatPercent(rec.strategy_vs_bh_pct)}
                      </p>
                    </div>
                    <div className="text-center p-2 rounded-lg bg-muted/30">
                      <p className="text-xs text-muted-foreground">Win Rate</p>
                      <p className="text-sm font-bold">
                        {rec.win_rate?.toFixed(0) ?? '—'}%
                      </p>
                    </div>
                  </div>

                  {/* Strategy Comparison */}
                  {rec.strategy_comparison && (
                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground">
                        ${rec.strategy_comparison.initial_capital.toLocaleString()} start + 
                        ${rec.strategy_comparison.monthly_contribution.toLocaleString()}/mo
                      </p>
                      <div className="space-y-1.5">
                        {rec.strategy_comparison.ranked_by_return.slice(0, 4).map((stratKey) => {
                          const strat = rec.strategy_comparison!.strategies[stratKey];
                          if (!strat) return null;
                          const isWinner = stratKey === rec.strategy_comparison!.winner;
                          return (
                            <div 
                              key={stratKey}
                              className={cn(
                                "flex items-center justify-between p-2 rounded-lg text-xs",
                                isWinner ? "bg-success/10 border border-success/30" : "bg-muted/20"
                              )}
                            >
                              <span className={cn("font-medium", isWinner && "text-success")}>
                                {strat.name}
                              </span>
                              <div className="text-right">
                                <span className="font-bold">
                                  ${(strat.final_value / 1000).toFixed(0)}k
                                </span>
                                <span className="text-muted-foreground ml-1">
                                  ({formatPercent(strat.total_return_pct)})
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </details>
            )}

            <Separator />

            {/* Fundamentals */}
            <details className="group" open>
              <summary className="flex items-center justify-between cursor-pointer py-2">
                <div className="flex items-center gap-2">
                  <Building2 className="h-4 w-4 text-primary" />
                  <span className="text-sm font-semibold">Fundamentals</span>
                </div>
                <ChevronDown className="h-4 w-4 text-muted-foreground group-open:rotate-180 transition-transform" />
              </summary>
              
              <div className="pt-3 grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground">Market Cap</p>
                  <p className="text-sm font-medium">{formatMarketCap(rec.market_cap)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Sector</p>
                  <p className="text-sm font-medium">{rec.sector || '—'}</p>
                </div>
                {rec.intrinsic_value != null && (
                  <div>
                    <p className="text-xs text-muted-foreground">Fair Value</p>
                    <div className="flex items-center gap-1">
                      <p className="text-sm font-medium">{formatPrice(rec.intrinsic_value)}</p>
                      {rec.valuation_status && (
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-[10px]",
                            rec.valuation_status === 'undervalued' && "text-success border-success/50",
                            rec.valuation_status === 'overvalued' && "text-danger border-danger/50"
                          )}
                        >
                          {rec.valuation_status}
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
                {rec.upside_pct != null && (
                  <div>
                    <p className="text-xs text-muted-foreground">Upside</p>
                    <p className={cn(
                      "text-sm font-medium",
                      rec.upside_pct >= 0 ? "text-success" : "text-danger"
                    )}>
                      {formatPercent(rec.upside_pct)}
                    </p>
                  </div>
                )}
                {rec.next_earnings_date && (
                  <div>
                    <p className="text-xs text-muted-foreground flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      Next Earnings
                    </p>
                    <p className="text-sm font-medium">
                      {new Date(rec.next_earnings_date).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric'
                      })}
                    </p>
                  </div>
                )}
              </div>

              {/* Domain Warnings */}
              {rec.domain_warnings && rec.domain_warnings.length > 0 && (
                <div className="mt-3 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600 mt-0.5" />
                    <div className="text-xs">
                      <p className="font-medium text-amber-700 dark:text-amber-500">Warnings</p>
                      <p className="text-muted-foreground">
                        {rec.domain_warnings.slice(0, 2).join(', ')}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </details>

            {/* Quant Evidence (collapsible) */}
            {rec.quant_evidence && (
              <>
                <Separator />
                <details className="group">
                  <summary className="flex items-center justify-between cursor-pointer py-2">
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-primary" />
                      <span className="text-sm font-semibold">Quant Evidence</span>
                      {rec.quant_gate_pass && (
                        <CheckCircle2 className="h-3.5 w-3.5 text-success" />
                      )}
                    </div>
                    <ChevronDown className="h-4 w-4 text-muted-foreground group-open:rotate-180 transition-transform" />
                  </summary>
                  
                  <div className="pt-3 grid grid-cols-3 gap-2 text-xs">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">P(Outperf)</p>
                          <p className="font-bold">{(rec.quant_evidence.p_outperf * 100).toFixed(0)}%</p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Probability of outperforming market</TooltipContent>
                    </Tooltip>
                    
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">DSR</p>
                          <p className="font-bold">{(rec.quant_evidence.dsr * 100).toFixed(0)}%</p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Deflated Sharpe Ratio</TooltipContent>
                    </Tooltip>
                    
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">P(Recovery)</p>
                          <p className="font-bold">{(rec.quant_evidence.p_recovery * 100).toFixed(0)}%</p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Probability of price recovery</TooltipContent>
                    </Tooltip>
                    
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">Expected</p>
                          <p className={cn(
                            "font-bold",
                            rec.quant_evidence.expected_value >= 0 ? "text-success" : "text-danger"
                          )}>
                            {formatPercent(rec.quant_evidence.expected_value * 100)}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Expected value from this position</TooltipContent>
                    </Tooltip>
                    
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">CVaR 5%</p>
                          <p className="font-bold text-danger">
                            {formatPercent(rec.quant_evidence.cvar_5 * 100)}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Conditional Value at Risk (worst 5% scenarios)</TooltipContent>
                    </Tooltip>
                    
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="text-center p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-muted-foreground">CI Range</p>
                          <p className="font-bold">
                            {formatPercent(rec.quant_evidence.ci_low * 100)} to {formatPercent(rec.quant_evidence.ci_high * 100)}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>95% confidence interval for returns</TooltipContent>
                    </Tooltip>
                  </div>
                </details>
              </>
            )}

            {/* AI Verdict */}
            {(rec.ai_summary || rec.ai_rating) && (
              <>
                <Separator />
                <details className="group" open>
                  <summary className="flex items-center justify-between cursor-pointer py-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold">AI Verdict</span>
                      {rec.ai_rating && (
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-[10px]",
                            rec.ai_rating === 'strong_buy' && "border-success/50 text-success bg-success/5",
                            rec.ai_rating === 'buy' && "border-success/50 text-success bg-success/5",
                            rec.ai_rating === 'hold' && "border-muted-foreground/50 text-muted-foreground",
                            rec.ai_rating === 'sell' && "border-danger/50 text-danger bg-danger/5",
                            rec.ai_rating === 'strong_sell' && "border-danger/50 text-danger bg-danger/5",
                          )}
                        >
                          {rec.ai_rating.replace('_', ' ').toUpperCase()}
                        </Badge>
                      )}
                    </div>
                    <ChevronDown className="h-4 w-4 text-muted-foreground group-open:rotate-180 transition-transform" />
                  </summary>
                  <div className="pt-2">
                    {rec.ai_summary && (
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {rec.ai_summary}
                      </p>
                    )}
                  </div>
                </details>
              </>
            )}
          </div>
        </ScrollArea>
      </Card>
    </motion.div>
  );
}

// ============================================================================
// Skeleton
// ============================================================================

export function StockDetailPanelSkeleton() {
  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4 border-b border-border/50">
        <div className="flex items-start gap-3">
          <Skeleton className="h-12 w-12 rounded-lg" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-5 w-20" />
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-6 w-24" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 space-y-4">
        <div className="flex gap-1">
          {PERIODS.map(({ days }) => (
            <Skeleton key={days} className="h-7 w-10" />
          ))}
        </div>
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-24 w-full" />
      </CardContent>
    </Card>
  );
}
