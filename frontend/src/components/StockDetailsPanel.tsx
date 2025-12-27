import { useMemo, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Line,
  ComposedChart,
  ReferenceDot,
  Label as RechartsLabel,
} from 'recharts';
import { ChartTooltip, SimpleChartTooltipContent } from '@/components/ui/chart';
import { CHART_LINE_ANIMATION, CHART_TRENDLINE_ANIMATION, CHART_ANIMATION } from '@/lib/chartConfig';
import type { DipStock, ChartDataPoint, StockInfo, BenchmarkType, ComparisonChartData, SignalTrigger, DipAnalysis, CurrentSignals } from '@/services/api';
import { getSignalTriggers, getDipAnalysis, getCurrentSignals } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ComparisonChart } from '@/components/ComparisonChart';
import { 
  TrendingUp, 
  TrendingDown, 
  Building2, 
  DollarSign, 
  BarChart3, 
  Calendar,
  ExternalLink,
  Activity,
  Percent,
  Target,
  Sparkles,
  PiggyBank,
  Scale,
  Banknote,
  Users,
  Zap,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  TrendingUpDown,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { StockLogo } from '@/components/StockLogo';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';

interface AiData {
  ai_rating: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell' | null;
  ai_reasoning: string | null;
  domain_analysis?: string | null;
  // Domain-specific analysis (sector-aware)
  domain_context?: string | null;
  domain_adjustment?: number | null;
  domain_adjustment_reason?: string | null;
  domain_risk_level?: string | null;
  domain_risk_factors?: string[] | null;
  domain_recovery_days?: number | null;
  domain_warnings?: string[] | null;
  volatility_regime?: string | null;
  volatility_percentile?: number | null;
  vs_sector_performance?: number | null;
  sector?: string | null;
}

interface StockDetailsPanelProps {
  stock: DipStock | null;
  chartData: ChartDataPoint[];
  stockInfo: StockInfo | null;
  chartPeriod: number;
  onPeriodChange: (days: number) => void;
  isLoadingChart: boolean;
  isLoadingInfo: boolean;
  onClose: () => void;
  // Benchmark props (optional)
  benchmark?: BenchmarkType;
  comparisonData?: ComparisonChartData[];
  isLoadingBenchmark?: boolean;
  // AI props (optional)
  aiData?: AiData | null;
  isLoadingAi?: boolean;
}

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toFixed(0)}`;
}

const periods = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
];

export function StockDetailsPanel({
  stock,
  chartData,
  stockInfo,
  chartPeriod,
  onPeriodChange,
  isLoadingChart,
  isLoadingInfo,
  onClose: _onClose,
  benchmark,
  comparisonData = [],
  isLoadingBenchmark = false,
  aiData,
  isLoadingAi = false,
}: StockDetailsPanelProps) {
  // Track when dots should be visible (after chart animation completes)
  const [dotsVisible, setDotsVisible] = useState(false);
  
  // Signal triggers for buy signal markers
  const [signalTriggers, setSignalTriggers] = useState<SignalTrigger[]>([]);
  const [showSignals, setShowSignals] = useState(true);
  
  // Quant analysis state
  const [dipAnalysis, setDipAnalysis] = useState<DipAnalysis | null>(null);
  const [currentSignals, setCurrentSignals] = useState<CurrentSignals | null>(null);
  const [isLoadingQuant, setIsLoadingQuant] = useState(false);
  
  // Fetch signal triggers when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      setSignalTriggers([]);
      return;
    }
    
    getSignalTriggers(stock.symbol, Math.max(chartPeriod + 30, 365))
      .then(setSignalTriggers)
      .catch(() => setSignalTriggers([]));
  }, [stock?.symbol, chartPeriod]);
  
  // Fetch quant analysis when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      setDipAnalysis(null);
      setCurrentSignals(null);
      return;
    }
    
    setIsLoadingQuant(true);
    Promise.all([
      getDipAnalysis(stock.symbol).catch(() => null),
      getCurrentSignals(stock.symbol).catch(() => null),
    ]).then(([dip, signals]) => {
      setDipAnalysis(dip);
      setCurrentSignals(signals);
      setIsLoadingQuant(false);
    });
  }, [stock?.symbol]);
  
  // When chart period or data changes, hide dots and show them after animation completes
  useEffect(() => {
    setDotsVisible(false);
    const timer = setTimeout(() => {
      setDotsVisible(true);
    }, CHART_ANIMATION.animationDuration + 50); // Add 50ms buffer
    return () => clearTimeout(timer);
  }, [chartPeriod, chartData]);

  const formattedChartData = useMemo(() => {
    return chartData.map((point) => ({
      ...point,
      displayDate: new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
    }));
  }, [chartData]);

  // Find ref high point index for marker
  const refHighIndex = useMemo(() => {
    if (chartData.length === 0) return -1;
    const refDate = chartData[0]?.ref_high_date;
    if (!refDate) return -1;
    return formattedChartData.findIndex(p => p.date === refDate);
  }, [chartData, formattedChartData]);

  // Get the display date for ref high point (for ReferenceDot x value)
  const refHighDisplayDate = useMemo(() => {
    if (refHighIndex < 0 || !formattedChartData[refHighIndex]) return null;
    return formattedChartData[refHighIndex].displayDate;
  }, [refHighIndex, formattedChartData]);

  // Get the display date for current price (last point)
  const currentDisplayDate = useMemo(() => {
    if (formattedChartData.length === 0) return null;
    return formattedChartData[formattedChartData.length - 1].displayDate;
  }, [formattedChartData]);

  // Get the current price for the reference line
  const currentPrice = useMemo(() => {
    if (formattedChartData.length === 0) return null;
    return formattedChartData[formattedChartData.length - 1]?.close;
  }, [formattedChartData]);

  // Get the ref high price
  const refHighPrice = useMemo(() => {
    if (chartData.length === 0) return null;
    return chartData[0]?.ref_high ?? null;
  }, [chartData]);

  // Create chart data with trendline, reference lines, and animated dot positions
  const chartDataWithTrendline = useMemo(() => {
    if (formattedChartData.length === 0) return [];
    
    return formattedChartData.map((point, index) => {
      // Only add trendline values at peak and current points
      let trendline: number | null = null;
      if (index === refHighIndex && refHighIndex >= 0) {
        trendline = point.close;
      } else if (index === formattedChartData.length - 1) {
        trendline = point.close;
      }
      
      return { 
        ...point, 
        trendline,
        // Horizontal reference lines (constant value across all points for smooth animation)
        currentPriceLine: currentPrice,
        refHighLine: refHighPrice,
        // Scatter point values - only non-null at specific indices for dot rendering
        refHighDot: index === refHighIndex ? point.close : null,
        currentDot: index === formattedChartData.length - 1 ? point.close : null,
      };
    });
  }, [formattedChartData, refHighIndex, currentPrice, refHighPrice]);

  // Find signal trigger points that match chart dates
  const signalPoints = useMemo(() => {
    if (!showSignals || signalTriggers.length === 0 || formattedChartData.length === 0) return [];
    
    // Create a map of date -> signal for quick lookup
    const signalMap = new Map<string, SignalTrigger>();
    signalTriggers.forEach(s => signalMap.set(s.date, s));
    
    // Find chart points that have matching signals
    return formattedChartData
      .filter(point => signalMap.has(point.date))
      .map(point => ({
        ...point,
        signal: signalMap.get(point.date)!,
      }));
  }, [showSignals, signalTriggers, formattedChartData]);

  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [chartData]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'var(--success)' : 'var(--danger)';

  // Use chart's latest close price if stock.last_price is missing/zero (quant data)
  const displayPrice = stock?.last_price && stock.last_price > 0 
    ? stock.last_price 
    : (chartData.length > 0 ? chartData[chartData.length - 1].close : 0);

  if (!stock) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent className="text-center text-muted-foreground py-12">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p>Select a stock to view details</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.2 }}
      className="h-full"
    >
      <Card className="h-full flex flex-col overflow-hidden">
        {/* Header - Fixed */}
        <CardHeader className="py-2 md:py-4 px-3 md:px-6 shrink-0 border-b border-border/50">
          <div className="flex flex-col gap-1">
            {/* Top row: Logo + Symbol + Price */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <StockLogo symbol={stock.symbol} size="lg" />
                <CardTitle className="text-xl md:text-2xl">{stock.symbol}</CardTitle>
              </div>
              <div className="text-right shrink-0">
                <div className="text-xl md:text-2xl font-bold font-mono">
                  ${displayPrice.toFixed(2)}
                </div>
                <div className={`flex items-center justify-end gap-1 text-sm ${
                  isPositive ? 'text-success' : 'text-danger'
                }`}>
                  {isPositive ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <TrendingDown className="h-4 w-4" />
                  )}
                  <span className="font-medium">
                    {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
            {/* Second row: Stock name (aligned with logo) */}
            <p className="text-xs md:text-sm text-muted-foreground">
              {stock.name || stock.symbol}
            </p>
          </div>
        </CardHeader>

        {/* Scrollable Content - Chart + Data */}
        <ScrollArea className="flex-1 min-h-0">
          {/* Chart Section */}
          <div className="px-3 md:px-6 py-2">
            {/* Period Selector + Dip Summary */}
            <div className="flex items-center justify-between mb-2">
              {/* Period Selector */}
              <div className="flex gap-1">
                {periods.map((p) => (
                  <Button
                    key={p.days}
                    variant={chartPeriod === p.days ? 'default' : 'ghost'}
                    size="sm"
                    className="h-6 px-2 text-xs"
                    onClick={() => onPeriodChange(p.days)}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
              {/* Compact Dip Info + Signals Toggle */}
              {stock && (
                <div className="flex items-center gap-3 text-xs">
                  <span className="font-medium text-danger">-{(stock.depth * 100).toFixed(1)}%</span>
                  {stock.days_since_dip && (
                    <span className="text-muted-foreground">{stock.days_since_dip}d</span>
                  )}
                  {/* Signals toggle */}
                  <div className="flex items-center gap-1.5 ml-1 border-l border-border pl-2">
                    <Zap className="h-3 w-3 text-success" />
                    <Switch
                      id="signals-toggle"
                      checked={showSignals}
                      onCheckedChange={setShowSignals}
                      className="h-4 w-7 data-[state=checked]:bg-success"
                    />
                    {signalTriggers.length > 0 && (
                      <span className="text-muted-foreground">({signalTriggers.length})</span>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {/* Recovery Banner */}
            {chartData[0]?.since_dip !== null && chartData[0]?.since_dip !== undefined && (
              <div className="flex items-center gap-3 py-1 px-2 rounded-md border border-border/50 bg-muted/30 mb-2 text-xs">
                <div className="flex items-center gap-1.5">
                  <TrendingDown className="h-3 w-3 text-muted-foreground" />
                  <span className="text-muted-foreground">Dip:</span>
                  <span className="font-medium text-danger">-{(stock.depth * 100).toFixed(1)}%</span>
                </div>
                {stock.days_since_dip && (
                  <div className="flex items-center gap-1.5">
                    <Calendar className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">{stock.days_since_dip}d</span>
                  </div>
                )}
                <div className="flex items-center gap-1.5 ml-auto">
                  <Target className="h-3 w-3 text-muted-foreground" />
                  <span className="text-muted-foreground">Recovery:</span>
                  <span className={`font-medium ${chartData[chartData.length - 1]?.since_dip && chartData[chartData.length - 1].since_dip! > 0 ? 'text-success' : 'text-danger'}`}>
                    {chartData[chartData.length - 1]?.since_dip ? (chartData[chartData.length - 1].since_dip! * 100).toFixed(1) : '0'}%
                  </span>
                </div>
              </div>
            )}

            {/* Chart */}
            <div className="h-40 lg:h-64 w-full">
              {/* Show comparison chart when benchmark is selected, otherwise show main chart */}
              {benchmark ? (
                <ComparisonChart
                  data={comparisonData}
                  stockSymbol={stock.symbol}
                  benchmark={benchmark}
                  isLoading={isLoadingBenchmark}
                  height="100%"
                  compact
                />
              ) : (
                // Always keep chart mounted for smooth interpolation animations
                <div className="relative h-full w-full">
                  {/* Loading overlay - fades in smoothly, keeps chart visible underneath */}
                  <div 
                    className={`absolute inset-0 bg-background/40 backdrop-blur-[1px] z-10 flex items-center justify-center transition-opacity duration-200 ${
                      isLoadingChart ? 'opacity-100' : 'opacity-0 pointer-events-none'
                    }`}
                  >
                    <div className="h-5 w-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  </div>
                  {formattedChartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                      {/* No key prop - keep chart mounted for smooth animation interpolation */}
                      <ComposedChart data={chartDataWithTrendline}>
                    <defs>
                      {/* Stable gradient ID (no period) for smooth transitions */}
                      <linearGradient id={`gradient-${stock.symbol}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="displayDate"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 10 }}
                      tickMargin={8}
                      minTickGap={40}
                    />
                    <YAxis
                      domain={['dataMin', 'dataMax']}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 10 }}
                      tickMargin={8}
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                      width={45}
                    />
                    {/* Current price reference line - animated horizontal line */}
                    <Line
                      type="linear"
                      dataKey="currentPriceLine"
                      stroke="var(--danger)"
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      strokeOpacity={0.7}
                      dot={false}
                      activeDot={false}
                      {...CHART_TRENDLINE_ANIMATION}
                    />
                    {/* Peak price reference line - animated horizontal line */}
                    <Line
                      type="linear"
                      dataKey="refHighLine"
                      stroke="var(--success)"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      strokeOpacity={0.5}
                      dot={false}
                      activeDot={false}
                      {...CHART_TRENDLINE_ANIMATION}
                    />
                    {/* Trendline connecting peak to current */}
                    <Line
                      type="linear"
                      dataKey="trendline"
                      stroke="var(--muted-foreground)"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      strokeOpacity={0.4}
                      dot={false}
                      activeDot={false}
                      connectNulls={true}
                      {...CHART_TRENDLINE_ANIMATION}
                    />
                    <ChartTooltip
                      content={
                        <SimpleChartTooltipContent
                          className="min-w-[140px]"
                          formatter={(value: number, name: string, payload: Record<string, unknown>) => {
                            // Only show for the 'close' data key
                            if (name !== 'close') return null;
                            return (
                              <div className="flex flex-col gap-1">
                                <div className="flex items-center justify-between gap-4">
                                  <span className="text-muted-foreground">Price</span>
                                  <span className="font-mono font-medium tabular-nums text-foreground">
                                    ${value.toFixed(2)}
                                  </span>
                                </div>
                                {payload.drawdown !== null && payload.drawdown !== undefined && (
                                  <div className="flex items-center justify-between gap-4">
                                    <span className="text-muted-foreground">Drawdown</span>
                                    <span className="font-mono font-medium tabular-nums text-danger">
                                      {((payload.drawdown as number) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                )}
                                {payload.since_dip !== null && payload.since_dip !== undefined && (
                                  <div className="flex items-center justify-between gap-4">
                                    <span className="text-muted-foreground">Since Dip</span>
                                    <span className={`font-mono font-medium tabular-nums ${(payload.since_dip as number) >= 0 ? 'text-success' : 'text-danger'}`}>
                                      {(payload.since_dip as number) >= 0 ? '+' : ''}{((payload.since_dip as number) * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                )}
                              </div>
                            );
                          }}
                          labelFormatter={(label: string) => (
                            <span className="font-medium text-foreground">{label}</span>
                          )}
                        />
                      }
                    />
                    <Area
                      type="linear"
                      dataKey="close"
                      stroke={chartColor}
                      strokeWidth={2}
                      fill={`url(#gradient-${stock.symbol})`}
                      dot={false}
                      activeDot={{
                        r: 5,
                        fill: chartColor,
                        stroke: 'var(--background)',
                        strokeWidth: 2,
                      }}
                      {...CHART_LINE_ANIMATION}
                    />
                    {/* Ref high dot (green) - appears after line animation completes */}
                    {dotsVisible && refHighDisplayDate && refHighPrice !== null && (
                      <ReferenceDot
                        x={refHighDisplayDate}
                        y={refHighPrice}
                        r={5}
                        fill="var(--success)"
                        stroke="var(--background)"
                        strokeWidth={2}
                      />
                    )}
                    {/* Current price dot (red) - appears after line animation completes */}
                    {dotsVisible && currentDisplayDate && currentPrice !== null && (
                      <ReferenceDot
                        x={currentDisplayDate}
                        y={currentPrice}
                        r={5}
                        fill="var(--danger)"
                        stroke="var(--background)"
                        strokeWidth={2}
                      />
                    )}
                    {/* Buy signal trigger dots with tooltip */}
                    {dotsVisible && showSignals && signalPoints.map((point, idx) => (
                      <ReferenceDot
                        key={`signal-${idx}`}
                        x={point.displayDate}
                        y={point.close}
                        r={8}
                        fill="var(--success)"
                        stroke="var(--background)"
                        strokeWidth={2}
                        shape={(props: { cx?: number; cy?: number }) => {
                          const { cx = 0, cy = 0 } = props;
                          const signal = point.signal;
                          const tooltipText = `${signal.signal_name}\n` +
                            `Win Rate: ${(signal.win_rate * 100).toFixed(0)}%\n` +
                            `Avg Return: ${signal.avg_return_pct >= 0 ? '+' : ''}${signal.avg_return_pct.toFixed(1)}%\n` +
                            `Hold: ${signal.holding_days} days\n` +
                            `Price: $${signal.price.toFixed(2)}`;
                          return (
                            <g style={{ cursor: 'pointer' }}>
                              <title>{tooltipText}</title>
                              <circle
                                cx={cx}
                                cy={cy}
                                r={8}
                                fill="var(--success)"
                                stroke="var(--background)"
                                strokeWidth={2}
                              />
                            </g>
                          );
                        }}
                      >
                        <RechartsLabel
                          position="top"
                          offset={10}
                          style={{
                            fontSize: 9,
                            fill: 'var(--foreground)',
                            fontWeight: 600,
                          }}
                        >
                          {`${(point.signal.win_rate * 100).toFixed(0)}%`}
                        </RechartsLabel>
                      </ReferenceDot>
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-muted-foreground">
                      No chart data available
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Data Section - Simplified for Decision Making */}
          <CardContent className="pt-3 pb-4 px-3 md:px-6">
            {/* HERO: Opportunity Score Card */}
            {(currentSignals || dipAnalysis) && !isLoadingQuant && (
              <div className={cn(
                "p-4 rounded-xl mb-4 text-center",
                currentSignals?.overall_action === 'STRONG_BUY' && "bg-gradient-to-br from-success/20 to-success/10 border-2 border-success/40",
                currentSignals?.overall_action === 'BUY' && "bg-gradient-to-br from-success/15 to-success/5 border border-success/30",
                currentSignals?.overall_action === 'HOLD' && "bg-gradient-to-br from-muted/40 to-muted/20 border border-border",
                currentSignals?.overall_action === 'SELL' && "bg-gradient-to-br from-warning/15 to-warning/5 border border-warning/30",
                currentSignals?.overall_action === 'STRONG_SELL' && "bg-gradient-to-br from-danger/20 to-danger/10 border-2 border-danger/40",
                !currentSignals?.overall_action && "bg-gradient-to-br from-muted/40 to-muted/20 border border-border"
              )}>
                <div className="flex items-center justify-center gap-2 mb-2">
                  {currentSignals?.overall_action === 'STRONG_BUY' && <Zap className="h-6 w-6 text-success" />}
                  {currentSignals?.overall_action === 'BUY' && <TrendingUp className="h-6 w-6 text-success" />}
                  {currentSignals?.overall_action === 'HOLD' && <Activity className="h-5 w-5 text-muted-foreground" />}
                  {currentSignals?.overall_action === 'SELL' && <TrendingDown className="h-5 w-5 text-warning" />}
                  {currentSignals?.overall_action === 'STRONG_SELL' && <XCircle className="h-6 w-6 text-danger" />}
                  <span className={cn(
                    "text-2xl font-bold",
                    currentSignals?.overall_action?.includes('BUY') && "text-success",
                    currentSignals?.overall_action === 'HOLD' && "text-foreground",
                    currentSignals?.overall_action?.includes('SELL') && "text-danger",
                  )}>
                    {currentSignals?.overall_action?.replace('_', ' ') || 'Analyzing...'}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  {currentSignals?.reasoning}
                </p>
              </div>
            )}

            {isLoadingQuant && (
              <div className="p-4 rounded-xl mb-4 bg-muted/20 border border-border">
                <Skeleton className="h-8 w-32 mx-auto mb-2" />
                <Skeleton className="h-4 w-3/4 mx-auto" />
              </div>
            )}

            {/* Quick Stats Row - Only the essentials */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground">Dip</p>
                <p className="text-lg font-bold text-danger">-{(stock.depth * 100).toFixed(0)}%</p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground">Win Rate</p>
                <p className={cn(
                  "text-lg font-bold",
                  currentSignals && currentSignals.buy_signals.length > 0 
                    ? (currentSignals.buy_signals[0].win_rate > 0.6 ? "text-success" : "text-foreground")
                    : "text-muted-foreground"
                )}>
                  {currentSignals?.buy_signals?.[0]?.win_rate 
                    ? `${(currentSignals.buy_signals[0].win_rate * 100).toFixed(0)}%`
                    : '—'}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-xs text-muted-foreground">Exp. Return</p>
                <p className={cn(
                  "text-lg font-bold",
                  currentSignals && currentSignals.buy_signals.length > 0 
                    ? (currentSignals.buy_signals[0].avg_return > 0 ? "text-success" : "text-danger")
                    : "text-muted-foreground"
                )}>
                  {currentSignals?.buy_signals?.[0]?.avg_return 
                    ? `+${currentSignals.buy_signals[0].avg_return.toFixed(0)}%`
                    : '—'}
                </p>
              </div>
            </div>

            {/* Risk & Context Card */}
            {(dipAnalysis || aiData) && (
              <div className="p-3 rounded-lg bg-muted/30 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Analysis</span>
                  <div className="flex gap-1">
                    {aiData?.volatility_regime && (
                      <Badge 
                        variant="outline" 
                        className={cn(
                          "text-xs",
                          aiData.volatility_regime === 'extreme' && "border-danger text-danger",
                          aiData.volatility_regime === 'high' && "border-warning text-warning",
                          aiData.volatility_regime === 'normal' && "border-muted-foreground text-muted-foreground",
                          aiData.volatility_regime === 'low' && "border-success text-success",
                        )}
                      >
                        {aiData.volatility_regime} vol
                      </Badge>
                    )}
                    {aiData?.domain_risk_level && (
                      <Badge 
                        variant="outline" 
                        className={cn(
                          "text-xs",
                          aiData.domain_risk_level === 'high' && "border-danger text-danger",
                          aiData.domain_risk_level === 'medium' && "border-warning text-warning",
                          aiData.domain_risk_level === 'low' && "border-success text-success",
                        )}
                      >
                        {aiData.domain_risk_level} risk
                      </Badge>
                    )}
                    {dipAnalysis?.dip_type && (
                      <Badge 
                        variant={dipAnalysis.dip_type === 'overreaction' ? 'default' : 'secondary'}
                        className="text-xs"
                      >
                        {dipAnalysis.dip_type === 'overreaction' ? '📉 Overreaction' : dipAnalysis.dip_type.replace('_', ' ')}
                      </Badge>
                    )}
                  </div>
                </div>
                
                {/* Domain context - sector-specific insights */}
                {aiData?.domain_context && (
                  <p className="text-xs text-foreground leading-relaxed mb-2 font-medium">
                    🏢 {aiData.domain_context}
                  </p>
                )}
                
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {dipAnalysis?.reasoning || aiData?.ai_reasoning || 'No analysis available'}
                </p>
                
                {/* Sector performance comparison */}
                {aiData?.vs_sector_performance !== undefined && aiData?.vs_sector_performance !== null && (
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">vs Sector:</span>
                    <span className={cn(
                      "text-xs font-medium",
                      aiData.vs_sector_performance > 0 ? "text-success" : "text-danger"
                    )}>
                      {aiData.vs_sector_performance > 0 ? '+' : ''}{aiData.vs_sector_performance.toFixed(1)}%
                    </span>
                    {aiData.domain_recovery_days && (
                      <>
                        <span className="text-xs text-muted-foreground">•</span>
                        <span className="text-xs text-muted-foreground">
                          ~{aiData.domain_recovery_days}d typical recovery
                        </span>
                      </>
                    )}
                  </div>
                )}
                
                {/* Key warnings if any */}
                {aiData?.domain_warnings && aiData.domain_warnings.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-border/50">
                    {aiData.domain_warnings.slice(0, 2).map((w, i) => (
                      <p key={i} className="text-xs text-warning flex items-center gap-1">
                        <AlertTriangle className="h-3 w-3 shrink-0" />
                        {w}
                      </p>
                    ))}
                  </div>
                )}
                
                {/* Risk factors */}
                {aiData?.domain_risk_factors && aiData.domain_risk_factors.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-border/50">
                    <span className="text-xs text-muted-foreground">Key risks: </span>
                    <span className="text-xs text-foreground">
                      {aiData.domain_risk_factors.slice(0, 4).join(' • ')}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Active Signals - Compact */}
            {currentSignals && (currentSignals.buy_signals.length > 0 || currentSignals.sell_signals.length > 0) && (
              <div className="flex flex-wrap gap-1 mb-4">
                {currentSignals.buy_signals.slice(0, 3).map((sig, i) => (
                  <Badge key={i} variant="outline" className="text-xs border-success/40 text-success bg-success/5">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    {sig.name}
                  </Badge>
                ))}
                {currentSignals.sell_signals.slice(0, 2).map((sig, i) => (
                  <Badge key={i} variant="outline" className="text-xs border-warning/40 text-warning bg-warning/5">
                    <AlertTriangle className="h-3 w-3 mr-1" />
                    {sig.name}
                  </Badge>
                ))}
              </div>
            )}

            <Separator className="my-3" />
            
            {/* Company Info - Collapsed by default */}
            <details className="group">
              <summary className="flex items-center justify-between cursor-pointer py-2 text-sm font-medium">
                <span>Company Details</span>
                <span className="text-xs text-muted-foreground group-open:hidden">Click to expand</span>
              </summary>
              <div className="pt-2 space-y-3">
                {/* Basic metrics grid */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  <StatItem
                    icon={Building2}
                    label="Market Cap"
                    value={formatMarketCap(stock.market_cap ?? stockInfo?.market_cap ?? null)}
                  />
                  <StatItem
                    icon={BarChart3}
                    label="P/E Ratio"
                    value={stock.pe_ratio?.toFixed(2) ?? stockInfo?.pe_ratio?.toFixed(2) ?? '—'}
                  />
                  <StatItem
                    icon={Calendar}
                    label="Days in Dip"
                    value={stock.days_since_dip?.toString() ?? '—'}
                  />
                </div>

                {/* Company summary */}
                {stockInfo && (
                  <>
                    {stockInfo.sector && (
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="secondary">{stockInfo.sector}</Badge>
                        {stockInfo.industry && (
                          <span className="text-xs text-muted-foreground">{stockInfo.industry}</span>
                        )}
                      </div>
                    )}
                    {(stockInfo.summary_ai || stockInfo.summary) && (
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        {stockInfo.summary_ai || stockInfo.summary}
                      </p>
                    )}
                    {stockInfo.website && (
                      <a
                        href={stockInfo.website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                      >
                        <ExternalLink className="h-3 w-3" />
                        {stockInfo.website.replace(/^https?:\/\/(www\.)?/, '').split('/')[0]}
                      </a>
                    )}
                  </>
                )}
              </div>
            </details>
          </CardContent>
        </ScrollArea>
      </Card>
    </motion.div>
  );
}

interface StatItemProps {
  icon: React.ElementType;
  label: string;
  value: string;
  valueColor?: string;
}

function StatItem({ icon: Icon, label, value, valueColor }: StatItemProps) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground truncate">{label}</p>
        <p className={`text-sm font-medium truncate ${valueColor || ''}`}>{value}</p>
      </div>
    </div>
  );
}
