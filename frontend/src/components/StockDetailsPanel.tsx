import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Line,
  ComposedChart,
  ReferenceDot,
  ReferenceLine,
  Label as RechartsLabel,
} from 'recharts';
import { ChartTooltip, SimpleChartTooltipContent } from '@/components/ui/chart';
import { CHART_LINE_ANIMATION, CHART_TRENDLINE_ANIMATION, CHART_ANIMATION } from '@/lib/chartConfig';
import type { DipStock, ChartDataPoint, StockInfo, BenchmarkType, ComparisonChartData, SignalTrigger, SignalTriggersResponse, DipAnalysis, CurrentSignals, AgentAnalysis, SymbolFundamentals, StrategySignalResponse, DipEntryResponse } from '@/services/api';
import { getSignalTriggers, getDipAnalysis, getCurrentSignals, getAgentAnalysis, getSymbolFundamentals, getStrategySignal, getDipEntry } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ComparisonChart } from '@/components/ComparisonChart';
import { DipThresholdChart } from '@/components/DipThresholdChart';
import { 
  TrendingUp, 
  TrendingDown, 
  Building2, 
  DollarSign, 
  BarChart3, 
  Calendar,
  ExternalLink,
  Activity,
  Target,
  Sparkles,
  Zap,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Building,
  Landmark,
  Shield,
  ChevronDown,
  ChevronUp,
  ArrowUpCircle,
  ArrowDownCircle,
  MinusCircle,
  CircleCheck,
  CircleAlert,
  CircleX,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { StockLogo } from '@/components/StockLogo';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

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
  { label: '5Y', days: 1825 },
];

/** Target number of points for all chart periods - keeps animations consistent */
const NORMALIZED_CHART_POINTS = 180;

const deferStateUpdate = (callback: () => void) => {
  Promise.resolve().then(callback);
};

/**
 * Normalize chart data to a fixed number of points for consistent animations.
 * Downsamples if too many points, upsamples (linear interpolation) if too few.
 */
function normalizeChartData<T extends { close: number; date: string }>(
  data: T[], 
  targetPoints: number = NORMALIZED_CHART_POINTS
): T[] {
  if (data.length === 0) return data;
  if (data.length === targetPoints) return data;
  
  // Downsampling: use LTTB-like algorithm
  if (data.length > targetPoints) {
    const result: T[] = [data[0]];
    const bucketSize = (data.length - 2) / (targetPoints - 2);
    
    for (let i = 1; i < targetPoints - 1; i++) {
      const start = Math.floor((i - 1) * bucketSize) + 1;
      const end = Math.floor(i * bucketSize) + 1;
      
      let maxDeviation = 0;
      let selectedIdx = start;
      const bucketSlice = data.slice(start, Math.min(end + 1, data.length - 1));
      
      if (bucketSlice.length > 0) {
        const meanClose = bucketSlice.reduce((s, p) => s + p.close, 0) / bucketSlice.length;
        bucketSlice.forEach((point, idx) => {
          const deviation = Math.abs(point.close - meanClose);
          if (deviation > maxDeviation) {
            maxDeviation = deviation;
            selectedIdx = start + idx;
          }
        });
      }
      
      if (data[selectedIdx]) {
        result.push(data[selectedIdx]);
      }
    }
    
    result.push(data[data.length - 1]);
    return result;
  }
  
  // Upsampling: use linear interpolation to fill gaps
  const result: T[] = [];
  const ratio = (data.length - 1) / (targetPoints - 1);
  
  for (let i = 0; i < targetPoints; i++) {
    const srcIndex = i * ratio;
    const lowerIdx = Math.floor(srcIndex);
    const upperIdx = Math.min(Math.ceil(srcIndex), data.length - 1);
    const fraction = srcIndex - lowerIdx;
    
    if (lowerIdx === upperIdx || fraction === 0) {
      result.push(data[lowerIdx]);
    } else {
      // Interpolate between two points
      const lower = data[lowerIdx];
      const upper = data[upperIdx];
      result.push({
        ...lower,
        close: lower.close + (upper.close - lower.close) * fraction,
        // Keep the date from the lower point for interpolated values
      });
    }
  }
  
  return result;
}

export function StockDetailsPanel({
  stock,
  chartData,
  stockInfo,
  chartPeriod,
  onPeriodChange,
  isLoadingChart,
  isLoadingInfo: _isLoadingInfo,
  onClose: _onClose,
  benchmark,
  comparisonData = [],
  isLoadingBenchmark = false,
  aiData: _aiData,
  isLoadingAi: _isLoadingAi,
}: StockDetailsPanelProps) {
  // Track when dots should be visible (after chart animation completes)
  const [dotsVisible, setDotsVisible] = useState(false);
  
  // Signal triggers for buy signal markers with benchmark comparison
  const [signalsResponse, setSignalsResponse] = useState<SignalTriggersResponse | null>(null);
  const [showSignals, setShowSignals] = useState(true);
  
  // Extract triggers from response for backward compatibility
  const signalTriggers = signalsResponse?.triggers ?? [];
  
  // Quant analysis state
  const [_dipAnalysis, setDipAnalysis] = useState<DipAnalysis | null>(null);
  const [currentSignals, setCurrentSignals] = useState<CurrentSignals | null>(null);
  const [isLoadingQuant, setIsLoadingQuant] = useState(false);
  
  // AI Agents and Fundamentals state
  const [agentAnalysis, setAgentAnalysis] = useState<AgentAnalysis | null>(null);
  const [fundamentals, setFundamentals] = useState<SymbolFundamentals | null>(null);
  const [isLoadingAgents, setIsLoadingAgents] = useState(false);
  const [showAllVerdicts, setShowAllVerdicts] = useState(false);
  const [expandedVerdicts, setExpandedVerdicts] = useState<Set<string>>(new Set());
  
  // Strategy signal state (quant optimizer)
  const [strategySignal, setStrategySignal] = useState<StrategySignalResponse | null>(null);
  const [isLoadingStrategy, setIsLoadingStrategy] = useState(false);
  
  // Dip entry optimizer state
  const [dipEntry, setDipEntry] = useState<DipEntryResponse | null>(null);
  const [isLoadingDipEntry, setIsLoadingDipEntry] = useState(false);
  
  // Fetch signal triggers when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      deferStateUpdate(() => setSignalsResponse(null));
      return;
    }
    
    // Cap at 730 days (API limit) - signals beyond 2 years are too historical to be useful
    const lookbackDays = Math.min(730, Math.max(chartPeriod + 30, 365));
    getSignalTriggers(stock.symbol, lookbackDays)
      .then(setSignalsResponse)
      .catch(() => setSignalsResponse(null));
  }, [stock?.symbol, chartPeriod]);
  
  // Fetch quant analysis when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      deferStateUpdate(() => {
        setDipAnalysis(null);
        setCurrentSignals(null);
      });
      return;
    }
    
    deferStateUpdate(() => setIsLoadingQuant(true));
    Promise.all([
      getDipAnalysis(stock.symbol).catch(() => null),
      getCurrentSignals(stock.symbol).catch(() => null),
    ]).then(([dip, signals]) => {
      setDipAnalysis(dip);
      setCurrentSignals(signals);
      setIsLoadingQuant(false);
    });
  }, [stock?.symbol]);
  
  // Fetch AI agents and fundamentals when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      deferStateUpdate(() => {
        setAgentAnalysis(null);
        setFundamentals(null);
      });
      return;
    }
    
    deferStateUpdate(() => {
      setIsLoadingAgents(true);
      setShowAllVerdicts(false);
    });
    Promise.all([
      getAgentAnalysis(stock.symbol).catch(() => null),
      getSymbolFundamentals(stock.symbol).catch(() => null),
    ]).then(([agents, funds]) => {
      setAgentAnalysis(agents);
      setFundamentals(funds);
      setIsLoadingAgents(false);
    });
  }, [stock?.symbol]);
  
  // Fetch strategy signal (quant optimizer) when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      deferStateUpdate(() => setStrategySignal(null));
      return;
    }
    
    deferStateUpdate(() => setIsLoadingStrategy(true));
    getStrategySignal(stock.symbol)
      .then(setStrategySignal)
      .catch(() => setStrategySignal(null))
      .finally(() => setIsLoadingStrategy(false));
  }, [stock?.symbol]);
  
  // Fetch dip entry analysis when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      deferStateUpdate(() => setDipEntry(null));
      return;
    }
    
    deferStateUpdate(() => setIsLoadingDipEntry(true));
    getDipEntry(stock.symbol)
      .then(setDipEntry)
      .catch(() => setDipEntry(null))
      .finally(() => setIsLoadingDipEntry(false));
  }, [stock?.symbol]);
  
  // When chart period or data changes, hide dots and show them after animation completes
  useEffect(() => {
    deferStateUpdate(() => setDotsVisible(false));
    const timer = setTimeout(() => {
      setDotsVisible(true);
    }, CHART_ANIMATION.animationDuration + 50); // Add 50ms buffer
    return () => clearTimeout(timer);
  }, [chartPeriod, chartData]);

  // Normalize and format chart data - React Compiler handles memoization
  const formattedChartData = (() => {
    // Normalize all periods to same number of points for consistent animations
    // This ensures smooth morphing when switching between periods
    const normalizedData = normalizeChartData(chartData, NORMALIZED_CHART_POINTS);
    
    return normalizedData.map((point) => ({
      ...point,
      displayDate: new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
    }));
  })();

  // Find ref high point index for marker
  const refHighIndex = (() => {
    if (chartData.length === 0) return -1;
    const refDate = chartData[0]?.ref_high_date;
    if (!refDate) return -1;
    return formattedChartData.findIndex(p => p.date === refDate);
  })();

  // Get the display date for ref high point (for ReferenceDot x value)
  const refHighDisplayDate = refHighIndex >= 0 && formattedChartData[refHighIndex]
    ? formattedChartData[refHighIndex].displayDate
    : null;

  // Get the display date for current price (last point)
  const currentDisplayDate = formattedChartData.length > 0
    ? formattedChartData[formattedChartData.length - 1].displayDate
    : null;

  // Get the current price for the reference line
  const currentPrice = formattedChartData.length > 0
    ? formattedChartData[formattedChartData.length - 1]?.close
    : null;

  // Get the ref high price
  const refHighPrice = chartData.length > 0 ? chartData[0]?.ref_high ?? null : null;

  // Create chart data with trendline, reference lines, and animated dot positions
  const chartDataWithTrendline = (() => {
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
  })();

  // Check if the stock has a valid optimized strategy (not just "dip" fallback)
  const hasValidStrategy = strategySignal && strategySignal.strategy_name !== 'dip';
  
  // Only show signals if they beat buy-and-hold (have positive edge)
  const signalsBeatBuyHold = signalsResponse?.beats_buy_hold ?? false;

  // Find signal trigger points that match chart dates (for technical strategy)
  const signalPoints = (() => {
    // Don't show signals if:
    // 1. User toggled them off
    // 2. Strategy is "dip" (no real backtested signals)
    // 3. Signal doesn't beat buy-and-hold (no edge = useless signal)
    if (!showSignals || !hasValidStrategy || !signalsBeatBuyHold || signalTriggers.length === 0 || formattedChartData.length === 0) return [];
    
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
  })();

  // Find dip signal trigger points that match chart dates (for dip strategy)
  // Only show when no valid technical strategy is available
  const dipSignalPoints = (() => {
    // Show dip signals when:
    // 1. User has signals toggled on
    // 2. No valid technical strategy OR strategy doesn't beat B&H
    // 3. Dip entry data with signal triggers is available
    const showDipSignals = showSignals && (!hasValidStrategy || !signalsBeatBuyHold);
    const dipTriggers = dipEntry?.signal_triggers ?? [];
    
    if (!showDipSignals || dipTriggers.length === 0 || formattedChartData.length === 0) return [];
    
    // Create a map of date -> signal for quick lookup
    const dipSignalMap = new Map<string, typeof dipTriggers[0]>();
    dipTriggers.forEach(s => dipSignalMap.set(s.date, s));
    
    // Find chart points that have matching signals
    return formattedChartData
      .filter(point => dipSignalMap.has(point.date))
      .map(point => ({
        ...point,
        signal: dipSignalMap.get(point.date)!,
      }));
  })();

  // Calculate price change for chart color
  const priceChange = (() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  })();

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
            {/* Second row: Stock name + Earnings badge */}
            <div className="flex items-center gap-2 flex-wrap">
              <p className="text-xs md:text-sm text-muted-foreground">
                {stock.name || stock.symbol}
              </p>
              {/* Upcoming Earnings Badge */}
              {fundamentals?.next_earnings_date && (() => {
                const earningsDate = new Date(fundamentals.next_earnings_date);
                const today = new Date();
                const daysUntil = Math.ceil((earningsDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
                if (daysUntil > 0 && daysUntil <= 7) {
                  return (
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 border-amber-500/50 text-amber-600 bg-amber-50 dark:bg-amber-950/30 flex items-center gap-1">
                      <Calendar className="h-2.5 w-2.5" /> Earnings in {daysUntil}d
                    </Badge>
                  );
                }
                return null;
              })()}
            </div>
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
                  {/* Signals toggle - only show if stock has valid strategy */}
                  {hasValidStrategy && (
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
                  )}
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
                    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={100}>
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
                    {dotsVisible && showSignals && signalPoints.map((point, idx) => {
                      // Calculate expected value: win_rate * avg_return (risk-adjusted return)
                      const expectedValue = point.signal.win_rate * point.signal.avg_return_pct;
                      const isEntry = point.signal.signal_type === 'entry';
                      const dotColor = isEntry ? 'var(--success)' : 'var(--danger)';
                      const signalIcon = isEntry ? '▲' : '▼';
                      const typeLabel = isEntry ? 'BUY' : 'SELL';
                      
                      return (
                      <ReferenceDot
                        key={`signal-${idx}`}
                        x={point.displayDate}
                        y={point.close}
                        r={isEntry ? 8 : 6}
                        fill={dotColor}
                        stroke="var(--background)"
                        strokeWidth={2}
                        shape={(props: { cx?: number; cy?: number }) => {
                          const { cx = 0, cy = 0 } = props;
                          const signal = point.signal;
                          const tooltipText = isEntry 
                            ? `${signalIcon} ${typeLabel}: ${signal.signal_name}\n` +
                              `Win Rate: ${(signal.win_rate * 100).toFixed(0)}%\n` +
                              `Avg Return: ${signal.avg_return_pct >= 0 ? '+' : ''}${signal.avg_return_pct.toFixed(1)}%\n` +
                              `Expected Value: ${expectedValue >= 0 ? '+' : ''}${expectedValue.toFixed(1)}%\n` +
                              `Hold: ${signal.holding_days} days\n` +
                              `Entry Price: $${signal.price.toFixed(2)}`
                            : `${signalIcon} ${typeLabel}: ${signal.signal_name}\n` +
                              `Trade Return: ${signal.avg_return_pct >= 0 ? '+' : ''}${signal.avg_return_pct.toFixed(1)}%\n` +
                              `Exit Price: $${signal.price.toFixed(2)}`;
                          return (
                            <g style={{ cursor: 'pointer' }}>
                              <title>{tooltipText}</title>
                              <circle
                                cx={cx}
                                cy={cy}
                                r={isEntry ? 8 : 6}
                                fill={dotColor}
                                stroke="var(--background)"
                                strokeWidth={2}
                              />
                            </g>
                          );
                        }}
                      >
                        {isEntry && (
                        <RechartsLabel
                          position="top"
                          offset={10}
                          style={{
                            fontSize: 9,
                            fill: 'var(--foreground)',
                            fontWeight: 600,
                          }}
                        >
                          {`${expectedValue >= 0 ? '+' : ''}${expectedValue.toFixed(0)}% EV`}
                        </RechartsLabel>
                        )}
                      </ReferenceDot>
                    );})}
                    {/* Dip strategy buy/sell signal dots (shown when no valid technical strategy) */}
                    {dotsVisible && showSignals && dipSignalPoints.map((point, idx) => {
                      const isEntry = point.signal.signal_type === 'entry';
                      const dotColor = isEntry ? 'var(--success)' : 'var(--danger)';
                      const signalIcon = isEntry ? '▲' : '▼';
                      const typeLabel = isEntry ? 'BUY DIP' : 'SELL';
                      const thresholdPct = Math.abs(point.signal.threshold_pct * 100).toFixed(0);
                      
                      return (
                      <ReferenceDot
                        key={`dip-signal-${idx}`}
                        x={point.displayDate}
                        y={point.close}
                        r={isEntry ? 8 : 6}
                        fill={dotColor}
                        stroke="var(--background)"
                        strokeWidth={2}
                        shape={(props: { cx?: number; cy?: number }) => {
                          const { cx = 0, cy = 0 } = props;
                          const signal = point.signal;
                          const tooltipText = isEntry 
                            ? `${signalIcon} ${typeLabel}\n` +
                              `Dip Threshold: -${thresholdPct}%\n` +
                              `Entry Price: $${signal.price.toFixed(2)}`
                            : `${signalIcon} ${typeLabel}\n` +
                              `Return: ${(signal.return_pct * 100) >= 0 ? '+' : ''}${(signal.return_pct * 100).toFixed(1)}%\n` +
                              `Held: ${signal.holding_days} days\n` +
                              `Exit Price: $${signal.price.toFixed(2)}`;
                          return (
                            <g style={{ cursor: 'pointer' }}>
                              <title>{tooltipText}</title>
                              <circle
                                cx={cx}
                                cy={cy}
                                r={isEntry ? 8 : 6}
                                fill={dotColor}
                                stroke="var(--background)"
                                strokeWidth={2}
                              />
                            </g>
                          );
                        }}
                      >
                        {isEntry && (
                        <RechartsLabel
                          position="top"
                          offset={10}
                          style={{
                            fontSize: 9,
                            fill: 'var(--foreground)',
                            fontWeight: 600,
                          }}
                        >
                          {`-${thresholdPct}% dip`}
                        </RechartsLabel>
                        )}
                      </ReferenceDot>
                    );})}
                    {/* Risk-Adjusted Entry Price Line (primary) */}
                    {dipEntry && dipEntry.optimal_entry_price != null && dipEntry.optimal_entry_price > 0 && (
                      <ReferenceLine
                        y={dipEntry.optimal_entry_price}
                        stroke="var(--primary)"
                        strokeWidth={2}
                        strokeDasharray="6 4"
                        strokeOpacity={0.8}
                      >
                        <RechartsLabel
                          value={`Risk $${dipEntry.optimal_entry_price?.toFixed(2) ?? '—'}`}
                          position="right"
                          style={{
                            fontSize: 10,
                            fill: 'var(--primary)',
                            fontWeight: 600,
                          }}
                        />
                      </ReferenceLine>
                    )}
                    {/* Max Profit Entry Price Line (if different from risk-adjusted) */}
                    {dipEntry && dipEntry.max_profit_entry_price != null && dipEntry.max_profit_entry_price > 0 && 
                     dipEntry.max_profit_threshold !== dipEntry.optimal_dip_threshold && (
                      <ReferenceLine
                        y={dipEntry.max_profit_entry_price}
                        stroke="var(--chart-2)"
                        strokeWidth={2}
                        strokeDasharray="3 3"
                        strokeOpacity={0.7}
                      >
                        <RechartsLabel
                          value={`Profit $${dipEntry.max_profit_entry_price?.toFixed(2) ?? '—'}`}
                          position="left"
                          style={{
                            fontSize: 10,
                            fill: 'var(--chart-2)',
                            fontWeight: 600,
                          }}
                        />
                      </ReferenceLine>
                    )}
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

          {/* Data Section - Complete Analysis Dashboard */}
          <CardContent className="pt-3 pb-4 px-3 md:px-6">
            {/* HERO: Overall Recommendation Card */}
            {(currentSignals || agentAnalysis) && !isLoadingQuant && !isLoadingAgents && (
              <div className={cn(
                "p-4 rounded-xl mb-4",
                (currentSignals?.overall_action === 'STRONG_BUY' || agentAnalysis?.overall_signal === 'strong_buy') && "bg-gradient-to-br from-success/20 to-success/5 border-2 border-success/40",
                (currentSignals?.overall_action === 'BUY' || agentAnalysis?.overall_signal === 'buy') && "bg-gradient-to-br from-success/15 to-success/5 border border-success/30",
                (currentSignals?.overall_action === 'HOLD' || agentAnalysis?.overall_signal === 'hold') && "bg-gradient-to-br from-muted/40 to-muted/20 border border-border",
                (currentSignals?.overall_action === 'SELL' || agentAnalysis?.overall_signal === 'sell') && "bg-gradient-to-br from-warning/15 to-warning/5 border border-warning/30",
                (currentSignals?.overall_action === 'STRONG_SELL' || agentAnalysis?.overall_signal === 'strong_sell') && "bg-gradient-to-br from-danger/20 to-danger/10 border-2 border-danger/40",
                !currentSignals?.overall_action && !agentAnalysis?.overall_signal && "bg-gradient-to-br from-muted/40 to-muted/20 border border-border"
              )}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {(currentSignals?.overall_action === 'STRONG_BUY' || agentAnalysis?.overall_signal === 'strong_buy') && <Zap className="h-6 w-6 text-success" />}
                    {(currentSignals?.overall_action === 'BUY' || agentAnalysis?.overall_signal === 'buy') && <TrendingUp className="h-6 w-6 text-success" />}
                    {(currentSignals?.overall_action === 'HOLD' || agentAnalysis?.overall_signal === 'hold') && <Activity className="h-5 w-5 text-muted-foreground" />}
                    {(currentSignals?.overall_action === 'SELL' || agentAnalysis?.overall_signal === 'sell') && <TrendingDown className="h-5 w-5 text-warning" />}
                    {(currentSignals?.overall_action === 'STRONG_SELL' || agentAnalysis?.overall_signal === 'strong_sell') && <XCircle className="h-6 w-6 text-danger" />}
                    <span className={cn(
                      "text-2xl font-bold",
                      (currentSignals?.overall_action?.includes('BUY') || ['strong_buy', 'buy'].includes(agentAnalysis?.overall_signal || '')) && "text-success",
                      (currentSignals?.overall_action === 'HOLD' || agentAnalysis?.overall_signal === 'hold') && "text-foreground",
                      (currentSignals?.overall_action?.includes('SELL') || ['strong_sell', 'sell'].includes(agentAnalysis?.overall_signal || '')) && "text-danger",
                    )}>
                      {(currentSignals?.overall_action || agentAnalysis?.overall_signal || 'ANALYZING').replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  {/* Show counts matching the signal source being used */}
                  {currentSignals?.overall_action ? (
                    <div className="flex gap-2 text-xs items-center">
                      <span className="text-success flex items-center gap-0.5"><ArrowUpCircle className="h-3 w-3" /> {currentSignals.buy_signals?.length || 0}</span>
                      <span className="text-danger flex items-center gap-0.5"><ArrowDownCircle className="h-3 w-3" /> {currentSignals.sell_signals?.length || 0}</span>
                    </div>
                  ) : agentAnalysis && (
                    <div className="flex gap-2 text-xs items-center">
                      <span className="text-success flex items-center gap-0.5"><ArrowUpCircle className="h-3 w-3" /> {agentAnalysis.bullish_count || 0}</span>
                      <span className="text-muted-foreground flex items-center gap-0.5"><MinusCircle className="h-3 w-3" /> {agentAnalysis.neutral_count || 0}</span>
                      <span className="text-danger flex items-center gap-0.5"><ArrowDownCircle className="h-3 w-3" /> {agentAnalysis.bearish_count || 0}</span>
                    </div>
                  )}
                </div>
                {currentSignals?.reasoning && (
                  <p className="text-sm text-muted-foreground mt-2">{currentSignals.reasoning}</p>
                )}
              </div>
            )}

            {(isLoadingQuant || isLoadingAgents) && (
              <div className="p-4 rounded-xl mb-4 bg-muted/20 border border-border">
                <Skeleton className="h-8 w-40 mb-2" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            )}

            {/* Quick Stats - Compact Inline */}
            <div className="flex items-center gap-4 mb-4 text-sm">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 cursor-help">
                    <TrendingDown className="h-3.5 w-3.5 text-danger" />
                    <span className="font-bold text-danger">-{(stock.depth * 100).toFixed(0)}%</span>
                    <span className="text-muted-foreground text-xs">from high</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Current price is {(stock.depth * 100).toFixed(1)}% below 52-week high</p>
                </TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 cursor-help">
                    <Calendar className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="font-bold">{stock.days_since_dip || 0}</span>
                    <span className="text-muted-foreground text-xs">days in dip</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Stock has been in this dip for {stock.days_since_dip || 0} days</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* PROMINENT: Intrinsic Value + Dip Entry Row */}
            {(fundamentals?.intrinsic_value || dipEntry) && (
              <div className="grid grid-cols-2 gap-2 mb-4">
                {/* Intrinsic Value Card */}
                {fundamentals?.intrinsic_value && fundamentals?.upside_pct != null && (
                  <div className={cn(
                    "p-3 rounded-lg border",
                    fundamentals.valuation_status === 'undervalued'
                      ? "bg-success/10 border-success/30" 
                      : fundamentals.valuation_status === 'overvalued'
                        ? "bg-danger/10 border-danger/30"
                        : "bg-muted/30 border-border"
                  )}>
                    <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                      <DollarSign className="h-3 w-3" />
                      Intrinsic Value
                    </p>
                    <p className={cn(
                      "text-xl font-bold",
                      fundamentals.valuation_status === 'undervalued' ? "text-success" : 
                      fundamentals.valuation_status === 'overvalued' ? "text-danger" : ""
                    )}>
                      ${fundamentals.intrinsic_value.toFixed(2)}
                    </p>
                    <p className={cn(
                      "text-sm font-medium",
                      (fundamentals.upside_pct ?? 0) >= 0 ? "text-success" : "text-danger"
                    )}>
                      {(fundamentals.upside_pct ?? 0) >= 0 ? '+' : ''}{(fundamentals.upside_pct ?? 0).toFixed(0)}% upside
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-1">
                      {fundamentals.intrinsic_value_method === 'analyst' 
                        ? `${fundamentals.num_analyst_opinions || 0} analysts` 
                        : fundamentals.intrinsic_value_method === 'peg' 
                          ? 'PEG-based' 
                          : fundamentals.intrinsic_value_method === 'graham'
                            ? 'Graham #'
                            : fundamentals.intrinsic_value_method || 'DCF'}
                    </p>
                  </div>
                )}
                
                {/* Dip Entry Card - Shows the optimal entry based on backtested dip strategy */}
                {dipEntry && dipEntry.max_profit_entry_price != null && (
                  <div className={cn(
                    "p-3 rounded-lg border",
                    dipEntry.is_buy_now && "bg-success/10 border-success/30",
                    !dipEntry.is_buy_now && dipEntry.current_drawdown_pct <= -0.10 && "bg-amber-500/10 border-amber-500/30",
                    !dipEntry.is_buy_now && dipEntry.current_drawdown_pct > -0.10 && "bg-muted/30 border-border",
                  )}>
                    {/* Primary: Max Profit Entry (this is what the backtest is based on) */}
                    <div>
                      <p className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                        <Target className="h-3 w-3" />
                        Optimal Dip Entry
                      </p>
                      <p className="text-xl font-bold text-primary">${dipEntry.max_profit_entry_price?.toFixed(2) ?? '—'}</p>
                      <p className="text-sm text-muted-foreground">
                        Buy at {Math.abs((dipEntry.max_profit_threshold ?? 0) * 100).toFixed(0)}% dip
                        {dipEntry.backtest && ` • ${dipEntry.backtest.n_trades ?? 0} trades backtested`}
                      </p>
                    </div>
                    
                    {/* Buy Signal */}
                    <p className={cn(
                      "text-[10px] mt-2 pt-2 border-t border-border/50",
                      dipEntry.is_buy_now ? "text-success font-medium" : "text-muted-foreground"
                    )}>
                      {dipEntry.is_buy_now ? "✓ Buy now" : `Wait for $${(displayPrice - dipEntry.max_profit_entry_price).toFixed(2)} more drop`}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Dip Threshold Analysis Chart */}
            {dipEntry && dipEntry.threshold_analysis && dipEntry.threshold_analysis.length > 0 && (
              <div className="mb-4">
                <DipThresholdChart
                  thresholdStats={dipEntry.threshold_analysis}
                  optimalThreshold={dipEntry.optimal_dip_threshold}
                  maxProfitThreshold={dipEntry.max_profit_threshold || dipEntry.optimal_dip_threshold}
                  currentDrawdown={dipEntry.current_drawdown_pct}
                  height={220}
                />
              </div>
            )}

            {/* Domain-Specific Metrics (Banks, REITs, Insurance) */}
            {fundamentals && fundamentals.domain && fundamentals.domain !== 'stock' && (
              <div className="p-3 rounded-lg bg-gradient-to-br from-primary/5 to-primary/10 border border-primary/20 mb-4">
                <div className="flex items-center gap-2 mb-3">
                  {fundamentals.domain === 'bank' && <Landmark className="h-4 w-4 text-primary" />}
                  {fundamentals.domain === 'reit' && <Building className="h-4 w-4 text-primary" />}
                  {fundamentals.domain === 'insurer' && <Shield className="h-4 w-4 text-primary" />}
                  {fundamentals.domain === 'utility' && <Zap className="h-4 w-4 text-primary" />}
                  <span className="text-sm font-semibold">{fundamentals.domain.charAt(0).toUpperCase() + fundamentals.domain.slice(1)} Metrics</span>
                </div>
                
                {/* Bank Metrics */}
                {fundamentals.domain === 'bank' && (
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-muted-foreground">Net Interest Margin</p>
                      <p className="text-lg font-bold">
                        {fundamentals.net_interest_margin 
                          ? `${(fundamentals.net_interest_margin * 100).toFixed(2)}%`
                          : '—'}
                      </p>
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        {fundamentals.net_interest_margin && fundamentals.net_interest_margin > 0.03 
                          ? <><CircleCheck className="h-3 w-3 text-success" /> Above avg (3%)</>
                          : fundamentals.net_interest_margin && fundamentals.net_interest_margin > 0.02 
                            ? <><CircleAlert className="h-3 w-3 text-warning" /> Average</>
                            : fundamentals.net_interest_margin 
                              ? <><CircleX className="h-3 w-3 text-danger" /> Below avg</>
                              : null}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Net Interest Income</p>
                      <p className="text-lg font-bold">
                        {fundamentals.net_interest_income 
                          ? `$${(fundamentals.net_interest_income / 1e9).toFixed(1)}B`
                          : '—'}
                      </p>
                      <p className="text-[10px] text-muted-foreground">annualized</p>
                    </div>
                  </div>
                )}
                
                {/* REIT Metrics */}
                {fundamentals.domain === 'reit' && (
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <p className="text-xs text-muted-foreground">FFO</p>
                      <p className="text-lg font-bold">
                        {fundamentals.ffo 
                          ? `$${(fundamentals.ffo / 1e9).toFixed(2)}B`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">FFO/Share</p>
                      <p className="text-lg font-bold">
                        {fundamentals.ffo_per_share 
                          ? `$${fundamentals.ffo_per_share.toFixed(2)}`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">P/FFO</p>
                      <p className={cn(
                        "text-lg font-bold",
                        fundamentals.p_ffo && fundamentals.p_ffo < 15 ? "text-success" : 
                        fundamentals.p_ffo && fundamentals.p_ffo > 20 ? "text-warning" : ""
                      )}>
                        {fundamentals.p_ffo 
                          ? fundamentals.p_ffo.toFixed(1)
                          : '—'}
                      </p>
                      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
                        {fundamentals.p_ffo && fundamentals.p_ffo < 15 
                          ? <><CircleCheck className="h-3 w-3 text-success" /> Undervalued</>
                          : fundamentals.p_ffo && fundamentals.p_ffo > 20 
                            ? <><CircleAlert className="h-3 w-3 text-warning" /> Expensive</>
                            : null}
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Insurance Metrics */}
                {fundamentals.domain === 'insurer' && (
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <p className="text-xs text-muted-foreground">Loss Ratio</p>
                      <p className="text-lg font-bold">
                        {fundamentals.loss_ratio 
                          ? `${(fundamentals.loss_ratio * 100).toFixed(0)}%`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Expense Ratio</p>
                      <p className="text-lg font-bold">
                        {fundamentals.expense_ratio 
                          ? `${(fundamentals.expense_ratio * 100).toFixed(0)}%`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Combined Ratio</p>
                      <p className={cn(
                        "text-lg font-bold",
                        fundamentals.combined_ratio && fundamentals.combined_ratio < 1 ? "text-success" : "text-danger"
                      )}>
                        {fundamentals.combined_ratio 
                          ? `${(fundamentals.combined_ratio * 100).toFixed(0)}%`
                          : '—'}
                      </p>
                      <p className="text-[10px] text-muted-foreground">
                        {fundamentals.combined_ratio && fundamentals.combined_ratio < 1 
                          ? '✅ Profitable' 
                          : fundamentals.combined_ratio 
                            ? '❌ Losing money'
                            : ''}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* AI Persona Verdicts */}
            {agentAnalysis && agentAnalysis.verdicts.length > 0 && (() => {
              // Calculate aggregated rating from all verdicts
              const signalScores: Record<string, number> = {
                'strong_buy': 2, 'buy': 1, 'bullish': 1,
                'hold': 0, 'neutral': 0,
                'sell': -1, 'bearish': -1, 'strong_sell': -2
              };
              const totalScore = agentAnalysis.verdicts.reduce((sum, v) => sum + (signalScores[v.signal] || 0), 0);
              const avgScore = totalScore / agentAnalysis.verdicts.length;
              const aggregatedSignal = avgScore >= 1.5 ? 'Strong Buy' : avgScore >= 0.5 ? 'Buy' : avgScore >= -0.5 ? 'Hold' : avgScore >= -1.5 ? 'Sell' : 'Strong Sell';
              const bullishCount = agentAnalysis.verdicts.filter(v => ['strong_buy', 'buy', 'bullish'].includes(v.signal)).length;
              const bearishCount = agentAnalysis.verdicts.filter(v => ['strong_sell', 'sell', 'bearish'].includes(v.signal)).length;
              
              return (
              <div className="mb-4">
                <div 
                  className="flex items-center justify-between cursor-pointer py-2"
                  onClick={() => setShowAllVerdicts(!showAllVerdicts)}
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <span className="text-sm font-semibold">AI Analyst Verdicts</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{agentAnalysis.verdicts.length} analysts</span>
                    {showAllVerdicts ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </div>
                </div>
                
                {/* Aggregated Consensus */}
                <div className={cn(
                  "p-3 rounded-lg mb-3 flex items-center justify-between",
                  avgScore >= 0.5 && "bg-success/10 border border-success/30",
                  avgScore < 0.5 && avgScore > -0.5 && "bg-muted/30 border border-border",
                  avgScore <= -0.5 && "bg-danger/10 border border-danger/30",
                )}>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Consensus Rating</p>
                    <p className={cn(
                      "text-lg font-bold",
                      avgScore >= 0.5 && "text-success",
                      avgScore < 0.5 && avgScore > -0.5 && "text-muted-foreground",
                      avgScore <= -0.5 && "text-danger",
                    )}>
                      {aggregatedSignal}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center gap-3 text-sm">
                      <span className="flex items-center gap-1 text-success"><ArrowUpCircle className="h-3.5 w-3.5" /> {bullishCount}</span>
                      <span className="flex items-center gap-1 text-muted-foreground"><MinusCircle className="h-3.5 w-3.5" /> {agentAnalysis.verdicts.length - bullishCount - bearishCount}</span>
                      <span className="flex items-center gap-1 text-danger"><ArrowDownCircle className="h-3.5 w-3.5" /> {bearishCount}</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-1">
                      Avg confidence: {Math.round(agentAnalysis.verdicts.reduce((s, v) => s + v.confidence, 0) / agentAnalysis.verdicts.length)}%
                    </p>
                  </div>
                </div>
                
                {/* Show 2 verdicts by default, scrollable list when expanded */}
                <div className={cn(
                  "space-y-2",
                  showAllVerdicts ? "max-h-[250px] overflow-y-auto pr-1" : ""
                )}>
                  {(showAllVerdicts ? agentAnalysis.verdicts : agentAnalysis.verdicts.slice(0, 2)).map((verdict) => {
                    const isExpanded = expandedVerdicts.has(verdict.agent_id);
                    return (
                    <div 
                      key={verdict.agent_id}
                      className={cn(
                        "p-2 rounded-lg border cursor-pointer transition-all hover:shadow-sm",
                        ['strong_buy', 'buy', 'bullish'].includes(verdict.signal) && "bg-success/5 border-success/20",
                        ['hold', 'neutral'].includes(verdict.signal) && "bg-muted/30 border-border",
                        ['strong_sell', 'sell', 'bearish'].includes(verdict.signal) && "bg-danger/5 border-danger/20",
                      )}
                      onClick={() => {
                        setExpandedVerdicts(prev => {
                          const next = new Set(prev);
                          if (next.has(verdict.agent_id)) {
                            next.delete(verdict.agent_id);
                          } else {
                            next.add(verdict.agent_id);
                          }
                          return next;
                        });
                      }}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-xs font-bold">
                            {verdict.agent_name.split(' ').map(n => n[0]).join('')}
                          </div>
                          <span className="text-sm font-medium">{verdict.agent_name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="outline"
                            className={cn(
                              "text-xs",
                              ['strong_buy', 'buy', 'bullish'].includes(verdict.signal) && "border-success text-success",
                              ['hold', 'neutral'].includes(verdict.signal) && "border-muted-foreground text-muted-foreground",
                              ['strong_sell', 'sell', 'bearish'].includes(verdict.signal) && "border-danger text-danger",
                            )}
                          >
                            {verdict.signal.replace('_', ' ')}
                          </Badge>
                          <span className="text-xs text-muted-foreground">{verdict.confidence}%</span>
                          <ChevronDown className={cn("h-3 w-3 text-muted-foreground transition-transform", isExpanded && "rotate-180")} />
                        </div>
                      </div>
                      <p className={cn(
                        "text-xs text-muted-foreground leading-relaxed",
                        !isExpanded && "line-clamp-2"
                      )}>
                        {verdict.reasoning}
                      </p>
                      {!isExpanded && verdict.reasoning.length > 120 && (
                        <span className="text-[10px] text-primary mt-1 inline-block">Click to expand</span>
                      )}
                    </div>
                  )})}
                </div>
                
                {!showAllVerdicts && agentAnalysis.verdicts.length > 2 && (
                  <button 
                    onClick={() => setShowAllVerdicts(true)}
                    className="text-xs text-primary hover:underline mt-2 w-full text-center"
                  >
                    Show {agentAnalysis.verdicts.length - 2} more analysts...
                  </button>
                )}
              </div>
            );
            })()}

            {isLoadingAgents && (
              <div className="mb-4 space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-16 w-full" />
                <Skeleton className="h-16 w-full" />
              </div>
            )}

            {/* Active Signals */}
            {currentSignals && (currentSignals.buy_signals.length > 0 || currentSignals.sell_signals.length > 0) && (
              <div className="mb-4">
                <p className="text-xs text-muted-foreground mb-2">Active Signals</p>
                <div className="flex flex-wrap gap-1">
                  {currentSignals.buy_signals.map((sig, i) => (
                    <Tooltip key={`buy-${i}`}>
                      <TooltipTrigger asChild>
                        <Badge variant="outline" className="text-xs border-success/40 text-success bg-success/5 cursor-help">
                          <CheckCircle2 className="h-3 w-3 mr-1" />
                          {sig.name}
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{sig.description || `Buy signal: ${sig.name}`}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Value: {typeof sig.value === 'number' ? sig.value.toFixed(2) : sig.value}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                  {currentSignals.sell_signals.map((sig, i) => (
                    <Tooltip key={`sell-${i}`}>
                      <TooltipTrigger asChild>
                        <Badge variant="outline" className="text-xs border-warning/40 text-warning bg-warning/5 cursor-help">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          {sig.name}
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{sig.description || `Caution: ${sig.name}`}</p>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              </div>
            )}

            {/* Quant Strategy Signal - Only show if beats B&H */}
            {strategySignal && strategySignal.benchmarks.beats_buy_hold && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-primary" />
                  <span className="text-sm font-semibold">Optimized Strategy</span>
                  <Badge variant="outline" className="text-[10px] border-success/50 text-success bg-success/5">
                    ✓ Beats B&H
                  </Badge>
                </div>
                
                {/* Main signal card */}
                <div className={cn(
                  "p-3 rounded-lg border mb-3",
                  strategySignal.signal.type === 'BUY' && strategySignal.signal.has_active && "bg-success/10 border-success/30",
                  strategySignal.signal.type === 'SELL' && "bg-danger/10 border-danger/30",
                  strategySignal.signal.type === 'HOLD' && "bg-muted/30 border-border",
                  strategySignal.signal.type === 'WAIT' && "bg-amber-500/10 border-amber-500/30",
                  strategySignal.signal.type === 'WATCH' && "bg-blue-500/10 border-blue-500/30",
                )}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline"
                        className={cn(
                          "text-xs font-semibold",
                          strategySignal.signal.type === 'BUY' && "border-success text-success",
                          strategySignal.signal.type === 'SELL' && "border-danger text-danger",
                          strategySignal.signal.type === 'HOLD' && "border-muted-foreground text-muted-foreground",
                          strategySignal.signal.type === 'WAIT' && "border-amber-500 text-amber-600",
                          strategySignal.signal.type === 'WATCH' && "border-blue-500 text-blue-600",
                        )}
                      >
                        {strategySignal.signal.has_active ? '● ' : ''}{strategySignal.signal.type}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {strategySignal.strategy_name.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                    {strategySignal.signal.reason}
                  </p>
                </div>
                
                {/* Performance metrics */}
                <div className="grid grid-cols-4 gap-2 mb-3">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Win Rate</p>
                    <p className={cn(
                      "text-sm font-bold",
                      strategySignal.metrics.win_rate >= 60 && "text-success",
                      strategySignal.metrics.win_rate < 50 && "text-danger"
                    )}>
                      {strategySignal.metrics.win_rate.toFixed(0)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">2025 YTD</p>
                    <p className={cn(
                      "text-sm font-bold",
                      strategySignal.recency.current_year_return_pct >= 0 ? "text-success" : "text-danger"
                    )}>
                      {strategySignal.recency.current_year_return_pct >= 0 ? '+' : ''}
                      {strategySignal.recency.current_year_return_pct.toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Sharpe</p>
                    <p className={cn(
                      "text-sm font-bold",
                      strategySignal.metrics.sharpe_ratio >= 1.5 && "text-success",
                      strategySignal.metrics.sharpe_ratio < 0.5 && "text-danger"
                    )}>
                      {strategySignal.metrics.sharpe_ratio.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">vs B&H</p>
                    <p className={cn(
                      "text-sm font-bold",
                      strategySignal.benchmarks.vs_buy_hold >= 0 ? "text-success" : "text-danger"
                    )}>
                      {strategySignal.benchmarks.vs_buy_hold >= 0 ? '+' : ''}
                      {strategySignal.benchmarks.vs_buy_hold.toFixed(0)}%
                    </p>
                  </div>
                </div>
                
                {/* Fundamental health */}
                {!strategySignal.fundamentals.healthy && strategySignal.fundamentals.concerns.length > 0 && (
                  <div className="flex items-start gap-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600 mt-0.5 shrink-0" />
                    <div>
                      <p className="text-xs font-medium text-amber-700 dark:text-amber-500">Fundamental Concerns</p>
                      <p className="text-[10px] text-muted-foreground">
                        {strategySignal.fundamentals.concerns.slice(0, 2).join(', ')}
                        {strategySignal.fundamentals.concerns.length > 2 && ` +${strategySignal.fundamentals.concerns.length - 2} more`}
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Healthy fundamentals badge */}
                {strategySignal.fundamentals.healthy && (
                  <div className="flex items-center gap-1.5 text-xs text-success">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Fundamentals pass quality filters</span>
                  </div>
                )}
                
                {/* Stats footer */}
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/50 text-[10px] text-muted-foreground">
                  <span>{strategySignal.metrics.n_trades} trades • {strategySignal.indicators_used.slice(0, 2).join(', ')}</span>
                  {strategySignal.optimized_at && (
                    <span>Updated {new Date(strategySignal.optimized_at).toLocaleDateString()}</span>
                  )}
                </div>
              </div>
            )}
            
            {(isLoadingStrategy || isLoadingDipEntry) && !strategySignal && !dipEntry && (
              <div className="mb-4 space-y-2">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-4 w-4 rounded-full" />
                  <Skeleton className="h-4 w-32" />
                </div>
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            )}

            {isLoadingDipEntry && (
              <div className="mb-4 space-y-2">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-4 w-4 rounded-full" />
                  <Skeleton className="h-4 w-36" />
                </div>
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            )}

            {/* Dip Strategy Backtest - Show when no optimized strategy beats B&H */}
            {dipEntry?.backtest && (!strategySignal?.benchmarks.beats_buy_hold || !hasValidStrategy) && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingDown className="h-4 w-4 text-primary" />
                  <span className="text-sm font-semibold">Dip Buying Strategy</span>
                  <Badge variant="outline" className="text-[10px] border-muted-foreground/50 text-muted-foreground bg-muted/20">
                    Historical
                  </Badge>
                </div>
                
                {/* Strategy explanation */}
                <div className={cn(
                  "p-3 rounded-lg border mb-3",
                  dipEntry.is_buy_now && "bg-success/10 border-success/30",
                  !dipEntry.is_buy_now && "bg-muted/30 border-border",
                )}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline"
                        className={cn(
                          "text-xs font-semibold",
                          dipEntry.is_buy_now ? "border-success text-success" : "border-muted-foreground text-muted-foreground",
                        )}
                      >
                        {dipEntry.is_buy_now ? '● BUY' : '○ WAIT'}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        Target: -{((dipEntry.max_profit_threshold ?? 0) * 100).toFixed(0)}% from 52w high
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {dipEntry.is_buy_now 
                      ? <>Currently -{((dipEntry.current_drawdown_pct ?? 0) * 100).toFixed(1)}% from peak. Buy signal active.</>
                      : <>Currently -{((dipEntry.current_drawdown_pct ?? 0) * 100).toFixed(1)}% from peak. Need -{((dipEntry.max_profit_threshold ?? 0) * 100).toFixed(0)}% to trigger buy.</>
                    }
                  </p>
                </div>
                
                {/* Performance metrics */}
                <div className="grid grid-cols-4 gap-2 mb-3">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Win Rate</p>
                    <p className={cn(
                      "text-sm font-bold",
                      (dipEntry.backtest.win_rate ?? 0) >= 0.60 && "text-success",
                      (dipEntry.backtest.win_rate ?? 0) < 0.50 && "text-danger"
                    )}>
                      {((dipEntry.backtest.win_rate ?? 0) * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">{(dipEntry.backtest.years_tested ?? 0).toFixed(0)}yr Return</p>
                    <p className={cn(
                      "text-sm font-bold",
                      (dipEntry.backtest.strategy_return ?? 0) >= 0 ? "text-success" : "text-danger"
                    )}>
                      {(dipEntry.backtest.strategy_return ?? 0) >= 0 ? '+' : ''}
                      {((dipEntry.backtest.strategy_return ?? 0) * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Optimal Hold</p>
                    <p className="text-sm font-bold">
                      {dipEntry.backtest.optimal_holding_days ?? 90}d
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">vs B&H</p>
                    <p className={cn(
                      "text-sm font-bold",
                      (dipEntry.backtest.vs_buy_hold ?? 0) >= 0 ? "text-success" : "text-danger"
                    )}>
                      {(dipEntry.backtest.vs_buy_hold ?? 0) >= 0 ? '+' : ''}
                      {((dipEntry.backtest.vs_buy_hold ?? 0) * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
                
                {/* Stats footer */}
                <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>{dipEntry.backtest.n_trades ?? 0} trades over {(dipEntry.backtest.years_tested ?? 0).toFixed(1)} years</span>
                  <span>Avg {((dipEntry.backtest.avg_return_per_trade ?? 0) * 100).toFixed(1)}% per trade</span>
                </div>
              </div>
            )}

            <Separator className="my-3" />
            
            {/* Fundamentals & Company Info */}
            <details className="group">
              <summary className="flex items-center justify-between cursor-pointer py-2 text-sm font-medium">
                <span>Fundamentals & Company</span>
                <ChevronDown className="h-4 w-4 text-muted-foreground group-open:rotate-180 transition-transform" />
              </summary>
              <div className="pt-2 space-y-3">
                {/* Key metrics grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <StatItem
                    icon={Building2}
                    label="Market Cap"
                    value={formatMarketCap(stock.market_cap ?? stockInfo?.market_cap ?? null)}
                    tooltip="Total value of all outstanding shares. Larger companies tend to be more stable but may grow slower."
                  />
                  <StatItem
                    icon={BarChart3}
                    label="P/E Ratio"
                    value={fundamentals?.pe_ratio?.toFixed(2) ?? stock.pe_ratio?.toFixed(2) ?? '—'}
                    tooltip="Price-to-Earnings ratio. Shows how much investors pay per dollar of earnings. Lower (<15) = cheaper, higher (>25) = growth priced in."
                  />
                  <StatItem
                    icon={Target}
                    label="Fwd P/E"
                    value={fundamentals?.forward_pe?.toFixed(2) ?? '—'}
                    tooltip="Forward P/E uses expected future earnings. If lower than P/E, earnings are expected to grow."
                  />
                  <StatItem
                    icon={DollarSign}
                    label="Target Price"
                    value={fundamentals?.target_mean_price ? `$${fundamentals.target_mean_price.toFixed(0)}` : '—'}
                    tooltip="Average analyst price target. Compare to current price to see potential upside/downside."
                  />
                </div>
                
                {/* Valuation & Health Metrics */}
                {fundamentals && (
                  <div className="grid grid-cols-4 gap-2 text-center">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-[10px] text-muted-foreground">P/B Ratio</p>
                          <p className={cn(
                            "text-sm font-medium",
                            fundamentals.price_to_book && Number(fundamentals.price_to_book) < 1 ? "text-success" :
                            fundamentals.price_to_book && Number(fundamentals.price_to_book) > 3 ? "text-warning" : ""
                          )}>
                            {fundamentals.price_to_book != null ? Number(fundamentals.price_to_book).toFixed(2) : '—'}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[200px]">
                        <p className="text-xs">Price-to-Book: Stock price vs book value. Under 1 = trading below asset value (potentially undervalued).</p>
                      </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-[10px] text-muted-foreground">D/E Ratio</p>
                          <p className={cn(
                            "text-sm font-medium",
                            fundamentals.debt_to_equity && Number(fundamentals.debt_to_equity) < 0.5 ? "text-success" :
                            fundamentals.debt_to_equity && Number(fundamentals.debt_to_equity) > 1.5 ? "text-danger" : ""
                          )}>
                            {fundamentals.debt_to_equity != null ? Number(fundamentals.debt_to_equity).toFixed(2) : '—'}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[200px]">
                        <p className="text-xs">Debt-to-Equity: How much debt vs shareholder equity. Under 0.5 = conservative, over 1.5 = high leverage.</p>
                      </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-[10px] text-muted-foreground">Current Ratio</p>
                          <p className={cn(
                            "text-sm font-medium",
                            fundamentals.current_ratio && Number(fundamentals.current_ratio) > 1.5 ? "text-success" :
                            fundamentals.current_ratio && Number(fundamentals.current_ratio) < 1 ? "text-danger" : ""
                          )}>
                            {fundamentals.current_ratio != null ? Number(fundamentals.current_ratio).toFixed(2) : '—'}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[200px]">
                        <p className="text-xs">Current Ratio: Assets vs liabilities due within 1 year. Above 1.5 = healthy, below 1 = liquidity risk.</p>
                      </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="p-2 rounded bg-muted/30 cursor-help">
                          <p className="text-[10px] text-muted-foreground">PEG Ratio</p>
                          <p className={cn(
                            "text-sm font-medium",
                            fundamentals.peg_ratio && Number(fundamentals.peg_ratio) < 1 ? "text-success" :
                            fundamentals.peg_ratio && Number(fundamentals.peg_ratio) > 2 ? "text-warning" : ""
                          )}>
                            {fundamentals.peg_ratio != null ? Number(fundamentals.peg_ratio).toFixed(2) : '—'}
                          </p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-[200px]">
                        <p className="text-xs">P/E to Growth: Under 1 = growth is cheap, 1-2 = fair value, over 2 = growth is expensive.</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                )}
                
                {/* Intrinsic Value - now shown prominently at top of panel */}
                
                {/* Growth & Returns */}
                {fundamentals && (
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded bg-muted/30">
                      <p className="text-[10px] text-muted-foreground">Revenue Growth</p>
                      <p className={cn(
                        "text-sm font-medium",
                        fundamentals.revenue_growth?.includes('-') ? "text-danger" : "text-success"
                      )}>
                        {fundamentals.revenue_growth || '—'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-muted/30">
                      <p className="text-[10px] text-muted-foreground">Profit Margin</p>
                      <p className={cn(
                        "text-sm font-medium",
                        fundamentals.profit_margin?.includes('-') ? "text-danger" : "text-success"
                      )}>
                        {fundamentals.profit_margin || '—'}
                      </p>
                    </div>
                    <div className="p-2 rounded bg-muted/30">
                      <p className="text-[10px] text-muted-foreground">ROE</p>
                      <p className={cn(
                        "text-sm font-medium",
                        fundamentals.return_on_equity?.includes('-') ? "text-danger" : "text-success"
                      )}>
                        {fundamentals.return_on_equity || '—'}
                      </p>
                    </div>
                  </div>
                )}

                {/* Analyst consensus */}
                {fundamentals?.recommendation && (
                  <div className="flex items-center gap-3">
                    <Badge 
                      variant={fundamentals.recommendation === 'buy' ? 'default' : 'secondary'}
                      className={cn(
                        fundamentals.recommendation === 'buy' && "bg-success",
                        fundamentals.recommendation === 'sell' && "bg-danger",
                      )}
                    >
                      {fundamentals.recommendation.toUpperCase()}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {fundamentals.num_analyst_opinions} analyst{fundamentals.num_analyst_opinions !== 1 ? 's' : ''}
                    </span>
                    {fundamentals.next_earnings_date && (
                      <span className="text-xs text-muted-foreground">
                        • Earnings: {new Date(fundamentals.next_earnings_date).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                )}

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
  tooltip?: string;
}

function StatItem({ icon: Icon, label, value, valueColor, tooltip }: StatItemProps) {
  const content = (
    <div className={cn(
      "flex items-center gap-2 p-2 rounded-lg bg-muted/50",
      tooltip && "cursor-help"
    )}>
      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground truncate">{label}</p>
        <p className={`text-sm font-medium truncate ${valueColor || ''}`}>{value}</p>
      </div>
    </div>
  );

  if (tooltip) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent className="max-w-[250px]">
          <p className="text-xs">{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    );
  }

  return content;
}
