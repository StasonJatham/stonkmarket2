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
import type { DipStock, ChartDataPoint, StockInfo, BenchmarkType, ComparisonChartData, SignalTrigger, SignalTriggersResponse, DipAnalysis, CurrentSignals, AgentAnalysis, SymbolFundamentals } from '@/services/api';
import { getSignalTriggers, getDipAnalysis, getCurrentSignals, getAgentAnalysis, getSymbolFundamentals } from '@/services/api';
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
];

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
  
  // Fetch signal triggers when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      setSignalsResponse(null);
      return;
    }
    
    getSignalTriggers(stock.symbol, Math.max(chartPeriod + 30, 365))
      .then(setSignalsResponse)
      .catch(() => setSignalsResponse(null));
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
  
  // Fetch AI agents and fundamentals when stock changes
  useEffect(() => {
    if (!stock?.symbol) {
      setAgentAnalysis(null);
      setFundamentals(null);
      return;
    }
    
    setIsLoadingAgents(true);
    setShowAllVerdicts(false);
    Promise.all([
      getAgentAnalysis(stock.symbol).catch(() => null),
      getSymbolFundamentals(stock.symbol).catch(() => null),
    ]).then(([agents, funds]) => {
      setAgentAnalysis(agents);
      setFundamentals(funds);
      setIsLoadingAgents(false);
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
                    {dotsVisible && showSignals && signalPoints.map((point, idx) => {
                      // Calculate expected value: win_rate * avg_return (risk-adjusted return)
                      const expectedValue = point.signal.win_rate * point.signal.avg_return_pct;
                      const isEntry = point.signal.signal_type === 'entry';
                      const dotColor = isEntry ? 'var(--success)' : 'var(--danger)';
                      const emoji = isEntry ? '🟢' : '🔴';
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
                            ? `${emoji} ${typeLabel}: ${signal.signal_name}\n` +
                              `Win Rate: ${(signal.win_rate * 100).toFixed(0)}%\n` +
                              `Avg Return: ${signal.avg_return_pct >= 0 ? '+' : ''}${signal.avg_return_pct.toFixed(1)}%\n` +
                              `Expected Value: ${expectedValue >= 0 ? '+' : ''}${expectedValue.toFixed(1)}%\n` +
                              `Hold: ${signal.holding_days} days\n` +
                              `Entry Price: $${signal.price.toFixed(2)}`
                            : `${emoji} ${typeLabel}: ${signal.signal_name}\n` +
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
                  {agentAnalysis && (
                    <div className="flex gap-1 text-xs">
                      <span className="text-success">{agentAnalysis.bullish_count || 0} 👍</span>
                      <span className="text-muted-foreground">{agentAnalysis.neutral_count || 0} —</span>
                      <span className="text-danger">{agentAnalysis.bearish_count || 0} 👎</span>
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

            {/* Quick Stats Row */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 text-center cursor-help">
                    <p className="text-xs text-muted-foreground">Dip Depth</p>
                    <p className="text-lg font-bold text-danger">-{(stock.depth * 100).toFixed(0)}%</p>
                    <p className="text-[10px] text-muted-foreground">from 52w high</p>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Current price is {(stock.depth * 100).toFixed(1)}% below 52-week high</p>
                </TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 text-center cursor-help">
                    <p className="text-xs text-muted-foreground">Historical Win</p>
                    <p className={cn(
                      "text-lg font-bold",
                      signalTriggers.length > 0 && signalTriggers[0].win_rate > 0.6 ? "text-success" : "text-foreground"
                    )}>
                      {signalTriggers.length > 0 
                        ? `${(signalTriggers[0].win_rate * 100).toFixed(0)}%`
                        : '—'}
                    </p>
                    <p className="text-[10px] text-muted-foreground">signals profitable</p>
                  </div>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p>{signalTriggers.length > 0 
                    ? `${(signalTriggers[0].win_rate * 100).toFixed(0)}% of similar buy signals historically led to profits. Based on ${signalTriggers.length} signals.`
                    : 'No historical signal data available'}</p>
                </TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-3 rounded-lg bg-muted/50 text-center cursor-help">
                    <p className="text-xs text-muted-foreground">Avg Return</p>
                    <p className={cn(
                      "text-lg font-bold",
                      signalTriggers.length > 0 && signalTriggers[0].avg_return_pct > 0 ? "text-success" : "text-danger"
                    )}>
                      {signalTriggers.length > 0 
                        ? `+${signalTriggers[0].avg_return_pct.toFixed(0)}%`
                        : '—'}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{signalTriggers.length > 0 ? `${signalTriggers[0].holding_days}d hold` : 'expected'}</p>
                  </div>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p>{signalTriggers.length > 0 
                    ? `Average return of +${signalTriggers[0].avg_return_pct.toFixed(1)}% when holding for ${signalTriggers[0].holding_days} days after similar signals`
                    : 'No historical return data available'}</p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Domain-Specific Metrics (Banks, REITs, Insurance) */}
            {fundamentals && fundamentals.domain !== 'stock' && (
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
                      <p className="text-[10px] text-muted-foreground">
                        {fundamentals.net_interest_margin && fundamentals.net_interest_margin > 0.03 
                          ? '✅ Above avg (3%)' 
                          : fundamentals.net_interest_margin && fundamentals.net_interest_margin > 0.02 
                            ? '⚠️ Average' 
                            : fundamentals.net_interest_margin 
                              ? '❌ Below avg'
                              : ''}
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
                      <p className="text-[10px] text-muted-foreground">
                        {fundamentals.p_ffo && fundamentals.p_ffo < 15 
                          ? '✅ Undervalued' 
                          : fundamentals.p_ffo && fundamentals.p_ffo > 20 
                            ? '⚠️ Expensive'
                            : ''}
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
            {agentAnalysis && agentAnalysis.verdicts.length > 0 && (
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
                
                {/* Top 3 verdicts always visible */}
                <div className="space-y-2">
                  {(showAllVerdicts ? agentAnalysis.verdicts : agentAnalysis.verdicts.slice(0, 3)).map((verdict) => (
                    <div 
                      key={verdict.agent_id}
                      className={cn(
                        "p-2 rounded-lg border",
                        ['strong_buy', 'buy', 'bullish'].includes(verdict.signal) && "bg-success/5 border-success/20",
                        ['hold', 'neutral'].includes(verdict.signal) && "bg-muted/30 border-border",
                        ['strong_sell', 'sell', 'bearish'].includes(verdict.signal) && "bg-danger/5 border-danger/20",
                      )}
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
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                        {verdict.reasoning}
                      </p>
                    </div>
                  ))}
                </div>
                
                {!showAllVerdicts && agentAnalysis.verdicts.length > 3 && (
                  <button 
                    onClick={() => setShowAllVerdicts(true)}
                    className="text-xs text-primary hover:underline mt-2 w-full text-center"
                  >
                    Show {agentAnalysis.verdicts.length - 3} more analysts...
                  </button>
                )}
              </div>
            )}

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
                  />
                  <StatItem
                    icon={BarChart3}
                    label="P/E Ratio"
                    value={fundamentals?.pe_ratio?.toFixed(2) ?? stock.pe_ratio?.toFixed(2) ?? '—'}
                  />
                  <StatItem
                    icon={Target}
                    label="Fwd P/E"
                    value={fundamentals?.forward_pe?.toFixed(2) ?? '—'}
                  />
                  <StatItem
                    icon={DollarSign}
                    label="Target Price"
                    value={fundamentals?.target_mean_price ? `$${fundamentals.target_mean_price.toFixed(0)}` : '—'}
                  />
                </div>
                
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

