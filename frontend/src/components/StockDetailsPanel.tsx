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
                    {/* Buy signal trigger dots (green with lightning icon) */}
                    {dotsVisible && showSignals && signalPoints.map((point, idx) => (
                      <ReferenceDot
                        key={`signal-${idx}`}
                        x={point.displayDate}
                        y={point.close}
                        r={8}
                        fill="var(--success)"
                        stroke="var(--background)"
                        strokeWidth={2}
                      >
                        <RechartsLabel
                          position="top"
                          offset={10}
                          style={{
                            fontSize: 10,
                            fill: 'var(--foreground)',
                            fontWeight: 600,
                          }}
                        >
                          {`${(point.signal.drawdown_pct * 100).toFixed(0)}%`}
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

          {/* Data Section */}
          <CardContent className="pt-3 pb-4 px-3 md:px-6">
            {/* Key Stats */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3">
              <StatItem
                icon={TrendingDown}
                label="Dip Depth"
                value={`-${(stock.depth * 100).toFixed(1)}%`}
                valueColor="text-danger"
              />
              <StatItem
                icon={Calendar}
                label="Days in Dip"
                value={stock.days_since_dip?.toString() ?? '—'}
              />
              <StatItem
                icon={DollarSign}
                label="52W High"
                value={stock.high_52w ? `$${stock.high_52w.toFixed(2)}` : '—'}
              />
              <StatItem
                icon={DollarSign}
                label="52W Low"
                value={stock.low_52w ? `$${stock.low_52w.toFixed(2)}` : '—'}
              />
              <StatItem
                icon={Building2}
                label="Market Cap"
                value={formatMarketCap(stock.market_cap ?? stockInfo?.market_cap ?? null)}
              />
              <StatItem
                icon={BarChart3}
                label="P/E Ratio"
                value={stock.pe_ratio?.toFixed(2) ?? '—'}
              />
            </div>

            {/* Extended Fundamentals from stockInfo */}
            {stockInfo && (
              <>
                <Separator className="my-3" />
                
                {/* Valuation & Risk Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3 mb-3">
                  {stockInfo.forward_pe !== null && stockInfo.forward_pe !== undefined && (
                    <StatItem
                      icon={Activity}
                      label="Forward P/E"
                      value={stockInfo.forward_pe.toFixed(2)}
                    />
                  )}
                  {stockInfo.peg_ratio !== null && stockInfo.peg_ratio !== undefined && (
                    <StatItem
                      icon={Activity}
                      label="PEG Ratio"
                      value={stockInfo.peg_ratio.toFixed(2)}
                      valueColor={stockInfo.peg_ratio < 1 ? 'text-success' : stockInfo.peg_ratio > 2 ? 'text-danger' : undefined}
                    />
                  )}
                  {stockInfo.dividend_yield !== null && stockInfo.dividend_yield !== undefined && (
                    <StatItem
                      icon={Percent}
                      label="Div Yield"
                      value={`${(stockInfo.dividend_yield * 100).toFixed(2)}%`}
                      valueColor={stockInfo.dividend_yield > 0 ? 'text-success' : undefined}
                    />
                  )}
                  {stockInfo.beta !== null && stockInfo.beta !== undefined && (
                    <StatItem
                      icon={Activity}
                      label="Beta"
                      value={stockInfo.beta.toFixed(2)}
                    />
                  )}
                </div>

                {/* Profitability Metrics */}
                {(stockInfo.profit_margin !== null || stockInfo.gross_margin !== null || stockInfo.return_on_equity !== null) && (
                  <>
                    <p className="text-xs font-medium text-muted-foreground mb-2">Profitability</p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3 mb-3">
                      {stockInfo.profit_margin !== null && stockInfo.profit_margin !== undefined && (
                        <StatItem
                          icon={Percent}
                          label="Profit Margin"
                          value={`${(stockInfo.profit_margin * 100).toFixed(1)}%`}
                          valueColor={stockInfo.profit_margin > 0.15 ? 'text-success' : stockInfo.profit_margin < 0.05 ? 'text-danger' : undefined}
                        />
                      )}
                      {stockInfo.gross_margin !== null && stockInfo.gross_margin !== undefined && (
                        <StatItem
                          icon={Percent}
                          label="Gross Margin"
                          value={`${(stockInfo.gross_margin * 100).toFixed(1)}%`}
                          valueColor={stockInfo.gross_margin > 0.4 ? 'text-success' : stockInfo.gross_margin < 0.2 ? 'text-danger' : undefined}
                        />
                      )}
                      {stockInfo.return_on_equity !== null && stockInfo.return_on_equity !== undefined && (
                        <StatItem
                          icon={PiggyBank}
                          label="ROE"
                          value={`${(stockInfo.return_on_equity * 100).toFixed(1)}%`}
                          valueColor={stockInfo.return_on_equity > 0.15 ? 'text-success' : stockInfo.return_on_equity < 0.05 ? 'text-danger' : undefined}
                        />
                      )}
                      {stockInfo.revenue_growth !== null && stockInfo.revenue_growth !== undefined && (
                        <StatItem
                          icon={TrendingUp}
                          label="Revenue Growth"
                          value={`${(stockInfo.revenue_growth * 100).toFixed(1)}%`}
                          valueColor={stockInfo.revenue_growth > 0 ? 'text-success' : 'text-danger'}
                        />
                      )}
                    </div>
                  </>
                )}

                {/* Financial Health Metrics */}
                {(stockInfo.debt_to_equity !== null || stockInfo.current_ratio !== null || stockInfo.free_cash_flow !== null) && (
                  <>
                    <p className="text-xs font-medium text-muted-foreground mb-2">Financial Health</p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3 mb-3">
                      {stockInfo.debt_to_equity !== null && stockInfo.debt_to_equity !== undefined && (
                        <StatItem
                          icon={Scale}
                          label="Debt/Equity"
                          value={stockInfo.debt_to_equity.toFixed(2)}
                          valueColor={stockInfo.debt_to_equity < 50 ? 'text-success' : stockInfo.debt_to_equity > 150 ? 'text-danger' : undefined}
                        />
                      )}
                      {stockInfo.current_ratio !== null && stockInfo.current_ratio !== undefined && (
                        <StatItem
                          icon={Scale}
                          label="Current Ratio"
                          value={stockInfo.current_ratio.toFixed(2)}
                          valueColor={stockInfo.current_ratio > 1.5 ? 'text-success' : stockInfo.current_ratio < 1 ? 'text-danger' : undefined}
                        />
                      )}
                      {stockInfo.free_cash_flow !== null && stockInfo.free_cash_flow !== undefined && (
                        <StatItem
                          icon={Banknote}
                          label="Free Cash Flow"
                          value={formatMarketCap(stockInfo.free_cash_flow)}
                          valueColor={stockInfo.free_cash_flow > 0 ? 'text-success' : 'text-danger'}
                        />
                      )}
                    </div>
                  </>
                )}

                {/* Analyst Info */}
                {(stockInfo.target_mean_price !== null || stockInfo.num_analyst_opinions !== null) && (
                  <>
                    <p className="text-xs font-medium text-muted-foreground mb-2">Analyst Estimates</p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3 mb-3">
                      {stockInfo.target_mean_price !== null && stockInfo.target_mean_price !== undefined && (
                        <StatItem
                          icon={Target}
                          label="Price Target"
                          value={`$${stockInfo.target_mean_price.toFixed(2)}`}
                          valueColor={stockInfo.target_mean_price > displayPrice ? 'text-success' : 'text-danger'}
                        />
                      )}
                      {stockInfo.num_analyst_opinions !== null && stockInfo.num_analyst_opinions !== undefined && (
                        <StatItem
                          icon={Users}
                          label="Analysts"
                          value={stockInfo.num_analyst_opinions.toString()}
                        />
                      )}
                    </div>
                  </>
                )}
                
                <div className="space-y-3">
                  {stockInfo.sector && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <Badge variant="secondary">{stockInfo.sector}</Badge>
                      {stockInfo.industry && (
                        <span className="text-sm text-muted-foreground">{stockInfo.industry}</span>
                      )}
                    </div>
                  )}

                  {(stockInfo.summary_ai || stockInfo.summary) && (
                    <div className="mt-2 p-3 bg-muted/30 rounded-lg">
                      <p className="text-xs font-medium text-muted-foreground mb-1">About the Company</p>
                      <p className="text-sm leading-relaxed">
                        {stockInfo.summary_ai || stockInfo.summary}
                      </p>
                    </div>
                  )}

                  {stockInfo.recommendation && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">Analyst Rating:</span>
                      <Badge variant={
                        stockInfo.recommendation.includes('buy') ? 'default' :
                        stockInfo.recommendation.includes('sell') ? 'destructive' : 'secondary'
                      }>
                        {stockInfo.recommendation.replace('_', ' ').toUpperCase()}
                      </Badge>
                    </div>
                  )}

                  {stockInfo.website && (
                    <a
                      href={stockInfo.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ExternalLink className="h-3 w-3" />
                      Visit website
                    </a>
                  )}
                </div>
              </>
            )}

            {isLoadingInfo && (
              <>
                <Separator className="my-4" />
                <div className="space-y-2">
                  <Skeleton className="h-5 w-24" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                </div>
              </>
            )}

            {/* Quant Signals Section */}
            {(dipAnalysis || currentSignals || isLoadingQuant) && (
              <>
                <Separator className="my-4" />
                <div className="p-3 bg-gradient-to-br from-chart-1/5 to-chart-2/10 rounded-lg border border-chart-1/20">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUpDown className="h-4 w-4 text-chart-1" />
                    <span className="text-sm font-medium">Quant Signals</span>
                  </div>
                  {isLoadingQuant ? (
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-full" />
                      <Skeleton className="h-4 w-3/4" />
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Overall Action */}
                      {currentSignals && (
                        <div className="flex items-center gap-2">
                          {currentSignals.overall_action === 'STRONG_BUY' && (
                            <Badge className="bg-success text-success-foreground">
                              <CheckCircle2 className="h-3 w-3 mr-1" />
                              Strong Buy
                            </Badge>
                          )}
                          {currentSignals.overall_action === 'BUY' && (
                            <Badge className="bg-success/80 text-success-foreground">
                              <TrendingUp className="h-3 w-3 mr-1" />
                              Buy
                            </Badge>
                          )}
                          {currentSignals.overall_action === 'HOLD' && (
                            <Badge variant="secondary">
                              <Activity className="h-3 w-3 mr-1" />
                              Hold
                            </Badge>
                          )}
                          {currentSignals.overall_action === 'SELL' && (
                            <Badge className="bg-warning text-warning-foreground">
                              <TrendingDown className="h-3 w-3 mr-1" />
                              Sell
                            </Badge>
                          )}
                          {currentSignals.overall_action === 'STRONG_SELL' && (
                            <Badge variant="destructive">
                              <XCircle className="h-3 w-3 mr-1" />
                              Strong Sell
                            </Badge>
                          )}
                          <span className="text-xs text-muted-foreground">
                            {currentSignals.reasoning}
                          </span>
                        </div>
                      )}

                      {/* Dip Analysis */}
                      {dipAnalysis && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Dip Type:</span>
                            <Badge 
                              variant={
                                dipAnalysis.dip_type === 'overreaction'
                                  ? 'default' 
                                  : dipAnalysis.dip_type === 'fundamental_decline' 
                                    ? 'destructive' 
                                    : 'secondary'
                              }
                              className="text-xs"
                            >
                              {dipAnalysis.dip_type.toUpperCase().replace('_', ' ')}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Current Dip:</span>
                            <span className="font-mono font-medium text-danger">
                              -{dipAnalysis.current_drawdown_pct.toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Typical Dip:</span>
                            <span className="font-mono">{dipAnalysis.typical_dip_pct.toFixed(1)}%</span>
                          </div>
                          {dipAnalysis.is_unusually_deep && (
                            <div className="flex items-center gap-1 text-xs text-warning">
                              <AlertTriangle className="h-3 w-3" />
                              Unusually deep ({dipAnalysis.dip_zscore.toFixed(1)}σ from typical)
                            </div>
                          )}
                          {/* Technical indicators */}
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Technical Score:</span>
                            <span className={`font-mono font-medium ${dipAnalysis.technical_score > 0 ? 'text-success' : dipAnalysis.technical_score < 0 ? 'text-danger' : ''}`}>
                              {dipAnalysis.technical_score > 0 ? '+' : ''}{(dipAnalysis.technical_score * 100).toFixed(0)}%
                            </span>
                          </div>
                          {dipAnalysis.momentum_divergence && (
                            <div className="flex items-center gap-1 text-xs text-success">
                              <TrendingUp className="h-3 w-3" />
                              Bullish divergence detected
                            </div>
                          )}
                          {dipAnalysis.trend_broken && (
                            <div className="flex items-center gap-1 text-xs text-warning">
                              <AlertTriangle className="h-3 w-3" />
                              Long-term trend broken
                            </div>
                          )}
                          {dipAnalysis.recovery_probability > 0.6 && (
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">Recovery Prob:</span>
                              <span className="font-mono text-success">{(dipAnalysis.recovery_probability * 100).toFixed(0)}%</span>
                            </div>
                          )}
                          <p className="text-xs text-muted-foreground">
                            {dipAnalysis.reasoning}
                          </p>
                        </div>
                      )}

                      {/* Active Signals */}
                      {currentSignals && currentSignals.buy_signals.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-success mb-1">Active Buy Signals:</p>
                          <div className="flex flex-wrap gap-1">
                            {currentSignals.buy_signals.map((sig, i) => (
                              <Badge key={i} variant="outline" className="text-xs border-success/30 text-success">
                                {sig.name}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {currentSignals && currentSignals.sell_signals.length > 0 && (
                        <div>
                          <p className="text-xs font-medium text-warning mb-1">Active Sell Signals:</p>
                          <div className="flex flex-wrap gap-1">
                            {currentSignals.sell_signals.map((sig, i) => (
                              <Badge key={i} variant="outline" className="text-xs border-warning/30 text-warning">
                                {sig.name}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}

            {/* AI Analysis Section */}
            {(aiData?.ai_reasoning || aiData?.domain_analysis || aiData?.domain_context || isLoadingAi) && (
              <>
                <Separator className="my-4" />
                <div className="p-3 bg-gradient-to-br from-primary/5 to-primary/10 rounded-lg border border-primary/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">AI Analysis</span>
                  </div>
                  {isLoadingAi ? (
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-full" />
                      <Skeleton className="h-3 w-4/5" />
                      <Skeleton className="h-3 w-3/4" />
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Domain-specific quant analysis */}
                      {aiData?.domain_context && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <p className="text-xs font-medium text-primary">Sector Analysis</p>
                            {aiData.sector && (
                              <Badge variant="secondary" className="text-[10px] py-0">
                                {aiData.sector}
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs leading-relaxed text-muted-foreground">
                            {aiData.domain_context}
                          </p>
                          
                          {/* Domain adjustment reason */}
                          {aiData.domain_adjustment_reason && (
                            <p className="text-xs italic text-muted-foreground">
                              {aiData.domain_adjustment_reason}
                            </p>
                          )}
                          
                          {/* Key metrics row */}
                          <div className="flex flex-wrap gap-2 mt-2">
                            {aiData.volatility_regime && (
                              <Badge 
                                variant="outline" 
                                className={cn(
                                  "text-[10px] py-0",
                                  aiData.volatility_regime === 'extreme' && "border-red-500 text-red-500",
                                  aiData.volatility_regime === 'high' && "border-orange-500 text-orange-500",
                                  aiData.volatility_regime === 'normal' && "border-blue-500 text-blue-500",
                                  aiData.volatility_regime === 'low' && "border-green-500 text-green-500",
                                )}
                              >
                                Vol: {aiData.volatility_regime}
                              </Badge>
                            )}
                            {aiData.domain_risk_level && (
                              <Badge 
                                variant="outline" 
                                className={cn(
                                  "text-[10px] py-0",
                                  aiData.domain_risk_level === 'high' && "border-red-500 text-red-500",
                                  aiData.domain_risk_level === 'medium' && "border-yellow-500 text-yellow-500",
                                  aiData.domain_risk_level === 'low' && "border-green-500 text-green-500",
                                )}
                              >
                                Risk: {aiData.domain_risk_level}
                              </Badge>
                            )}
                            {aiData.vs_sector_performance !== null && aiData.vs_sector_performance !== undefined && (
                              <Badge 
                                variant="outline" 
                                className={cn(
                                  "text-[10px] py-0",
                                  aiData.vs_sector_performance >= 0 ? "border-green-500 text-green-500" : "border-red-500 text-red-500",
                                )}
                              >
                                vs Sector: {aiData.vs_sector_performance >= 0 ? '+' : ''}{aiData.vs_sector_performance.toFixed(1)}%
                              </Badge>
                            )}
                            {aiData.domain_recovery_days && (
                              <Badge variant="outline" className="text-[10px] py-0">
                                Typical recovery: {aiData.domain_recovery_days}d
                              </Badge>
                            )}
                          </div>
                          
                          {/* Domain warnings */}
                          {aiData.domain_warnings && aiData.domain_warnings.length > 0 && (
                            <div className="mt-2 space-y-1">
                              {aiData.domain_warnings.map((warning, i) => (
                                <p key={i} className="text-[10px] text-orange-500 flex items-center gap-1">
                                  <AlertTriangle className="h-3 w-3" />
                                  {warning}
                                </p>
                              ))}
                            </div>
                          )}
                          
                          {/* Risk factors */}
                          {aiData.domain_risk_factors && aiData.domain_risk_factors.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              <span className="text-[10px] text-muted-foreground">Key risks:</span>
                              {aiData.domain_risk_factors.slice(0, 3).map((factor, i) => (
                                <Badge key={i} variant="secondary" className="text-[10px] py-0">
                                  {factor}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Fallback to old domain_analysis if no domain_context */}
                      {!aiData?.domain_context && aiData?.domain_analysis && (
                        <div>
                          <p className="text-xs font-medium text-primary mb-1">Sector Analysis</p>
                          <p className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                            {aiData.domain_analysis}
                          </p>
                        </div>
                      )}
                      {/* General AI reasoning */}
                      {aiData?.ai_reasoning && (
                        <div>
                          {(aiData?.domain_analysis || aiData?.domain_context) && (
                            <p className="text-xs font-medium text-primary mb-1">General Analysis</p>
                          )}
                          <p className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                            {aiData.ai_reasoning}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}
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
