import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  getStockChart,
  getStockInfo,
  type DipSignal, 
  type ChartDataPoint,
  type StockInfo,
} from '@/services/api';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from '@/components/ui/sheet';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { HelpCircle } from 'lucide-react';
import { 
  TrendingDown, 
  BarChart3,
  Activity,
  Target,
  Gauge,
  Clock,
  Building2,
  DollarSign,
  TrendingUp,
  Percent,
  ThumbsUp,
  ThumbsDown,
  Minus,
  Info,
  ExternalLink,
} from 'lucide-react';
import {
  Area,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ReferenceDot,
} from 'recharts';

// Chart timeframe options
const CHART_TIMEFRAMES = [
  { value: 7, label: '1W' },
  { value: 30, label: '1M' },
  { value: 90, label: '3M' },
  { value: 365, label: '1Y' },
  { value: 1825, label: '5Y' },
  { value: -1, label: 'All' }, // Special value for max available
];

// Dip type badge - shows whether dip is stock-specific, mixed, or market-wide
function getDipTypeBadge(dipType: string, colorblindMode: boolean) {
  const config: Record<string, { color: string; colorblindColor: string; label: string; desc: string }> = {
    STOCK_SPECIFIC: { 
      color: 'bg-success/20 text-success border-success/30', 
      colorblindColor: 'bg-blue-500/20 text-blue-500 border-blue-500/30',
      label: 'Stock',
      desc: 'This dip is specific to this stock, not market-wide' 
    },
    MIXED: { 
      color: 'bg-chart-2/20 text-chart-2 border-chart-2/30', 
      colorblindColor: 'bg-purple-500/20 text-purple-500 border-purple-500/30',
      label: 'Mixed',
      desc: 'Part stock-specific, part market-driven dip' 
    },
    MARKET_DIP: { 
      color: 'bg-chart-4/20 text-chart-4 border-chart-4/30', 
      colorblindColor: 'bg-orange-500/20 text-orange-500 border-orange-500/30',
      label: 'Market',
      desc: 'This dip is primarily driven by overall market decline' 
    },
  };
  const item = config[dipType] || { color: '', colorblindColor: '', label: dipType, desc: '' };
  return (
    <Badge variant="outline" className={`${colorblindMode ? item.colorblindColor : item.color} font-medium`} title={item.desc}>
      {item.label}
    </Badge>
  );
}

// Score badge with colorblind-safe colors
function getScoreBadgeClass(score: number, colorblindMode: boolean): string {
  if (score >= 70) {
    return colorblindMode ? 'bg-blue-500/20 text-blue-500' : 'bg-success/20 text-success';
  }
  if (score >= 50) {
    return colorblindMode ? 'bg-purple-500/20 text-purple-500' : 'bg-chart-4/20 text-chart-4';
  }
  if (score >= 30) {
    return colorblindMode ? 'bg-orange-500/20 text-orange-500' : 'bg-chart-2/20 text-chart-2';
  }
  return 'bg-muted text-muted-foreground';
}

// Progress bar with colorblind-safe colors
function getProgressColor(colorblindMode: boolean, type: 'quality' | 'stability'): string {
  if (colorblindMode) {
    return type === 'quality' ? 'bg-blue-500' : 'bg-purple-500';
  }
  return type === 'quality' ? 'bg-success' : 'bg-chart-2';
}

function formatPercent(value: number, decimals = 1): string {
  const pct = value * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(decimals)}%`;
}

function formatScore(value: number): string {
  return value.toFixed(0);
}

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toLocaleString()}`;
}

interface SignalDetailsSheetProps {
  signal: DipSignal | null;
  isOpen: boolean;
  onClose: () => void;
  colorblindMode: boolean;
}

export function SignalDetailsSheet({
  signal,
  isOpen,
  onClose,
  colorblindMode,
}: SignalDetailsSheetProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [isLoadingInfo, setIsLoadingInfo] = useState(false);
  const [chartTimeframe, setChartTimeframe] = useState(365);
  
  // Unique gradient ID to avoid conflicts
  const gradientId = `sheet-chart-gradient-${signal?.ticker || 'default'}`;
  
  // Load stock info when signal changes
  useEffect(() => {
    if (!signal || !isOpen) return;

    const ticker = signal.ticker;
    setStockInfo(null);
    setIsLoadingInfo(true);
    
    async function loadInfo() {
      try {
        const info = await getStockInfo(ticker);
        setStockInfo(info);
      } catch (err) {
        console.error('Failed to load stock info:', err);
      } finally {
        setIsLoadingInfo(false);
      }
    }

    loadInfo();
  }, [signal, isOpen]);

  // Load chart data when signal or timeframe changes
  useEffect(() => {
    if (!signal || !isOpen) return;

    const ticker = signal.ticker;
    setChartData([]);
    
    async function loadChart() {
      setIsLoadingChart(true);
      
      try {
        // -1 means "All" - use max available (1825 days = 5 years)
        const days = chartTimeframe === -1 ? 1825 : chartTimeframe;
        const chart = await getStockChart(ticker, days);
        setChartData(chart || []);
      } catch (err) {
        console.error('Failed to load chart data:', err);
        setChartData([]);
      } finally {
        setIsLoadingChart(false);
      }
    }

    loadChart();
  }, [signal, isOpen, chartTimeframe]);

  if (!signal) return null;

  // Calculate if stock is up or down over the chart period
  const isStockUp = chartData.length >= 2 && chartData[chartData.length - 1].close >= chartData[0].close;
  
  // Theme-aware colors using CSS variables like StockDetailsPanel
  const chartColor = isStockUp ? 'var(--success)' : 'var(--danger)';

  // Get the reference high from the data (first point has ref_high value)
  const refHigh = chartData.length > 0 ? chartData[0]?.ref_high : null;
  const refHighDate = chartData.length > 0 ? chartData[0]?.ref_high_date : null;

  // Get current price (last point)
  const currentPrice = chartData.length > 0 ? chartData[chartData.length - 1]?.close : null;
  const currentPointIndex = chartData.length - 1;

  // Format chart data with display dates
  const formattedChartData = chartData.map(point => ({
    ...point,
    displayDate: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }));

  // Find ref high point index for marker
  const refHighIndex = refHighDate 
    ? formattedChartData.findIndex(p => p.date === refHighDate)
    : -1;

  // Create chart data with trendline connecting peak to current
  const chartDataWithTrendline = formattedChartData.map((point, index) => {
    let trendline: number | null = null;
    if (index === refHighIndex && refHighIndex >= 0) {
      trendline = point.close;
    } else if (index === formattedChartData.length - 1) {
      trendline = point.close;
    }
    return { ...point, trendline };
  });

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.05, delayChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 8 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.2 } }
  };

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent 
        side="right" 
        className="w-full sm:max-w-xl p-0 flex flex-col h-full overflow-hidden"
      >
        {/* Fixed Header */}
        <div className="shrink-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
          <SheetHeader className="p-6 pb-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <SheetTitle className="flex items-center gap-3 flex-wrap">
                  <span className="text-2xl font-bold">{signal.ticker}</span>
                  {getDipTypeBadge(signal.dip_class, colorblindMode)}
                </SheetTitle>
                <SheetDescription className="mt-1 truncate">
                  {stockInfo?.name || 'Loading company info...'}
                </SheetDescription>
              </div>
              {/* Score Badge */}
              <div className={`shrink-0 px-4 py-2 rounded-xl ${getScoreBadgeClass(signal.final_score, colorblindMode)}`}>
                <p className="text-3xl font-bold leading-none">{formatScore(signal.final_score)}</p>
                <p className="text-[10px] uppercase tracking-wide mt-1 opacity-80">Score</p>
              </div>
            </div>
          </SheetHeader>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto">
          <motion.div 
            className="p-6 space-y-6"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {/* Quick Stats Grid */}
            <TooltipProvider delayDuration={300}>
              <motion.div variants={itemVariants} className="grid grid-cols-3 gap-3">
                <div className="bg-muted/40 rounded-xl p-4 text-center relative">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="absolute top-2 right-2 text-muted-foreground/50 hover:text-muted-foreground cursor-help" tabIndex={-1}>
                        <HelpCircle className="h-3.5 w-3.5" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[200px]">
                      <p>How much the stock has dropped from its recent high within the lookback window.</p>
                    </TooltipContent>
                  </Tooltip>
                  <TrendingDown className={`h-5 w-5 mx-auto mb-2 ${colorblindMode ? 'text-orange-500' : 'text-danger'}`} />
                  <p className={`text-xl font-bold ${colorblindMode ? 'text-orange-500' : 'text-danger'}`}>
                    {formatPercent(-signal.dip_stock)}
                  </p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Dip</p>
                </div>
                <div className="bg-muted/40 rounded-xl p-4 text-center relative">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="absolute top-2 right-2 text-muted-foreground/50 hover:text-muted-foreground cursor-help" tabIndex={-1}>
                        <HelpCircle className="h-3.5 w-3.5" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[200px]">
                      <p>The stock's dip minus the market's dip. Higher means the stock fell more than the market.</p>
                    </TooltipContent>
                  </Tooltip>
                  <Target className={`h-5 w-5 mx-auto mb-2 ${signal.excess_dip > 0 ? (colorblindMode ? 'text-blue-500' : 'text-success') : 'text-muted-foreground'}`} />
                  <p className={`text-xl font-bold ${signal.excess_dip > 0 ? (colorblindMode ? 'text-blue-500' : 'text-success') : 'text-muted-foreground'}`}>
                    {formatPercent(signal.excess_dip)}
                  </p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Excess</p>
                </div>
                <div className="bg-muted/40 rounded-xl p-4 text-center relative">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="absolute top-2 right-2 text-muted-foreground/50 hover:text-muted-foreground cursor-help" tabIndex={-1}>
                        <HelpCircle className="h-3.5 w-3.5" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[200px]">
                      <p>Number of consecutive days the stock has been in dip territory (&gt;10% below peak).</p>
                    </TooltipContent>
                  </Tooltip>
                  <Clock className="h-5 w-5 mx-auto mb-2 text-muted-foreground" />
                  <p className="text-xl font-bold">
                    {signal.persist_days > 0 ? signal.persist_days : '—'}
                  </p>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide mt-1">Days</p>
                </div>
              </motion.div>
            </TooltipProvider>

            {/* Price Chart Section */}
            <motion.div variants={itemVariants} className="bg-muted/20 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  <h3 className="text-sm font-semibold">Price History</h3>
                </div>
                {/* Timeframe Picker */}
                <div className="flex gap-1">
                  {CHART_TIMEFRAMES.map((tf) => (
                    <Button
                      key={tf.value}
                      variant={chartTimeframe === tf.value ? 'secondary' : 'ghost'}
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => setChartTimeframe(tf.value)}
                    >
                      {tf.label}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="h-56 w-full">
                {isLoadingChart ? (
                  <div className="h-full w-full flex flex-col gap-2">
                    <Skeleton className="flex-1 w-full rounded-lg" />
                    <div className="flex justify-between">
                      <Skeleton className="h-3 w-12" />
                      <Skeleton className="h-3 w-12" />
                      <Skeleton className="h-3 w-12" />
                    </div>
                  </div>
                ) : formattedChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart 
                      data={chartDataWithTrendline} 
                      margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={chartColor} stopOpacity={0.3}>
                            <animate attributeName="stop-color" to={chartColor} dur="0.3s" fill="freeze" />
                          </stop>
                          <stop offset="100%" stopColor={chartColor} stopOpacity={0}>
                            <animate attributeName="stop-color" to={chartColor} dur="0.3s" fill="freeze" />
                          </stop>
                        </linearGradient>
                      </defs>
                      <XAxis
                        dataKey="displayDate"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                        minTickGap={60}
                        dy={8}
                      />
                      <YAxis
                        domain={['dataMin', 'dataMax']}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                        tickFormatter={(v) => `$${v.toFixed(0)}`}
                        width={50}
                        dx={-5}
                      />
                      {/* Current price reference line (red dashed) */}
                      {currentPrice && (
                        <ReferenceLine
                          y={currentPrice}
                          stroke="var(--danger)"
                          strokeDasharray="5 5"
                          strokeOpacity={0.7}
                        />
                      )}
                      {/* Peak price reference line (green dashed) */}
                      {refHigh && (
                        <ReferenceLine
                          y={refHigh}
                          stroke="var(--success)"
                          strokeDasharray="3 3"
                          strokeOpacity={0.5}
                        />
                      )}
                      {/* Trendline connecting peak to current */}
                      <Line
                        type="linear"
                        dataKey="trendline"
                        stroke="var(--muted-foreground)"
                        strokeWidth={1}
                        strokeDasharray="3 3"
                        strokeOpacity={0.4}
                        dot={false}
                        connectNulls={true}
                        isAnimationActive={false}
                      />
                      {/* Mark the current price point (red dot) */}
                      {currentPointIndex >= 0 && formattedChartData[currentPointIndex] && (
                        <ReferenceDot
                          x={formattedChartData[currentPointIndex].displayDate}
                          y={formattedChartData[currentPointIndex].close}
                          r={4}
                          fill="var(--danger)"
                          stroke="var(--background)"
                          strokeWidth={2}
                        />
                      )}
                      {/* Mark the ref high point (green dot) */}
                      {refHighIndex >= 0 && formattedChartData[refHighIndex] && (
                        <ReferenceDot
                          x={formattedChartData[refHighIndex].displayDate}
                          y={formattedChartData[refHighIndex].close}
                          r={4}
                          fill="var(--success)"
                          stroke="var(--background)"
                          strokeWidth={2}
                        />
                      )}
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: 'hsl(var(--background) / 0.95)',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px',
                          backdropFilter: 'blur(8px)',
                          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                        }}
                        labelStyle={{ color: 'hsl(var(--foreground))' }}
                        formatter={(value: number, name: string) => {
                          if (name === 'trendline') return null;
                          return [`$${value.toFixed(2)}`, 'Price'];
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="close"
                        stroke={chartColor}
                        fill={`url(#${gradientId})`}
                        strokeWidth={2}
                        animationDuration={800}
                        animationEasing="ease-out"
                        isAnimationActive={true}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                    <div className="text-center">
                      <Activity className="h-8 w-8 mx-auto mb-2 opacity-40" />
                      <p>No chart data available</p>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Scores Section */}
            <motion.div variants={itemVariants} className="space-y-4">
              <div className="flex items-center gap-2">
                <Gauge className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-semibold">Score Breakdown</h3>
              </div>
              
              <div className="grid gap-3">
                {/* Quality Score */}
                <div className="bg-muted/20 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`p-1.5 rounded-lg ${colorblindMode ? 'bg-blue-500/15' : 'bg-success/15'}`}>
                        <BarChart3 className={`h-4 w-4 ${colorblindMode ? 'text-blue-500' : 'text-success'}`} />
                      </div>
                      <span className="font-medium">Quality</span>
                    </div>
                    <span className="font-bold text-lg">{formatScore(signal.quality_score)}</span>
                  </div>
                  <div className="w-full bg-muted/50 rounded-full h-2.5 overflow-hidden">
                    <motion.div 
                      className={`h-full rounded-full ${getProgressColor(colorblindMode, 'quality')}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(signal.quality_score, 100)}%` }}
                      transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Profitability, balance sheet, cash flow & growth
                  </p>
                </div>

                {/* Stability Score */}
                <div className="bg-muted/20 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`p-1.5 rounded-lg ${colorblindMode ? 'bg-purple-500/15' : 'bg-chart-2/15'}`}>
                        <Activity className={`h-4 w-4 ${colorblindMode ? 'text-purple-500' : 'text-chart-2'}`} />
                      </div>
                      <span className="font-medium">Stability</span>
                    </div>
                    <span className="font-bold text-lg">{formatScore(signal.stability_score)}</span>
                  </div>
                  <div className="w-full bg-muted/50 rounded-full h-2.5 overflow-hidden">
                    <motion.div 
                      className={`h-full rounded-full ${getProgressColor(colorblindMode, 'stability')}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(signal.stability_score, 100)}%` }}
                      transition={{ duration: 0.6, ease: 'easeOut', delay: 0.3 }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Volatility, beta & historical drawdowns
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Dip Analysis */}
            <motion.div variants={itemVariants} className="space-y-4">
              <div className="flex items-center gap-2">
                <TrendingDown className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-semibold">Dip Analysis</h3>
              </div>
              
              <div className="bg-muted/20 rounded-xl divide-y divide-border/50">
                <div className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-1.5 rounded-lg ${colorblindMode ? 'bg-orange-500/15' : 'bg-danger/15'}`}>
                      <TrendingDown className={`h-4 w-4 ${colorblindMode ? 'text-orange-500' : 'text-danger'}`} />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Dip Magnitude</p>
                      <p className="text-xs text-muted-foreground">Drop from recent peak</p>
                    </div>
                  </div>
                  <span className={`font-bold text-lg ${colorblindMode ? 'text-orange-500' : 'text-danger'}`}>
                    {formatPercent(-signal.dip_stock)}
                  </span>
                </div>
                
                <div className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-1.5 rounded-lg ${signal.excess_dip > 0 ? (colorblindMode ? 'bg-blue-500/15' : 'bg-success/15') : 'bg-muted'}`}>
                      <Target className={`h-4 w-4 ${signal.excess_dip > 0 ? (colorblindMode ? 'text-blue-500' : 'text-success') : 'text-muted-foreground'}`} />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Excess vs Market</p>
                      <p className="text-xs text-muted-foreground">Outpacing market decline</p>
                    </div>
                  </div>
                  <span className={`font-bold text-lg ${signal.excess_dip > 0 ? (colorblindMode ? 'text-blue-500' : 'text-success') : 'text-muted-foreground'}`}>
                    {formatPercent(signal.excess_dip)}
                  </span>
                </div>
                
                <div className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-1.5 rounded-lg bg-muted">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Persistence</p>
                      <p className="text-xs text-muted-foreground">Time in dip territory</p>
                    </div>
                  </div>
                  <span className="font-bold text-lg">
                    {signal.persist_days > 0 ? `${signal.persist_days}d` : 'New'}
                  </span>
                </div>
              </div>
            </motion.div>

            {/* Company Info */}
            {isLoadingInfo ? (
              <motion.div variants={itemVariants} className="space-y-4">
                <div className="flex items-center gap-2">
                  <Building2 className="h-4 w-4 text-muted-foreground" />
                  <h3 className="text-sm font-semibold">Company Profile</h3>
                </div>
                <div className="space-y-3">
                  <Skeleton className="h-6 w-32" />
                  <Skeleton className="h-24 w-full" />
                  <div className="grid grid-cols-2 gap-3">
                    <Skeleton className="h-20 w-full rounded-xl" />
                    <Skeleton className="h-20 w-full rounded-xl" />
                    <Skeleton className="h-20 w-full rounded-xl" />
                    <Skeleton className="h-20 w-full rounded-xl" />
                  </div>
                </div>
              </motion.div>
            ) : stockInfo ? (
              <motion.div variants={itemVariants} className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Building2 className="h-4 w-4 text-muted-foreground" />
                    <h3 className="text-sm font-semibold">Company Profile</h3>
                  </div>
                  {stockInfo.website && (
                    <a 
                      href={stockInfo.website} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-xs text-primary hover:underline flex items-center gap-1"
                    >
                      Website <ExternalLink className="h-3 w-3" />
                    </a>
                  )}
                </div>
                
                {/* Sector & Industry */}
                <div className="flex flex-wrap gap-2">
                  {stockInfo.sector && (
                    <Badge variant="secondary" className="text-xs">
                      {stockInfo.sector}
                    </Badge>
                  )}
                  {stockInfo.industry && (
                    <Badge variant="outline" className="text-xs">
                      {stockInfo.industry}
                    </Badge>
                  )}
                </div>

                {/* Company Summary */}
                {(stockInfo.summary_ai || stockInfo.summary) && (
                  <div className="bg-muted/20 rounded-xl p-4">
                    <div className="flex items-start gap-2 mb-2">
                      <Info className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">About</h4>
                    </div>
                    <div className="max-h-32 overflow-y-auto scrollbar-thin">
                      <p className="text-sm text-muted-foreground leading-relaxed pr-2">
                        {stockInfo.summary_ai || stockInfo.summary}
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Valuation Metrics */}
                <div className="space-y-3">
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2">
                    <DollarSign className="h-3 w-3" />
                    Valuation
                  </h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Market Cap</p>
                      <p className="font-medium text-sm">{formatMarketCap(stockInfo.market_cap)}</p>
                    </div>
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">P/E Ratio</p>
                      <p className="font-medium text-sm">{stockInfo.pe_ratio?.toFixed(1) || '—'}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {stockInfo.pe_ratio ? (stockInfo.pe_ratio > 25 ? 'Above avg' : stockInfo.pe_ratio < 15 ? 'Below avg' : 'Average') : ''}
                      </p>
                    </div>
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Forward P/E</p>
                      <p className="font-medium text-sm">{stockInfo.forward_pe?.toFixed(1) || '—'}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {stockInfo.forward_pe && stockInfo.pe_ratio 
                          ? (stockInfo.forward_pe < stockInfo.pe_ratio ? 'Growth expected' : 'Slowing growth') 
                          : ''}
                      </p>
                    </div>
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Dividend Yield</p>
                      <p className="font-medium text-sm flex items-center gap-1">
                        {stockInfo.dividend_yield 
                          ? `${(stockInfo.dividend_yield * 100).toFixed(2)}%` 
                          : '—'}
                        {stockInfo.dividend_yield && stockInfo.dividend_yield > 0.03 && (
                          <Percent className="h-3 w-3 text-success" />
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Risk & Trading Metrics */}
                <div className="space-y-3">
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2">
                    <Activity className="h-3 w-3" />
                    Risk & Trading
                  </h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Beta</p>
                      <p className="font-medium text-sm">{stockInfo.beta?.toFixed(2) || '—'}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {stockInfo.beta 
                          ? (stockInfo.beta > 1.2 ? 'High volatility' : stockInfo.beta < 0.8 ? 'Low volatility' : 'Market-like') 
                          : ''}
                      </p>
                    </div>
                    <div className="bg-muted/20 rounded-xl p-4">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Avg Volume</p>
                      <p className="font-medium text-sm">
                        {stockInfo.avg_volume 
                          ? `${(stockInfo.avg_volume / 1e6).toFixed(1)}M` 
                          : '—'}
                      </p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        {stockInfo.avg_volume 
                          ? (stockInfo.avg_volume > 10e6 ? 'Very liquid' : stockInfo.avg_volume > 1e6 ? 'Good liquidity' : 'Lower liquidity') 
                          : ''}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Analyst Recommendation */}
                {stockInfo.recommendation && (
                  <div className="bg-muted/20 rounded-xl p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Analyst Consensus</p>
                        <p className="font-medium text-sm capitalize flex items-center gap-2">
                          {stockInfo.recommendation.toLowerCase().includes('buy') && (
                            <ThumbsUp className={`h-4 w-4 ${colorblindMode ? 'text-blue-500' : 'text-success'}`} />
                          )}
                          {stockInfo.recommendation.toLowerCase().includes('sell') && (
                            <ThumbsDown className={`h-4 w-4 ${colorblindMode ? 'text-orange-500' : 'text-danger'}`} />
                          )}
                          {stockInfo.recommendation.toLowerCase().includes('hold') && (
                            <Minus className="h-4 w-4 text-muted-foreground" />
                          )}
                          {stockInfo.recommendation.replace(/_/g, ' ')}
                        </p>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={
                          stockInfo.recommendation.toLowerCase().includes('buy') 
                            ? (colorblindMode ? 'border-blue-500/50 text-blue-500' : 'border-success/50 text-success')
                            : stockInfo.recommendation.toLowerCase().includes('sell')
                            ? (colorblindMode ? 'border-orange-500/50 text-orange-500' : 'border-danger/50 text-danger')
                            : 'border-muted-foreground/50'
                        }
                      >
                        {stockInfo.recommendation.toLowerCase().includes('strong') ? 'Strong' : 'Moderate'}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Based on aggregated analyst ratings. Not investment advice.
                    </p>
                  </div>
                )}
              </motion.div>
            ) : null}

            {/* What This Dip Means - Contextual Analysis */}
            <motion.div variants={itemVariants} className="space-y-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-semibold">What This Means</h3>
              </div>
              
              <div className="bg-gradient-to-br from-primary/5 to-transparent rounded-xl p-4 border border-primary/10">
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {signal.dip_class === 'STOCK_SPECIFIC' && (
                    <>
                      This stock is experiencing a <strong className="text-foreground">company-specific decline</strong> that 
                      significantly exceeds the broader market movement. This could indicate an overreaction to news, 
                      earnings, or sector rotation—potentially creating a buying opportunity if fundamentals remain strong.
                    </>
                  )}
                  {signal.dip_class === 'MIXED' && (
                    <>
                      This decline is <strong className="text-foreground">partly market-driven and partly stock-specific</strong>. 
                      The stock is falling faster than the market, suggesting some company-specific pressure. 
                      Recovery may depend on both market conditions and company catalysts.
                    </>
                  )}
                  {signal.dip_class === 'MARKET_DIP' && (
                    <>
                      This stock is <strong className="text-foreground">moving with the broader market</strong>. 
                      The decline is primarily driven by market-wide sentiment rather than company-specific issues. 
                      Quality companies in market dips often recover when sentiment improves.
                    </>
                  )}
                </p>
                
                {/* Score Interpretation */}
                <div className="mt-4 pt-4 border-t border-border/50">
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {signal.final_score >= 70 && (
                      <>
                        With a <strong className="text-foreground">score of {formatScore(signal.final_score)}</strong>, 
                        this ranks as a <strong className={colorblindMode ? 'text-blue-500' : 'text-success'}>strong opportunity</strong>—a 
                        significant dip in a quality, stable company.
                      </>
                    )}
                    {signal.final_score >= 50 && signal.final_score < 70 && (
                      <>
                        With a <strong className="text-foreground">score of {formatScore(signal.final_score)}</strong>, 
                        this is a <strong className="text-chart-4">solid opportunity</strong> worth investigating further.
                      </>
                    )}
                    {signal.final_score >= 30 && signal.final_score < 50 && (
                      <>
                        With a <strong className="text-foreground">score of {formatScore(signal.final_score)}</strong>, 
                        this is a <strong className="text-chart-2">moderate opportunity</strong>. Some concerns about 
                        quality or stability may be limiting the score.
                      </>
                    )}
                    {signal.final_score < 30 && (
                      <>
                        With a <strong className="text-foreground">score of {formatScore(signal.final_score)}</strong>, 
                        this ranks <strong className="text-muted-foreground">lower</strong>. The dip may not be significant enough, 
                        or there are quality/stability concerns worth investigating.
                      </>
                    )}
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Bottom padding for scroll */}
            <div className="h-4" />
          </motion.div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
