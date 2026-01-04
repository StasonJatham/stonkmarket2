import { useState, useEffect, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Line,
  ComposedChart,
  ReferenceDot,
} from 'recharts';
import { ChartTooltip, SimpleChartTooltipContent } from '@/components/ui/chart';
import { CHART_LINE_ANIMATION, CHART_TRENDLINE_ANIMATION, CHART_ANIMATION } from '@/lib/chartConfig';
import { useStockDetail, useStockChart } from '@/features/market-data/api/queries';
import type { StockInfo, DipCard } from '@/features/market-data/api/schemas';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { 
  TrendingDown,
  TrendingUp,
  ArrowLeft,
  Building2,
  DollarSign,
  BarChart3,
  AlertTriangle,
  Calendar,
  ExternalLink,
  Activity,
  Percent,
  Target,
  Sparkles,
  Globe,
  LineChart,
  Scale,
  Users,
  ThumbsUp,
  ThumbsDown,
  MinusCircle,
} from 'lucide-react';
import { StockLogo } from '@/components/StockLogo';

// Format large numbers
function formatNumber(value: number | null | undefined, options?: { decimals?: number; prefix?: string; suffix?: string }): string {
  if (value === null || value === undefined) return '—';
  const { decimals = 2, prefix = '', suffix = '' } = options || {};
  
  if (Math.abs(value) >= 1e12) return `${prefix}${(value / 1e12).toFixed(decimals)}T${suffix}`;
  if (Math.abs(value) >= 1e9) return `${prefix}${(value / 1e9).toFixed(decimals)}B${suffix}`;
  if (Math.abs(value) >= 1e6) return `${prefix}${(value / 1e6).toFixed(decimals)}M${suffix}`;
  if (Math.abs(value) >= 1e3) return `${prefix}${(value / 1e3).toFixed(decimals)}K${suffix}`;
  return `${prefix}${value.toFixed(decimals)}${suffix}`;
}

function formatPrice(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return `$${value.toFixed(2)}`;
}

function formatPercent(value: number | null | undefined, asDecimal = true): string {
  if (value === null || value === undefined) return '—';
  const pct = asDecimal ? value * 100 : value;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
}

// Generate stock-specific structured data
function generateStockJsonLd(symbol: string, info: StockInfo | null, dipCard: DipCard | null) {
  const baseSchema = {
    '@context': 'https://schema.org',
    '@type': 'FinancialProduct',
    'name': info?.name || symbol,
    'description': `Stock analysis and dip tracking for ${info?.name || symbol} (${symbol}). View recovery potential, price charts, and AI-powered insights.`,
    'category': 'Stock',
    'provider': {
      '@type': 'Organization',
      'name': 'StonkMarket',
      'url': 'https://stonkmarket.de',
    },
  };

  if (dipCard) {
    return {
      ...baseSchema,
      'additionalProperty': [
        {
          '@type': 'PropertyValue',
          'name': 'Dip Percentage',
          'value': `${(dipCard.dip_pct * 100).toFixed(1)}%`,
        },
        {
          '@type': 'PropertyValue',
          'name': 'Days Below High',
          'value': dipCard.days_below?.toString() || 'N/A',
        },
      ],
    };
  }

  return baseSchema;
}

// Chart period options
const CHART_PERIODS = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: '2Y', days: 730 },
  { label: '5Y', days: 1825 },
];

// Stat item component
interface StatItemProps {
  icon: React.ElementType;
  label: string;
  value: string;
  valueColor?: string;
}

function StatItem({ icon: Icon, label, value, valueColor }: StatItemProps) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 hover:bg-muted/70 transition-colors">
      <div className="p-2 rounded-md bg-background">
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-sm font-semibold truncate ${valueColor || ''}`}>{value}</p>
      </div>
    </div>
  );
}

export function StockDetailPage() {
  const { symbol } = useParams<{ symbol: string }>();
  const upperSymbol = symbol?.toUpperCase() || '';
  const [chartPeriod, setChartPeriod] = useState(365);

  // Use TanStack Query for all data fetching
  const { 
    info, 
    chartData, 
    dipCard, 
    agentAnalysis,
    isLoading,
    isError,
  } = useStockDetail(upperSymbol, chartPeriod);

  // Separate chart query for period changes (will use cached data when period is in cache)
  const chartQuery = useStockChart(upperSymbol, chartPeriod);

  // Use chart data from the dedicated chart query when available (for period changes)
  const displayChartData = chartQuery.data ?? chartData;

  // Track when dots should be visible (after chart animation completes)
  const [dotsVisible, setDotsVisible] = useState(false);
  
  // Generate a key for the current chart state to trigger re-render of dots visibility
  const chartKey = `${chartPeriod}-${displayChartData.length}`;
  
  // When chart period or data changes, hide dots and show them after animation completes
  useEffect(() => {
    // Use a micro-task to avoid synchronous setState warning from React Compiler
    const rafId = requestAnimationFrame(() => {
      setDotsVisible(false);
    });
    const timer = setTimeout(() => {
      setDotsVisible(true);
    }, CHART_ANIMATION.animationDuration + 50);
    return () => {
      cancelAnimationFrame(rafId);
      clearTimeout(timer);
    };
  }, [chartKey]);

  // SEO for stock detail page
  useSEO({
    title: info ? `${info.name} (${upperSymbol}) - Stock Analysis` : `${upperSymbol} Stock Analysis`,
    description: info 
      ? `Track ${info.name} (${upperSymbol}) stock dips and recovery potential. Current sector: ${info.sector || 'N/A'}. View price charts and AI analysis.`
      : `Stock analysis and dip tracking for ${upperSymbol}. View recovery potential and price charts.`,
    keywords: `${upperSymbol}, ${info?.name || ''}, stock analysis, dip tracking, recovery potential, ${info?.sector || 'stocks'}`,
    canonical: `/stock/${upperSymbol.toLowerCase()}`,
    jsonLd: [
      generateBreadcrumbJsonLd([
        { name: 'Home', url: '/' },
        { name: 'Stocks', url: '/' },
        { name: upperSymbol, url: `/stock/${upperSymbol.toLowerCase()}` },
      ]),
      generateStockJsonLd(upperSymbol, info ?? null, dipCard ?? null),
    ],
  });

  // Format chart data with display dates
  const formattedChartData = useMemo(() => {
    return displayChartData.map((point) => ({
      ...point,
      displayDate: new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
    }));
  }, [displayChartData]);

  // Find ref high point index for marker
  const refHighIndex = useMemo(() => {
    if (displayChartData.length === 0) return -1;
    const refDate = displayChartData[0]?.ref_high_date;
    if (!refDate) return -1;
    return formattedChartData.findIndex(p => p.date === refDate);
  }, [displayChartData, formattedChartData]);

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

  // Get current price for reference line
  const currentPrice = useMemo(() => {
    if (formattedChartData.length === 0) return null;
    return formattedChartData[formattedChartData.length - 1]?.close;
  }, [formattedChartData]);

  // Get the ref high price
  const refHighPrice = useMemo(() => {
    if (displayChartData.length === 0) return null;
    return displayChartData[0]?.ref_high ?? null;
  }, [displayChartData]);

  // Create chart data with trendline, reference lines, and animated dot positions
  const chartDataWithTrendline = useMemo(() => {
    if (formattedChartData.length === 0) return [];
    
    return formattedChartData.map((point, index) => {
      let trendline: number | null = null;
      if (index === refHighIndex && refHighIndex >= 0) {
        trendline = point.close;
      } else if (index === formattedChartData.length - 1) {
        trendline = point.close;
      }
      
      return { 
        ...point, 
        trendline,
        // Horizontal reference lines for smooth animation
        currentPriceLine: currentPrice,
        refHighLine: refHighPrice,
        // Scatter point values - only non-null at specific indices for dot rendering
        refHighDot: index === refHighIndex ? point.close : null,
        currentDot: index === formattedChartData.length - 1 ? point.close : null,
      };
    });
  }, [formattedChartData, refHighIndex, currentPrice, refHighPrice]);

  // Calculate price change for chart period
  const priceChange = useMemo(() => {
    if (displayChartData.length < 2) return 0;
    const first = displayChartData[0].close;
    const last = displayChartData[displayChartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [displayChartData]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'var(--success)' : 'var(--danger)';

  // Handle chart period change - just update state, query handles fetching
  const handlePeriodChange = (days: number) => {
    setChartPeriod(days);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid gap-4 md:grid-cols-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
        <Skeleton className="h-[400px]" />
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <AlertTriangle className="h-16 w-16 text-muted-foreground" />
        <h1 className="text-2xl font-bold">Stock Not Found</h1>
        <p className="text-muted-foreground">Stock not found or data unavailable</p>
        <Button asChild>
          <Link to="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
      </div>
    );
  }

  const dipPct = dipCard?.dip_pct ? (dipCard.dip_pct * 100).toFixed(1) : null;
  const minDipPct = dipCard?.min_dip_pct ? (dipCard.min_dip_pct * 100).toFixed(1) : null;
  const currentPriceValue = dipCard?.current_price || displayChartData[displayChartData.length - 1]?.close;
  const refHigh = dipCard?.ref_high || displayChartData[0]?.ref_high;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Back button */}
      <Button variant="ghost" size="sm" asChild>
        <Link to="/">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Link>
      </Button>

      {/* Hero Section */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Stock Header */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <StockLogo symbol={upperSymbol} size="xl" className="shrink-0" />
                  <div>
                    <div className="flex items-center gap-3">
                      <CardTitle className="text-3xl font-bold">
                        {info?.name || upperSymbol}
                      </CardTitle>
                    </div>
                    <CardDescription className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="font-mono">{upperSymbol}</Badge>
                      {info?.sector && (
                        <Badge variant="secondary">
                          <Building2 className="mr-1 h-3 w-3" />
                          {info.sector}
                        </Badge>
                      )}
                      {info?.industry && (
                        <span className="text-sm text-muted-foreground">{info.industry}</span>
                      )}
                    </CardDescription>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-3xl font-bold font-mono">
                    {formatPrice(currentPriceValue)}
                  </div>
                  <div className={`flex items-center justify-end gap-1 text-sm ${isPositive ? 'text-success' : 'text-danger'}`}>
                    {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                    <span className="font-medium">{formatPercent(priceChange / 100)}</span>
                    <span className="text-muted-foreground">({CHART_PERIODS.find(p => p.days === chartPeriod)?.label})</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            
            {/* Dip Summary Banner */}
            {dipPct && (
              <CardContent className="pt-0">
                <div className="flex flex-wrap items-center gap-4 p-3 rounded-lg border border-danger/20 bg-danger/5">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-danger" />
                    <span className="text-sm text-muted-foreground">Current Dip:</span>
                    <span className="font-bold text-danger">-{dipPct}%</span>
                  </div>
                  {minDipPct && (
                    <div className="flex items-center gap-2">
                      <Target className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Max Dip:</span>
                      <span className="font-semibold">-{minDipPct}%</span>
                    </div>
                  )}
                  {dipCard?.days_below && (
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Days in Dip:</span>
                      <span className="font-semibold">{dipCard.days_below}</span>
                    </div>
                  )}
                  {refHigh && (
                    <div className="flex items-center gap-2 ml-auto">
                      <span className="text-sm text-muted-foreground">Ref High:</span>
                      <span className="font-semibold font-mono">{formatPrice(refHigh)}</span>
                    </div>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
        </div>

        {/* Quick Stats Card */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Quick Stats</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between items-center py-1">
              <span className="text-sm text-muted-foreground">Market Cap</span>
              <span className="font-semibold">{formatNumber(info?.market_cap, { prefix: '$' })}</span>
            </div>
            <Separator />
            <div className="flex justify-between items-center py-1">
              <span className="text-sm text-muted-foreground">P/E Ratio</span>
              <span className="font-semibold">{info?.pe_ratio?.toFixed(2) || '—'}</span>
            </div>
            <Separator />
            <div className="flex justify-between items-center py-1">
              <span className="text-sm text-muted-foreground">Forward P/E</span>
              <span className="font-semibold">{info?.forward_pe?.toFixed(2) || '—'}</span>
            </div>
            <Separator />
            <div className="flex justify-between items-center py-1">
              <span className="text-sm text-muted-foreground">Dividend Yield</span>
              <span className={`font-semibold ${info?.dividend_yield && info.dividend_yield > 0 ? 'text-success' : ''}`}>
                {info?.dividend_yield ? `${(info.dividend_yield * 100).toFixed(2)}%` : '—'}
              </span>
            </div>
            <Separator />
            <div className="flex justify-between items-center py-1">
              <span className="text-sm text-muted-foreground">Beta</span>
              <span className="font-semibold">{info?.beta?.toFixed(2) || '—'}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Price Chart */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <CardTitle className="flex items-center gap-2">
              <LineChart className="h-5 w-5" />
              Price Chart
            </CardTitle>
            <div className="flex gap-1">
              {CHART_PERIODS.map((p) => (
                <Button
                  key={p.days}
                  variant={chartPeriod === p.days ? 'default' : 'ghost'}
                  size="sm"
                  className="h-7 px-2.5 text-xs"
                  onClick={() => handlePeriodChange(p.days)}
                >
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
          <CardDescription>
            Price history with dip reference lines
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80 md:h-96">
            {chartDataWithTrendline.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartDataWithTrendline}>
                  <defs>
                    <linearGradient id={`gradient-${upperSymbol}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="displayDate"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 11 }}
                    tickMargin={8}
                    minTickGap={50}
                  />
                  <YAxis
                    domain={['dataMin', 'dataMax']}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 11 }}
                    tickMargin={8}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                    width={50}
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
                    type="monotone"
                    dataKey="close"
                    stroke={chartColor}
                    strokeWidth={2}
                    fill={`url(#gradient-${upperSymbol})`}
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
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                No chart data available
              </div>
            )}
          </div>
          
          {/* Chart Legend */}
          <div className="flex flex-wrap items-center gap-4 mt-4 pt-4 border-t text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-success" />
              <span>Reference High</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-danger" />
              <span>Current Price</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-0 border-t-2 border-dashed border-muted-foreground" />
              <span>Dip Trendline</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fundamental Data Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Valuation Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Scale className="h-4 w-4" />
              Valuation
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            <StatItem 
              icon={DollarSign} 
              label="Market Cap" 
              value={formatNumber(info?.market_cap, { prefix: '$' })} 
            />
            <StatItem 
              icon={BarChart3} 
              label="P/E Ratio (TTM)" 
              value={info?.pe_ratio?.toFixed(2) || '—'} 
            />
            <StatItem 
              icon={BarChart3} 
              label="Forward P/E" 
              value={info?.forward_pe?.toFixed(2) || '—'} 
            />
          </CardContent>
        </Card>

        {/* Trading Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Trading
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            <StatItem 
              icon={BarChart3} 
              label="Avg Volume" 
              value={formatNumber(info?.avg_volume)} 
            />
            <StatItem 
              icon={Activity} 
              label="Beta" 
              value={info?.beta?.toFixed(2) || '—'} 
            />
            <StatItem 
              icon={Percent} 
              label="Dividend Yield" 
              value={info?.dividend_yield ? `${(info.dividend_yield * 100).toFixed(2)}%` : '—'} 
              valueColor={info?.dividend_yield && info.dividend_yield > 0 ? 'text-success' : undefined}
            />
          </CardContent>
        </Card>

        {/* Price Range */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              52-Week Range
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            <StatItem 
              icon={TrendingUp} 
              label="52W High" 
              value={formatPrice(dipCard?.ref_high)} 
              valueColor="text-success"
            />
            <StatItem 
              icon={TrendingDown} 
              label="Current Price" 
              value={formatPrice(currentPriceValue)} 
            />
            {dipPct && (
              <StatItem 
                icon={Target} 
                label="Distance from High" 
                value={`-${dipPct}%`} 
                valueColor="text-danger"
              />
            )}
          </CardContent>
        </Card>
      </div>

      {/* AI Analysis & Company Info */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* AI Analysis */}
        {dipCard?.ai_reasoning && (
          <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
                {dipCard.ai_reasoning}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Company Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Building2 className="h-5 w-5" />
              About the Company
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {info?.sector && (
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{info.sector}</Badge>
                {info.industry && (
                  <span className="text-sm text-muted-foreground">{info.industry}</span>
                )}
              </div>
            )}
            
            {(info?.summary_ai || info?.summary) && (
              <p className="text-sm leading-relaxed text-muted-foreground">
                {info.summary_ai || info.summary}
              </p>
            )}

            {info?.recommendation && (
              <div className="flex items-center gap-2 pt-2">
                <span className="text-sm text-muted-foreground">Analyst Rating:</span>
                <Badge variant={
                  info.recommendation.includes('buy') ? 'default' :
                  info.recommendation.includes('sell') ? 'destructive' : 'secondary'
                }>
                  {info.recommendation.replace('_', ' ').toUpperCase()}
                </Badge>
              </div>
            )}

            {info?.website && (
              <a
                href={info.website}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
              >
                <Globe className="h-4 w-4" />
                Visit website
                <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </CardContent>
        </Card>
      </div>

      {/* AI Agent Personas Analysis */}
      {agentAnalysis && agentAnalysis.verdicts.length > 0 && (
        <Card className="bg-gradient-to-br from-violet-500/5 to-purple-500/10 border-violet-500/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5 text-violet-500" />
              Investment Persona Analyses
              <Badge 
                variant={
                  agentAnalysis.overall_signal === 'bullish' ? 'default' :
                  agentAnalysis.overall_signal === 'bearish' ? 'destructive' : 'secondary'
                }
                className="ml-auto"
              >
                {agentAnalysis.bullish_count} Bullish • {agentAnalysis.bearish_count} Bearish • {agentAnalysis.neutral_count} Neutral
              </Badge>
            </CardTitle>
            <CardDescription>
              What famous investors might think about this stock
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {agentAnalysis.verdicts.map((verdict) => (
                <div 
                  key={verdict.agent_name}
                  className="p-4 rounded-lg bg-background/50 border"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-semibold">{verdict.agent_name}</h4>
                      <p className="text-xs text-muted-foreground">
                        Confidence: {verdict.confidence}%
                      </p>
                    </div>
                    <Badge 
                      variant={
                        verdict.signal === 'bullish' ? 'default' :
                        verdict.signal === 'bearish' ? 'destructive' : 'secondary'
                      }
                      className="flex items-center gap-1"
                    >
                      {verdict.signal === 'bullish' && <ThumbsUp className="h-3 w-3" />}
                      {verdict.signal === 'bearish' && <ThumbsDown className="h-3 w-3" />}
                      {verdict.signal === 'neutral' && <MinusCircle className="h-3 w-3" />}
                      {verdict.signal.toUpperCase()}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {verdict.reasoning}
                  </p>
                  {verdict.key_factors.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {verdict.key_factors.slice(0, 3).map((factor, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {factor}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Disclaimer */}
      <Card className="bg-muted/50 border-dashed">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
            <div className="text-sm text-muted-foreground">
              <strong className="text-foreground">Disclaimer:</strong> This is not financial advice. 
              All information is for educational purposes only. Past performance does not guarantee 
              future results. Always do your own research and consult a financial advisor before 
              making investment decisions.
            </div>
          </div>
        </CardContent>
      </Card>

      {/* External Links */}
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" asChild>
          <a 
            href={`https://finance.yahoo.com/quote/${upperSymbol}`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Yahoo Finance
            <ExternalLink className="ml-2 h-3 w-3" />
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a 
            href={`https://www.google.com/finance/quote/${upperSymbol}:NASDAQ`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Google Finance
            <ExternalLink className="ml-2 h-3 w-3" />
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a 
            href={`https://www.tradingview.com/symbols/${upperSymbol}`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            TradingView
            <ExternalLink className="ml-2 h-3 w-3" />
          </a>
        </Button>
      </div>
    </motion.div>
  );
}
