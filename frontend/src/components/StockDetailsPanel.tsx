import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
  Line,
  ComposedChart,
} from 'recharts';
import type { DipStock, ChartDataPoint, StockInfo, BenchmarkType, ComparisonChartData } from '@/services/api';
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
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { StockLogo } from '@/components/StockLogo';

interface AiData {
  ai_rating: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell' | null;
  ai_reasoning: string | null;
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
  const formattedChartData = useMemo(() => {
    return chartData.map((point) => ({
      ...point,
      displayDate: new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
    }));
  }, [chartData]);

  // Find current/last point index for the red marker (showing current dip position)
  const currentPointIndex = useMemo(() => {
    if (formattedChartData.length === 0) return -1;
    return formattedChartData.length - 1; // Last point = current price
  }, [formattedChartData]);

  // Find ref high point index for marker
  const refHighIndex = useMemo(() => {
    if (chartData.length === 0) return -1;
    const refDate = chartData[0]?.ref_high_date;
    if (!refDate) return -1;
    return formattedChartData.findIndex(p => p.date === refDate);
  }, [chartData, formattedChartData]);

  // Create chart data with trendline connecting peak to current
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
      return { ...point, trendline };
    });
  }, [formattedChartData, refHighIndex]);

  // Get the current price for the reference line
  const currentPrice = useMemo(() => {
    if (formattedChartData.length === 0) return null;
    return formattedChartData[formattedChartData.length - 1]?.close;
  }, [formattedChartData]);

  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [chartData]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'var(--success)' : 'var(--danger)';

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
                  ${stock.last_price.toFixed(2)}
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

          {/* Period Selector */}
          <div className="flex items-center justify-between mt-2 md:mt-4">
            <div className="flex gap-1">
              {periods.map((p) => (
                <Button
                  key={p.days}
                  variant={chartPeriod === p.days ? 'default' : 'ghost'}
                  size="sm"
                  className="h-7 px-2.5 text-xs"
                  onClick={() => onPeriodChange(p.days)}
                >
                  {p.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>

        {/* Scrollable Content - Chart + Data */}
        <ScrollArea className="flex-1 min-h-0">
          {/* Chart Section */}
          <div className="px-3 md:px-6 py-2">
            {/* Dip Summary Banner */}
            {stock && (
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
                {chartData[0]?.since_dip !== null && chartData[0]?.since_dip !== undefined && (
                  <div className="flex items-center gap-1.5 ml-auto">
                    <Target className="h-3 w-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Recovery:</span>
                    <span className={`font-medium ${chartData[chartData.length - 1]?.since_dip && chartData[chartData.length - 1].since_dip! > 0 ? 'text-success' : 'text-danger'}`}>
                      {chartData[chartData.length - 1]?.since_dip ? (chartData[chartData.length - 1].since_dip! * 100).toFixed(1) : '0'}%
                    </span>
                  </div>
                )}
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
              ) : isLoadingChart ? (
                <Skeleton className="h-full w-full" />
              ) : formattedChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                  <ComposedChart 
                    key={`${stock.symbol}-${chartPeriod}`}
                    data={chartDataWithTrendline}
                  >
                    <defs>
                      <linearGradient id={`gradient-${stock.symbol}-${chartPeriod}`} x1="0" y1="0" x2="0" y2="1">
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
                    {/* Current price reference line (aligns with red dot) */}
                    {currentPrice && (
                      <ReferenceLine
                        y={currentPrice}
                        stroke="var(--danger)"
                        strokeDasharray="5 5"
                        strokeOpacity={0.7}
                      />
                    )}
                    {/* Peak price reference line (aligns with green dot) */}
                    {chartData[0]?.ref_high && (
                      <ReferenceLine
                        y={chartData[0].ref_high}
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
                    {/* Mark the current price point (red - shows where we are now in the dip) */}
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
                    {/* Mark the ref high point */}
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
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--background))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                        fontSize: '12px',
                        padding: '8px 12px',
                      }}
                      labelStyle={{
                        color: 'hsl(var(--foreground))',
                      }}
                      itemStyle={{
                        color: 'hsl(var(--foreground))',
                      }}
                      formatter={(value: number, name: string, props) => {
                        if (name === 'trendline') return null; // Hide trendline from tooltip
                        const point = props.payload;
                        const items: [string, string][] = [
                          [`$${value.toFixed(2)}`, 'Price']
                        ];
                        if (point.drawdown !== null) {
                          items.push([`${(point.drawdown * 100).toFixed(1)}%`, 'Drawdown']);
                        }
                        if (point.since_dip !== null) {
                          items.push([`${(point.since_dip * 100).toFixed(1)}%`, 'Since Dip']);
                        }
                        return items;
                      }}
                      labelFormatter={(label) => label}
                    />
                    <Area
                      type="linear"
                      dataKey="close"
                      stroke={chartColor}
                      strokeWidth={2}
                      fill={`url(#gradient-${stock.symbol}-${chartPeriod})`}
                      isAnimationActive={true}
                      animationDuration={800}
                      animationEasing="ease-out"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                  No chart data available
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
                value={formatMarketCap(stock.market_cap)}
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
                
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3 mb-3">
                  {stockInfo.forward_pe !== null && stockInfo.forward_pe !== undefined && (
                    <StatItem
                      icon={Activity}
                      label="Forward P/E"
                      value={stockInfo.forward_pe.toFixed(2)}
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

            {/* AI Analysis Section */}
            {(aiData?.ai_reasoning || isLoadingAi) && (
              <>
                <Separator className="my-4" />
                <div className="p-3 bg-gradient-to-br from-primary/5 to-primary/10 rounded-lg border border-primary/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium">AI Analysis</span>
                    {aiData?.ai_rating && (
                      <Badge 
                        variant={
                          aiData.ai_rating === 'strong_buy' || aiData.ai_rating === 'buy' 
                            ? 'default' 
                            : aiData.ai_rating === 'strong_sell' || aiData.ai_rating === 'sell'
                              ? 'destructive'
                              : 'secondary'
                        }
                        className="ml-auto"
                      >
                        {aiData.ai_rating.replace('_', ' ').toUpperCase()}
                      </Badge>
                    )}
                  </div>
                  {isLoadingAi ? (
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-full" />
                      <Skeleton className="h-3 w-4/5" />
                      <Skeleton className="h-3 w-3/4" />
                    </div>
                  ) : aiData?.ai_reasoning ? (
                    <p className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                      {aiData.ai_reasoning}
                    </p>
                  ) : null}
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
