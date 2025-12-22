import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
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
  X,
  Activity,
  Percent,
  Target
} from 'lucide-react';
import { Button } from '@/components/ui/button';

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
  onClose,
  benchmark,
  comparisonData = [],
  isLoadingBenchmark = false,
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

  // Find dip start point index for marker
  const dipStartIndex = useMemo(() => {
    if (chartData.length === 0) return -1;
    const dipDate = chartData[0]?.dip_start_date;
    if (!dipDate) return -1;
    return formattedChartData.findIndex(p => p.date === dipDate);
  }, [chartData, formattedChartData]);

  // Find ref high point index for marker
  const refHighIndex = useMemo(() => {
    if (chartData.length === 0) return -1;
    const refDate = chartData[0]?.ref_high_date;
    if (!refDate) return -1;
    return formattedChartData.findIndex(p => p.date === refDate);
  }, [chartData, formattedChartData]);

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
        <CardHeader className="pb-2 shrink-0 border-b border-border/50">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <CardTitle className="text-2xl">{stock.symbol}</CardTitle>
                <Button variant="ghost" size="icon" className="h-6 w-6 md:hidden" onClick={onClose}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-muted-foreground truncate mt-0.5">
                {stock.name || stock.symbol}
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold font-mono">
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

          {/* Period Selector */}
          <div className="flex items-center justify-between mt-4">
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

        {/* Chart Section - Fixed */}
        <div className="shrink-0 px-4 md:px-6 py-3 border-b border-border/50">
          {/* Dip Summary Banner */}
          {stock && (
            <div className="flex items-center gap-3 py-1.5 px-2.5 rounded-md border border-border/50 bg-muted/30 mb-3 text-xs">
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
          <div className="h-52 lg:h-60">
            {isLoadingChart ? (
              <Skeleton className="h-full w-full" />
            ) : formattedChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart 
                  key={`${stock.symbol}-${chartPeriod}`}
                  data={formattedChartData}
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
                  {/* Dip threshold line */}
                  {chartData[0]?.threshold && (
                    <ReferenceLine
                      y={chartData[0].threshold}
                      stroke="var(--danger)"
                      strokeDasharray="5 5"
                      strokeOpacity={0.7}
                      label={{ 
                        value: 'Dip Threshold', 
                        position: 'right', 
                        fontSize: 10, 
                        fill: 'var(--danger)',
                        opacity: 0.7
                      }}
                    />
                  )}
                  {/* 52-week high reference line */}
                  {chartData[0]?.ref_high && (
                    <ReferenceLine
                      y={chartData[0].ref_high}
                      stroke="var(--success)"
                      strokeDasharray="3 3"
                      strokeOpacity={0.5}
                    />
                  )}
                  {/* Mark the dip start point */}
                  {dipStartIndex >= 0 && formattedChartData[dipStartIndex] && (
                    <ReferenceDot
                      x={formattedChartData[dipStartIndex].displayDate}
                      y={formattedChartData[dipStartIndex].close}
                      r={6}
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
                      r={5}
                      fill="var(--success)"
                      stroke="var(--background)"
                      strokeWidth={2}
                    />
                  )}
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--background)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px',
                      fontSize: '12px',
                      padding: '8px 12px',
                    }}
                    formatter={(value: number, _name: string, props) => {
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
                    type="monotone"
                    dataKey="close"
                    stroke={chartColor}
                    strokeWidth={2}
                    fill={`url(#gradient-${stock.symbol}-${chartPeriod})`}
                    isAnimationActive={true}
                    animationDuration={800}
                    animationEasing="ease-out"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                No chart data available
              </div>
            )}
          </div>

          {/* Comparison Chart (when benchmark is selected) */}
          {benchmark && (
            <div className="mt-4">
              <ComparisonChart
                data={comparisonData}
                stockSymbol={stock.symbol}
                benchmark={benchmark}
                isLoading={isLoadingBenchmark}
              />
            </div>
          )}
        </div>

        {/* Data Section - Scrollable */}
        <ScrollArea className="flex-1 min-h-0">
          <CardContent className="pt-4 px-4 md:px-6">
            {/* Key Stats */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
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
                <Separator className="my-4" />
                
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
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

                  {stockInfo.summary && (
                    <div className="mt-2 p-3 bg-muted/30 rounded-lg">
                      <p className="text-xs font-medium text-muted-foreground mb-1">About the Company</p>
                      <p className="text-sm leading-relaxed">
                        {stockInfo.summary}
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
