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
  X
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

  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [chartData]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'hsl(var(--success))' : 'hsl(var(--danger))';

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
        {/* Header */}
        <CardHeader className="pb-2 shrink-0">
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

        <ScrollArea className="flex-1">
          <CardContent className="pt-0">
            {/* Chart */}
            <div className="h-48 mt-2">
              {isLoadingChart ? (
                <Skeleton className="h-full w-full" />
              ) : formattedChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={formattedChartData}>
                    <defs>
                      <linearGradient id={`gradient-detail`} x1="0" y1="0" x2="0" y2="1">
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
                    {chartData[0]?.threshold && (
                      <ReferenceLine
                        y={chartData[0].threshold}
                        stroke="hsl(var(--muted-foreground))"
                        strokeDasharray="3 3"
                        strokeOpacity={0.5}
                      />
                    )}
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--background))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                      formatter={(value) => [`$${Number(value).toFixed(2)}`, 'Price']}
                    />
                    <Area
                      type="monotone"
                      dataKey="close"
                      stroke={chartColor}
                      strokeWidth={2}
                      fill={`url(#gradient-detail)`}
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

            <Separator className="my-4" />

            {/* Key Stats */}
            <div className="grid grid-cols-2 gap-3">
              <StatItem
                icon={TrendingDown}
                label="Dip Depth"
                value={`${(stock.depth * 100).toFixed(2)}%`}
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

            {/* Stock Info */}
            {stockInfo && (
              <>
                <Separator className="my-4" />
                
                <div className="space-y-3">
                  {stockInfo.sector && (
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{stockInfo.sector}</Badge>
                      {stockInfo.industry && (
                        <span className="text-sm text-muted-foreground">{stockInfo.industry}</span>
                      )}
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

                  {stockInfo.summary && (
                    <div className="mt-3">
                      <p className="text-sm text-muted-foreground leading-relaxed line-clamp-4">
                        {stockInfo.summary}
                      </p>
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
