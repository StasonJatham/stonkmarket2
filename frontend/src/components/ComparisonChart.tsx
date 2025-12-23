import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
} from 'recharts';
import { CHART_LINE_ANIMATION } from '@/lib/chartConfig';
import type { ComparisonChartData, BenchmarkType } from '@/services/api';
import { getBenchmarkName } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  ChartContainer, 
  ChartTooltip, 
  ChartTooltipContent,
  type ChartConfig 
} from '@/components/ui/chart';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';

interface ComparisonChartProps {
  data: ComparisonChartData[];
  stockSymbol: string;
  stockName?: string | null;
  benchmark: BenchmarkType;
  isLoading?: boolean;
  height?: number | string;
  compact?: boolean; // Render without Card wrapper
}

export function ComparisonChart({
  data,
  stockSymbol,
  stockName,
  benchmark,
  isLoading,
  height = 200,
  compact = false,
}: ComparisonChartProps) {
  const { colorblindMode, customColors } = useTheme();
  
  // Get colors from theme
  const stockColor = colorblindMode ? '#3b82f6' : customColors.up;
  const benchmarkColor = '#888888'; // Neutral gray for benchmark

  const chartConfig: ChartConfig = useMemo(() => ({
    stockNormalized: {
      label: stockSymbol,
      color: stockColor,
    },
    benchmarkNormalized: {
      label: getBenchmarkName(benchmark) || 'Benchmark',
      color: benchmarkColor,
    },
  }), [stockSymbol, benchmark, stockColor, benchmarkColor]);

  const { stockPerformance, benchmarkPerformance, outperformance } = useMemo(() => {
    if (data.length < 2) return { stockPerformance: 0, benchmarkPerformance: 0, outperformance: 0 };
    
    const lastPoint = data[data.length - 1];
    const stockPerf = lastPoint.stockNormalized;
    const benchmarkPerf = lastPoint.benchmarkNormalized ?? 0;
    
    return {
      stockPerformance: stockPerf,
      benchmarkPerformance: benchmarkPerf,
      outperformance: stockPerf - benchmarkPerf,
    };
  }, [data]);

  if (isLoading) {
    if (compact) {
      return <Skeleton className="h-full w-full rounded-lg" />;
    }
    return (
      <Card>
        <CardHeader className="pb-2">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-32 mt-1" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[200px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (data.length === 0) {
    if (compact) {
      return (
        <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
          Loading comparison...
        </div>
      );
    }
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-[200px] text-muted-foreground">
          No chart data available
        </CardContent>
      </Card>
    );
  }

  const hasBenchmark = benchmark && data.some(d => d.benchmarkNormalized !== undefined);

  // Compact mode: render without Card wrapper
  if (compact) {
    return (
      <div className="flex flex-col h-full">
        {/* Compact header */}
        <div className="flex items-center justify-between text-sm shrink-0 pb-2">
          <span className="text-muted-foreground">
            {stockSymbol} vs {getBenchmarkName(benchmark)}
          </span>
          {hasBenchmark && (
            <div className="flex items-center gap-2">
              <span className={stockPerformance >= 0 ? 'text-success' : 'text-danger'}>
                {stockPerformance >= 0 ? '+' : ''}{stockPerformance.toFixed(1)}%
              </span>
              <span className="text-muted-foreground">vs</span>
              <span className={benchmarkPerformance >= 0 ? 'text-success' : 'text-danger'}>
                {benchmarkPerformance >= 0 ? '+' : ''}{benchmarkPerformance.toFixed(1)}%
              </span>
              <Badge 
                variant={outperformance >= 0 ? 'default' : 'destructive'}
                className="text-xs px-1.5 py-0"
              >
                {outperformance >= 0 ? '+' : ''}{outperformance.toFixed(1)}%
              </Badge>
            </div>
          )}
        </div>
        {/* Chart */}
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%" minWidth={0} debounce={50}>
            <LineChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="displayDate"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 9 }}
              tickMargin={4}
              minTickGap={50}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 9 }}
              tickMargin={4}
              tickFormatter={(value) => `${value.toFixed(0)}%`}
              width={35}
            />
            <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" strokeOpacity={0.3} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--background) / 0.95)',
                backdropFilter: 'blur(8px)',
                border: '1px solid hsl(var(--border))',
                borderRadius: '6px',
                fontSize: '11px',
                padding: '6px 10px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
              }}
              labelStyle={{
                color: 'hsl(var(--foreground))',
              }}
              itemStyle={{
                color: 'hsl(var(--foreground))',
              }}
              formatter={(value, name) => {
                if (value === undefined) return ['', ''];
                const label = name === 'stockNormalized' ? stockSymbol : getBenchmarkName(benchmark);
                return [`${Number(value).toFixed(2)}%`, label];
              }}
              labelFormatter={(label) => label}
            />
            <Line
              type="monotone"
              dataKey="stockNormalized"
              stroke={stockColor}
              strokeWidth={2}
              dot={false}
              {...CHART_LINE_ANIMATION}
            />
            {hasBenchmark && (
              <Line
                type="monotone"
                dataKey="benchmarkNormalized"
                stroke={benchmarkColor}
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
                {...CHART_LINE_ANIMATION}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
        </div>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              {stockSymbol}
              {hasBenchmark && (
                <span className="text-muted-foreground font-normal">
                  vs {getBenchmarkName(benchmark)}
                </span>
              )}
            </CardTitle>
            {stockName && (
              <CardDescription>{stockName}</CardDescription>
            )}
          </div>
          
          {hasBenchmark && (
            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Stock</p>
                <p className={`text-sm font-semibold ${stockPerformance >= 0 ? 'text-success' : 'text-danger'}`}>
                  {stockPerformance >= 0 ? '+' : ''}{stockPerformance.toFixed(2)}%
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground">{getBenchmarkName(benchmark)}</p>
                <p className={`text-sm font-semibold ${benchmarkPerformance >= 0 ? 'text-success' : 'text-danger'}`}>
                  {benchmarkPerformance >= 0 ? '+' : ''}{benchmarkPerformance.toFixed(2)}%
                </p>
              </div>
              <Badge 
                variant={outperformance >= 0 ? 'default' : 'destructive'}
                className="flex items-center gap-1"
              >
                {outperformance >= 0 ? (
                  <TrendingUp className="h-3 w-3" />
                ) : (
                  <TrendingDown className="h-3 w-3" />
                )}
                {outperformance >= 0 ? '+' : ''}{outperformance.toFixed(2)}%
              </Badge>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className={`h-[${height}px] w-full`}>
          <ResponsiveContainer width="100%" height={height as number} minWidth={0} debounce={50}>
            <LineChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="stockGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={stockColor} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={stockColor} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={benchmarkColor} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={benchmarkColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="hsl(var(--border))" 
                vertical={false}
              />
              
              <XAxis
                dataKey="displayDate"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10 }}
                tickMargin={8}
                minTickGap={40}
              />
              
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10 }}
                tickMargin={8}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
                width={45}
              />
              
              <ReferenceLine 
                y={0} 
                stroke="hsl(var(--muted-foreground))" 
                strokeDasharray="3 3"
                strokeOpacity={0.5}
              />
              
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    formatter={(value, name) => {
                      const label = name === 'stockNormalized' ? stockSymbol : getBenchmarkName(benchmark);
                      return [`${Number(value).toFixed(2)}%`, label];
                    }}
                  />
                }
              />
              
              {hasBenchmark && (
                <Legend 
                  verticalAlign="top" 
                  height={36}
                  formatter={(value) => (
                    <span className="text-xs text-muted-foreground">
                      {value === 'stockNormalized' ? stockSymbol : getBenchmarkName(benchmark)}
                    </span>
                  )}
                />
              )}
              
              <Line
                type="monotone"
                dataKey="stockNormalized"
                stroke={stockColor}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                {...CHART_LINE_ANIMATION}
              />
              
              {hasBenchmark && (
                <Line
                  type="monotone"
                  dataKey="benchmarkNormalized"
                  stroke={benchmarkColor}
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                  {...CHART_LINE_ANIMATION}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
