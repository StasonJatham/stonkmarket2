import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { AggregatedPerformance, BenchmarkType } from '@/services/api';
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
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';

interface PortfolioChartProps {
  data: AggregatedPerformance[];
  benchmark: BenchmarkType;
  stockCount: number;
  isLoading?: boolean;
  height?: number;
}

export function PortfolioChart({
  data,
  benchmark,
  stockCount,
  isLoading,
  height = 350,
}: PortfolioChartProps) {
  const chartConfig: ChartConfig = useMemo(() => ({
    portfolioAvg: {
      label: 'Dip Portfolio',
      color: 'hsl(var(--chart-1))',
    },
    portfolioMin: {
      label: 'Min',
      color: 'hsl(var(--chart-3))',
    },
    portfolioMax: {
      label: 'Max',
      color: 'hsl(var(--chart-3))',
    },
    benchmarkValue: {
      label: getBenchmarkName(benchmark) || 'Benchmark',
      color: 'hsl(var(--chart-2))',
    },
  }), [benchmark]);

  const { portfolioPerformance, benchmarkPerformance, outperformance } = useMemo(() => {
    if (data.length < 2) return { portfolioPerformance: 0, benchmarkPerformance: 0, outperformance: 0 };
    
    const lastPoint = data[data.length - 1];
    const portfolioPerf = lastPoint.portfolioAvg;
    const benchmarkPerf = lastPoint.benchmarkValue ?? 0;
    
    return {
      portfolioPerformance: portfolioPerf,
      benchmarkPerformance: benchmarkPerf,
      outperformance: portfolioPerf - benchmarkPerf,
    };
  }, [data]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <Skeleton className="h-6 w-64" />
          <Skeleton className="h-4 w-48 mt-1" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[350px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (data.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center h-[350px] text-muted-foreground">
          <BarChart3 className="h-12 w-12 mb-4 opacity-20" />
          <p>No aggregated data available</p>
        </CardContent>
      </Card>
    );
  }

  const hasBenchmark = benchmark && data.some(d => d.benchmarkValue !== undefined);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="text-xl flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-muted-foreground" />
              Dip Portfolio Performance
            </CardTitle>
            <CardDescription>
              Aggregated performance of {stockCount} stocks in dip
              {hasBenchmark && ` vs ${getBenchmarkName(benchmark)}`}
            </CardDescription>
          </div>
          
          <div className="flex items-center gap-3 flex-wrap">
            <div className="text-right">
              <p className="text-xs text-muted-foreground">Portfolio Avg</p>
              <p className={`text-lg font-bold ${portfolioPerformance >= 0 ? 'text-success' : 'text-danger'}`}>
                {portfolioPerformance >= 0 ? '+' : ''}{portfolioPerformance.toFixed(2)}%
              </p>
            </div>
            
            {hasBenchmark && (
              <>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground">{getBenchmarkName(benchmark)}</p>
                  <p className={`text-lg font-bold ${benchmarkPerformance >= 0 ? 'text-success' : 'text-danger'}`}>
                    {benchmarkPerformance >= 0 ? '+' : ''}{benchmarkPerformance.toFixed(2)}%
                  </p>
                </div>
                
                <Badge 
                  variant={outperformance >= 0 ? 'default' : 'destructive'}
                  className="flex items-center gap-1 text-sm py-1"
                >
                  {outperformance >= 0 ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <TrendingDown className="h-4 w-4" />
                  )}
                  {outperformance >= 0 ? 'Outperforming' : 'Underperforming'} by {Math.abs(outperformance).toFixed(2)}%
                </Badge>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className={`h-[${height}px] w-full`}>
          <ResponsiveContainer width="100%" height={height}>
            <AreaChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="rangeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--chart-3))" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="hsl(var(--chart-3))" stopOpacity={0.02} />
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
                tick={{ fontSize: 11 }}
                tickMargin={8}
                minTickGap={50}
              />
              
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11 }}
                tickMargin={8}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
                width={50}
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
                      let label = '';
                      if (name === 'portfolioAvg') label = 'Portfolio Avg';
                      else if (name === 'portfolioMin') label = 'Min';
                      else if (name === 'portfolioMax') label = 'Max';
                      else if (name === 'benchmarkValue') label = getBenchmarkName(benchmark) || 'Benchmark';
                      return [`${Number(value).toFixed(2)}%`, label];
                    }}
                  />
                }
              />
              
              <Legend 
                verticalAlign="top" 
                height={36}
                formatter={(value) => {
                  const labels: Record<string, string> = {
                    portfolioAvg: 'Portfolio Avg',
                    portfolioMin: 'Range (Min/Max)',
                    benchmarkValue: getBenchmarkName(benchmark) || 'Benchmark',
                  };
                  return <span className="text-xs text-muted-foreground">{labels[value] || value}</span>;
                }}
              />
              
              {/* Min/Max range area */}
              <Area
                type="monotone"
                dataKey="portfolioMax"
                stroke="none"
                fill="url(#rangeGradient)"
                fillOpacity={1}
              />
              <Area
                type="monotone"
                dataKey="portfolioMin"
                stroke="none"
                fill="hsl(var(--background))"
                fillOpacity={1}
              />
              
              {/* Portfolio average line with fill */}
              <Area
                type="monotone"
                dataKey="portfolioAvg"
                stroke="hsl(var(--chart-1))"
                strokeWidth={2.5}
                fill="url(#portfolioGradient)"
                fillOpacity={1}
              />
              
              {/* Benchmark line */}
              {hasBenchmark && (
                <Line
                  type="monotone"
                  dataKey="benchmarkValue"
                  stroke="hsl(var(--chart-2))"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
