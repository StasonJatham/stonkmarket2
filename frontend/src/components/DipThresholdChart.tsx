/**
 * DipThresholdChart - Visualize the relationship between dip thresholds and expected returns
 * 
 * Shows a multi-axis chart with:
 * - X-axis: Dip threshold (e.g., -5%, -10%, -15%, ...)
 * - Y-axis left: Avg return (%) and Total profit (%)
 * - Y-axis right: Number of occurrences
 * - Reference lines for optimal thresholds (risk-adjusted and max profit)
 * 
 * Helps users understand the trade-off:
 * - Shallow dips: More frequent but lower returns
 * - Deep dips: Higher returns but fewer opportunities
 */

import { useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import type { DipThresholdStats } from '@/services/api';

interface DipThresholdChartProps {
  thresholdStats: DipThresholdStats[];
  optimalThreshold: number;  // Risk-adjusted optimal
  maxProfitThreshold: number;  // Max profit optimal
  currentDrawdown?: number;
  height?: number;
}

export function DipThresholdChart({
  thresholdStats,
  optimalThreshold,
  maxProfitThreshold,
  currentDrawdown,
  height = 300,
}: DipThresholdChartProps) {
  // Transform data for the chart
  // Filter to reasonable thresholds (>= -50%) and with at least 2 occurrences
  const chartData = useMemo(() => {
    return thresholdStats
      .filter(s => s.threshold >= -50 && s.occurrences >= 2)
      .map(s => ({
        threshold: s.threshold,
        thresholdLabel: `${s.threshold}%`,
        avgReturn: s.avg_return,
        totalProfit: s.total_profit,
        totalProfitCompounded: s.total_profit_compounded,
        totalProfitAtRecovery: s.total_profit_at_recovery,
        avgDaysAtRecovery: s.avg_days_at_recovery,
        occurrences: s.occurrences,
        recoveryDays: s.avg_days_to_threshold,
        velocity: s.avg_recovery_velocity,
        winRate: s.win_rate * 100,
        mae: Math.abs(s.avg_further_drawdown),
      }))
      .sort((a, b) => b.threshold - a.threshold);  // -1% to -50% (left to right)
  }, [thresholdStats]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: { 
    active?: boolean; 
    payload?: Array<{ value: number; name: string; color: string }>; 
    label?: string 
  }) => {
    if (!active || !payload || payload.length === 0) return null;
    
    const data = chartData.find(d => d.thresholdLabel === label);
    if (!data) return null;
    
    return (
      <div className="bg-popover border border-border rounded-lg p-3 shadow-lg text-sm">
        <p className="font-semibold mb-2">{label} Dip Threshold</p>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Occurrences:</span>
            <span className="font-mono">{data.occurrences}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Avg Return (90d):</span>
            <span className="font-mono text-success">{data.avgReturn.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Total Profit (90d):</span>
            <span className="font-mono text-muted-foreground">{data.totalProfit.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Total Profit @ Rec:</span>
            <span className="font-mono text-primary">{data.totalProfitAtRecovery.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Avg Days to Rec:</span>
            <span className="font-mono">{data.avgDaysAtRecovery.toFixed(0)}d</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Avg MAE (pain):</span>
            <span className="font-mono text-danger">-{data.mae.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-muted-foreground">Win Rate:</span>
            <span className="font-mono">{data.winRate.toFixed(0)}%</span>
          </div>
        </div>
      </div>
    );
  };

  if (chartData.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Threshold Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] flex items-center justify-center text-muted-foreground text-sm">
            Insufficient data for threshold analysis
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Dip Threshold Analysis</CardTitle>
        <CardDescription className="text-xs">
          How dip depth affects returns (90-day holding period)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart
            data={chartData}
            margin={{ top: 20, right: 40, left: 0, bottom: 20 }}
          >
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke="hsl(var(--border))" 
              vertical={false}
            />
            
            <XAxis
              dataKey="thresholdLabel"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10 }}
              interval={2}  // Show every 3rd label to avoid crowding
            />
            
            {/* Left Y-axis: Returns */}
            <YAxis
              yAxisId="left"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10 }}
              tickFormatter={(value) => `${value}%`}
              width={45}
            />
            
            {/* Right Y-axis: Occurrences */}
            <YAxis
              yAxisId="right"
              orientation="right"
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10 }}
              width={35}
            />
            
            <Tooltip content={<CustomTooltip />} />
            
            <Legend 
              wrapperStyle={{ fontSize: 10 }}
              iconSize={8}
            />
            
            {/* Bar: Number of occurrences */}
            <Bar
              yAxisId="right"
              dataKey="occurrences"
              name="Occurrences"
              fill="hsl(var(--muted))"
              opacity={0.4}
              radius={[2, 2, 0, 0]}
            />
            
            {/* Line: Average Return */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="avgReturn"
              name="Avg Return"
              stroke="hsl(var(--success))"
              strokeWidth={2}
              dot={{ r: 3, fill: 'hsl(var(--success))' }}
              activeDot={{ r: 5 }}
            />
            
            {/* Line: Total Profit at Recovery (primary metric) */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="totalProfitAtRecovery"
              name="Total Profit @ Rec"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={{ r: 3, fill: 'hsl(var(--primary))' }}
              activeDot={{ r: 5 }}
            />
            
            {/* Reference line: Risk-Adjusted Optimal */}
            <ReferenceLine
              x={`${optimalThreshold}%`}
              yAxisId="left"
              stroke="hsl(var(--chart-1))"
              strokeWidth={2}
              strokeDasharray="3 3"
              label={{
                value: '🎯 Risk',
                position: 'top',
                style: { fontSize: 9, fill: 'hsl(var(--chart-1))' },
              }}
            />
            
            {/* Reference line: Max Profit Optimal (only if different) */}
            {maxProfitThreshold !== optimalThreshold && (
              <ReferenceLine
                x={`${maxProfitThreshold}%`}
                yAxisId="left"
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                strokeDasharray="3 3"
                label={{
                  value: '💰 Profit',
                  position: 'top',
                  style: { fontSize: 9, fill: 'hsl(var(--chart-2))' },
                }}
              />
            )}
            
            {/* Reference line: Current Drawdown */}
            {currentDrawdown && currentDrawdown <= -5 && (
              <ReferenceLine
                x={`${Math.round(currentDrawdown)}%`}
                yAxisId="left"
                stroke="hsl(var(--danger))"
                strokeWidth={2}
                label={{
                  value: 'Now',
                  position: 'top',
                  style: { fontSize: 9, fill: 'hsl(var(--danger))' },
                }}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
        
        {/* Legend for reference lines */}
        <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-[hsl(var(--chart-1))]" style={{ borderStyle: 'dashed' }} />
            <span className="text-muted-foreground">Risk-Adjusted</span>
          </div>
          {maxProfitThreshold !== optimalThreshold && (
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-[hsl(var(--chart-2))]" style={{ borderStyle: 'dashed' }} />
              <span className="text-muted-foreground">Max Profit</span>
            </div>
          )}
          {currentDrawdown && currentDrawdown <= -5 && (
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-[hsl(var(--danger))]" />
              <span className="text-muted-foreground">Current</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
