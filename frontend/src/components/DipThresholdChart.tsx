/**
 * DipThresholdChart - Visualize the relationship between dip thresholds and expected returns
 * 
 * Split into two clear charts:
 * 1. Returns Chart: Avg return and win rate by threshold
 * 2. Opportunity Chart: Frequency and total profit potential
 * 
 * Helps users understand the trade-off:
 * - Shallow dips: More frequent but lower returns
 * - Deep dips: Higher returns but fewer opportunities
 */

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
  Area,
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

type DipThresholdChartDatum = {
  threshold: number;
  thresholdLabel: string;
  avgReturn: number;
  totalProfit: number;
  totalProfitCompounded: number;
  totalProfitAtRecovery: number;
  avgDaysAtRecovery: number;
  occurrences: number;
  recoveryDays: number;
  velocity: number;
  winRate: number;
  mae: number;
};

export function DipThresholdChart({
  thresholdStats,
  optimalThreshold,
  maxProfitThreshold,
  currentDrawdown,
  height = 200,
}: DipThresholdChartProps) {
  // Transform data for the chart
  // Filter to reasonable thresholds (>= -50%) and with at least 2 occurrences
  // Note: API returns decimal format (0.15 = 15%), so we multiply by 100 for display
  const chartData = (() => {
    return thresholdStats
      .filter(s => s.threshold >= -0.50 && s.occurrences >= 2)  // -0.50 = -50%
      .map(s => ({
        threshold: s.threshold * 100,  // -0.15 -> -15
        thresholdLabel: `${(s.threshold * 100).toFixed(0)}%`,
        avgReturn: s.avg_return * 100,  // 0.42 -> 42%
        totalProfit: s.total_profit * 100,
        totalProfitCompounded: s.total_profit_compounded * 100,
        totalProfitAtRecovery: s.total_profit_at_recovery * 100,
        avgDaysAtRecovery: s.avg_days_at_recovery,
        occurrences: s.occurrences,
        recoveryDays: s.avg_days_to_threshold,
        velocity: s.avg_recovery_velocity,
        winRate: s.win_rate * 100,  // 0.857 -> 85.7%
        mae: Math.abs(s.avg_further_drawdown * 100),  // 0.21 -> 21%
      }))
      .sort((a, b) => b.threshold - a.threshold);  // -1% to -50% (left to right)
  })();

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

  // Optimal thresholds for reference lines (used in ReferenceLine components)
  void optimalThreshold; // Reserved for future optimal line display
  const maxProfitLabel = `${Math.round(maxProfitThreshold * 100)}%`;
  const currentLabel = currentDrawdown ? `${Math.round(currentDrawdown * 100)}%` : null;

  return (
    <div className="space-y-4">
      {/* Chart 1: Returns & Win Rate */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Returns by Dip Depth</CardTitle>
          <CardDescription className="text-xs">
            Expected returns at each threshold level
          </CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart
              data={chartData}
              margin={{ top: 15, right: 10, left: 0, bottom: 5 }}
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
                interval={2}
              />
              
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10 }}
                tickFormatter={(value) => `${value}%`}
                width={40}
              />
              
              <Tooltip content={<ReturnsTooltip />} />
              
              <Legend 
                wrapperStyle={{ fontSize: 10 }}
                iconSize={8}
              />
              
              {/* Area: Win Rate (background) */}
              <Area
                type="monotone"
                dataKey="winRate"
                name="Win Rate"
                fill="hsl(var(--primary))"
                fillOpacity={0.15}
                stroke="hsl(var(--primary))"
                strokeWidth={1}
                strokeOpacity={0.5}
              />
              
              {/* Line: Average Return */}
              <Line
                type="monotone"
                dataKey="avgReturn"
                name="Avg Return"
                stroke="hsl(var(--success))"
                strokeWidth={2}
                dot={{ r: 2, fill: 'hsl(var(--success))' }}
                activeDot={{ r: 4 }}
              />
              
              {/* Reference line: Max Profit Optimal */}
              <ReferenceLine
                x={maxProfitLabel}
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                strokeDasharray="4 2"
                label={{
                  value: '💰 Best',
                  position: 'top',
                  style: { fontSize: 9, fill: 'hsl(var(--chart-2))' },
                }}
              />
              
              {/* Reference line: Current Drawdown */}
              {currentLabel && currentDrawdown && currentDrawdown <= -0.05 && (
                <ReferenceLine
                  x={currentLabel}
                  stroke="hsl(var(--danger))"
                  strokeWidth={2}
                  label={{
                    value: '📍 Now',
                    position: 'top',
                    style: { fontSize: 9, fill: 'hsl(var(--danger))' },
                  }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Chart 2: Frequency & Profit Potential */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Trade Frequency</CardTitle>
          <CardDescription className="text-xs">
            Historical occurrences and total profit at each level
          </CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart
              data={chartData}
              margin={{ top: 15, right: 35, left: 0, bottom: 5 }}
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
                interval={2}
              />
              
              {/* Left Y-axis: Total Profit */}
              <YAxis
                yAxisId="left"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10 }}
                tickFormatter={(value) => `${value}%`}
                width={40}
              />
              
              {/* Right Y-axis: Occurrences */}
              <YAxis
                yAxisId="right"
                orientation="right"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 10 }}
                width={30}
              />
              
              <Tooltip content={<FrequencyTooltip />} />
              
              <Legend 
                wrapperStyle={{ fontSize: 10 }}
                iconSize={8}
              />
              
              {/* Bar: Number of occurrences */}
              <Bar
                yAxisId="right"
                dataKey="occurrences"
                name="# Trades"
                fill="hsl(var(--muted))"
                opacity={0.5}
                radius={[2, 2, 0, 0]}
              />
              
              {/* Line: Total Profit */}
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="totalProfitAtRecovery"
                name="Total Profit"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={{ r: 2, fill: 'hsl(var(--primary))' }}
                activeDot={{ r: 4 }}
              />
              
              {/* Reference line: Max Profit Optimal */}
              <ReferenceLine
                x={maxProfitLabel}
                yAxisId="left"
                stroke="hsl(var(--chart-2))"
                strokeWidth={2}
                strokeDasharray="4 2"
                label={{
                  value: '💰 Best',
                  position: 'top',
                  style: { fontSize: 9, fill: 'hsl(var(--chart-2))' },
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}

// Tooltip for Returns chart
function ReturnsTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: DipThresholdChartDatum }> }) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-popover border border-border rounded-lg p-2.5 shadow-lg text-xs">
      <p className="font-semibold mb-1.5">{data.thresholdLabel} Dip</p>
      <div className="space-y-1">
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Avg Return:</span>
          <span className="font-mono text-success">+{data.avgReturn.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Win Rate:</span>
          <span className="font-mono">{data.winRate.toFixed(0)}%</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Max Pain:</span>
          <span className="font-mono text-danger">-{data.mae.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

// Tooltip for Frequency chart
function FrequencyTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: DipThresholdChartDatum }> }) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-popover border border-border rounded-lg p-2.5 shadow-lg text-xs">
      <p className="font-semibold mb-1.5">{data.thresholdLabel} Dip</p>
      <div className="space-y-1">
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Occurrences:</span>
          <span className="font-mono">{data.occurrences}</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Total Profit:</span>
          <span className="font-mono text-primary">+{data.totalProfitAtRecovery.toFixed(0)}%</span>
        </div>
        <div className="flex justify-between gap-3">
          <span className="text-muted-foreground">Avg Recovery:</span>
          <span className="font-mono">{data.avgDaysAtRecovery.toFixed(0)}d</span>
        </div>
      </div>
    </div>
  );
}
