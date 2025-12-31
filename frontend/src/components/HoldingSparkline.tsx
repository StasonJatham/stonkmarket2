import { useMemo } from 'react';
import {
  Area,
  AreaChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { useTheme } from '@/context/ThemeContext';
import type { HoldingSparklineData } from '@/services/api';

// Default colors
const DEFAULT_COLORS = {
  up: '#22c55e', // Green
  down: '#ef4444', // Red
};

// Colorblind-friendly palette (blue/orange)
const COLORBLIND_COLORS = {
  up: '#3b82f6', // Blue
  down: '#f97316', // Orange
};

/**
 * Get chart colors respecting theme and colorblind mode.
 * Uses context values directly to avoid race conditions with CSS variable updates.
 */
function useChartColors() {
  const { colorblindMode, customColors } = useTheme();

  return useMemo(() => {
    // Use colorblind palette when enabled
    if (colorblindMode) {
      return { success: COLORBLIND_COLORS.up, danger: COLORBLIND_COLORS.down, muted: '#6b7280' };
    }
    
    // Use custom colors if set
    if (customColors && (customColors.up !== DEFAULT_COLORS.up || customColors.down !== DEFAULT_COLORS.down)) {
      return { success: customColors.up, danger: customColors.down, muted: '#6b7280' };
    }
    
    // Default green/red
    return { success: DEFAULT_COLORS.up, danger: DEFAULT_COLORS.down, muted: '#6b7280' };
  }, [colorblindMode, customColors]);
}

interface HoldingSparklineProps {
  data: HoldingSparklineData | null;
  width?: number;
  height?: number;
}

/**
 * Mini sparkline chart for portfolio holdings.
 * 
 * Shows price history with:
 * - Theme-aware colors (green for positive, red for negative)
 * - Color-blind friendly palette
 * - Dots marking trade entry/exit points
 */
export function HoldingSparkline({
  data,
  width = 120,
  height = 40,
}: HoldingSparklineProps) {
  // Get theme-aware colors for SVG chart (respects colorblind mode)
  const chartColors = useChartColors();
  
  // Determine color based on performance
  const isPositive = data?.change_pct ? data.change_pct >= 0 : null;
  const chartColor = isPositive === null 
    ? chartColors.muted 
    : isPositive 
      ? chartColors.success 
      : chartColors.danger;

  // Transform data for recharts
  const chartData = useMemo(() => {
    if (!data?.prices.length) return [];
    
    return data.prices.map((p) => ({
      date: p.date,
      price: p.close,
    }));
  }, [data?.prices]);

  // Find trade markers that fall on chart dates
  const tradeMarkers = useMemo(() => {
    if (!data?.trades.length || !chartData.length) return [];
    
    const chartDates = new Set(chartData.map((d) => d.date));
    
    return data.trades
      .filter((t) => chartDates.has(t.date))
      .map((t) => {
        const point = chartData.find((d) => d.date === t.date);
        return {
          ...t,
          price: point?.price ?? t.price,
        };
      });
  }, [data?.trades, chartData]);

  if (!data || chartData.length < 2) {
    // Show placeholder for insufficient data
    return (
      <div 
        className="flex items-center justify-center text-muted-foreground/50"
        style={{ width, height }}
      >
        <svg viewBox="0 0 24 8" className="w-full h-2 opacity-30">
          <line
            x1="0"
            y1="4"
            x2="24"
            y2="4"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="2 2"
          />
        </svg>
      </div>
    );
  }

  return (
    <div style={{ width, height }} className="relative">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 2, right: 2, left: 2, bottom: 2 }}
        >
          <defs>
            <linearGradient id={`sparkGradient-${data.symbol}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={chartColor} stopOpacity={0.4} />
              <stop offset="100%" stopColor={chartColor} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          
          <Tooltip
            content={({ payload }) => {
              if (!payload?.[0]?.payload) return null;
              const { date, price } = payload[0].payload;
              return (
                <div className="bg-popover border rounded px-2 py-1 text-xs shadow-md">
                  <div className="text-muted-foreground">{date}</div>
                  <div className="font-semibold">${price.toFixed(2)}</div>
                </div>
              );
            }}
            cursor={false}
          />
          
          <Area
            type="monotone"
            dataKey="price"
            stroke={chartColor}
            strokeWidth={1.5}
            fill={`url(#sparkGradient-${data.symbol})`}
            isAnimationActive={false}
          />
          
          {/* Trade entry/exit markers */}
          {tradeMarkers.map((trade, idx) => (
            <ReferenceDot
              key={`${trade.date}-${idx}`}
              x={trade.date}
              y={trade.price}
              r={4}
              fill={trade.side === 'buy' ? chartColors.success : chartColors.danger}
              stroke="#fff"
              strokeWidth={1.5}
              isFront
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      
      {/* Change indicator badge */}
      {data.change_pct !== null && (
        <div
          className="absolute -top-1 -right-1 text-[9px] font-semibold px-1 rounded"
          style={{ 
            color: chartColor,
            backgroundColor: `${chartColor}33`,
          }}
        >
          {isPositive ? '+' : ''}{data.change_pct.toFixed(0)}%
        </div>
      )}
    </div>
  );
}

export default HoldingSparkline;
