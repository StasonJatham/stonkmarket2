import { useMemo } from 'react';
import {
  Area,
  AreaChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { HoldingSparklineData } from '@/services/api';

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
  // Determine color based on performance
  const { strokeColor, fillColor, isPositive } = useMemo(() => {
    if (!data?.change_pct) {
      // Neutral color when no change data
      return {
        strokeColor: 'hsl(var(--muted-foreground))',
        fillColor: 'hsl(var(--muted))',
        isPositive: null,
      };
    }
    
    const positive = data.change_pct >= 0;
    
    // Using CSS variables for theme awareness
    // These map to the chart colors in globals.css which are color-blind friendly
    if (positive) {
      return {
        strokeColor: 'hsl(var(--chart-2))', // Green variant
        fillColor: 'hsl(var(--chart-2) / 0.2)',
        isPositive: true,
      };
    } else {
      return {
        strokeColor: 'hsl(var(--chart-5))', // Red variant  
        fillColor: 'hsl(var(--chart-5) / 0.2)',
        isPositive: false,
      };
    }
  }, [data?.change_pct]);

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
              <stop offset="0%" stopColor={strokeColor} stopOpacity={0.4} />
              <stop offset="100%" stopColor={strokeColor} stopOpacity={0.05} />
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
            stroke={strokeColor}
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
              fill={trade.side === 'buy' ? 'hsl(var(--chart-2))' : 'hsl(var(--chart-5))'}
              stroke="hsl(var(--background))"
              strokeWidth={1.5}
              isFront
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      
      {/* Change indicator badge */}
      {data.change_pct !== null && (
        <div
          className={`absolute -top-1 -right-1 text-[9px] font-semibold px-1 rounded ${
            isPositive
              ? 'bg-chart-2/20 text-chart-2'
              : 'bg-chart-5/20 text-chart-5'
          }`}
          style={{ 
            color: strokeColor,
            backgroundColor: fillColor,
          }}
        >
          {isPositive ? '+' : ''}{data.change_pct.toFixed(0)}%
        </div>
      )}
    </div>
  );
}

export default HoldingSparkline;
