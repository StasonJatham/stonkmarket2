import { useMemo, useState, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';
import { CHART_LINE_ANIMATION } from '@/lib/chartConfig';
import type { ChartDataPoint, SignalTrigger, SignalTriggersResponse } from '@/services/api';
import { getSignalTriggers } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Zap, CircleArrowUp } from 'lucide-react';

interface StockChartProps {
  symbol: string;
  data: ChartDataPoint[];
  isLoading?: boolean;
  showSignalMarkers?: boolean;
}

export function StockChart({ symbol, data, isLoading, showSignalMarkers = true }: StockChartProps) {
  const [signalsResponse, setSignalsResponse] = useState<SignalTriggersResponse | null>(null);
  const [showMarkers, setShowMarkers] = useState(showSignalMarkers);
  const [isLoadingSignals, setIsLoadingSignals] = useState(false);
  
  // Fetch signal triggers when symbol changes
  useEffect(() => {
    if (!symbol || !showMarkers) return;
    
    setIsLoadingSignals(true);
    // Cap at 730 days (API limit)
    const lookbackDays = Math.min(730, Math.max(180, data.length));
    getSignalTriggers(symbol, lookbackDays)
      .then(setSignalsResponse)
      .catch(() => setSignalsResponse(null))
      .finally(() => setIsLoadingSignals(false));
  }, [symbol, data.length, showMarkers]);
  
  // Extract triggers from response, but only if signals beat buy-and-hold
  const signals = (signalsResponse?.beats_buy_hold ?? false) ? (signalsResponse?.triggers ?? []) : [];
  
  // Merge chart data with signal triggers
  const chartData = useMemo(() => {
    const signalMap = new Map<string, SignalTrigger>();
    signals.forEach(s => signalMap.set(s.date, s));
    
    return data.map((point) => {
      const formattedDate = new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      });
      const signalTrigger = signalMap.get(point.date);
      
      return {
        ...point,
        date: formattedDate,
        originalDate: point.date,
        signalTrigger,
        // For the reference dots
        signalPrice: signalTrigger ? point.close : undefined,
      };
    });
  }, [data, signals]);
  
  // Find data points that have signals (only if strategy beats buy-and-hold)
  const signalPoints = useMemo(() => 
    chartData.filter(d => d.signalTrigger != null),
    [chartData]
  );

  const priceChange = useMemo(() => {
    if (data.length < 2) return 0;
    const first = data[0].close;
    const last = data[data.length - 1].close;
    return ((last - first) / first) * 100;
  }, [data]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? 'hsl(var(--success))' : 'hsl(var(--danger))';

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-24" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">{symbol}</CardTitle>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold">
              ${data[data.length - 1]?.close.toFixed(2)}
            </span>
            <span
              className={`text-sm font-medium ${
                isPositive ? 'text-success' : 'text-danger'
              }`}
            >
              {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
            </span>
          </div>
        </div>
        
        {/* Signal toggle and legend */}
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center gap-2">
            <Switch
              id="show-signals"
              checked={showMarkers}
              onCheckedChange={setShowMarkers}
              disabled={isLoadingSignals}
            />
            <Label htmlFor="show-signals" className="text-sm text-muted-foreground cursor-pointer">
              Show buy signals
            </Label>
          </div>
          
          {showMarkers && signalPoints.length > 0 && (
            <Badge variant="outline" className="gap-1 text-xs">
              <Zap className="h-3 w-3 text-success" />
              {signalPoints.length} signal{signalPoints.length > 1 ? 's' : ''}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id={`gradient-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12 }}
                tickMargin={8}
                minTickGap={40}
              />
              <YAxis
                domain={['dataMin', 'dataMax']}
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12 }}
                tickMargin={8}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
                width={60}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
                formatter={(value, _name, props) => {
                  const trigger = props.payload?.signalTrigger as SignalTrigger | undefined;
                  if (trigger) {
                    // Calculate expected value: win_rate * avg_return
                    const expectedValue = trigger.win_rate * trigger.avg_return_pct;
                    return [
                      <div key="price" className="flex flex-col gap-1">
                        <span>${Number(value).toFixed(2)}</span>
                        <span className="text-success text-xs font-medium flex items-center gap-1">
                          <CircleArrowUp className="h-3 w-3" /> {trigger.signal_name}
                        </span>
                        <span className="text-muted-foreground text-xs">
                          {Math.round(trigger.win_rate * 100)}% win, {expectedValue >= 0 ? '+' : ''}{expectedValue.toFixed(1)}% EV
                        </span>
                      </div>,
                      'Price'
                    ];
                  }
                  return [`$${Number(value).toFixed(2)}`, 'Price'];
                }}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={chartColor}
                strokeWidth={2}
                fill={`url(#gradient-${symbol})`}
                {...CHART_LINE_ANIMATION}
              />
              
              {/* Signal marker dots */}
              {showMarkers && signalPoints.map((point, idx) => (
                <ReferenceDot
                  key={`signal-${idx}`}
                  x={point.date}
                  y={point.close}
                  r={5}
                  fill="hsl(var(--success))"
                  stroke="hsl(var(--background))"
                  strokeWidth={2}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        {/* Signal legend below chart */}
        {showMarkers && signalPoints.length > 0 && (
          <div className="mt-3 pt-3 border-t">
            <p className="text-xs text-muted-foreground mb-2">
              <span className="inline-flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-success" />
                Buy signals (based on historical data only - no look-ahead)
              </span>
            </p>
            <div className="flex flex-wrap gap-1">
              {signalPoints.slice(-5).map((point, idx) => (
                <Badge 
                  key={idx} 
                  variant="secondary" 
                  className="text-[10px] px-1.5 py-0.5"
                >
                  {point.originalDate}: {point.signalTrigger?.signal_name}
                </Badge>
              ))}
              {signalPoints.length > 5 && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0.5">
                  +{signalPoints.length - 5} more
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
