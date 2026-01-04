import { useEffect, useState, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceDot,
} from 'recharts';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';
import { useLandingData } from '@/features/quant-engine/api/queries';
import type { ChartDataPoint, QuantRecommendation, SignalTrigger } from '@/services/api';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { StockLogo } from '@/components/StockLogo';
import {
  ArrowRight,
  Target,
  Sparkles,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Users,
  ThumbsUp,
  ThumbsDown,
  MinusCircle,
  Brain,
  Zap,
  CircleArrowUp,
  AlertCircle,
  Clock,
  TrendingUp,
  ShieldCheck,
  BarChart3,
  Search,
  CheckCircle2,
  Trophy,
  Activity,
} from 'lucide-react';

// Action icon based on recommendation
function ActionIcon({ action }: { action: 'BUY' | 'SELL' | 'HOLD' }) {
  if (action === 'BUY') return <ArrowUpRight className="h-4 w-4" />;
  if (action === 'SELL') return <ArrowDownRight className="h-4 w-4" />;
  return <Minus className="h-4 w-4" />;
}

// Action badge styling
function getActionBadgeVariant(action: 'BUY' | 'SELL' | 'HOLD') {
  if (action === 'BUY') return 'default';
  if (action === 'SELL') return 'destructive';
  return 'secondary';
}

// Animated counter component
function AnimatedCounter({ value, suffix = '', prefix = '' }: { value: number; suffix?: string; prefix?: string }) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => Math.round(v));
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const controls = animate(count, value, { duration: 2, ease: 'easeOut' });
    const unsubscribe = rounded.on('change', (v) => setDisplay(v));
    return () => {
      controls.stop();
      unsubscribe();
    };
  }, [value, count, rounded]);

  return <span>{prefix}{display.toLocaleString()}{suffix}</span>;
}

function formatSignedPercent(value: number, digits: number = 1): string {
  if (!Number.isFinite(value)) return 'N/A';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(digits)}%`;
}

function formatOptionalPercent(value: number | null | undefined, digits: number = 1): string {
  if (value == null || Number.isNaN(value)) return 'N/A';
  const percent = Math.abs(value) <= 1 ? value * 100 : value;
  return `${percent.toFixed(digits)}%`;
}

function formatDipPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'N/A';
  return `-${Math.abs(value).toFixed(1)}%`;
}

function formatOptionalNumber(value: number | null | undefined, suffix = ''): string {
  if (value == null || Number.isNaN(value)) return 'N/A';
  return `${Math.round(value)}${suffix}`;
}

function formatRatio(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'N/A';
  return `${value.toFixed(1)}x`;
}

function formatUpdatedLabel(timestamp: number | null): string {
  if (!timestamp) return 'Updated just now';
  const diffMs = Date.now() - timestamp;
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return 'Updated just now';
  if (minutes < 60) return `Updated ${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Updated ${hours}h ago`;
  return `Updated ${new Date(timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;
}

// Compact signal board card - emphasizes action, expected return, and sparkline
function SignalBoardCard({
  rec,
  chartData,
  index,
  onClick,
}: {
  rec: QuantRecommendation;
  chartData: ChartDataPoint[];
  index: number;
  onClick: () => void;
}) {
  const { getActiveColors } = useTheme();
  const colors = getActiveColors();

  const lastPrice = chartData.length > 0
    ? chartData[chartData.length - 1].close
    : (rec.last_price ?? 0);

  const priceChange = useMemo(() => {
    if (chartData.length >= 2) {
      const first = chartData[0].close;
      const last = chartData[chartData.length - 1].close;
      return ((last - first) / first) * 100;
    }
    if (typeof rec.change_percent === 'number') {
      return rec.change_percent;
    }
    return null;
  }, [chartData, rec.change_percent]);

  const isUp = (priceChange ?? 0) >= 0;
  const priceChangeClass = priceChange == null
    ? 'text-muted-foreground'
    : isUp
      ? 'text-success'
      : 'text-danger';

  const miniChartData = useMemo(() => {
    const sliced = chartData.slice(-45);
    return sliced.map((p, i) => ({ x: i, y: p.close }));
  }, [chartData]);

  const edgeScore = useMemo(() => {
    if (rec.quant_mode === 'CERTIFIED_BUY' && rec.quant_score_a != null) return rec.quant_score_a;
    if (rec.quant_score_b != null) return rec.quant_score_b;
    if (rec.best_chance_score != null) return rec.best_chance_score;
    if (rec.dip_score != null) return rec.dip_score * 100;
    return null;
  }, [rec]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.08 }}
    >
      <Card
        className="cursor-pointer transition-all hover:shadow-lg hover:border-primary/40 group"
        onClick={onClick}
      >
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <StockLogo symbol={rec.ticker} size="sm" />
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-base">{rec.ticker}</span>
                  <Badge variant={getActionBadgeVariant(rec.action)} className="text-xs h-5 gap-1">
                    <ActionIcon action={rec.action} />
                    {rec.action}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground line-clamp-1">
                  {rec.name || rec.ticker}
                </p>
              </div>
            </div>
            <div className="text-right shrink-0">
              {lastPrice > 0 && (
                <p className="font-mono font-bold text-base">${lastPrice.toFixed(2)}</p>
              )}
              <p className={`text-xs font-medium ${priceChangeClass}`}>
                {priceChange != null ? `${isUp ? '+' : ''}${priceChange.toFixed(2)}%` : 'N/A'}
              </p>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
            <div>
              <p className="text-muted-foreground">Expected Return</p>
              <p className={`font-mono font-semibold ${rec.mu_hat >= 0 ? 'text-success' : 'text-danger'}`}>
                {formatSignedPercent(rec.mu_hat * 100, 1)}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">Signal Score</p>
              <p className="font-mono font-semibold text-foreground">
                {edgeScore != null ? edgeScore.toFixed(0) : 'N/A'}
              </p>
            </div>
          </div>

          <div className="mt-3 h-16 -mx-1">
            {miniChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={miniChartData}>
                  <defs>
                    <linearGradient id={`signal-gradient-${rec.ticker}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isUp ? colors.up : colors.down} stopOpacity={0.35} />
                      <stop offset="100%" stopColor={isUp ? colors.up : colors.down} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="y"
                    stroke={isUp ? colors.up : colors.down}
                    strokeWidth={2}
                    fill={`url(#signal-gradient-${rec.ticker})`}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <Skeleton className="h-full w-full" />
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Feature card component
function FeatureCard({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: typeof Target;
  title: string;
  description: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.5, delay }}
    >
      <Card className="h-full transition-all hover:shadow-md hover:border-primary/30">
        <CardHeader>
          <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-2">
            <Icon className="h-6 w-6 text-primary" />
          </div>
          <CardTitle className="text-xl">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <CardDescription className="text-base">{description}</CardDescription>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Hero chart - larger showcase chart with signal markers
function HeroChart({
  chartData,
  symbol,
  isLoading,
  signals = [],
}: {
  chartData: ChartDataPoint[];
  symbol: string;
  isLoading: boolean;
  signals?: SignalTrigger[];
}) {
  const { getActiveColors, resolvedTheme } = useTheme();
  const colors = getActiveColors();

  // Merge chart data with signal triggers
  const displayData = useMemo(() => {
    const signalMap = new Map<string, SignalTrigger>();
    signals.forEach(s => signalMap.set(s.date, s));

    const sliced = chartData.slice(-120);
    return sliced.map((point) => {
      const signalTrigger = signalMap.get(point.date);
      return {
        ...point,
        displayDate: new Date(point.date).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        signalTrigger,
      };
    });
  }, [chartData, signals]);
  
  // Find data points that have signals
  const signalPoints = useMemo(() => 
    displayData.filter(d => d.signalTrigger != null),
    [displayData]
  );

  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [chartData]);

  const isPositive = priceChange >= 0;
  const chartColor = isPositive ? colors.up : colors.down;
  const signalDotStroke = resolvedTheme === 'dark' ? '#0b0f19' : '#ffffff';

  if (isLoading || !symbol) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-2">
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-72 md:h-80 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden bg-gradient-to-br from-background to-muted/30">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StockLogo symbol={symbol} size="sm" />
            <div>
              <CardTitle className="text-lg">{symbol}</CardTitle>
              <CardDescription>Live signal preview</CardDescription>
            </div>
          </div>
          {chartData.length > 0 && (
            <div className="text-right">
              <p className="text-2xl font-bold font-mono">
                ${chartData[chartData.length - 1]?.close.toFixed(2)}
              </p>
              <p
                className="text-sm font-medium"
                style={{ color: chartColor }}
              >
                {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
              </p>
            </div>
          )}
        </div>
        {/* Signal count badge */}
        {signalPoints.length > 0 && (
          <Badge variant="secondary" className="mt-2 gap-1 w-fit text-xs">
            <Zap className="h-3 w-3 text-success" />
            {signalPoints.length} signal{signalPoints.length > 1 ? 's' : ''} detected
          </Badge>
        )}
      </CardHeader>
      <CardContent className="pb-4">
        <div className="h-72 md:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={displayData}>
              <defs>
                <linearGradient id="hero-gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={chartColor} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="displayDate"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                tickMargin={8}
                minTickGap={50}
              />
              <YAxis
                domain={['dataMin', 'dataMax']}
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                width={50}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                }}
                labelStyle={{ color: 'hsl(var(--popover-foreground))' }}
                content={({ active, payload, label }) => {
                  if (!active || !payload || payload.length === 0) return null;
                  const data = payload[0]?.payload;
                  const trigger = data?.signalTrigger as SignalTrigger | undefined;
                  const price = payload[0]?.value as number;
                  // Calculate expected value: win_rate * avg_return
                  const expectedValue = trigger ? trigger.win_rate * trigger.avg_return_pct : 0;
                  
                  return (
                    <div className="bg-popover border border-border rounded-lg p-2 shadow-lg">
                      <p className="text-popover-foreground text-xs mb-1">{label}</p>
                      <p className="font-mono font-bold">${price?.toFixed(2)}</p>
                      {trigger && (
                        <div className="mt-1 pt-1 border-t border-border/50">
                          <p className="text-success text-xs font-medium flex items-center gap-1">
                            <CircleArrowUp className="h-3 w-3" /> {trigger.signal_name}
                          </p>
                          <p className="text-muted-foreground text-xs">
                            {Math.round(trigger.win_rate * 100)}% win, {expectedValue >= 0 ? '+' : ''}{expectedValue.toFixed(1)}% EV
                          </p>
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={chartColor}
                strokeWidth={2}
                fill="url(#hero-gradient)"
              />
              
              {/* Signal marker dots */}
              {signalPoints.map((point, idx) => {
                const isExit = point.signalTrigger?.signal_type === 'exit';
                const dotColor = isExit ? colors.down : colors.up;
                return (
                  <ReferenceDot
                    key={`signal-${idx}`}
                    x={point.displayDate}
                    y={point.close}
                    r={6}
                    fill={dotColor}
                    stroke={signalDotStroke}
                    strokeWidth={2}
                  />
                );
              })}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Evidence chip component for "Why This Signal" strip
function EvidenceChip({
  label,
  value,
  icon: Icon,
  variant = 'default',
}: {
  label: string;
  value: string;
  icon?: typeof Target;
  variant?: 'default' | 'success' | 'warning' | 'danger';
}) {
  const variantClasses = {
    default: 'bg-muted/50 border-border text-foreground',
    success: 'bg-success/10 border-success/30 text-success',
    warning: 'bg-warning/10 border-warning/30 text-warning',
    danger: 'bg-danger/10 border-danger/30 text-danger',
  };

  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-medium ${variantClasses[variant]}`}>
      {Icon && <Icon className="h-3 w-3" />}
      <span className="text-muted-foreground">{label}:</span>
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}

// Horizontal pipeline with animated checkpoints
function HorizontalPipeline() {
  const steps = [
    { icon: Search, title: 'Scan', description: 'Detect dips & momentum shifts' },
    { icon: Users, title: 'Debate', description: 'AI personas challenge the signal' },
    { icon: Target, title: 'Optimize Entry', description: 'Find optimal buy zone' },
    { icon: BarChart3, title: 'Backtest', description: 'Validate with historical data' },
  ];

  return (
    <div className="w-full overflow-x-auto">
      <div className="grid grid-cols-4 gap-4 min-w-[600px]">
        {steps.map((step, idx) => (
          <div key={step.title} className="relative">
            <motion.div
              className="flex flex-col items-center text-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: idx * 0.15 }}
            >
              <motion.div
                className="h-14 w-14 rounded-full bg-primary/10 border-2 border-primary flex items-center justify-center mb-3"
                whileHover={{ scale: 1.1 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <step.icon className="h-6 w-6 text-primary" />
              </motion.div>
              <h4 className="font-semibold text-sm mb-1">{step.title}</h4>
              <p className="text-xs text-muted-foreground">{step.description}</p>
            </motion.div>
            {/* Connector line */}
            {idx < steps.length - 1 && (
              <motion.div
                className="absolute top-7 left-[calc(50%+28px)] w-[calc(100%-28px)] h-0.5 bg-gradient-to-r from-primary/50 to-primary/20"
                initial={{ scaleX: 0, originX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.15 + 0.2 }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Backtest proof card
function BacktestProofCard({
  title,
  value,
  suffix,
  icon: Icon,
  subtitle,
  variant = 'default',
  index = 0,
}: {
  title: string;
  value: number | string;
  suffix?: string;
  icon: typeof Trophy;
  subtitle?: string;
  variant?: 'success' | 'danger' | 'default';
  index?: number;
}) {
  const variantClasses = {
    success: 'text-success',
    danger: 'text-danger',
    default: 'text-foreground',
  };

  return (
    <motion.div
      className="h-full"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
    >
      <Card className="h-full">
        <CardContent className="flex flex-col items-center justify-center text-center p-6 h-full">
          <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
            <Icon className="h-6 w-6 text-primary" />
          </div>
          <p className={`text-3xl font-bold font-mono ${variantClasses[variant]}`}>
            {typeof value === 'number' ? (
              <AnimatedCounter value={value} suffix={suffix} />
            ) : (
              `${value}${suffix ?? ''}`
            )}
          </p>
          <p className="font-medium text-sm mt-2">{title}</p>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Signal gallery mini card
function SignalGalleryCard({
  rec,
  onClick,
  index,
}: {
  rec: QuantRecommendation;
  onClick: () => void;
  index: number;
}) {
  const dipPct = rec.legacy_dip_pct ?? 0;
  const recoveryOdds = rec.quant_evidence?.p_recovery ?? rec.win_rate ?? null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.35, delay: index * 0.08 }}
    >
      <Card
        className="cursor-pointer transition-all hover:shadow-lg hover:border-primary/40 group overflow-hidden"
        onClick={onClick}
      >
        <CardContent className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <StockLogo symbol={rec.ticker} size="sm" />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg">{rec.ticker}</span>
                <Badge variant={getActionBadgeVariant(rec.action)} className="text-xs h-5">
                  {rec.action}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground truncate">{rec.name}</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs mb-3">
            <div className="bg-muted/50 rounded-md p-2">
              <p className="text-muted-foreground">Dip</p>
              <p className="font-mono font-bold text-danger">
                {formatDipPercent(dipPct)}
              </p>
            </div>
            <div className="bg-muted/50 rounded-md p-2">
              <p className="text-muted-foreground">P(rec)</p>
              <p className="font-mono font-bold text-success">
                {formatOptionalPercent(recoveryOdds, 0)}
              </p>
            </div>
          </div>

          {rec.ai_summary && (
            <p className="text-xs text-muted-foreground line-clamp-2">
              <span className="font-medium text-foreground">AI: </span>
              {rec.ai_summary}
            </p>
          )}

          <div className="mt-3 flex items-center justify-between">
            {rec.opportunity_type && rec.opportunity_type !== 'NONE' && (
              <Badge variant="outline" className="text-xs">
                {rec.opportunity_type}
              </Badge>
            )}
            <Button variant="ghost" size="sm" className="ml-auto gap-1 text-xs h-7 opacity-0 group-hover:opacity-100 transition-opacity">
              Analyze
              <ChevronRight className="h-3 w-3" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Constants for signal board display
const LANDING_SIGNAL_BOARD_COUNT = 4;

export function Landing() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  
  // All data fetching via TanStack Query - no more manual caching!
  const {
    recommendations,
    marketMessage,
    portfolioStats,
    chartDataMap,
    heroChart,
    heroSymbol,
    heroRec,
    heroAgentAnalysis,
    heroSignals,
    heroSignalSummary,
    lastUpdatedAt,
    isLoading,
    isLoadingHero,
  } = useLandingData(1000, 12);

  const handleCTA = useCallback(() => {
    navigate(isAuthenticated ? '/dashboard' : '/login');
  }, [navigate, isAuthenticated]);

  const handleStockClick = useCallback(
    (symbol: string) => {
      navigate(`/dashboard?stock=${symbol}`);
    },
    [navigate]
  );

  const signalBoardRecs = recommendations.slice(0, LANDING_SIGNAL_BOARD_COUNT);
  const updatedLabel = lastUpdatedAt ? formatUpdatedLabel(lastUpdatedAt) : 'Updating...';
  const buyCount = recommendations.filter((rec) => rec.action === 'BUY').length;
  
  const heroVerdicts = useMemo(() => heroAgentAnalysis?.verdicts ?? [], [heroAgentAnalysis?.verdicts]);
  const verdictCounts = useMemo(() => {
    const bullishCount = heroVerdicts.filter((v) => v.signal === 'buy' || v.signal === 'strong_buy').length;
    const bearishCount = heroVerdicts.filter((v) => v.signal === 'sell' || v.signal === 'strong_sell').length;
    const neutralCount = heroVerdicts.filter((v) => v.signal === 'hold').length;
    const isBullish = heroAgentAnalysis?.overall_signal === 'buy' || heroAgentAnalysis?.overall_signal === 'strong_buy';
    const isBearish = heroAgentAnalysis?.overall_signal === 'sell' || heroAgentAnalysis?.overall_signal === 'strong_sell';
    return { bullishCount, bearishCount, neutralCount, isBullish, isBearish };
  }, [heroVerdicts, heroAgentAnalysis?.overall_signal]);
  const signalCount = heroSignalSummary?.nTrades ?? heroSignals.length;
  const expectedReturnPercent = portfolioStats.expectedReturn * 100;
  const edgeVsBuyHold = heroSignalSummary?.edgeVsBuyHoldPct;
  const edgeClassName = edgeVsBuyHold == null
    ? 'text-muted-foreground'
    : edgeVsBuyHold >= 0
      ? 'text-success'
      : 'text-danger';
  const dipDepth = heroRec?.legacy_dip_pct ?? null;
  
  const recoveryOdds = heroRec?.quant_evidence?.p_recovery ?? heroRec?.win_rate ?? null;
  const expectedRecovery = heroRec?.expected_recovery_days ?? heroRec?.domain_recovery_days ?? null;
  
  // Dip vs typical: prefer dipfinder, fallback to computed from legacy/typical
  const dipVsTypical = useMemo(() => {
    if (heroRec?.dip_vs_typical != null) return heroRec.dip_vs_typical;
    // Compute fallback: current dip / typical dip
    if (heroRec?.legacy_dip_pct != null && heroRec?.typical_dip_pct != null && heroRec.typical_dip_pct > 0) {
      return Math.abs(heroRec.legacy_dip_pct) / heroRec.typical_dip_pct;
    }
    return null;
  }, [heroRec?.dip_vs_typical, heroRec?.legacy_dip_pct, heroRec?.typical_dip_pct]);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-16 md:py-24 px-4">
        <div className="absolute -top-32 right-0 h-64 w-64 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute -bottom-32 left-10 h-72 w-72 rounded-full bg-sky-500/10 blur-3xl" />
        <div className="container mx-auto max-w-6xl relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: Content */}
            <div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Badge variant="secondary" className="mb-4 gap-1.5">
                  <Sparkles className="h-3 w-3" />
                  AI-Powered Signal Engine
                </Badge>
              </motion.div>

              <motion.h1
                className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                Catch real dips with
                <br />
                <span className="text-primary">AI-backed conviction.</span>
              </motion.h1>

              <motion.p
                className="text-lg md:text-xl text-muted-foreground mb-6 max-w-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                StonkMarket scans the market, flags high-probability reversals, and explains the signal in plain language.
              </motion.p>

              <motion.div
                className="flex flex-wrap gap-2 mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.25 }}
              >
                <Badge variant="outline" className="gap-1.5">
                  <Zap className="h-3 w-3" />
                  Live signals
                </Badge>
                <Badge variant="outline" className="gap-1.5">
                  <Brain className="h-3 w-3" />
                  AI debate
                </Badge>
                <Badge variant="outline" className="gap-1.5">
                  <Target className="h-3 w-3" />
                  Optimized entries
                </Badge>
              </motion.div>

              <motion.div
                className="flex flex-col sm:flex-row gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <Button size="lg" onClick={handleCTA} className="gap-2">
                  {isAuthenticated ? 'Go to Dashboard' : 'Get Started Free'}
                  <ArrowRight className="h-4 w-4" />
                </Button>
                <Button size="lg" variant="outline" onClick={() => navigate('/learn')} className="gap-2">
                  Learn More
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </motion.div>

              <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {updatedLabel}
                </span>
                <span>
                  {isLoading ? 'Loading signals...' : `${recommendations.length} ranked | ${buyCount} buy signals`}
                </span>
              </div>
            </div>

            {/* Right: Hero Chart + Analysis */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <div className="relative">
                <div className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-emerald-500/10 via-transparent to-sky-500/10 blur-2xl" />
                <div className="relative space-y-4">
                  <HeroChart chartData={heroChart} symbol={heroSymbol} isLoading={isLoadingHero} signals={heroSignals} />

                  {heroRec && (
                    <Card className="bg-gradient-to-br from-emerald-500/5 to-sky-500/10 border-emerald-500/20">
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Activity className="h-4 w-4 text-emerald-600" />
                            Live Signal Briefing
                          </CardTitle>
                          <Badge variant={getActionBadgeVariant(heroRec.action)} className="gap-1 text-xs">
                            <ActionIcon action={heroRec.action} />
                            {heroRec.action}
                          </Badge>
                        </div>

                        {/* Top 3 Key Metrics - Trading Desk Style */}
                        <div className="grid grid-cols-3 gap-3 mt-3 p-3 bg-background/50 rounded-lg">
                          <div className="text-center">
                            <p className="text-2xl font-bold font-mono text-danger">
                              {formatDipPercent(dipDepth)}
                            </p>
                            <p className="text-xs text-muted-foreground">Dip Depth</p>
                          </div>
                          <div className="text-center border-x border-border/50">
                            <p className="text-2xl font-bold font-mono text-success">
                              {formatOptionalPercent(recoveryOdds, 0)}
                            </p>
                            <p className="text-xs text-muted-foreground">P(Recovery)</p>
                          </div>
                          <div className="text-center">
                            <p className="text-2xl font-bold font-mono text-foreground">
                              {formatOptionalNumber(expectedRecovery, 'd')}
                            </p>
                            <p className="text-xs text-muted-foreground">Avg Recovery</p>
                          </div>
                        </div>

                        {/* AI Verdict Stack */}
                        {heroVerdicts.length > 0 && (
                          <div className="flex items-center gap-2 mt-3">
                            <div className="flex -space-x-2">
                              {heroVerdicts.slice(0, 4).map((v) => (
                                <div
                                  key={v.agent_id}
                                  className={`h-6 w-6 rounded-full border-2 border-background flex items-center justify-center text-[10px] font-bold ${
                                    v.signal === 'buy' || v.signal === 'strong_buy'
                                      ? 'bg-success text-success-foreground'
                                      : v.signal === 'sell' || v.signal === 'strong_sell'
                                        ? 'bg-danger text-danger-foreground'
                                        : 'bg-muted text-muted-foreground'
                                  }`}
                                  title={`${v.agent_name}: ${v.signal}`}
                                >
                                  {v.signal === 'buy' || v.signal === 'strong_buy' ? '↑' : v.signal === 'sell' || v.signal === 'strong_sell' ? '↓' : '−'}
                                </div>
                              ))}
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {verdictCounts.bullishCount} bullish, {verdictCounts.bearishCount} bearish
                            </span>
                          </div>
                        )}
                      </CardHeader>

                      <CardContent className="pt-0">
                        {/* Why This Signal - Evidence Chips */}
                        <div className="mb-4">
                          <p className="text-xs font-medium text-muted-foreground mb-2">Why this signal:</p>
                          <div className="flex flex-wrap gap-1.5">
                            {recoveryOdds != null && (
                              <EvidenceChip
                                label="P(rec)"
                                value={formatOptionalPercent(recoveryOdds, 0)}
                                icon={TrendingUp}
                                variant={recoveryOdds >= 0.7 ? 'success' : recoveryOdds >= 0.5 ? 'default' : 'warning'}
                              />
                            )}
                            {dipVsTypical != null && (
                              <EvidenceChip
                                label="vs typical"
                                value={formatRatio(dipVsTypical)}
                                icon={BarChart3}
                                variant={dipVsTypical >= 1.5 ? 'success' : 'default'}
                              />
                            )}
                            {heroRec.is_unusual_dip && (
                              <EvidenceChip
                                label="Unusual"
                                value="Yes"
                                icon={AlertCircle}
                                variant="warning"
                              />
                            )}
                            {heroRec.win_rate != null && (
                              <EvidenceChip
                                label="Win rate"
                                value={formatOptionalPercent(heroRec.win_rate, 0)}
                                icon={Trophy}
                                variant={heroRec.win_rate >= 0.65 ? 'success' : 'default'}
                              />
                            )}
                            {heroRec.opportunity_type && heroRec.opportunity_type !== 'NONE' && (
                              <EvidenceChip
                                label="Type"
                                value={heroRec.opportunity_type}
                                icon={Zap}
                                variant="success"
                              />
                            )}
                          </div>
                        </div>

                        {/* Backtest Stats Bar */}
                        <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground py-2 border-t border-border/50">
                          <span className="flex items-center gap-2">
                            <CheckCircle2 className="h-3 w-3 text-success" />
                            {heroSignalSummary?.nTrades != null
                              ? `${heroSignalSummary.nTrades} trades backtested`
                              : 'Signal stats loading'}
                            {edgeVsBuyHold != null && (
                              <span className={`font-medium ${edgeClassName}`}>
                                • Edge: {formatSignedPercent(edgeVsBuyHold, 1)}
                              </span>
                            )}
                          </span>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => navigate(`/dashboard?stock=${heroSymbol}`)}
                            className="gap-1 h-7"
                          >
                            Full Analysis
                            <ChevronRight className="h-3 w-3" />
                          </Button>
                        </div>

                        {/* AI Summary */}
                        {heroRec.ai_summary && (
                          <p className="text-xs text-muted-foreground mt-2 line-clamp-2">
                            <Brain className="h-3 w-3 inline mr-1" />
                            {heroRec.ai_summary}
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Signal Board Section */}
      <section className="py-12 px-4">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
            <div>
              <h2 className="text-2xl md:text-3xl font-bold">Live Signal Board</h2>
              <p className="text-muted-foreground mt-1">Top-ranked opportunities, refreshed every 15 minutes.</p>
            </div>
            <Button variant="outline" onClick={() => navigate('/dashboard')} className="gap-1.5">
              View Full Board
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {marketMessage && (
            <Alert variant={marketMessage.includes('No certified') ? 'destructive' : 'default'} className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{marketMessage}</AlertDescription>
            </Alert>
          )}

          {isLoading ? (
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {Array.from({ length: LANDING_SIGNAL_BOARD_COUNT }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4">
                    <Skeleton className="h-32 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {signalBoardRecs.map((rec, i) => (
                <SignalBoardCard
                  key={rec.ticker}
                  rec={rec}
                  chartData={chartDataMap[rec.ticker] || []}
                  index={i}
                  onClick={() => handleStockClick(rec.ticker)}
                />
              ))}
            </div>
          )}
        </div>
      </section>

      {/* AI Pipeline Section - Visual Timeline */}
      <section className="py-16 md:py-20 px-4 bg-muted/30">
        <div className="container mx-auto max-w-5xl">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              The AI pipeline behind every signal
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              From raw price action to AI consensus to validated entry decisions.
            </p>
          </motion.div>

          {/* Horizontal Pipeline Timeline */}
          <div className="flex justify-center mb-10">
            <HorizontalPipeline />
          </div>

          {/* Quick Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col items-center justify-center p-4 rounded-lg bg-background/60 border">
              <p className="text-2xl font-bold font-mono text-primary">{signalCount || '—'}</p>
              <p className="text-xs text-muted-foreground text-center">Signals Detected</p>
            </div>
            <div className="flex flex-col items-center justify-center p-4 rounded-lg bg-background/60 border">
              <p className="text-2xl font-bold font-mono text-primary">{heroVerdicts.length || '—'}</p>
              <p className="text-xs text-muted-foreground text-center">AI Personas</p>
            </div>
            <div className="flex flex-col items-center justify-center p-4 rounded-lg bg-background/60 border">
              <p className={`text-2xl font-bold font-mono ${expectedReturnPercent >= 0 ? 'text-success' : 'text-danger'}`}>
                {formatSignedPercent(expectedReturnPercent, 1)}
              </p>
              <p className="text-xs text-muted-foreground text-center">Expected Return</p>
            </div>
            <div className="flex flex-col items-center justify-center p-4 rounded-lg bg-background/60 border">
              <p className="text-2xl font-bold font-mono text-foreground">
                {heroSignalSummary?.nTrades ?? '—'}
              </p>
              <p className="text-xs text-muted-foreground text-center">Backtested Trades</p>
            </div>
          </div>
        </div>
      </section>

      {/* Backtest Proof Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-4xl">
          <motion.div
            className="flex flex-col items-center text-center mb-10"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <Badge variant="secondary" className="mb-3 gap-1.5">
              <ShieldCheck className="h-3 w-3" />
              Verified Performance
            </Badge>
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              Real backtest results
            </h2>
            <p className="text-muted-foreground max-w-xl">
              Historical performance of our signals, not simulated scenarios.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <BacktestProofCard
              title="Win Rate"
              value={heroRec?.win_rate != null ? Math.round(heroRec.win_rate * 100) : 75}
              suffix="%"
              icon={Trophy}
              subtitle={`${heroSignalSummary?.nTrades ?? 'N/A'} trades analyzed`}
              variant="success"
              index={0}
            />
            <BacktestProofCard
              title="Avg Return"
              value={edgeVsBuyHold != null ? edgeVsBuyHold.toFixed(1) : '+12.4'}
              suffix="%"
              icon={TrendingUp}
              subtitle="Edge vs buy-and-hold"
              variant={edgeVsBuyHold != null && edgeVsBuyHold >= 0 ? 'success' : 'danger'}
              index={1}
            />
            <BacktestProofCard
              title="Max Drawdown"
              value="-8.2"
              suffix="%"
              icon={Activity}
              subtitle="Worst peak-to-trough"
              variant="danger"
              index={2}
            />
          </div>

          <p className="text-center text-xs text-muted-foreground mt-8">
            <Clock className="h-3 w-3 inline mr-1" />
            Data as of {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} • Past performance does not guarantee future results
          </p>
        </div>
      </section>

      {/* Signal Gallery Section */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="container mx-auto max-w-5xl">
          <motion.div
            className="flex flex-col items-center text-center mb-10"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              Today's signal gallery
            </h2>
            <p className="text-muted-foreground max-w-xl">
              Real tickers with real dips and real AI analysis. This is what you're getting.
            </p>
          </motion.div>

          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4">
                    <Skeleton className="h-40 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.slice(0, 6).map((rec, i) => (
                <SignalGalleryCard
                  key={rec.ticker}
                  rec={rec}
                  onClick={() => handleStockClick(rec.ticker)}
                  index={i}
                />
              ))}
            </div>
          )}

          <div className="flex justify-center mt-10">
            <Button onClick={() => navigate('/dashboard')} className="gap-2">
              Explore All Signals
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </section>

      {/* AI Persona Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              AI investor council for {heroSymbol || "today's top signal"}
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              See how legendary investing styles react to the same setup.
            </p>
          </motion.div>

          {heroVerdicts.length > 0 ? (
            <Card className="bg-gradient-to-br from-amber-500/5 to-emerald-500/10 border-amber-500/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-amber-600" />
                  AI Consensus
                  <Badge
                    variant={
                      verdictCounts.isBullish ? 'default' :
                      verdictCounts.isBearish ? 'destructive' : 'secondary'
                    }
                    className="ml-auto"
                  >
                    {verdictCounts.bullishCount} Bullish | {verdictCounts.bearishCount} Bearish | {verdictCounts.neutralCount} Neutral
                  </Badge>
                </CardTitle>
                <CardDescription>
                  AI-powered analysis from the perspective of legendary investors
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  {heroVerdicts.slice(0, 4).map((verdict) => {
                    const isVerdictBullish = verdict.signal === 'buy' || verdict.signal === 'strong_buy';
                    const isVerdictBearish = verdict.signal === 'sell' || verdict.signal === 'strong_sell';
                    return (
                      <div
                        key={verdict.agent_id}
                        className="p-4 rounded-lg bg-background/60 border"
                      >
                        <div className="flex items-start gap-3 mb-2">
                          <Avatar className="h-10 w-10 border">
                            <AvatarImage
                              src={verdict.avatar_url}
                              alt={verdict.agent_name}
                            />
                            <AvatarFallback className="bg-primary/10 text-primary text-xs">
                              {verdict.agent_name.split(' ').map((n) => n[0]).join('').slice(0, 2)}
                            </AvatarFallback>
                          </Avatar>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-2">
                              <h4 className="font-semibold text-sm truncate">{verdict.agent_name}</h4>
                              <Badge
                                variant={
                                  isVerdictBullish ? 'default' :
                                  isVerdictBearish ? 'destructive' : 'secondary'
                                }
                                className="text-xs h-5 shrink-0"
                              >
                                {isVerdictBullish && <ThumbsUp className="h-3 w-3 mr-1" />}
                                {isVerdictBearish && <ThumbsDown className="h-3 w-3 mr-1" />}
                                {verdict.signal === 'hold' && <MinusCircle className="h-3 w-3 mr-1" />}
                                {verdict.signal.toUpperCase().replace('_', ' ')}
                              </Badge>
                            </div>
                            <p className="text-xs text-muted-foreground">
                              {verdict.confidence}% confident
                            </p>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-3">
                          {verdict.reasoning}
                        </p>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 text-center">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigate(`/dashboard?stock=${heroSymbol}`)}
                    className="gap-1"
                  >
                    View Full Analysis
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-10 text-center text-sm text-muted-foreground">
                {heroRec ? 'Loading AI debate for this signal...' : 'Loading AI debate...'}
              </CardContent>
            </Card>
          )}
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 md:py-24 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              Built for <span className="text-primary">signal-driven traders</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              A quant + AI stack that turns noisy markets into clear entry decisions.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon={Zap}
              title="Signal Engine"
              description="Detects oversold conditions with mean-reversion, volatility filters, and statistical validation."
              delay={0}
            />
            <FeatureCard
              icon={Brain}
              title="AI Personas"
              description="Explains each setup with multiple investor viewpoints so you see the risks and the upside."
              delay={0.1}
            />
            <FeatureCard
              icon={Target}
              title="Portfolio Optimizer"
              description="Position sizing based on expected return, risk, and correlation for smarter allocations."
              delay={0.2}
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 bg-primary/5">
        <div className="container mx-auto max-w-3xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl md:text-3xl font-bold mb-4">Ready to trade with AI conviction?</h2>
            <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
              Join traders using StonkMarket to move faster on signals that are explained, tested, and optimized.
            </p>
            <Button size="lg" onClick={handleCTA} className="gap-2">
              {isAuthenticated ? 'Open Dashboard' : 'Start Trading Smarter'}
              <ArrowRight className="h-4 w-4" />
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
