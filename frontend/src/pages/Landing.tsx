import { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import {
  AreaChart,
  Area,
  Line,
  ComposedChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import { useTheme } from '@/context/ThemeContext';
import { useAuth } from '@/context/AuthContext';
import {
  getStockChart,
  getQuantRecommendations,
  type ChartDataPoint,
  type QuantRecommendation,
} from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { StockLogo } from '@/components/StockLogo';
import {
  ArrowRight,
  Target,
  Activity,
  LineChart,
  Sparkles,
  ChevronRight,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
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

// Featured stock card with chart - using quant recommendations
function FeaturedStockCard({
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
  
  // Calculate chart-based trend
  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    const first = chartData[0].close;
    const last = chartData[chartData.length - 1].close;
    return ((last - first) / first) * 100;
  }, [chartData]);
  
  const lastPrice = chartData.length > 0 ? chartData[chartData.length - 1].close : 0;
  const isUp = priceChange >= 0;
  
  // Enhanced mini chart data with peak reference
  const chartWithRef = useMemo(() => {
    const sliced = chartData.slice(-60);
    if (sliced.length === 0) return [];
    
    const maxPrice = Math.max(...sliced.map(p => p.close));
    return sliced.map((p, i) => ({ 
      x: i, 
      y: p.close,
      peak: maxPrice,
    }));
  }, [chartData]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
    >
      <Card
        className="cursor-pointer transition-all hover:shadow-lg hover:border-primary/50 overflow-hidden group"
        onClick={onClick}
      >
        <CardContent className="p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <StockLogo symbol={rec.ticker} size="md" />
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-lg">{rec.ticker}</span>
                  <Badge variant={getActionBadgeVariant(rec.action)} className="text-xs h-5 gap-1">
                    <ActionIcon action={rec.action} />
                    {rec.action}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground line-clamp-1">
                  {rec.name || rec.ticker}
                </p>
              </div>
            </div>
            <div className="text-right">
              {lastPrice > 0 && (
                <p className="font-mono font-bold text-lg">
                  ${lastPrice.toFixed(2)}
                </p>
              )}
              <p className={`text-sm font-medium ${isUp ? 'text-success' : 'text-danger'}`}>
                {isUp ? '+' : ''}{priceChange.toFixed(2)}%
              </p>
            </div>
          </div>

          {/* Mini chart with peak reference line */}
          <div className="h-20 -mx-2 mb-3">
            {chartWithRef.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartWithRef}>
                  <defs>
                    <linearGradient id={`gradient-${rec.ticker}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isUp ? colors.up : colors.down} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={isUp ? colors.up : colors.down} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  {/* Peak reference line (52w high) */}
                  <Line
                    type="linear"
                    dataKey="peak"
                    stroke={colors.up}
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    strokeOpacity={0.4}
                    dot={false}
                    isAnimationActive={false}
                  />
                  {/* Price area */}
                  <Area
                    type="monotone"
                    dataKey="y"
                    stroke={isUp ? colors.up : colors.down}
                    strokeWidth={2}
                    fill={`url(#gradient-${rec.ticker})`}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <Skeleton className="h-full w-full" />
            )}
          </div>

          {/* Quant metrics */}
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="text-center">
              <p className="text-muted-foreground text-xs">μ̂ (E[R])</p>
              <p className={`font-mono font-medium ${rec.mu_hat >= 0 ? 'text-success' : 'text-danger'}`}>
                {rec.mu_hat > 0 ? '+' : ''}{(rec.mu_hat * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Dip Score</p>
              <p className={`font-mono font-medium ${(rec.dip_score ?? 0) < 0 ? 'text-success' : 'text-muted-foreground'}`}>
                {rec.dip_score?.toFixed(2) ?? 'N/A'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-muted-foreground text-xs">Utility</p>
              <p className={`font-mono font-medium ${rec.marginal_utility >= 0 ? 'text-success' : 'text-danger'}`}>
                {(rec.marginal_utility * 1000).toFixed(2)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Live ticker component - uses quant recommendations
function LiveTicker({ recommendations, chartDataMap }: { 
  recommendations: QuantRecommendation[]; 
  chartDataMap: Record<string, ChartDataPoint[]>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState(0);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const { getActiveColors } = useTheme();
  const colors = getActiveColors();

  const tickerItems = recommendations.slice(0, 30);
  const speed = 25;

  useEffect(() => {
    if (tickerItems.length === 0) return;

    const doAnimate = (timestamp: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      const delta = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;

      if (contentRef.current) {
        const contentWidth = contentRef.current.scrollWidth / 2;
        setOffset((prev) => {
          const newOffset = prev + (speed * delta) / 1000;
          return newOffset >= contentWidth ? 0 : newOffset;
        });
      }
      animationRef.current = requestAnimationFrame(doAnimate);
    };

    animationRef.current = requestAnimationFrame(doAnimate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [tickerItems.length]);

  if (tickerItems.length === 0) return null;

  const items = [...tickerItems, ...tickerItems];

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden border-y border-border bg-muted/30 py-3"
    >
      <div
        ref={contentRef}
        className="flex gap-6 whitespace-nowrap"
        style={{ transform: `translateX(-${offset}px)` }}
      >
        {items.map((rec, i) => {
          // Get chart data for this symbol to calculate price
          const chart = chartDataMap[rec.ticker] || [];
          const lastPrice = chart.length > 0 ? chart[chart.length - 1].close : 0;
          
          return (
            <div key={`${rec.ticker}-${i}`} className="flex items-center gap-2 px-3">
              <span className="font-semibold">{rec.ticker}</span>
              {lastPrice > 0 && (
                <span className="font-mono text-sm">${lastPrice.toFixed(2)}</span>
              )}
              <span
                className="flex items-center gap-1 text-sm font-medium"
                style={{ color: rec.action === 'BUY' ? colors.up : rec.action === 'SELL' ? colors.down : 'inherit' }}
              >
                <ActionIcon action={rec.action} />
                {rec.action}
              </span>
              {rec.mu_hat !== 0 && (
                <span
                  className="font-mono text-xs"
                  style={{ color: rec.mu_hat > 0 ? colors.up : colors.down }}
                >
                  {rec.mu_hat > 0 ? '+' : ''}{(rec.mu_hat * 100).toFixed(1)}%
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
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

// Hero chart - larger showcase chart
function HeroChart({
  chartData,
  symbol,
  isLoading,
}: {
  chartData: ChartDataPoint[];
  symbol: string;
  isLoading: boolean;
}) {
  const { getActiveColors } = useTheme();
  const colors = getActiveColors();

  const displayData = useMemo(() => {
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
  const chartColor = isPositive ? colors.up : colors.down;

  if (isLoading) {
    return (
      <Card className="overflow-hidden">
        <CardHeader className="pb-2">
          <Skeleton className="h-6 w-32" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
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
              <CardDescription>Live Chart Preview</CardDescription>
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
      </CardHeader>
      <CardContent className="pb-4">
        <div className="h-64">
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
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
              />
              <Area
                type="monotone"
                dataKey="close"
                stroke={chartColor}
                strokeWidth={2}
                fill="url(#hero-gradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Stats component - uses quant recommendations
function Stats({ recommendations, portfolioStats }: { 
  recommendations: QuantRecommendation[];
  portfolioStats: {
    expectedReturn: number;
    expectedRisk: number;
    totalTrades: number;
  };
}) {
  const stats = useMemo(() => {
    const buyCount = recommendations.filter((r) => r.action === 'BUY').length;
    const avgMuHat = recommendations.length > 0
      ? recommendations.reduce((sum, r) => sum + r.mu_hat, 0) / recommendations.length * 100
      : 0;
    
    return { 
      count: recommendations.length,
      buyCount,
      avgMuHat: Math.round(avgMuHat * 100) / 100,
    };
  }, [recommendations]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="grid grid-cols-3 gap-4 md:gap-8"
    >
      <Card className="text-center">
        <CardContent className="pt-6 pb-4">
          <p className="text-3xl md:text-4xl font-bold text-primary">
            <AnimatedCounter value={stats.count} suffix="" />
          </p>
          <p className="text-sm text-muted-foreground mt-1">Assets Analyzed</p>
        </CardContent>
      </Card>
      <Card className="text-center">
        <CardContent className="pt-6 pb-4">
          <p className="text-3xl md:text-4xl font-bold text-success">
            <AnimatedCounter value={stats.buyCount} />
          </p>
          <p className="text-sm text-muted-foreground mt-1">Buy Signals</p>
        </CardContent>
      </Card>
      <Card className="text-center">
        <CardContent className="pt-6 pb-4">
          <p className="text-3xl md:text-4xl font-bold text-primary">
            {portfolioStats.expectedReturn > 0 ? '+' : ''}
            <AnimatedCounter value={Math.round(portfolioStats.expectedReturn * 10000) / 100} suffix="%" />
          </p>
          <p className="text-sm text-muted-foreground mt-1">Portfolio E[R]</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function Landing() {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const [recommendations, setRecommendations] = useState<QuantRecommendation[]>([]);
  const [portfolioStats, setPortfolioStats] = useState({
    expectedReturn: 0,
    expectedRisk: 0,
    totalTrades: 0,
  });
  const [chartDataMap, setChartDataMap] = useState<Record<string, ChartDataPoint[]>>({});
  const [heroChart, setHeroChart] = useState<ChartDataPoint[]>([]);
  const [heroSymbol, setHeroSymbol] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingHero, setIsLoadingHero] = useState(true);

  // Fetch quant recommendations data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Fetch quant engine recommendations
        const data = await getQuantRecommendations(1000, 20);
        setRecommendations(data.recommendations);
        setPortfolioStats({
          expectedReturn: data.expected_portfolio_return,
          expectedRisk: data.expected_portfolio_risk,
          totalTrades: data.total_trades,
        });

        // Load charts for top recommendations
        const topRecs = data.recommendations.slice(0, 6);
        const chartPromises = topRecs.map(async (rec: QuantRecommendation) => {
          try {
            const chart = await getStockChart(rec.ticker, 90);
            return { symbol: rec.ticker, chart };
          } catch {
            return { symbol: rec.ticker, chart: [] as ChartDataPoint[] };
          }
        });

        const results = await Promise.all(chartPromises);
        const chartMap: Record<string, ChartDataPoint[]> = {};
        results.forEach(({ symbol, chart }: { symbol: string; chart: ChartDataPoint[] }) => {
          chartMap[symbol] = chart;
        });
        setChartDataMap(chartMap);

        // Set hero chart to first recommendation with a BUY action, or first overall
        const buyRec = data.recommendations.find(r => r.action === 'BUY') || data.recommendations[0];
        if (buyRec) {
          setHeroSymbol(buyRec.ticker);
          if (chartMap[buyRec.ticker]) {
            setHeroChart(chartMap[buyRec.ticker]);
          } else {
            const heroData = await getStockChart(buyRec.ticker, 180);
            setHeroChart(heroData);
          }
        }
      } catch (err) {
        console.error('Failed to load landing data:', err);
      } finally {
        setIsLoading(false);
        setIsLoadingHero(false);
      }
    };

    loadData();
  }, []);

  const handleCTA = useCallback(() => {
    navigate(isAuthenticated ? '/dashboard' : '/login');
  }, [navigate, isAuthenticated]);

  const handleStockClick = useCallback(
    (symbol: string) => {
      navigate(`/dashboard?symbol=${symbol}`);
    },
    [navigate]
  );

  const featuredRecs = recommendations.slice(0, 6);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-16 md:py-24 px-4">
        <div className="container mx-auto max-w-6xl">
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
                  AI-Powered Analysis
                </Badge>
              </motion.div>

              <motion.h1
                className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-6"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                Find the dip.
                <br />
                <span className="text-primary">Catch the wave.</span>
              </motion.h1>

              <motion.p
                className="text-lg md:text-xl text-muted-foreground mb-8 max-w-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                Quantitative stock analysis identifies oversold conditions before the bounce.
                Real-time signals powered by mean-variance optimization.
              </motion.p>

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
            </div>

            {/* Right: Hero Chart */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <HeroChart chartData={heroChart} symbol={heroSymbol} isLoading={isLoadingHero} />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Live Ticker */}
      <LiveTicker recommendations={recommendations} chartDataMap={chartDataMap} />

      {/* Stats Section */}
      <section className="py-12 px-4">
        <div className="container mx-auto max-w-4xl">
          <Stats recommendations={recommendations} portfolioStats={portfolioStats} />
        </div>
      </section>

      {/* Featured Stocks Section */}
      <section className="py-12 px-4 bg-muted/30">
        <div className="container mx-auto max-w-6xl">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl md:text-3xl font-bold">Today's Top Opportunities</h2>
              <p className="text-muted-foreground mt-1">Quant-ranked stocks by expected return</p>
            </div>
            <Button variant="ghost" onClick={() => navigate('/dashboard')} className="gap-1.5">
              View All
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {isLoading ? (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4">
                    <Skeleton className="h-32 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {featuredRecs.map((rec, i) => (
                <FeaturedStockCard
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
              Built for <span className="text-primary">serious traders</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Professional-grade tools for quantitative stock analysis and portfolio optimization.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon={LineChart}
              title="Real-Time Analysis"
              description="Monitor stocks with live momentum signals and technical indicators. Charts update automatically with the latest market data."
              delay={0}
            />
            <FeatureCard
              icon={Target}
              title="Dip Detection"
              description="Our algorithms identify oversold conditions using 52-week highs, volume patterns, and AI-powered sentiment analysis."
              delay={0.1}
            />
            <FeatureCard
              icon={Activity}
              title="Portfolio Optimization"
              description="Mean-variance optimization calculates risk-adjusted expected returns. Build efficient portfolios with quantitative precision."
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
            <h2 className="text-2xl md:text-3xl font-bold mb-4">Ready to find your edge?</h2>
            <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
              Join traders using StonkMarket to identify market opportunities before they disappear.
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
