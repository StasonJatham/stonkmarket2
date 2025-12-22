import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { PanInfo } from 'framer-motion';
import { 
  getDipCards, 
  voteDip,
  getTopSuggestions,
  voteForSuggestion,
  getStockChart,
  type DipCard, 
  type VoteType,
  type TopSuggestion,
  type ChartDataPoint
} from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  X, 
  Check, 
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Users,
  ThumbsUp,
  ThumbsDown,
  Lightbulb,
  DollarSign,
  MapPin,
  Clock,
  Flame
} from 'lucide-react';

// Swipe threshold in pixels
const SWIPE_THRESHOLD = 100;

// Spring transition for smooth animations
const springTransition = {
  type: "spring" as const,
  stiffness: 300,
  damping: 25,
};

// Exit animation variants based on direction - not used, removing to avoid lint errors
// const exitVariants = {
//   left: { x: -400, opacity: 0, rotate: -20, transition: { ...springTransition, duration: 0.4 } },
//   right: { x: 400, opacity: 0, rotate: 20, transition: { ...springTransition, duration: 0.4 } },
// };

// Vote button animations
const voteButtonVariants = {
  tap: { scale: 0.9 },
  hover: { scale: 1.1 },
  initial: { scale: 1 },
};

function formatPrice(value: number): string {
  return new Intl.NumberFormat('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

function formatDipPct(value: number): string {
  if (value === 0) return '0%';
  if (value < 1) return `-${value.toFixed(1)}%`;
  return `-${value.toFixed(0)}%`;
}

// Custom tooltip for chart
function ChartTooltip({ active, payload }: { active?: boolean; payload?: Array<{ value: number }> }) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg px-3 py-2 shadow-lg">
        <p className="text-sm font-medium">{formatPrice(payload[0].value)}</p>
      </div>
    );
  }
  return null;
}

// Sentiment indicator
function SentimentBar({ buyPct, sellPct }: { buyPct: number; sellPct: number }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <ThumbsUp className="w-3 h-3 text-success" />
          {buyPct.toFixed(0)}%
        </span>
        <span className="flex items-center gap-1">
          {sellPct.toFixed(0)}%
          <ThumbsDown className="w-3 h-3 text-danger" />
        </span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden flex">
        <div 
          className="bg-success transition-all duration-300" 
          style={{ width: `${buyPct}%` }}
        />
        <div 
          className="bg-danger transition-all duration-300" 
          style={{ width: `${sellPct}%` }}
        />
      </div>
    </div>
  );
}

// Individual swipeable card - Tinder-style!
function SwipeableCard({ 
  card, 
  onVote, 
  isTop,
  chartData,
}: { 
  card: DipCard; 
  onVote: (vote: VoteType) => void;
  isTop: boolean;
  chartData?: ChartDataPoint[];
}) {
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-20, 20]);
  const buyOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
  const sellOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
  const scale = useTransform(x, [-200, 0, 200], [0.95, 1, 0.95]);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > SWIPE_THRESHOLD) {
      onVote('buy');
    } else if (info.offset.x < -SWIPE_THRESHOLD) {
      onVote('sell');
    }
  };

  const buyPct = card.vote_counts.buy + card.vote_counts.sell > 0
    ? (card.vote_counts.buy / (card.vote_counts.buy + card.vote_counts.sell)) * 100
    : 50;
  const sellPct = 100 - buyPct;

  // Prepare chart data for mini chart
  const miniChartData = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    return chartData.slice(-60).map((p, i) => ({ x: i, y: p.close }));
  }, [chartData]);

  // Calculate "age" (days since peak)
  const stockAge = card.days_below || 0;

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none"
      style={{ x, rotate, scale, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      dragTransition={{ bounceStiffness: 300, bounceDamping: 20 }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.9, opacity: 0, y: 20 }}
      animate={{ 
        scale: isTop ? 1 : 0.92, 
        opacity: isTop ? 1 : 0.6,
        y: isTop ? 0 : 10,
      }}
      transition={springTransition}
      exit={{ 
        x: 400, 
        opacity: 0,
        rotate: 15,
      }}
      whileHover={isTop ? { scale: 1.02 } : {}}
    >
      {/* Swipe indicators - Match/Nope style */}
      <motion.div 
        className="absolute left-4 top-4 z-20 pointer-events-none"
        style={{ opacity: sellOpacity }}
      >
        <div className="border-4 border-danger rounded-lg px-4 py-1 rotate-[-20deg]">
          <span className="text-danger font-black text-3xl tracking-wider">NOPE</span>
        </div>
      </motion.div>
      <motion.div 
        className="absolute right-4 top-4 z-20 pointer-events-none"
        style={{ opacity: buyOpacity }}
      >
        <div className="border-4 border-success rounded-lg px-4 py-1 rotate-[20deg]">
          <span className="text-success font-black text-3xl tracking-wider">BUY!</span>
        </div>
      </motion.div>

      <Card className="h-full flex flex-col overflow-hidden bg-card border-0 shadow-2xl rounded-3xl">
        {/* Chart as "Profile Photo" - takes up top half */}
        <div className="relative h-[45%] bg-gradient-to-b from-muted/80 to-muted/20 overflow-hidden">
          {miniChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={miniChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                <defs>
                  <linearGradient id={`tinder-gradient-${card.symbol}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--danger)" stopOpacity={0.6} />
                    <stop offset="100%" stopColor="var(--danger)" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="x" hide />
                <YAxis hide domain={['dataMin', 'dataMax']} />
                <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'var(--muted-foreground)', strokeWidth: 1, strokeDasharray: '4 4' }} />
                <Area
                  type="monotone"
                  dataKey="y"
                  stroke="var(--danger)"
                  strokeWidth={3}
                  fill={`url(#tinder-gradient-${card.symbol})`}
                  isAnimationActive={true}
                  animationDuration={600}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center">
              <BarChart3 className="w-16 h-16 text-muted-foreground/30" />
            </div>
          )}
          
          {/* Gradient overlay for text readability */}
          <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-card to-transparent" />
          
          {/* Stock "name" overlay at bottom of photo */}
          <div className="absolute bottom-4 left-4 right-4">
            <div className="flex items-end justify-between">
              <div>
                <h2 className="text-4xl font-bold text-foreground drop-shadow-lg flex items-center gap-2">
                  {card.symbol}
                  {card.ai_rating === 'strong_buy' || card.ai_rating === 'buy' ? (
                    <Badge className="bg-blue-500 text-white border-0">
                      <Check className="w-3 h-3 mr-0.5" />
                      Verified
                    </Badge>
                  ) : null}
                </h2>
                {stockAge > 0 && (
                  <div className="flex items-center gap-3 text-muted-foreground mt-1">
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {stockAge} days in the dip
                    </span>
                  </div>
                )}
              </div>
              {card.dip_pct > 0 && (
                <div className="text-right">
                  <Badge variant="destructive" className="text-lg px-3 py-1 font-bold">
                    {formatDipPct(card.dip_pct)}
                  </Badge>
                </div>
              )}
            </div>
          </div>
          
          {/* Hot badge for popular stocks */}
          {card.vote_counts.buy > 5 && (
            <div className="absolute top-4 right-4">
              <Badge className="bg-gradient-to-r from-orange-500 to-pink-500 text-white border-0 gap-1">
                <Flame className="w-3 h-3" />
                Popular
              </Badge>
            </div>
          )}
        </div>

        {/* Profile Info - Tinder style */}
        <CardContent className="flex-1 p-5 flex flex-col gap-3 overflow-y-auto">
          {/* Name and sector like job title */}
          <div>
            <p className="text-lg font-medium">{card.name || card.symbol}</p>
            {card.sector && (
              <p className="text-muted-foreground flex items-center gap-1">
                <MapPin className="w-4 h-4" />
                {card.sector}
              </p>
            )}
          </div>

          {/* Bio - the star of the show! */}
          {card.tinder_bio ? (
            <div className="flex-1 py-3 border-y border-border/50">
              <p className="text-[15px] leading-relaxed">{card.tinder_bio.replace(/^"|"$/g, '')}</p>
            </div>
          ) : (
            <div className="flex-1 py-3 border-y border-border/50">
              <p className="text-[15px] leading-relaxed text-muted-foreground italic">
                Looking for investors who appreciate a good dip. 📉 {card.dip_pct > 0 ? `Currently ${formatDipPct(card.dip_pct)} off my peak - that's basically a sale, right?` : ''} Swipe right if you see my true value! 💸
              </p>
            </div>
          )}

          {/* Quick stats like Tinder interests */}
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="rounded-full">
              💰 {formatPrice(card.current_price)}
            </Badge>
            <Badge variant="outline" className="rounded-full">
              📈 ATH: {formatPrice(card.ref_high)}
            </Badge>
          </div>

          {/* Community sentiment as "mutual friends" */}
          <div className="mt-auto pt-2">
            <div className="flex items-center gap-2 mb-2 text-sm text-muted-foreground">
              <Users className="w-4 h-4" />
              <span>{card.vote_counts.buy + card.vote_counts.sell} investors voted</span>
              {buyPct > 60 && <span className="text-success">• Mostly bullish 🐂</span>}
              {sellPct > 60 && <span className="text-danger">• Mostly bearish 🐻</span>}
            </div>
            <SentimentBar buyPct={buyPct} sellPct={sellPct} />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// Suggestion card for voting mode
function SuggestionSwipeCard({ 
  suggestion, 
  onVote, 
  isTop,
}: { 
  suggestion: TopSuggestion; 
  onVote: (approve: boolean) => void;
  isTop: boolean;
}) {
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-15, 15]);
  const approveOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
  const skipOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > SWIPE_THRESHOLD) {
      onVote(true);
    } else if (info.offset.x < -SWIPE_THRESHOLD) {
      onVote(false);
    }
  };

  const scale = useTransform(x, [-200, 0, 200], [0.95, 1, 0.95]);

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none"
      style={{ x, rotate, scale, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      dragTransition={{ bounceStiffness: 300, bounceDamping: 20 }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.9, opacity: 0, y: 20 }}
      animate={{ 
        scale: isTop ? 1 : 0.92, 
        opacity: isTop ? 1 : 0.6,
        y: isTop ? 0 : 10,
      }}
      transition={springTransition}
      exit={{ 
        x: 400, 
        opacity: 0,
        rotate: 15,
      }}
      whileHover={isTop ? { scale: 1.02 } : {}}
    >
      {/* Swipe indicators */}
      <motion.div 
        className="absolute -left-4 top-1/2 -translate-y-1/2 z-20 pointer-events-none"
        style={{ opacity: skipOpacity }}
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
      >
        <motion.div 
          className="bg-muted text-muted-foreground p-3 rounded-full shadow-lg"
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <X className="w-8 h-8" />
        </motion.div>
      </motion.div>
      <motion.div 
        className="absolute -right-4 top-1/2 -translate-y-1/2 z-20 pointer-events-none"
        style={{ opacity: approveOpacity }}
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
      >
        <motion.div 
          className="bg-success text-white p-3 rounded-full shadow-lg"
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          <ThumbsUp className="w-8 h-8" />
        </motion.div>
      </motion.div>

      <Card className="h-full flex flex-col overflow-hidden bg-card border-border/50 shadow-xl hover:shadow-2xl transition-shadow duration-300">
        <CardContent className="flex-1 p-6 flex flex-col">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-3xl font-bold">{suggestion.symbol}</h2>
                <Badge variant="secondary" className="font-normal">
                  Suggestion
                </Badge>
              </div>
              <p className="text-muted-foreground mt-1">
                {suggestion.name || suggestion.symbol}
              </p>
            </div>
            <Badge variant="outline" className="bg-chart-4/20 text-chart-4 border-chart-4/30">
              <ThumbsUp className="w-3 h-3 mr-1" />
              {suggestion.vote_count} votes
            </Badge>
          </div>

          {/* Suggestion info */}
          <div className="flex-1 flex flex-col gap-4">
            <div className="bg-muted/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Lightbulb className="w-4 h-4 text-chart-4" />
                <span className="text-sm font-medium">Why this stock?</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Community suggested stock awaiting approval. Vote to add it to the tracking list!
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">Current Votes</p>
                <p className="text-xl font-semibold">{suggestion.vote_count}</p>
              </div>
              <div className="bg-muted/30 rounded-lg p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">Status</p>
                <Badge variant="secondary" className="mt-1">Pending</Badge>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="text-center text-sm text-muted-foreground border-t pt-4 mt-4">
            <p>Swipe right to vote, left to skip</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

type SwipeMode = 'dips' | 'suggestions';

export function DipSwipePage() {
  const [mode, setMode] = useState<SwipeMode>('dips');
  const [cards, setCards] = useState<DipCard[]>([]);
  const [suggestions, setSuggestions] = useState<TopSuggestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isVoting, setIsVoting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [votedCards, setVotedCards] = useState<Set<string>>(new Set());
  const [chartDataMap, setChartDataMap] = useState<Record<string, ChartDataPoint[]>>({});

  const loadCards = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (mode === 'dips') {
        const response = await getDipCards(true);
        setCards(response.cards);
      } else {
        const data = await getTopSuggestions(50);
        setSuggestions(data);
      }
      setCurrentIndex(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cards');
    } finally {
      setIsLoading(false);
    }
  }, [mode]);

  useEffect(() => {
    loadCards();
  }, [loadCards, mode]);

  // Reset index when mode changes
  useEffect(() => {
    setCurrentIndex(0);
    setVotedCards(new Set());
  }, [mode]);

  const currentCard = useMemo(() => cards[currentIndex], [cards, currentIndex]);
  const nextCard = useMemo(() => cards[currentIndex + 1], [cards, currentIndex]);
  const currentSuggestion = useMemo(() => suggestions[currentIndex], [suggestions, currentIndex]);
  const nextSuggestion = useMemo(() => suggestions[currentIndex + 1], [suggestions, currentIndex]);
  const totalItems = mode === 'dips' ? cards.length : suggestions.length;

  // Fetch chart data for current and next cards
  useEffect(() => {
    if (mode !== 'dips') return;
    
    const symbolsToFetch = [currentCard?.symbol, nextCard?.symbol].filter((s): s is string => 
      !!s && !chartDataMap[s]
    );
    
    if (symbolsToFetch.length === 0) return;
    
    symbolsToFetch.forEach(async (symbol) => {
      try {
        const data = await getStockChart(symbol, 90);
        setChartDataMap(prev => ({ ...prev, [symbol]: data }));
      } catch (err) {
        console.error(`Failed to fetch chart for ${symbol}:`, err);
      }
    });
  }, [mode, currentCard?.symbol, nextCard?.symbol, chartDataMap]);

  const handleVote = useCallback(async (vote: VoteType) => {
    if (!currentCard || isVoting) return;
    
    setIsVoting(true);
    try {
      await voteDip(currentCard.symbol, vote);
      setVotedCards(prev => new Set([...prev, currentCard.symbol]));
      setCurrentIndex(prev => prev + 1);
    } catch (err) {
      console.error('Vote failed:', err);
      // Still advance to next card
      setCurrentIndex(prev => prev + 1);
    } finally {
      setIsVoting(false);
    }
  }, [currentCard, isVoting]);

  const handleSuggestionVote = useCallback(async (approve: boolean) => {
    if (!currentSuggestion || isVoting) return;
    
    setIsVoting(true);
    try {
      if (approve) {
        await voteForSuggestion(currentSuggestion.symbol);
      }
      setVotedCards(prev => new Set([...prev, currentSuggestion.symbol]));
      setCurrentIndex(prev => prev + 1);
    } catch (err) {
      console.error('Vote failed:', err);
      setCurrentIndex(prev => prev + 1);
    } finally {
      setIsVoting(false);
    }
  }, [currentSuggestion, isVoting]);

  const handleSkip = useCallback(() => {
    setCurrentIndex(prev => Math.min(prev + 1, totalItems));
  }, [totalItems]);

  const handlePrevious = useCallback(() => {
    setCurrentIndex(prev => Math.max(prev - 1, 0));
  }, []);

  // Mode selector component
  const ModeSelector = () => (
    <Tabs value={mode} onValueChange={(v) => setMode(v as SwipeMode)} className="w-full mb-4">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="dips" className="gap-2">
          <DollarSign className="w-4 h-4" />
          Buy/Sell Dips
        </TabsTrigger>
        <TabsTrigger value="suggestions" className="gap-2">
          <Lightbulb className="w-4 h-4" />
          Vote Suggestions
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );

  // Loading state
  if (isLoading) {
    return (
      <div className="container max-w-lg mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
          <p className="text-muted-foreground">
            {mode === 'dips' ? 'Swipe right to buy, left to pass' : 'Vote on stock suggestions'}
          </p>
        </div>
        <ModeSelector />
        <Card className="h-[600px]">
          <CardContent className="p-6 space-y-4">
            <Skeleton className="h-8 w-24" />
            <Skeleton className="h-6 w-48" />
            <div className="grid grid-cols-2 gap-4">
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
            </div>
            <Skeleton className="h-32" />
            <Skeleton className="h-20" />
          </CardContent>
        </Card>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="container max-w-lg mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">DipSwipe</h1>
          <ModeSelector />
          <Card className="p-8">
            <p className="text-danger mb-4">{error}</p>
            <Button onClick={loadCards}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Try Again
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  // No more cards
  if (currentIndex >= totalItems) {
    return (
      <div className="container max-w-lg mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">DipSwipe</h1>
          <ModeSelector />
          <Card className="p-8">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <h2 className="text-xl font-semibold mb-2">All caught up!</h2>
            <p className="text-muted-foreground mb-4">
              {mode === 'dips' 
                ? `You've reviewed all ${cards.length} stocks in the dip.`
                : `You've reviewed all ${suggestions.length} suggestions.`}
            </p>
            <p className="text-sm text-muted-foreground mb-6">
              Voted on {votedCards.size} items
            </p>
            <Button onClick={loadCards}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Start Over
            </Button>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="container max-w-lg mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-4">
        <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
        <p className="text-muted-foreground">
          {mode === 'dips' ? 'Swipe right to buy, left to pass' : 'Vote on community suggestions'}
        </p>
      </div>

      {/* Mode Selector */}
      <ModeSelector />

      {/* Progress */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <Badge variant="outline">
          {currentIndex + 1} / {totalItems}
        </Badge>
        <Badge variant="secondary">
          {votedCards.size} voted
        </Badge>
      </div>

      {/* Card stack */}
      <div className="relative h-[600px] mb-6">
        <AnimatePresence mode="popLayout">
          {mode === 'dips' ? (
            <>
              {nextCard && (
                <SwipeableCard
                  key={nextCard.symbol}
                  card={nextCard}
                  chartData={chartDataMap[nextCard.symbol]}
                  onVote={() => {}}
                  isTop={false}
                />
              )}
              {currentCard && (
                <SwipeableCard
                  key={currentCard.symbol}
                  card={currentCard}
                  chartData={chartDataMap[currentCard.symbol]}
                  onVote={handleVote}
                  isTop={true}
                />
              )}
            </>
          ) : (
            <>
              {nextSuggestion && (
                <SuggestionSwipeCard
                  key={nextSuggestion.symbol}
                  suggestion={nextSuggestion}
                  onVote={() => {}}
                  isTop={false}
                />
              )}
              {currentSuggestion && (
                <SuggestionSwipeCard
                  key={currentSuggestion.symbol}
                  suggestion={currentSuggestion}
                  onVote={handleSuggestionVote}
                  isTop={true}
                />
              )}
            </>
          )}
        </AnimatePresence>
      </div>

      {/* Action buttons */}
      <div className="flex items-center justify-center gap-4">
        <Button
          variant="outline"
          size="icon"
          className="w-12 h-12 rounded-full"
          onClick={handlePrevious}
          disabled={currentIndex === 0}
        >
          <ChevronLeft className="w-6 h-6" />
        </Button>
        
        {mode === 'dips' ? (
          <>
            <motion.button
              className="w-16 h-16 rounded-full bg-danger text-white flex items-center justify-center shadow-lg"
              variants={voteButtonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => handleVote('sell')}
              disabled={isVoting}
            >
              <X className="w-8 h-8" />
            </motion.button>
            
            <Button
              variant="ghost"
              size="icon"
              className="w-12 h-12 rounded-full"
              onClick={handleSkip}
            >
              <ChevronRight className="w-6 h-6" />
            </Button>
            
            <motion.button
              className="w-16 h-16 rounded-full bg-success text-white flex items-center justify-center shadow-lg"
              variants={voteButtonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => handleVote('buy')}
              disabled={isVoting}
            >
              <Check className="w-8 h-8" />
            </motion.button>
          </>
        ) : (
          <>
            <motion.button
              className="w-16 h-16 rounded-full bg-muted text-muted-foreground flex items-center justify-center shadow-lg"
              variants={voteButtonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => handleSuggestionVote(false)}
              disabled={isVoting}
            >
              <X className="w-8 h-8" />
            </motion.button>
            
            <Button
              variant="ghost"
              size="icon"
              className="w-12 h-12 rounded-full"
              onClick={handleSkip}
            >
              <ChevronRight className="w-6 h-6" />
            </Button>
            
            <motion.button
              className="w-16 h-16 rounded-full bg-success text-white flex items-center justify-center shadow-lg"
              variants={voteButtonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => handleSuggestionVote(true)}
              disabled={isVoting}
            >
              <ThumbsUp className="w-8 h-8" />
            </motion.button>
          </>
        )}

        <Button
          variant="outline"
          size="icon"
          className="w-12 h-12 rounded-full"
          onClick={loadCards}
        >
          <RefreshCw className="w-5 h-5" />
        </Button>
      </div>

      {/* Instructions */}
      <div className="text-center mt-8 text-sm text-muted-foreground">
        <p>Drag card left or right, or use buttons below</p>
      </div>
    </div>
  );
}
