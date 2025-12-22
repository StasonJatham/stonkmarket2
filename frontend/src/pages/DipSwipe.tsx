import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import { useIsMobile } from '@/hooks/useIsMobile';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis } from 'recharts';
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
  Building2,
  Calendar,
  Flame,
  TrendingDown,
  Award
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

function formatCompactNumber(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toString();
}

// Get stock logo URL (free API)
function getStockLogoUrl(symbol: string): string {
  // Using logo.clearbit.com or a fallback
  const cleanSymbol = symbol.replace('.', '-').toLowerCase();
  // Map common symbols to company domains
  const domainMap: Record<string, string> = {
    'aapl': 'apple.com',
    'msft': 'microsoft.com',
    'googl': 'google.com',
    'goog': 'google.com',
    'amzn': 'amazon.com',
    'meta': 'meta.com',
    'tsla': 'tesla.com',
    'nvda': 'nvidia.com',
    'amd': 'amd.com',
    'intc': 'intel.com',
    'nflx': 'netflix.com',
    'crm': 'salesforce.com',
    'orcl': 'oracle.com',
    'adbe': 'adobe.com',
    'csco': 'cisco.com',
    'ibm': 'ibm.com',
    'pypl': 'paypal.com',
    'dis': 'disney.com',
    'nke': 'nike.com',
    'ko': 'coca-cola.com',
    'pep': 'pepsico.com',
    'wmt': 'walmart.com',
    'jpm': 'jpmorgan.com',
    'v': 'visa.com',
    'ma': 'mastercard.com',
    'bac': 'bankofamerica.com',
    'gs': 'goldmansachs.com',
    'ms': 'morganstanley.com',
  };
  const domain = domainMap[cleanSymbol] || `${cleanSymbol}.com`;
  return `https://logo.clearbit.com/${domain}`;
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
  onSwipeComplete,
}: { 
  card: DipCard; 
  onVote: (vote: VoteType) => void;
  isTop: boolean;
  chartData?: ChartDataPoint[];
  onSwipeComplete?: () => void;
}) {
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-15, 15]);
  const buyOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
  const sellOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
  const [logoError, setLogoError] = useState(false);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > SWIPE_THRESHOLD) {
      onVote('buy');
    } else if (info.offset.x < -SWIPE_THRESHOLD) {
      onVote('sell');
    }
  };

  const totalVotes = card.vote_counts.buy + card.vote_counts.sell;
  const buyPct = totalVotes > 0 ? (card.vote_counts.buy / totalVotes) * 100 : 50;
  const sellPct = 100 - buyPct;

  // Prepare chart data for mini chart
  const miniChartData = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    return chartData.slice(-60).map((p, i) => ({ x: i, y: p.close }));
  }, [chartData]);

  // Days in dip (like "age")
  const daysInDip = card.days_below || 0;

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none"
      style={{ x, rotate, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      dragTransition={{ bounceStiffness: 300, bounceDamping: 20 }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.95, opacity: 0, y: 30 }}
      animate={{ 
        scale: 1, 
        opacity: isTop ? 1 : 0, // Hide non-top cards completely
        y: 0,
      }}
      transition={springTransition}
      exit={{ 
        x: x.get() > 0 ? 400 : -400, 
        opacity: 0,
        rotate: x.get() > 0 ? 15 : -15,
        transition: { duration: 0.3 }
      }}
      onAnimationComplete={(def) => {
        if (def === 'exit' && onSwipeComplete) {
          onSwipeComplete();
        }
      }}
    >
      {/* Swipe indicators - NOPE/BUY overlay */}
      <motion.div 
        className="absolute left-4 top-6 z-20 pointer-events-none"
        style={{ opacity: sellOpacity }}
      >
        <div className="border-4 border-danger rounded-lg px-4 py-1 rotate-[-15deg] bg-background/80">
          <span className="text-danger font-black text-2xl tracking-wider">NOPE</span>
        </div>
      </motion.div>
      <motion.div 
        className="absolute right-4 top-6 z-20 pointer-events-none"
        style={{ opacity: buyOpacity }}
      >
        <div className="border-4 border-success rounded-lg px-4 py-1 rotate-[15deg] bg-background/80">
          <span className="text-success font-black text-2xl tracking-wider">BUY!</span>
        </div>
      </motion.div>

      <Card className="h-full flex flex-col overflow-hidden bg-card border-0 shadow-2xl rounded-3xl">
        {/* Header Row - Symbol, Name, Logo, Age */}
        <div className="shrink-0 p-4 pb-2 flex items-start justify-between">
          <div className="flex items-center gap-3">
            {/* Company Logo */}
            <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center overflow-hidden border border-border/50">
              {!logoError ? (
                <img 
                  src={getStockLogoUrl(card.symbol)}
                  alt={card.symbol}
                  className="w-full h-full object-contain p-1"
                  onError={() => setLogoError(true)}
                />
              ) : (
                <Building2 className="w-6 h-6 text-muted-foreground" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="font-bold text-sm">
                  {card.symbol}
                </Badge>
                {card.ai_rating === 'strong_buy' || card.ai_rating === 'buy' ? (
                  <Badge className="bg-blue-500 text-white border-0 text-xs">
                    <Award className="w-3 h-3 mr-0.5" />
                    AI Pick
                  </Badge>
                ) : null}
              </div>
              <p className="text-sm text-foreground font-medium mt-0.5 line-clamp-1">
                {card.name || card.symbol}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-muted-foreground flex items-center gap-1 justify-end">
              <Calendar className="w-3 h-3" />
              {daysInDip}d in dip
            </p>
            {card.vote_counts.buy > 5 && (
              <Badge className="bg-gradient-to-r from-orange-500 to-pink-500 text-white border-0 text-xs mt-1">
                <Flame className="w-3 h-3 mr-0.5" />
                Hot
              </Badge>
            )}
          </div>
        </div>

        {/* Chart as "Profile Photo" */}
        <div className="relative flex-1 min-h-[180px] max-h-[240px] bg-gradient-to-b from-muted/30 to-transparent mx-4 rounded-xl overflow-hidden border border-border/30">
          {miniChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={miniChartData} margin={{ top: 20, right: 10, left: 10, bottom: 10 }}>
                <defs>
                  <linearGradient id={`tinder-gradient-${card.symbol}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--danger)" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="var(--danger)" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="x" hide />
                <YAxis hide domain={['dataMin', 'dataMax']} />
                <Area
                  type="monotone"
                  dataKey="y"
                  stroke="var(--danger)"
                  strokeWidth={2}
                  fill={`url(#tinder-gradient-${card.symbol})`}
                  isAnimationActive={true}
                  animationDuration={600}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center">
              <BarChart3 className="w-12 h-12 text-muted-foreground/30" />
            </div>
          )}
          
          {/* Dip percentage badge overlay */}
          <div className="absolute top-3 right-3">
            <Badge variant="destructive" className="text-base px-2.5 py-0.5 font-bold shadow-lg">
              <TrendingDown className="w-3.5 h-3.5 mr-1" />
              {formatDipPct(card.dip_pct)}
            </Badge>
          </div>
        </div>

        {/* Key Stats Chips */}
        <div className="shrink-0 px-4 py-2 flex flex-wrap gap-1.5">
          <Badge variant="outline" className="rounded-full text-xs">
            💰 {formatPrice(card.current_price)}
          </Badge>
          <Badge variant="outline" className="rounded-full text-xs">
            📈 Peak: {formatPrice(card.ref_high)}
          </Badge>
          {card.sector && (
            <Badge variant="outline" className="rounded-full text-xs">
              🏢 {card.sector}
            </Badge>
          )}
        </div>

        {/* Bio */}
        <div className="shrink-0 px-4 py-2 border-t border-border/30">
          <p className="text-sm leading-relaxed line-clamp-3 text-muted-foreground">
            {card.tinder_bio 
              ? card.tinder_bio.replace(/^"|"$/g, '')
              : `Looking for investors who appreciate a good dip. 📉 Currently ${formatDipPct(card.dip_pct)} off my peak. Swipe right if you see my potential! 💸`
            }
          </p>
        </div>

        {/* Vote Stats & Sentiment */}
        <div className="shrink-0 p-4 pt-2 border-t border-border/30 bg-muted/20">
          {/* Stats Row */}
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Users className="w-3.5 h-3.5" />
                {totalVotes} votes
              </span>
              <span className="flex items-center gap-1 text-success">
                <ThumbsUp className="w-3.5 h-3.5" />
                {card.vote_counts.buy}
              </span>
              <span className="flex items-center gap-1 text-danger">
                <ThumbsDown className="w-3.5 h-3.5" />
                {card.vote_counts.sell}
              </span>
            </div>
            {buyPct > 60 && <span className="text-success font-medium">🐂 Bullish</span>}
            {sellPct > 60 && <span className="text-danger font-medium">🐻 Bearish</span>}
          </div>
          
          {/* Sentiment Bar */}
          <SentimentBar buyPct={buyPct} sellPct={sellPct} />
        </div>
      </Card>
    </motion.div>
  );
}

// Suggestion card for voting mode - Tinder-style!
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
  const [logoError, setLogoError] = useState(false);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > SWIPE_THRESHOLD) {
      onVote(true);
    } else if (info.offset.x < -SWIPE_THRESHOLD) {
      onVote(false);
    }
  };

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none"
      style={{ x, rotate, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      dragTransition={{ bounceStiffness: 300, bounceDamping: 20 }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.95, opacity: 0, y: 30 }}
      animate={{ 
        scale: 1, 
        opacity: isTop ? 1 : 0,
        y: 0,
      }}
      transition={springTransition}
      exit={{ 
        x: x.get() > 0 ? 400 : -400, 
        opacity: 0,
        rotate: x.get() > 0 ? 15 : -15,
        transition: { duration: 0.3 }
      }}
    >
      {/* Swipe indicators - SKIP/VOTE overlay */}
      <motion.div 
        className="absolute left-4 top-6 z-20 pointer-events-none"
        style={{ opacity: skipOpacity }}
      >
        <div className="border-4 border-muted-foreground rounded-lg px-4 py-1 rotate-[-15deg] bg-background/80">
          <span className="text-muted-foreground font-black text-2xl tracking-wider">SKIP</span>
        </div>
      </motion.div>
      <motion.div 
        className="absolute right-4 top-6 z-20 pointer-events-none"
        style={{ opacity: approveOpacity }}
      >
        <div className="border-4 border-success rounded-lg px-4 py-1 rotate-[15deg] bg-background/80">
          <span className="text-success font-black text-2xl tracking-wider">VOTE!</span>
        </div>
      </motion.div>

      <Card className="h-full flex flex-col overflow-hidden bg-card border-0 shadow-2xl rounded-3xl">
        {/* Header Row - Symbol, Name, Logo */}
        <div className="shrink-0 p-4 pb-2 flex items-start justify-between">
          <div className="flex items-center gap-3">
            {/* Company Logo */}
            <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center overflow-hidden border border-border/50">
              {!logoError ? (
                <img 
                  src={getStockLogoUrl(suggestion.symbol)}
                  alt={suggestion.symbol}
                  className="w-full h-full object-contain p-1"
                  onError={() => setLogoError(true)}
                />
              ) : (
                <Building2 className="w-6 h-6 text-muted-foreground" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="font-bold text-sm">
                  {suggestion.symbol}
                </Badge>
                <Badge className="bg-chart-4/20 text-chart-4 border-chart-4/30 text-xs">
                  <Lightbulb className="w-3 h-3 mr-0.5" />
                  Suggestion
                </Badge>
              </div>
              <p className="text-sm text-foreground font-medium mt-0.5 line-clamp-1">
                {suggestion.name || suggestion.symbol}
              </p>
            </div>
          </div>
          <div className="text-right">
            <Badge variant="outline" className="bg-success/10 text-success border-success/30">
              <ThumbsUp className="w-3 h-3 mr-1" />
              {suggestion.vote_count}
            </Badge>
          </div>
        </div>

        {/* Placeholder Chart Area - showing pending state */}
        <div className="relative flex-1 min-h-[180px] max-h-[240px] bg-gradient-to-b from-muted/30 to-transparent mx-4 rounded-xl overflow-hidden border border-border/30 flex items-center justify-center">
          <div className="text-center">
            <BarChart3 className="w-16 h-16 mx-auto text-muted-foreground/30 mb-2" />
            <p className="text-sm text-muted-foreground">Chart available after approval</p>
          </div>
          
          {/* Vote count badge overlay */}
          <div className="absolute top-3 right-3">
            <Badge className="bg-chart-4 text-white text-base px-2.5 py-0.5 font-bold shadow-lg border-0">
              <Users className="w-3.5 h-3.5 mr-1" />
              {suggestion.vote_count} votes
            </Badge>
          </div>
        </div>

        {/* Key Stats Chips */}
        <div className="shrink-0 px-4 py-2 flex flex-wrap gap-1.5">
          {suggestion.sector && (
            <Badge variant="outline" className="rounded-full text-xs">
              🏢 {suggestion.sector}
            </Badge>
          )}
          <Badge variant="outline" className="rounded-full text-xs">
            ⏳ Pending Approval
          </Badge>
          <Badge variant="outline" className="rounded-full text-xs">
            🗳️ Community Pick
          </Badge>
        </div>

        {/* Bio/Summary */}
        <div className="shrink-0 px-4 py-2 border-t border-border/30">
          <p className="text-sm leading-relaxed line-clamp-3 text-muted-foreground">
            {suggestion.summary 
              ? suggestion.summary
              : `Hey there! 👋 I'm ${suggestion.name || suggestion.symbol} and I'm waiting to join the party. Vote for me to get tracked! The community thinks I have potential. 🚀`
            }
          </p>
        </div>

        {/* Vote Info & Call to Action */}
        <div className="shrink-0 p-4 pt-2 border-t border-border/30 bg-muted/20">
          <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Lightbulb className="w-3.5 h-3.5 text-chart-4" />
                Community Suggested
              </span>
            </div>
            {suggestion.vote_count >= 5 && (
              <span className="text-success font-medium">🔥 Trending</span>
            )}
          </div>
          
          {/* Progress to approval */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Votes needed for auto-approval</span>
              <span>{suggestion.vote_count}/10</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-success transition-all duration-300" 
                style={{ width: `${Math.min(suggestion.vote_count * 10, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

type SwipeMode = 'dips' | 'suggestions';

export function DipSwipePage() {
  const isMobile = useIsMobile();
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
  const totalItems = mode === 'dips' ? cards.length : suggestions.length;

  // Fetch chart data for current and next cards (prefetch next for smooth experience)
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
      <div className={`container max-w-lg mx-auto px-4 ${isMobile ? 'py-2' : 'py-8'}`}>
        {!isMobile && (
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
            <p className="text-muted-foreground">
              {mode === 'dips' ? 'Swipe right to buy, left to pass' : 'Vote on stock suggestions'}
            </p>
          </div>
        )}
        <ModeSelector />
        <Card className={isMobile ? 'h-[calc(100vh-120px)]' : 'h-[600px]'}>
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
      <div className={`container max-w-lg mx-auto px-4 ${isMobile ? 'py-2' : 'py-8'}`}>
        <div className="text-center">
          {!isMobile && <h1 className="text-3xl font-bold mb-4">DipSwipe</h1>}
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
      <div className={`container max-w-lg mx-auto px-4 ${isMobile ? 'py-2' : 'py-8'}`}>
        <div className="text-center">
          {!isMobile && <h1 className="text-3xl font-bold mb-4">DipSwipe</h1>}
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
    <div className={`container max-w-lg mx-auto px-4 ${isMobile ? 'py-2 overflow-hidden' : 'py-8'}`}>
      {/* Header - Hidden on mobile */}
      {!isMobile && (
        <div className="text-center mb-4">
          <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
          <p className="text-muted-foreground">
            {mode === 'dips' ? 'Swipe right to buy, left to pass' : 'Vote on community suggestions'}
          </p>
        </div>
      )}

      {/* Mode Selector */}
      <ModeSelector />

      {/* Progress - Simplified on mobile */}
      <div className="flex items-center justify-center gap-2 mb-2">
        <Badge variant="outline" className={isMobile ? 'text-xs' : ''}>
          {currentIndex + 1} / {totalItems}
        </Badge>
        {!isMobile && (
          <Badge variant="secondary">
            {votedCards.size} voted
          </Badge>
        )}
      </div>

      {/* Card stack - Full height on mobile */}
      <div className={`relative ${isMobile ? 'h-[calc(100vh-140px)]' : 'h-[520px]'} mb-2`}>
        <AnimatePresence mode="popLayout">
          {mode === 'dips' && currentCard && (
            <SwipeableCard
              key={currentCard.symbol}
              card={currentCard}
              chartData={chartDataMap[currentCard.symbol]}
              onVote={handleVote}
              isTop={true}
            />
          )}
          {mode === 'suggestions' && currentSuggestion && (
            <SuggestionSwipeCard
              key={currentSuggestion.symbol}
              suggestion={currentSuggestion}
              onVote={handleSuggestionVote}
              isTop={true}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Action buttons - Hidden on mobile (swipe only) */}
      {!isMobile && (
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
      )}

      {/* Instructions - Hidden on mobile */}
      {!isMobile && (
        <div className="text-center mt-8 text-sm text-muted-foreground">
          <p>Drag card left or right, or use buttons below</p>
        </div>
      )}
    </div>
  );
}
