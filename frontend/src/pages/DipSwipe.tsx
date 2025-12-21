import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import type { PanInfo } from 'framer-motion';
import { 
  getDipCards, 
  voteDip, 
  type DipCard, 
  type VoteType 
} from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  TrendingDown, 
  X, 
  Check, 
  Brain, 
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  BarChart3,
  Users,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';

// Swipe threshold in pixels
const SWIPE_THRESHOLD = 100;

// Vote button animations
const voteButtonVariants = {
  tap: { scale: 0.9 },
  hover: { scale: 1.05 },
};

function formatPrice(value: number): string {
  return new Intl.NumberFormat('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

// AI Rating badge component
function AIRatingBadge({ rating, confidence }: { rating: string | null; confidence: number | null }) {
  if (!rating) return null;
  
  const ratingColors: Record<string, string> = {
    strong_buy: 'bg-success/20 text-success border-success/30',
    buy: 'bg-success/10 text-success border-success/20',
    hold: 'bg-muted text-muted-foreground border-border',
    sell: 'bg-danger/10 text-danger border-danger/20',
    strong_sell: 'bg-danger/20 text-danger border-danger/30',
  };
  
  const ratingLabels: Record<string, string> = {
    strong_buy: 'Strong Buy',
    buy: 'Buy',
    hold: 'Hold',
    sell: 'Sell',
    strong_sell: 'Strong Sell',
  };
  
  return (
    <Badge 
      variant="outline" 
      className={`${ratingColors[rating] || ''} font-medium`}
    >
      <Sparkles className="w-3 h-3 mr-1" />
      {ratingLabels[rating] || rating}
      {confidence && <span className="ml-1 opacity-70">({confidence}/10)</span>}
    </Badge>
  );
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

// Individual swipeable card
function SwipeableCard({ 
  card, 
  onVote, 
  isTop,
}: { 
  card: DipCard; 
  onVote: (vote: VoteType) => void;
  isTop: boolean;
}) {
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-15, 15]);
  const buyOpacity = useTransform(x, [0, SWIPE_THRESHOLD], [0, 1]);
  const sellOpacity = useTransform(x, [-SWIPE_THRESHOLD, 0], [1, 0]);
  
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

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing"
      style={{ x, rotate, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.9}
      onDragEnd={handleDragEnd}
      initial={{ scale: isTop ? 1 : 0.95, opacity: isTop ? 1 : 0.5 }}
      animate={{ scale: isTop ? 1 : 0.95, opacity: isTop ? 1 : 0.5 }}
      exit={{ 
        x: 300, 
        opacity: 0,
        transition: { duration: 0.3 }
      }}
    >
      {/* Swipe indicators */}
      <motion.div 
        className="absolute -left-4 top-1/2 -translate-y-1/2 z-20 pointer-events-none"
        style={{ opacity: sellOpacity }}
      >
        <div className="bg-danger text-white p-3 rounded-full shadow-lg">
          <X className="w-8 h-8" />
        </div>
      </motion.div>
      <motion.div 
        className="absolute -right-4 top-1/2 -translate-y-1/2 z-20 pointer-events-none"
        style={{ opacity: buyOpacity }}
      >
        <div className="bg-success text-white p-3 rounded-full shadow-lg">
          <Check className="w-8 h-8" />
        </div>
      </motion.div>

      <Card className="h-full flex flex-col overflow-hidden bg-card border-border/50 shadow-xl">
        <CardContent className="flex-1 p-6 flex flex-col">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-3xl font-bold">{card.symbol}</h2>
                {card.sector && (
                  <Badge variant="secondary" className="font-normal">
                    {card.sector}
                  </Badge>
                )}
              </div>
              <p className="text-muted-foreground mt-1">
                {card.name || card.symbol}
              </p>
            </div>
            <AIRatingBadge rating={card.ai_rating} confidence={card.ai_confidence} />
          </div>

          {/* Price info */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-muted/50 rounded-lg p-4">
              <p className="text-sm text-muted-foreground mb-1">Current Price</p>
              <p className="text-2xl font-semibold">{formatPrice(card.current_price)}</p>
            </div>
            <div className="bg-danger/10 rounded-lg p-4">
              <p className="text-sm text-muted-foreground mb-1 flex items-center gap-1">
                <TrendingDown className="w-3 h-3 text-danger" />
                Dip from High
              </p>
              <p className="text-2xl font-semibold text-danger">
                -{card.dip_pct.toFixed(1)}%
              </p>
            </div>
          </div>

          {/* AI Bio */}
          {card.tinder_bio && (
            <div className="bg-accent/50 rounded-lg p-4 mb-4 flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">AI Analysis</span>
              </div>
              <p className="text-sm leading-relaxed">{card.tinder_bio}</p>
            </div>
          )}

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-2 text-center mb-4">
            <div className="bg-muted/30 rounded-lg p-2">
              <p className="text-xs text-muted-foreground">52W High</p>
              <p className="font-medium">{formatPrice(card.ref_high)}</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-2">
              <p className="text-xs text-muted-foreground">Days Below</p>
              <p className="font-medium">{card.days_below}</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-2">
              <p className="text-xs text-muted-foreground">Net Score</p>
              <p className={`font-medium ${card.vote_counts.net_score >= 0 ? 'text-success' : 'text-danger'}`}>
                {card.vote_counts.net_score >= 0 ? '+' : ''}{card.vote_counts.net_score}
              </p>
            </div>
          </div>

          {/* Community sentiment */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Community Sentiment</span>
            </div>
            <SentimentBar buyPct={buyPct} sellPct={sellPct} />
          </div>

          {/* AI Reasoning */}
          {card.ai_reasoning && (
            <div className="text-sm text-muted-foreground border-t pt-4">
              <p className="italic">{card.ai_reasoning}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function DipSwipePage() {
  const [cards, setCards] = useState<DipCard[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isVoting, setIsVoting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [votedCards, setVotedCards] = useState<Set<string>>(new Set());

  const loadCards = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getDipCards(true);
      setCards(response.cards);
      setCurrentIndex(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cards');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCards();
  }, [loadCards]);

  const currentCard = useMemo(() => cards[currentIndex], [cards, currentIndex]);
  const nextCard = useMemo(() => cards[currentIndex + 1], [cards, currentIndex]);

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

  const handleSkip = useCallback(() => {
    setCurrentIndex(prev => Math.min(prev + 1, cards.length));
  }, [cards.length]);

  const handlePrevious = useCallback(() => {
    setCurrentIndex(prev => Math.max(prev - 1, 0));
  }, []);

  // Loading state
  if (isLoading) {
    return (
      <div className="container max-w-lg mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
          <p className="text-muted-foreground">Swipe right to buy, left to pass</p>
        </div>
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
  if (currentIndex >= cards.length) {
    return (
      <div className="container max-w-lg mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">DipSwipe</h1>
          <Card className="p-8">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <h2 className="text-xl font-semibold mb-2">All caught up!</h2>
            <p className="text-muted-foreground mb-4">
              You've reviewed all {cards.length} stocks in the dip.
            </p>
            <p className="text-sm text-muted-foreground mb-6">
              Voted on {votedCards.size} stocks
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
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold mb-2">DipSwipe</h1>
        <p className="text-muted-foreground">
          Swipe right to buy, left to pass
        </p>
        <div className="flex items-center justify-center gap-2 mt-2">
          <Badge variant="outline">
            {currentIndex + 1} / {cards.length}
          </Badge>
          <Badge variant="secondary">
            {votedCards.size} voted
          </Badge>
        </div>
      </div>

      {/* Card stack */}
      <div className="relative h-[600px] mb-6">
        <AnimatePresence mode="popLayout">
          {nextCard && (
            <SwipeableCard
              key={nextCard.symbol}
              card={nextCard}
              onVote={() => {}}
              isTop={false}
            />
          )}
          {currentCard && (
            <SwipeableCard
              key={currentCard.symbol}
              card={currentCard}
              onVote={handleVote}
              isTop={true}
            />
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
