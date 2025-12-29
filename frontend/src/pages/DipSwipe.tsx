import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import { useIsMobile } from '@/hooks/useIsMobile';
import { useTheme } from '@/context/ThemeContext';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts';
import { CHART_LINE_ANIMATION } from '@/lib/chartConfig';
import type { PanInfo } from 'framer-motion';
import { 
  getDipCards, 
  voteDip,
  getTopSuggestions,
  getSuggestionSettings,
  voteForSuggestion,
  getStockChart,
  type DipCard, 
  type VoteType,
  type TopSuggestion,
  type ChartDataPoint
} from '@/services/api';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
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
  Calendar,
  Flame,
  TrendingDown,
  TrendingUp,
  Building2,
  BarChart2,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { StockLogo } from '@/components/StockLogo';

// Dynamic swipe threshold based on screen width
// Mobile: ~25% of screen width (responsive to quick gestures)
// Desktop: ~15% of card width (feels natural with mouse/trackpad)
const getSwipeThreshold = (screenWidth: number): number => {
  if (screenWidth < 480) return Math.max(80, screenWidth * 0.22);   // Small phones - more responsive
  if (screenWidth < 768) return Math.max(70, screenWidth * 0.18);   // Large phones/small tablets
  if (screenWidth < 1024) return Math.max(60, screenWidth * 0.12);  // Tablets
  return Math.min(120, Math.max(60, screenWidth * 0.06));           // Desktop
};
const VELOCITY_THRESHOLD = 300;  // Lower threshold for more responsive quick swipes

// Tinder-like spring physics - snappy but smooth
const springTransition = {
  type: "spring" as const,
  stiffness: 500,  // Higher = snappier response
  damping: 30,     // Balanced damping for natural feel
  mass: 0.5,       // Lower mass = faster acceleration
};

// Exit spring - fast throw off screen
const exitSpring = {
  type: "spring" as const,
  stiffness: 400,
  damping: 30,
  mass: 0.8,
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

// Format bio text to add paragraph breaks at natural points
function formatBioText(text: string): string {
  return text
    .replace(/([.!?])\s+/g, '$1\n\n')  // Add line break after sentences
    .replace(/\n{3,}/g, '\n\n')  // Normalize multiple line breaks
    .trim();
}

// Colorblind-friendly colors for dip indicator
const DIP_COLOR_COLORBLIND = '#0066CC'; // Blue - visible to all color blindness types
const DIP_COLOR_NORMAL = '#ef4444'; // Red for dip
const SUCCESS_COLOR = '#22c55e'; // Green

// Helper to convert hex to rgba
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Chart colors based on theme preference - respects color picker
function getChartColors(isColorblind: boolean, customColors?: { up: string; down: string }, isPositive: boolean = false) {
  // Colorblind mode takes priority
  if (isColorblind) {
    return {
      stroke: DIP_COLOR_COLORBLIND,
      gradientStart: 'rgba(0, 102, 204, 0.4)',
      gradientEnd: 'rgba(0, 102, 204, 0.05)',
    };
  }
  
  // Use custom colors if provided (from color picker)
  if (customColors) {
    const color = isPositive ? customColors.up : customColors.down;
    return {
      stroke: color,
      gradientStart: hexToRgba(color, 0.4),
      gradientEnd: hexToRgba(color, 0.05),
    };
  }
  
  // Fallback to defaults
  if (isPositive) {
    return {
      stroke: SUCCESS_COLOR,
      gradientStart: 'rgba(34, 197, 94, 0.4)',
      gradientEnd: 'rgba(34, 197, 94, 0.05)',
    };
  }
  return {
    stroke: DIP_COLOR_NORMAL,
    gradientStart: 'rgba(239, 68, 68, 0.4)',
    gradientEnd: 'rgba(239, 68, 68, 0.05)',
  };
}

// Individual swipeable card - Swipe-style!
function SwipeableCard({ 
  card, 
  onVote, 
  isTop,
  chartData,
  onSwipeComplete,
  colorblindMode,
  customColors,
}: { 
  card: DipCard; 
  onVote: (vote: VoteType) => void;
  isTop: boolean;
  chartData?: ChartDataPoint[];
  onSwipeComplete?: () => void;
  colorblindMode: boolean;
  customColors?: { up: string; down: string };
}) {
  // Dynamic threshold based on current screen width
  const [swipeThreshold, setSwipeThreshold] = useState(() => 
    typeof window !== 'undefined' ? getSwipeThreshold(window.innerWidth) : 100
  );
  
  // Track exit velocity for natural throw animation
  const [exitVelocity, setExitVelocity] = useState(0);
  
  // Update threshold on resize
  useEffect(() => {
    const handleResize = () => setSwipeThreshold(getSwipeThreshold(window.innerWidth));
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const x = useMotionValue(0);
  // More dramatic rotation that responds to velocity feel
  const rotate = useTransform(x, [-200, 0, 200], [-25, 0, 25]);
  // Scale down slightly as card moves away (depth effect)
  const scale = useTransform(x, [-300, 0, 300], [0.95, 1, 0.95]);
  const buyOpacity = useTransform(x, [0, swipeThreshold * 0.6], [0, 1]);
  const sellOpacity = useTransform(x, [-swipeThreshold * 0.6, 0], [1, 0]);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const offset = info.offset.x;
    const velocity = info.velocity.x;
    const absVelocity = Math.abs(velocity);
    
    // More responsive detection - velocity is key for natural feel
    const hasEnoughDistance = Math.abs(offset) > swipeThreshold;
    const hasQuickSwipe = Math.abs(offset) > swipeThreshold * 0.3 && absVelocity > VELOCITY_THRESHOLD;
    const hasStrongThrow = absVelocity > VELOCITY_THRESHOLD * 2; // Very fast swipe regardless of distance
    
    if (hasEnoughDistance || hasQuickSwipe || hasStrongThrow) {
      // Store velocity for exit animation
      setExitVelocity(velocity);
      if (offset > 0 || velocity > VELOCITY_THRESHOLD) {
        onVote('buy');
      } else {
        onVote('sell');
      }
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
  
  // Calculate exit distance based on velocity (faster = further)
  const getExitX = () => {
    const direction = x.get() > 0 || exitVelocity > 0 ? 1 : -1;
    const baseDistance = 500;
    const velocityBonus = Math.min(Math.abs(exitVelocity) * 0.3, 300);
    return direction * (baseDistance + velocityBonus);
  };

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none will-change-transform"
      style={{ x, rotate, scale, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.5}
      dragMomentum={true}
      dragTransition={{ 
        bounceStiffness: 600, 
        bounceDamping: 20,
        power: 0.3,
        timeConstant: 200,
      }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.92, opacity: 0, y: 50 }}
      animate={{ 
        scale: 1, 
        opacity: isTop ? 1 : 0,
        y: 0,
      }}
      transition={springTransition}
      exit={{ 
        x: getExitX(),
        opacity: 0,
        rotate: (x.get() > 0 || exitVelocity > 0) ? 30 : -30,
        transition: exitSpring,
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

      <Card className="h-full flex flex-col overflow-hidden bg-card border-0 shadow-2xl rounded-3xl p-0">
        {/* Header Row - Symbol, Name, Logo, Age */}
        <div className="shrink-0 px-4 pt-3 pb-1 flex items-start justify-between">
          <div className="flex items-center gap-2">
            {/* Company Logo */}
            <StockLogo symbol={card.symbol} size="md" />
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-base">
                  {card.symbol}
                </span>

              </div>
              <p className="text-xs text-muted-foreground font-medium mt-0.5 line-clamp-1">
                {card.name || 'Loading...'}
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

        {/* Chart as "Profile Photo" - compact, with tooltip */}
        <div className="relative shrink-0 h-[120px] mx-4 rounded-lg overflow-hidden">
          {miniChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={miniChartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <defs>
                  <linearGradient id={`swipe-gradient-${card.symbol}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={getChartColors(colorblindMode, customColors).gradientStart} />
                    <stop offset="100%" stopColor={getChartColors(colorblindMode, customColors).gradientEnd} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="x" hide />
                <YAxis hide domain={['dataMin', 'dataMax']} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload?.[0]?.value) {
                      return (
                        <div className="bg-background/95 border border-border px-2 py-1 rounded shadow-lg">
                          <span className="text-sm font-medium">{formatPrice(payload[0].value as number)}</span>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="y"
                  stroke={getChartColors(colorblindMode, customColors).stroke}
                  strokeWidth={2}
                  fill={`url(#swipe-gradient-${card.symbol})`}
                  {...CHART_LINE_ANIMATION}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center">
              <BarChart3 className="w-12 h-12 text-muted-foreground/30" />
            </div>
          )}
          
          {/* Dip percentage badge overlay */}
          <div className="absolute top-1 right-1">
            <Badge 
              className="text-sm px-2 py-0.5 font-bold shadow-lg text-white border-0"
              style={{ backgroundColor: colorblindMode ? DIP_COLOR_COLORBLIND : (customColors?.down || DIP_COLOR_NORMAL) }}
            >
              <TrendingDown className="w-3 h-3 mr-1" />
              Dip {formatDipPct(card.dip_pct)}
            </Badge>
          </div>
          {/* Timeframe badge */}
          <div className="absolute bottom-1 left-1">
            <Badge variant="outline" className="text-xs px-1.5 py-0 bg-background/80">
              90 days
            </Badge>
          </div>
        </div>

        {/* Key Stats Chips - compact */}
        <div className="shrink-0 px-4 py-2 flex flex-wrap gap-1">
          <Badge variant="outline" className="rounded-full text-xs px-2 py-0 flex items-center gap-1">
            <DollarSign className="h-3 w-3" /> {formatPrice(card.current_price)}
          </Badge>
          <Badge variant="outline" className="rounded-full text-xs px-2 py-0 flex items-center gap-1">
            <TrendingUp className="h-3 w-3" /> Peak: {formatPrice(card.ref_high)}
          </Badge>
          {card.sector && (
            <Badge variant="outline" className="rounded-full text-xs px-2 py-0 flex items-center gap-1">
              <Building2 className="h-3 w-3" /> {card.sector}
            </Badge>
          )}
          {card.ipo_year && (
            <Badge variant="outline" className="rounded-full text-xs px-2 py-0 flex items-center gap-1">
              <Calendar className="h-3 w-3" /> Since {card.ipo_year}
            </Badge>
          )}
          {card.ai_pending && (
            <Badge variant="secondary" className="rounded-full text-xs px-2 py-0 flex items-center gap-1 animate-pulse">
              <Sparkles className="h-3 w-3" /> AI Generating
            </Badge>
          )}
        </div>

        {/* Bio - flex-1 to take remaining space, larger text */}
        <div className="flex-1 px-4 py-3 border-t border-border/30 overflow-y-auto min-h-0">
          {card.ai_pending ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Generating AI insights...</span>
            </div>
          ) : (
            <p className="text-base leading-relaxed text-foreground whitespace-pre-wrap">
              {formatBioText(card.swipe_bio 
                ? card.swipe_bio.replace(/^"|"$/g, '')
                : `Looking for investors who appreciate a good dip. Currently ${formatDipPct(card.dip_pct)} off my peak. Swipe right if you see my potential!`
              )}
            </p>
          )}
          
          {/* AI Analysis - Expandable */}
          {card.ai_reasoning && (
            <details className="mt-2 group">
              <summary className="text-xs text-primary cursor-pointer hover:text-primary/80 flex items-center gap-1">
                <BarChart2 className="h-3 w-3" />
                <span className="group-open:hidden">View AI Analysis</span>
                <span className="hidden group-open:inline">Hide AI Analysis</span>
              </summary>
              <div className="mt-2 p-2 bg-muted/50 rounded-md border border-border/30">
                <p className="text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                  {card.ai_reasoning}
                </p>
              </div>
            </details>
          )}
        </div>

        {/* Vote Stats - thumbs only, no sentiment bar */}
        <div className="shrink-0 px-4 py-2 border-t border-border/30 bg-muted/20">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
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
            {buyPct > 60 && <span className="text-success font-medium flex items-center gap-1"><TrendingUp className="h-3 w-3" /> Bullish</span>}
            {sellPct > 60 && <span className="text-danger font-medium flex items-center gap-1"><TrendingDown className="h-3 w-3" /> Bearish</span>}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

// Suggestion card for voting mode - Swipe-style!
function SuggestionSwipeCard({ 
  suggestion, 
  onVote, 
  isTop,
  autoApproveVotes = 10,
}: { 
  suggestion: TopSuggestion; 
  onVote: (approve: boolean) => void;
  isTop: boolean;
  autoApproveVotes?: number;
}) {
  // Dynamic threshold based on current screen width
  const [swipeThreshold, setSwipeThreshold] = useState(() => 
    typeof window !== 'undefined' ? getSwipeThreshold(window.innerWidth) : 100
  );
  
  // Track exit velocity for natural throw animation
  const [exitVelocity, setExitVelocity] = useState(0);
  
  // Update threshold on resize
  useEffect(() => {
    const handleResize = () => setSwipeThreshold(getSwipeThreshold(window.innerWidth));
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const x = useMotionValue(0);
  // More dramatic rotation
  const rotate = useTransform(x, [-200, 0, 200], [-25, 0, 25]);
  // Scale down slightly as card moves away
  const scale = useTransform(x, [-300, 0, 300], [0.95, 1, 0.95]);
  const approveOpacity = useTransform(x, [0, swipeThreshold * 0.6], [0, 1]);
  const skipOpacity = useTransform(x, [-swipeThreshold * 0.6, 0], [1, 0]);
  
  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const offset = info.offset.x;
    const velocity = info.velocity.x;
    const absVelocity = Math.abs(velocity);
    
    // More responsive detection
    const hasEnoughDistance = Math.abs(offset) > swipeThreshold;
    const hasQuickSwipe = Math.abs(offset) > swipeThreshold * 0.3 && absVelocity > VELOCITY_THRESHOLD;
    const hasStrongThrow = absVelocity > VELOCITY_THRESHOLD * 2;
    
    if (hasEnoughDistance || hasQuickSwipe || hasStrongThrow) {
      setExitVelocity(velocity);
      if (offset > 0 || velocity > VELOCITY_THRESHOLD) {
        onVote(true);
      } else {
        onVote(false);
      }
    }
  };
  
  // Calculate exit distance based on velocity
  const getExitX = () => {
    const direction = x.get() > 0 || exitVelocity > 0 ? 1 : -1;
    const baseDistance = 500;
    const velocityBonus = Math.min(Math.abs(exitVelocity) * 0.3, 300);
    return direction * (baseDistance + velocityBonus);
  };

  return (
    <motion.div
      className="absolute inset-0 cursor-grab active:cursor-grabbing touch-none will-change-transform"
      style={{ x, rotate, scale, zIndex: isTop ? 10 : 0 }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.5}
      dragMomentum={true}
      dragTransition={{ 
        bounceStiffness: 600, 
        bounceDamping: 20,
        power: 0.3,
        timeConstant: 200,
      }}
      onDragEnd={handleDragEnd}
      initial={{ scale: 0.92, opacity: 0, y: 50 }}
      animate={{ 
        scale: 1, 
        opacity: isTop ? 1 : 0,
        y: 0,
      }}
      transition={springTransition}
      exit={{ 
        x: getExitX(),
        opacity: 0,
        rotate: (x.get() > 0 || exitVelocity > 0) ? 30 : -30,
        transition: exitSpring,
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

      <Card className="h-full flex flex-col overflow-hidden bg-card border-0 shadow-2xl rounded-3xl py-0">
        {/* Compact Header - Logo, Name, Stats in row */}
        <div className="shrink-0 px-4 pt-3 pb-2 flex items-start gap-3 bg-gradient-to-b from-chart-4/10 to-transparent">
          {/* Company Logo */}
          <StockLogo 
            symbol={suggestion.symbol} 
            size="xl" 
            className="shrink-0 border-2 border-chart-4/30 shadow-md" 
          />
          {/* Name & Badges */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="font-bold text-lg">{suggestion.symbol}</span>
              <Badge className="bg-chart-4/20 text-chart-4 border-chart-4/30 text-xs px-1.5 py-0">
                <Lightbulb className="w-2.5 h-2.5 mr-0.5" />
                Suggestion
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground font-medium truncate">
              {suggestion.name && suggestion.name !== suggestion.symbol ? suggestion.name : 'Community Suggestion'}
            </p>
            {/* Inline stats */}
            <div className="flex flex-wrap gap-1 mt-1">
              {suggestion.sector && (
                <span className="text-xs text-muted-foreground flex items-center gap-0.5"><Building2 className="h-3 w-3" /> {suggestion.sector}</span>
              )}
              {suggestion.ipo_year && (
                <span className="text-xs text-muted-foreground flex items-center gap-0.5">• <Calendar className="h-3 w-3" /> {suggestion.ipo_year}</span>
              )}
            </div>
          </div>
          {/* Vote count badge */}
          <div className="shrink-0 flex flex-col items-center">
            <div className="w-12 h-12 rounded-full bg-success/15 border border-success/30 flex flex-col items-center justify-center">
              <span className="text-lg font-bold text-success leading-none">{suggestion.vote_count}</span>
              <ThumbsUp className="w-3 h-3 text-success mt-0.5" />
            </div>
            {suggestion.vote_count >= 5 && (
              <Flame className="w-3 h-3 text-orange-500 mt-1" />
            )}
          </div>
        </div>

        {/* Bio/Summary - Takes up main space */}
        <div className="flex-1 min-h-0 px-4 py-3 overflow-y-auto">
          <p className="text-base leading-relaxed text-foreground whitespace-pre-wrap">
            {formatBioText(suggestion.summary 
              ? suggestion.summary
              : `Hey there! I'm ${suggestion.name || suggestion.symbol} and I'm waiting to join the party. Vote for me to get tracked! The community thinks I have potential.`
            )}
          </p>
        </div>

        {/* Progress to Approval - Compact footer */}
        <div className="shrink-0 px-4 py-2.5 border-t border-border/30 bg-muted/20">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-success to-chart-4 transition-all duration-300 rounded-full" 
                  style={{ width: `${Math.min((suggestion.vote_count / autoApproveVotes) * 100, 100)}%` }}
                />
              </div>
            </div>
            <span className="text-xs text-muted-foreground shrink-0">{suggestion.vote_count}/{autoApproveVotes}</span>
          </div>
          {suggestion.vote_count >= autoApproveVotes && (
            <p className="text-xs text-success mt-1 text-center font-medium">
              ✅ Eligible for auto-approval!
            </p>
          )}
        </div>
      </Card>
    </motion.div>
  );
}

type SwipeMode = 'dips' | 'suggestions';

export function DipSwipePage() {
  const isMobile = useIsMobile();
  const { colorblindMode, customColors } = useTheme();
  const [mode, setMode] = useState<SwipeMode>('dips');
  const [cards, setCards] = useState<DipCard[]>([]);
  const [suggestions, setSuggestions] = useState<TopSuggestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isVoting, setIsVoting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [votedCards, setVotedCards] = useState<Set<string>>(new Set());
  const [chartDataMap, setChartDataMap] = useState<Record<string, ChartDataPoint[]>>({});
  const [autoApproveVotes, setAutoApproveVotes] = useState(10);

  // Get current card for dynamic SEO
  const currentCard = mode === 'dips' ? cards[currentIndex] : null;

  // SEO - Dynamic meta based on current card
  useSEO({
    title: currentCard 
      ? `${currentCard.symbol} - Vote: Buy or Sell the Dip?`
      : 'DipSwipe - Swipe on Stock Dips',
    description: currentCard
      ? `${currentCard.symbol} is ${currentCard.dip_pct?.toFixed(0)}% below its high. AI says: ${currentCard.ai_rating}. Swipe right to buy, left to pass.`
      : 'Swipe through stock dips like a dating app. Vote on whether to buy or sell stocks that have dipped. Community-powered stock sentiment.',
    keywords: 'stock voting, dip buying, stock sentiment, community investing, swipe stocks',
    canonical: '/swipe',
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'DipSwipe', url: '/swipe' },
    ]),
  });

  // Fetch suggestion settings
  useEffect(() => {
    getSuggestionSettings()
      .then(settings => setAutoApproveVotes(settings.auto_approve_votes))
      .catch(() => setAutoApproveVotes(10));
  }, []);

  const loadCards = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (mode === 'dips') {
        // Exclude cards user has already voted on
        const response = await getDipCards(true, true);
        setCards(response.cards);
      } else {
        // Exclude suggestions user has already voted on
        const data = await getTopSuggestions(50, true);
        setSuggestions(data);
      }
      setCurrentIndex(0);
      setVotedCards(new Set()); // Reset local voted tracking on reload
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
      // Check if it's a cooldown/already voted error - skip silently
      const errMsg = err instanceof Error ? err.message : String(err);
      if (errMsg.includes('Already voted') || errMsg.includes('cooldown') || errMsg.includes('Try again')) {
        // Already voted - mark as voted and advance
        setVotedCards(prev => new Set([...prev, currentCard.symbol]));
        setCurrentIndex(prev => prev + 1);
      } else {
        // Other error - show it and still advance
        setError(`Vote failed: ${errMsg}`);
        setTimeout(() => setError(null), 3000);
        setCurrentIndex(prev => prev + 1);
      }
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

      {/* Progress indicator - shows position in deck */}
      <div className="flex items-center justify-center gap-2 mb-2">
        <Badge variant="outline" className={`${isMobile ? 'text-xs' : ''} flex items-center gap-1`} title="Card position / Total unvoted cards">
          <BarChart2 className="h-3 w-3" /> {currentIndex + 1} of {totalItems} {mode === 'dips' ? 'stocks' : 'suggestions'}
        </Badge>
      </div>

      {/* Card stack - Responsive height based on content */}
      <div className={`relative ${isMobile ? 'h-[calc(100vh-140px)]' : 'min-h-[500px] h-[calc(100vh-280px)] max-h-[800px]'} mb-2`}>
        <AnimatePresence mode="popLayout">
          {mode === 'dips' && currentCard && (
            <SwipeableCard
              key={currentCard.symbol}
              card={currentCard}
              chartData={chartDataMap[currentCard.symbol]}
              onVote={handleVote}
              isTop={true}
              colorblindMode={colorblindMode}
              customColors={customColors}
            />
          )}
          {mode === 'suggestions' && currentSuggestion && (
            <SuggestionSwipeCard
              key={currentSuggestion.symbol}
              suggestion={currentSuggestion}
              onVote={handleSuggestionVote}
              isTop={true}
              autoApproveVotes={autoApproveVotes}
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
