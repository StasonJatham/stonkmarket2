/**
 * StockCardV2 - Information-dense dashboard card for stock display.
 * 
 * Features:
 * - Mini sparkline chart as subtle background
 * - Company logo + ticker + name
 * - Current price with daily change
 * - Sector performance comparison (vs global sector ETF)
 * - Technical signal indicator (top optimized signal)
 * - Buying opportunity rating (composite score)
 * - Domain analysis result
 */

import { useMemo, memo, useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import { CHART_MINI_ANIMATION } from '@/lib/chartConfig';
import type { ChartDataPoint } from '@/services/api';
import { prefetchStockChart, prefetchStockInfo } from '@/services/api';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { StockLogo } from '@/components/StockLogo';
import { 
  TrendingUp, 
  TrendingDown, 
  ChevronRight, 
  Zap,
  Target,
  BarChart2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTheme } from '@/context/ThemeContext';

// Get chart colors respecting theme and colorblind mode
function useChartColors() {
  const { colorblindMode, customColors } = useTheme();
  const [colors, setColors] = useState({ success: '#22c55e', danger: '#ef4444' });
  
  useEffect(() => {
    // Read computed colors from CSS variables (respects theme and colorblind mode)
    const root = document.documentElement;
    const success = getComputedStyle(root).getPropertyValue('--success').trim();
    const danger = getComputedStyle(root).getPropertyValue('--danger').trim();
    
    if (success && danger) {
      setColors({ success, danger });
    } else if (colorblindMode) {
      // Fallback colorblind colors
      setColors({ success: '#3b82f6', danger: '#f97316' }); // Blue/Orange
    } else if (customColors) {
      setColors({ success: customColors.up, danger: customColors.down });
    }
  }, [colorblindMode, customColors]);
  
  return colors;
}

// ============================================================================
// Types
// ============================================================================

export interface StockCardData {
  // Core identity
  symbol: string;
  name: string | null;
  sector: string | null;
  
  // Price & performance
  last_price: number;
  change_percent: number | null;
  high_52w: number | null;
  low_52w: number | null;
  market_cap: number | null;
  
  // Dip metrics
  depth: number; // Fraction (0.15 = 15% dip)
  days_since_dip: number | null;
  dip_bucket: string | null;
  
  // Recovery and unusual dip metrics (quant engine backtest data)
  expected_recovery_days?: number | null; // Expected time to recover based on optimal holding period
  typical_dip_pct?: number | null; // Stock's typical dip size as percentage
  dip_vs_typical?: number | null; // Current dip / typical dip ratio (>1.5 = unusual)
  is_unusual_dip?: boolean; // True if current dip is significantly larger than typical
  opportunity_type?: 'OUTLIER' | 'BOUNCE' | 'BOTH' | 'NONE'; // Opportunity classification
  // Extreme Value Analysis (EVA) fields
  is_tail_event?: boolean; // True if current dip is an extreme tail event
  return_period_years?: number | null; // How rare? (years between similar events)
  regime_dip_percentile?: number | null; // Dip percentile within 'normal' regime
  win_rate?: number | null; // Historical win rate for similar signals
  
  // Technical signals
  top_signal?: {
    name: string;
    is_buy: boolean;
    strength: number; // 0-1
    description?: string;
  } | null;
  
  // Sector comparison
  sector_delta?: number | null; // % difference from sector ETF
  sector_etf?: string | null; // e.g., "XLF" for financials
  
  // Opportunity rating (composite)
  opportunity_score?: number | null; // 0-100
  opportunity_rating?: 'strong_buy' | 'buy' | 'hold' | 'avoid' | null;
  
  // AI analysis
  ai_rating?: string | null;
  ai_summary?: string | null;
  domain_analysis?: string | null;
  
  // Domain-specific analysis
  domain_context?: string | null;
  domain_risk_level?: string | null;
  volatility_regime?: string | null;
  
  // Signal metrics
  mu_hat?: number; // Expected return
  uncertainty?: number;
  marginal_utility?: number;
  
  // Intrinsic Value
  intrinsic_value?: number | null;
  upside_pct?: number | null;
  valuation_status?: 'undervalued' | 'fair' | 'overvalued' | null;
  
  // APUS + DOUS Dual-Mode Scoring
  quant_mode?: 'CERTIFIED_BUY' | 'DIP_ENTRY' | 'HOLD' | 'DOWNTREND' | null;
  quant_score_a?: number | null;  // Mode A (APUS) score 0-100
  quant_score_b?: number | null;  // Mode B (DOUS) score 0-100
  quant_gate_pass?: boolean;
  quant_evidence?: {
    p_outperf: number;
    ci_low: number;
    ci_high: number;
    dsr: number;
    median_edge: number;
    worst_regime_edge: number;
    cvar_5: number;
    fund_mom: number;
    val_z: number;
    event_risk: boolean;
    p_recovery: number;
    expected_value: number;
  } | null;
}

interface StockCardV2Props {
  stock: StockCardData;
  chartData?: ChartDataPoint[];
  isLoading?: boolean;
  isSelected?: boolean;
  compact?: boolean;
  onClick?: () => void;
}

// ============================================================================
// Sector ETF Mapping
// ============================================================================

const SECTOR_ETF_MAP: Record<string, { etf: string; name: string }> = {
  'Technology': { etf: 'XLK', name: 'Tech Select SPDR' },
  'Information Technology': { etf: 'XLK', name: 'Tech Select SPDR' },
  'Healthcare': { etf: 'XLV', name: 'Health Care SPDR' },
  'Health Care': { etf: 'XLV', name: 'Health Care SPDR' },
  'Financials': { etf: 'XLF', name: 'Financial SPDR' },
  'Financial Services': { etf: 'XLF', name: 'Financial SPDR' },
  'Consumer Discretionary': { etf: 'XLY', name: 'Consumer Disc. SPDR' },
  'Consumer Cyclical': { etf: 'XLY', name: 'Consumer Disc. SPDR' },
  'Consumer Staples': { etf: 'XLP', name: 'Consumer Staples SPDR' },
  'Consumer Defensive': { etf: 'XLP', name: 'Consumer Staples SPDR' },
  'Energy': { etf: 'XLE', name: 'Energy SPDR' },
  'Industrials': { etf: 'XLI', name: 'Industrial SPDR' },
  'Materials': { etf: 'XLB', name: 'Materials SPDR' },
  'Basic Materials': { etf: 'XLB', name: 'Materials SPDR' },
  'Real Estate': { etf: 'XLRE', name: 'Real Estate SPDR' },
  'Utilities': { etf: 'XLU', name: 'Utilities SPDR' },
  'Communication Services': { etf: 'XLC', name: 'Comm Services SPDR' },
  'Communication': { etf: 'XLC', name: 'Comm Services SPDR' },
};

export function getSectorETF(sector: string | null): { etf: string; name: string } | null {
  if (!sector) return null;
  return SECTOR_ETF_MAP[sector] || null;
}

// ============================================================================
// Formatting Utilities
// ============================================================================

function formatPrice(value: number | null): string {
  if (value === null || value === undefined) return '—';
  if (value >= 1000) return `$${value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  if (value >= 1) return `$${value.toFixed(2)}`;
  return `$${value.toFixed(4)}`;
}

function formatMarketCap(value: number | null): string {
  if (!value) return '—';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  return `$${value.toFixed(0)}`;
}

function formatPercent(value: number | null | undefined, showSign = true): string {
  if (value === null || value === undefined) return '—';
  const sign = showSign ? (value >= 0 ? '+' : '') : '';
  return `${sign}${value.toFixed(2)}%`;
}

function formatDipPercent(depth: number): string {
  const pct = depth * 100;
  return `-${pct.toFixed(1)}%`;
}

// ============================================================================
// Opportunity Score Calculation
// ============================================================================

function calculateOpportunityScore(stock: StockCardData): number {
  let score = 50; // Neutral baseline
  
  // Dip contribution (up to 25 points for deep dips)
  const dipPct = stock.depth * 100;
  if (dipPct > 5) score += Math.min(dipPct / 2, 25);
  
  // Signal strength contribution (up to 20 points)
  if (stock.top_signal?.is_buy && stock.top_signal.strength) {
    score += stock.top_signal.strength * 20;
  }
  
  // AI rating contribution (up to 15 points)
  if (stock.ai_rating) {
    const ratingBonus: Record<string, number> = {
      'strong_buy': 15,
      'buy': 10,
      'hold': 0,
      'sell': -10,
      'strong_sell': -15,
    };
    score += ratingBonus[stock.ai_rating] || 0;
  }
  
  // Expected return contribution (up to 10 points)
  if (stock.mu_hat && stock.mu_hat > 0) {
    score += Math.min(stock.mu_hat * 100, 10);
  }
  
  // Sector outperformance (up to 5 points)
  if (stock.sector_delta && stock.sector_delta < 0) {
    // Stock underperformed sector = potential recovery opportunity
    score += Math.min(Math.abs(stock.sector_delta) / 2, 5);
  }
  
  return Math.max(0, Math.min(100, score));
}

function getOpportunityRating(score: number, mode?: string | null): 'strong_buy' | 'buy' | 'hold' | 'avoid' {
  // If mode is HOLD or DOWNTREND, don't show as buy opportunity regardless of score
  if (mode === 'HOLD' || mode === 'DOWNTREND') {
    if (score >= 60) return 'hold';  // High score but not in dip = hold
    return 'hold';
  }
  if (score >= 75) return 'strong_buy';
  if (score >= 60) return 'buy';
  if (score >= 40) return 'hold';
  return 'avoid';
}

function getOpportunityColor(rating: string): string {
  switch (rating) {
    case 'strong_buy': return 'text-success';
    case 'buy': return 'text-success/80';
    case 'hold': return 'text-warning';
    case 'avoid': return 'text-danger';
    default: return 'text-muted-foreground';
  }
}

function getOpportunityLabel(rating: string): string {
  switch (rating) {
    case 'strong_buy': return 'Strong Buy';
    case 'buy': return 'Buy';
    case 'hold': return 'Hold';
    case 'avoid': return 'Avoid';
    default: return 'N/A';
  }
}

// ============================================================================
// Subcomponents
// ============================================================================

interface InfoTooltipProps {
  content: string;
  children: React.ReactNode;
}

function InfoTooltip({ content, children }: InfoTooltipProps) {
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent className="max-w-xs text-sm">
          <p>{content}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface SignalBadgeProps {
  signal: StockCardData['top_signal'];
}

function SignalBadge({ signal }: SignalBadgeProps) {
  if (!signal) return null;
  
  const strengthPct = Math.round(signal.strength * 100);
  
  return (
    <InfoTooltip content={signal.description || `${signal.name}: ${strengthPct}% signal strength`}>
      <Badge 
        variant={signal.is_buy ? 'default' : 'secondary'} 
        className={cn(
          'gap-1 text-[10px] px-1.5 py-0 h-5 cursor-help',
          signal.is_buy ? 'bg-emerald-500/20 text-emerald-600 hover:bg-emerald-500/30 border-emerald-500/30' : ''
        )}
      >
        <Zap className="h-3 w-3" />
        {signal.name}
      </Badge>
    </InfoTooltip>
  );
}

interface SectorDeltaProps {
  delta: number | null;
  sectorEtf: string | null;
  sector: string | null;
}

function SectorDelta({ delta, sectorEtf, sector }: SectorDeltaProps) {
  if (delta === null || !sectorEtf) return null;
  
  const isOutperforming = delta >= 0;
  const tooltipText = `${isOutperforming ? 'Outperforming' : 'Underperforming'} ${sector || 'sector'} benchmark (${sectorEtf}) by ${Math.abs(delta).toFixed(1)}%`;
  
  return (
    <InfoTooltip content={tooltipText}>
      <div className={cn(
        'flex items-center gap-0.5 text-[10px] cursor-help',
        isOutperforming ? 'text-success' : 'text-danger'
      )}>
        <BarChart2 className="h-3 w-3" />
        <span>{formatPercent(delta)}</span>
        <span className="text-muted-foreground">vs {sectorEtf}</span>
      </div>
    </InfoTooltip>
  );
}

interface OpportunityMeterProps {
  score: number;
  rating: string;
  compact?: boolean;
}

function OpportunityMeter({ score, rating, compact = false }: OpportunityMeterProps) {
  const tooltipContent = `Opportunity Score: ${Math.round(score)}/100. Combines dip depth, technical signals, AI analysis, and expected returns.`;
  
  return (
    <InfoTooltip content={tooltipContent}>
      <div className={cn('flex items-center gap-1.5 cursor-help', compact ? 'flex-col' : '')}>
        <div className="flex items-center gap-1">
          <Target className={cn('h-3.5 w-3.5', getOpportunityColor(rating))} />
          {!compact && (
            <span className={cn('text-xs font-medium', getOpportunityColor(rating))}>
              {getOpportunityLabel(rating)}
            </span>
          )}
        </div>
        <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
          <div 
            className={cn(
              'h-full transition-all duration-300 rounded-full',
              rating === 'strong_buy' ? 'bg-success' :
              rating === 'buy' ? 'bg-success/80' :
              rating === 'hold' ? 'bg-warning' : 'bg-danger'
            )}
            style={{ width: `${score}%` }}
          />
        </div>
      </div>
    </InfoTooltip>
  );
}

interface DipContextProps {
  depth: number;
  days: number | null;
  bucket: string | null;
  expectedRecoveryDays?: number | null;
  typicalDipPct?: number | null;
  dipVsTypical?: number | null;
  isUnusualDip?: boolean;
  opportunityType?: 'OUTLIER' | 'BOUNCE' | 'BOTH' | 'NONE';
  winRate?: number | null;
  // Extreme Value Analysis (EVA) fields
  isTailEvent?: boolean;
  returnPeriodYears?: number | null;
  regimeDipPercentile?: number | null;
}

// ============================================================================
function DipContext({ 
  depth, 
  days, 
  bucket, 
  expectedRecoveryDays, 
  typicalDipPct, 
  dipVsTypical, 
  isUnusualDip,
  opportunityType,
  winRate,
  isTailEvent,
  returnPeriodYears,
  regimeDipPercentile,
}: DipContextProps) {
  const dipPct = depth * 100;
  
  let severity: 'shallow' | 'moderate' | 'deep' | 'extreme' = 'shallow';
  if (bucket) {
    severity = bucket as typeof severity;
  } else if (dipPct >= 50) {
    severity = 'extreme';
  } else if (dipPct >= 35) {
    severity = 'deep';
  } else if (dipPct >= 20) {
    severity = 'moderate';
  }
  
  const severityColors = {
    shallow: 'text-warning',
    moderate: 'text-warning',
    deep: 'text-danger',
    very_deep: 'text-danger',
    extreme: 'text-danger',
  };
  
  // Build detailed tooltip
  let tooltipLines: string[] = [
    `${dipPct.toFixed(1)}% below recent high`,
  ];
  
  if (days !== null) {
    tooltipLines.push(`Persisting for ${days} days`);
  }
  
  // Add EVA context if this is a tail event
  if (isTailEvent && returnPeriodYears) {
    if (returnPeriodYears >= 50) {
      tooltipLines.push(`[RARE] Extreme event (once in 50+ years)`);
    } else if (returnPeriodYears >= 10) {
      tooltipLines.push(`[RARE] ~${Math.round(returnPeriodYears)}-year event`);
    } else {
      tooltipLines.push(`[UNUSUAL] ~${returnPeriodYears.toFixed(1)}-year event`);
    }
  }
  
  // Add regime percentile context
  if (regimeDipPercentile !== null && regimeDipPercentile !== undefined) {
    tooltipLines.push(`${regimeDipPercentile.toFixed(0)}th percentile in normal regime`);
  }
  
  // Add opportunity type context to tooltip
  if (opportunityType === 'OUTLIER') {
    tooltipLines.push(`[OUTLIER] Stable stock at rare statistical discount`);
  } else if (opportunityType === 'BOUNCE') {
    tooltipLines.push(`[BOUNCE] Mean-reversion potential from significant dip`);
  } else if (opportunityType === 'BOTH') {
    tooltipLines.push(`[BOTH] Unusual dip with high bounce potential`);
  }
  
  if (isUnusualDip && dipVsTypical) {
    tooltipLines.push(`${dipVsTypical.toFixed(1)}x larger than typical dip`);
  } else if (dipVsTypical) {
    tooltipLines.push(`${dipVsTypical.toFixed(1)}x vs typical dip (${typicalDipPct?.toFixed(1) || '?'}%)`);
  }
  
  if (expectedRecoveryDays) {
    tooltipLines.push(`Expected recovery: ~${expectedRecoveryDays} trading days`);
  }
  
  if (winRate) {
    tooltipLines.push(`Historical win rate: ${(winRate * 100).toFixed(0)}%`);
  }
  
  const tooltipContent = tooltipLines.join('. ');
  
  // Determine which badge to show based on opportunity_type (preferred) or is_unusual_dip (fallback)
  const showOpportunityBadge = opportunityType && opportunityType !== 'NONE';
  
  // Format return period for display
  const formatReturnPeriod = (years: number | null | undefined): string => {
    if (!years) return '';
    if (years >= 50) return '50y+';
    if (years >= 10) return `${Math.round(years)}y`;
    if (years >= 1) return `${years.toFixed(1)}y`;
    const months = Math.round(years * 12);
    return months <= 1 ? '~mo' : `${months}mo`;
  };
  
  return (
    <InfoTooltip content={tooltipContent}>
      <div className="flex items-center gap-1 cursor-help">
        <TrendingDown className={cn('h-3 w-3', severityColors[severity] || severityColors.moderate)} />
        <span className={cn('text-xs font-medium', severityColors[severity] || severityColors.moderate)}>
          {formatDipPercent(depth)}
        </span>
        {/* Show tail event badge if this is a rare event */}
        {isTailEvent && returnPeriodYears && (
          <span className="text-[9px] px-1 py-0.5 bg-rose-500/20 text-rose-500 rounded font-semibold">
            {formatReturnPeriod(returnPeriodYears)} EVENT
          </span>
        )}
        {/* Show opportunity type badge (OUTLIER/BOUNCE/BOTH) */}
        {showOpportunityBadge && !isTailEvent ? (
          opportunityType === 'OUTLIER' ? (
            <span className="text-[9px] px-1 py-0.5 bg-blue-500/20 text-blue-500 rounded font-semibold">
              OUTLIER
            </span>
          ) : opportunityType === 'BOUNCE' ? (
            <span className="text-[9px] px-1 py-0.5 bg-emerald-500/20 text-emerald-500 rounded font-semibold">
              BOUNCE
            </span>
          ) : opportunityType === 'BOTH' ? (
            <span className="text-[9px] px-1 py-0.5 bg-purple-500/20 text-purple-500 rounded font-semibold">
              BOTH
            </span>
          ) : null
        ) : !isTailEvent && isUnusualDip ? (
          <span className="text-[9px] px-1 py-0.5 bg-amber-500/20 text-amber-500 rounded font-semibold">
            UNUSUAL
          </span>
        ) : null}
        {/* Show regime percentile if available and significant */}
        {regimeDipPercentile !== null && regimeDipPercentile !== undefined && regimeDipPercentile >= 75 && !isTailEvent && (
          <span className="text-[10px] text-muted-foreground">
            P{Math.round(regimeDipPercentile)}
          </span>
        )}
        {expectedRecoveryDays && !showOpportunityBadge && !isUnusualDip && !isTailEvent && (
          <span className="text-[10px] text-muted-foreground">
            ~{expectedRecoveryDays}d
          </span>
        )}
        {days !== null && !expectedRecoveryDays && !showOpportunityBadge && !isUnusualDip && !isTailEvent && (
          <span className="text-[10px] text-muted-foreground">
            ({days}d)
          </span>
        )}
      </div>
    </InfoTooltip>
  );
}
// ============================================================================
// Mini Sparkline (Background Chart)
// ============================================================================

interface MiniSparklineProps {
  data: { x: number; y: number }[];
  color: string;
  id: string;
}

function MiniSparkline({ data, color, id }: MiniSparklineProps) {
  if (data.length === 0) return null;
  
  return (
    <div className="absolute bottom-0 left-0 right-0 h-1/2 opacity-60 pointer-events-none">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`sparkline-${id}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.4} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="y"
            stroke={color}
            strokeWidth={1.5}
            fill={`url(#sparkline-${id})`}
            {...CHART_MINI_ANIMATION}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export const StockCardV2 = memo(function StockCardV2({
  stock,
  chartData,
  isLoading,
  isSelected,
  compact = false,
  onClick,
}: StockCardV2Props) {
  const isPositive = (stock.change_percent ?? 0) >= 0;
  
  // Get theme-aware colors for SVG chart (respects colorblind mode)
  const chartColors = useChartColors();
  const chartColor = isPositive ? chartColors.success : chartColors.danger;
  
  // Calculate opportunity metrics
  const opportunityScore = stock.opportunity_score ?? calculateOpportunityScore(stock);
  const opportunityRating = stock.opportunity_rating ?? getOpportunityRating(opportunityScore, stock.quant_mode);
  
  // Prepare chart data for sparkline
  const sparklineData = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    return chartData.slice(-60).map((p, i) => ({ x: i, y: p.close }));
  }, [chartData]);
  
  // Prefetch on hover
  const handleMouseEnter = useCallback(() => {
    prefetchStockChart(stock.symbol, 90);
    prefetchStockInfo(stock.symbol);
  }, [stock.symbol]);
  
  if (isLoading) {
    return <StockCardV2Skeleton compact={compact} />;
  }
  
  return (
    <motion.div
      whileHover={{ scale: 1.005 }}
      whileTap={{ scale: 0.995 }}
      transition={{ duration: 0.12 }}
      className="h-full"
      onMouseEnter={handleMouseEnter}
    >
      <Card
        className={cn(
          'relative overflow-hidden cursor-pointer transition-all duration-200 hover:shadow-lg h-full',
          isSelected ? 'ring-2 ring-primary' : 'hover:border-primary/30'
        )}
        onClick={onClick}
      >
        {/* Sparkline Background */}
        <MiniSparkline data={sparklineData} color={chartColor} id={stock.symbol} />
        
        <CardContent className={cn('relative z-10', compact ? 'p-3' : 'p-4')}>
          {/* Header Row: Logo + Identity + Price */}
          <div className="flex items-start gap-3">
            {/* Logo */}
            <StockLogo symbol={stock.symbol} size={compact ? 'sm' : 'md'} />
            
            {/* Identity */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-bold text-base tracking-tight">{stock.symbol}</span>
                {stock.sector && (
                  <Badge variant="outline" className="text-[9px] px-1 py-0 h-4 font-normal">
                    {stock.sector}
                  </Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground truncate mt-0.5">
                {stock.name || stock.symbol}
              </p>
            </div>
            
            {/* Price Block */}
            <div className="text-right shrink-0">
              <div className="font-semibold text-base font-mono">
                {formatPrice(stock.last_price)}
              </div>
              {stock.change_percent !== null && stock.change_percent !== undefined ? (
                <div className={cn(
                  'flex items-center justify-end gap-0.5 text-xs',
                  stock.change_percent >= 0 ? 'text-success' : 'text-danger'
                )}>
                  {stock.change_percent >= 0 ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  <span className="font-medium">{formatPercent(stock.change_percent)}</span>
                </div>
              ) : (
                <span className="text-xs text-muted-foreground">—</span>
              )}
            </div>
          </div>
          
          {/* Metrics Row */}
          <div className="flex items-center gap-3 mt-3 pt-2 border-t border-border/40">
            {/* Dip Context with recovery and unusual dip info */}
            <DipContext 
              depth={stock.depth} 
              days={stock.days_since_dip} 
              bucket={stock.dip_bucket}
              expectedRecoveryDays={stock.expected_recovery_days}
              typicalDipPct={stock.typical_dip_pct}
              dipVsTypical={stock.dip_vs_typical}
              isUnusualDip={stock.is_unusual_dip}
              opportunityType={stock.opportunity_type}
              winRate={stock.win_rate}
              isTailEvent={stock.is_tail_event}
              returnPeriodYears={stock.return_period_years}
              regimeDipPercentile={stock.regime_dip_percentile}
            />
            
            {/* Sector Delta */}
            {stock.sector_delta !== undefined && (
              <SectorDelta 
                delta={stock.sector_delta} 
                sectorEtf={stock.sector_etf || getSectorETF(stock.sector)?.etf || null}
                sector={stock.sector}
              />
            )}
            
            {/* Market Cap - always show if available */}
            {!compact && stock.market_cap && (
              <div className="text-[10px] text-muted-foreground">
                <span className="text-foreground/60">MCap:</span>{' '}
                <span className="font-medium">{formatMarketCap(stock.market_cap)}</span>
              </div>
            )}
            
            <div className="flex-1" />
            
            {/* Opportunity Meter */}
            <OpportunityMeter score={opportunityScore} rating={opportunityRating} compact={compact} />
          </div>
          
          {/* Signals Row - only show unique info not shown elsewhere */}
          <div className="flex items-center gap-2 mt-2">
            {/* Intrinsic Value Badge */}
            {stock.valuation_status && stock.upside_pct != null && (
              <InfoTooltip content={`Intrinsic value: $${stock.intrinsic_value?.toFixed(2) ?? '?'} (${stock.upside_pct >= 0 ? '+' : ''}${stock.upside_pct.toFixed(0)}% vs current)`}>
                <Badge 
                  variant="outline" 
                  className={cn(
                    "gap-0.5 text-[10px] px-1.5 py-0 h-5 cursor-help",
                    stock.valuation_status === 'undervalued' && "border-success/50 text-success bg-success/10",
                    stock.valuation_status === 'overvalued' && "border-danger/50 text-danger bg-danger/10",
                    stock.valuation_status === 'fair' && "border-muted-foreground/50 text-muted-foreground"
                  )}
                >
                  {stock.valuation_status === 'undervalued' && <TrendingUp className="h-3 w-3" />}
                  {stock.valuation_status === 'overvalued' && <TrendingDown className="h-3 w-3" />}
                  {stock.upside_pct >= 0 ? '+' : ''}{stock.upside_pct.toFixed(0)}%
                </Badge>
              </InfoTooltip>
            )}
            
            {/* Technical Signal with description */}
            {stock.top_signal && <SignalBadge signal={stock.top_signal} />}
            
            {/* Win Rate - REMOVED: Strategy backtest win rate is misleading for dip stocks
               Users think it means "% of similar dips recovered" but it's actually
               "% of trades from a trading strategy were profitable" - different concept.
               TODO: Replace with dip-specific metrics like dip recovery rate when available */}
            
            <div className="flex-1" />
            
            {/* Navigate Arrow */}
            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
});

// ============================================================================
// Skeleton
// ============================================================================

export function StockCardV2Skeleton({ compact = false }: { compact?: boolean }) {
  return (
    <Card className="overflow-hidden h-full">
      <CardContent className={cn('relative', compact ? 'p-3' : 'p-4')}>
        <div className="flex items-start gap-3">
          <Skeleton className={cn(compact ? 'w-8 h-8' : 'w-10 h-10', 'rounded-lg')} />
          <div className="flex-1 space-y-1.5">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-3 w-28" />
          </div>
          <div className="text-right space-y-1">
            <Skeleton className="h-4 w-14 ml-auto" />
            <Skeleton className="h-3 w-10 ml-auto" />
          </div>
        </div>
        <div className="flex gap-3 mt-3 pt-2 border-t border-border/40">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-20" />
          <div className="flex-1" />
          <Skeleton className="h-4 w-16" />
        </div>
        <div className="flex gap-2 mt-2">
          <Skeleton className="h-5 w-16" />
          <Skeleton className="h-5 w-14" />
        </div>
      </CardContent>
    </Card>
  );
}

export default StockCardV2;
