/**
 * Zod schemas for market data API responses.
 * 
 * Runtime validation ensures type safety even when backend changes.
 * All API responses are validated before use.
 */

import { z } from 'zod';

// ============================================================================
// Stock Info Schema
// ============================================================================

export const StockInfoSchema = z.object({
  symbol: z.string(),
  name: z.string().nullable(),
  sector: z.string().nullable(),
  industry: z.string().nullable(),
  country: z.string().nullable(),
  market_cap: z.number().nullable(),
  current_price: z.number().nullable(),
  pe_ratio: z.number().nullable(),
  forward_pe: z.number().nullable(),
  peg_ratio: z.number().nullable(),
  dividend_yield: z.number().nullable(),
  beta: z.number().nullable(),
  avg_volume: z.number().nullable(),
  summary: z.string().nullable(),
  summary_ai: z.string().nullable(),
  website: z.string().nullable(),
  recommendation: z.string().nullable(),
  // Extended fundamentals
  profit_margin: z.number().nullable(),
  gross_margin: z.number().nullable(),
  return_on_equity: z.number().nullable(),
  debt_to_equity: z.number().nullable(),
  current_ratio: z.number().nullable(),
  revenue_growth: z.number().nullable(),
  free_cash_flow: z.number().nullable(),
  target_mean_price: z.number().nullable(),
  num_analyst_opinions: z.number().nullable(),
});

export type StockInfo = z.infer<typeof StockInfoSchema>;

// ============================================================================
// Chart Data Schema
// ============================================================================

export const ChartDataPointSchema = z.object({
  date: z.string(),
  close: z.number(),
  threshold: z.number().nullable(),
  ref_high: z.number().nullable(),
  drawdown: z.number().nullable(),
  since_dip: z.number().nullable(),
  ref_high_date: z.string().nullable(),
  dip_start_date: z.string().nullable(),
});

export type ChartDataPoint = z.infer<typeof ChartDataPointSchema>;

export const ChartDataSchema = z.array(ChartDataPointSchema);

// ============================================================================
// Dip Stock / Ranking Schema
// ============================================================================

export const DipStockSchema = z.object({
  symbol: z.string(),
  name: z.string().nullable(),
  depth: z.number(),
  last_price: z.number(),
  previous_close: z.number().nullable(),
  change_percent: z.number().nullable(),
  days_since_dip: z.number().nullable(),
  high_52w: z.number().nullable(),
  low_52w: z.number().nullable(),
  market_cap: z.number().nullable(),
  sector: z.string().nullable(),
  pe_ratio: z.number().nullable(),
  volume: z.number().nullable(),
  symbol_type: z.enum(['stock', 'index', 'etf']).optional(),
  updated_at: z.string().optional(),
  // UI calculations (added by frontend)
  dip_score: z.number().optional(),
  recovery_potential: z.number().optional(),
});

export type DipStock = z.infer<typeof DipStockSchema>;

// The ranking API returns an array directly, not wrapped in an object
export const RankingArraySchema = z.array(DipStockSchema);

// For backward compatibility, we transform the array to this structure
export const RankingResponseSchema = z.object({
  ranking: z.array(DipStockSchema),
  last_updated: z.string().nullable(),
});

export type RankingResponse = z.infer<typeof RankingResponseSchema>;

// ============================================================================
// Signal Triggers Schema
// ============================================================================

export const SignalTriggerSchema = z.object({
  signal_name: z.string(),
  trigger_date: z.string(),
  is_buy: z.boolean(),
  signal_value: z.number().nullable(),
  price_at_signal: z.number().nullable(),
});

export type SignalTrigger = z.infer<typeof SignalTriggerSchema>;

export const SignalTriggersResponseSchema = z.object({
  symbol: z.string(),
  triggers: z.array(SignalTriggerSchema),
  benchmark_triggers: z.array(SignalTriggerSchema).optional(),
});

export type SignalTriggersResponse = z.infer<typeof SignalTriggersResponseSchema>;

// ============================================================================
// Current Signals Schema
// ============================================================================

export const SignalInfoSchema = z.object({
  name: z.string(),
  value: z.number(),
  threshold: z.number(),
  description: z.string().optional(),
});

export type SignalInfo = z.infer<typeof SignalInfoSchema>;

export const CurrentSignalsSchema = z.object({
  symbol: z.string(),
  buy_signals: z.array(SignalInfoSchema),
  sell_signals: z.array(SignalInfoSchema),
  overall_action: z.enum(['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'SELL', 'STRONG_SELL']),
  reasoning: z.string(),
});

export type CurrentSignals = z.infer<typeof CurrentSignalsSchema>;

// ============================================================================
// Benchmark Schema
// ============================================================================

export const BenchmarkSchema = z.object({
  id: z.string(),
  symbol: z.string(),
});

export type Benchmark = z.infer<typeof BenchmarkSchema>;

// ============================================================================
// Fundamentals Schema
// ============================================================================

export const SymbolFundamentalsSchema = z.object({
  symbol: z.string(),
  name: z.string().nullable(),
  sector: z.string().nullable(),
  industry: z.string().nullable(),
  market_cap: z.number().nullable(),
  pe_ratio: z.number().nullable(),
  forward_pe: z.number().nullable(),
  peg_ratio: z.number().nullable(),
  dividend_yield: z.number().nullable(),
  profit_margin: z.number().nullable(),
  gross_margin: z.number().nullable(),
  revenue_growth: z.number().nullable(),
  earnings_growth: z.number().nullable(),
  debt_to_equity: z.number().nullable(),
  current_ratio: z.number().nullable(),
  return_on_equity: z.number().nullable(),
  return_on_assets: z.number().nullable(),
  free_cash_flow: z.number().nullable(),
  price_to_book: z.number().nullable(),
  price_to_sales: z.number().nullable(),
  enterprise_value: z.number().nullable(),
  ev_to_ebitda: z.number().nullable(),
  beta: z.number().nullable(),
  fifty_two_week_high: z.number().nullable(),
  fifty_two_week_low: z.number().nullable(),
  fifty_day_ma: z.number().nullable(),
  two_hundred_day_ma: z.number().nullable(),
  avg_volume: z.number().nullable(),
  current_price: z.number().nullable(),
  target_mean_price: z.number().nullable(),
  recommendation: z.string().nullable(),
  num_analyst_opinions: z.number().nullable(),
  next_earnings_date: z.string().nullable(),
  combined_ratio: z.number().nullable(),
  // Quality scores
  quality_score: z.number().nullable(),
  value_score: z.number().nullable(),
  growth_score: z.number().nullable(),
  momentum_score: z.number().nullable(),
  // Metadata
  last_updated: z.string().nullable(),
});

export type SymbolFundamentals = z.infer<typeof SymbolFundamentalsSchema>;

// ============================================================================
// Agent Analysis Schema (AI personas)
// ============================================================================

export const AgentVerdictSchema = z.object({
  agent_id: z.string(),
  agent_name: z.string(),
  agent_style: z.string().nullable().optional(),
  signal: z.enum(['strong_buy', 'buy', 'hold', 'sell', 'strong_sell', 'bullish', 'bearish', 'neutral']),
  confidence: z.number(),
  reasoning: z.string(),
  key_factors: z.array(z.string()),
  avatar_url: z.string().optional(),
});

export type AgentVerdict = z.infer<typeof AgentVerdictSchema>;

export const AgentAnalysisSchema = z.object({
  symbol: z.string(),
  analyzed_at: z.string(),
  verdicts: z.array(AgentVerdictSchema),
  overall_signal: z.enum(['strong_buy', 'buy', 'hold', 'sell', 'strong_sell', 'bullish', 'bearish', 'neutral']),
  overall_confidence: z.number(),
  summary: z.string().optional(),
  expires_at: z.string().optional(),
  bullish_count: z.number().optional(),
  bearish_count: z.number().optional(),
  neutral_count: z.number().optional(),
  agent_pending: z.boolean().optional(),
});

export type AgentAnalysis = z.infer<typeof AgentAnalysisSchema>;

// ============================================================================
// Vote Counts Schema
// ============================================================================

export const VoteCountsSchema = z.object({
  buy: z.number(),
  sell: z.number(),
  buy_weighted: z.number(),
  sell_weighted: z.number(),
  net_score: z.number(),
});

export type VoteCounts = z.infer<typeof VoteCountsSchema>;

// ============================================================================
// Dip Card Schema (for swipe cards / stock detail)
// ============================================================================

export const DipCardSchema = z.object({
  symbol: z.string(),
  name: z.string().nullable(),
  sector: z.string().nullable(),
  industry: z.string().nullable(),
  website: z.string().nullable(),
  current_price: z.number(),
  ref_high: z.number(),
  dip_pct: z.number(),
  days_below: z.number(),
  min_dip_pct: z.number().nullable(),
  opportunity_type: z.string().optional(), // 'OUTLIER' | 'BOUNCE' | 'BOTH' | 'NONE'
  is_tail_event: z.boolean().optional(),
  return_period_years: z.number().nullable().optional(),
  regime_dip_percentile: z.number().nullable().optional(),
  summary_ai: z.string().nullable(),
  swipe_bio: z.string().nullable(),
  ai_rating: z.enum(['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']).nullable(),
  ai_reasoning: z.string().nullable(),
  ai_confidence: z.number().nullable(),
  ai_pending: z.boolean().nullable().optional(),
  ai_task_id: z.string().nullable().optional(),
  vote_counts: VoteCountsSchema,
  ipo_year: z.number().nullable(),
});

export type DipCard = z.infer<typeof DipCardSchema>;

// ============================================================================
// Dip Analysis Schema
// ============================================================================

export const DipAnalysisSchema = z.object({
  symbol: z.string(),
  current_depth: z.number(),
  historical_dips: z.array(z.object({
    start_date: z.string(),
    end_date: z.string().nullable(),
    max_depth: z.number(),
    recovery_days: z.number().nullable(),
  })),
  avg_recovery_days: z.number().nullable(),
  current_vs_historical: z.string(),
});

export type DipAnalysis = z.infer<typeof DipAnalysisSchema>;

// ============================================================================
// Validation helpers
// ============================================================================

/**
 * Safe parse with logging for debugging
 */
export function safeParse<T>(schema: z.ZodSchema<T>, data: unknown, context?: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    console.error(`[Zod Validation Error] ${context || 'Unknown'}:`, result.error.issues);
    throw new Error(`Validation failed: ${result.error.issues.map((e) => e.message).join(', ')}`);
  }
  return result.data;
}
