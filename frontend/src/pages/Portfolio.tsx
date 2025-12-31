import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';
import {
  createPortfolio,
  getPortfolioAllocationRecommendation,
  getPortfolioAnalyticsJob,
  getPortfolioDetail,
  getPortfolios,
  getPortfolioRiskAnalytics,
  getStockInfo,
  runPortfolioAnalytics,
  updatePortfolio,
  deletePortfolio,
  upsertHolding,
  deleteHolding,
  type PortfolioAllocationRecommendation,
  type PortfolioAnalyticsJob,
  type PortfolioAnalyticsResponse,
  type Portfolio,
  type PortfolioDetail,
  type PortfolioRiskAnalyticsResponse,
  type StockInfo,
} from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { 
  Plus, 
  Trash2, 
  MoreHorizontal, 
  Pencil, 
  Wallet,
  TrendingUp,
  TrendingDown,
  Info,
  Upload,
  PieChart as PieChartIcon,
  BarChart3,
  Target,
  Sparkles,
  AlertTriangle,
  Brain,
  Check,
  ChevronsUpDown,
} from 'lucide-react';
import { BulkImportModal } from '@/components/BulkImportModal';
import { HoldingSparkline } from '@/components/HoldingSparkline';
import { AIPortfolioAnalysis } from '@/components/AIPortfolioAnalysis';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

/** Portfolio optimization strategies from skfolio */
const OPTIMIZATION_STRATEGIES = [
  { value: 'auto', label: 'Auto (Best Method)', description: 'Cross-validates all methods' },
  { value: 'max_sharpe', label: 'Max Sharpe Ratio', description: 'Best risk-adjusted returns' },
  { value: 'max_sortino', label: 'Max Sortino Ratio', description: 'Best downside-adjusted returns' },
  { value: 'min_variance', label: 'Min Variance', description: 'Minimize volatility' },
  { value: 'min_cvar', label: 'Min CVaR', description: 'Minimize tail risk' },
  { value: 'min_semi_variance', label: 'Min Semi-Variance', description: 'Minimize downside variance' },
  { value: 'risk_parity', label: 'Risk Parity', description: 'Equal risk contribution' },
  { value: 'cvar_parity', label: 'CVaR Parity', description: 'Equal tail-risk contribution' },
  { value: 'hrp', label: 'Hierarchical Risk Parity', description: 'Clustering-based allocation' },
  { value: 'herc', label: 'Hierarchical Equal Risk', description: 'Hierarchical equal risk' },
  { value: 'max_diversification', label: 'Max Diversification', description: 'Spread across assets' },
  { value: 'robust_cvar', label: 'Robust CVaR', description: 'Worst-case optimization' },
  { value: 'inverse_volatility', label: 'Inverse Volatility', description: 'Weight by 1/volatility' },
  { value: 'equal_weight', label: 'Equal Weight', description: 'Simple 1/N allocation' },
] as const;

/** Retail investor-friendly metric explanations */
const METRIC_TOOLTIPS = {
  cagr: 'Compound Annual Growth Rate – Your average yearly return if gains were reinvested. Think of it as "how much your money grew per year on average."',
  sharpe: 'Risk-adjusted return measure. Above 1.0 is good, above 2.0 is excellent. Higher means better returns for the risk taken.',
  sortino: 'Like Sharpe, but only counts bad volatility (losses). Higher is better – it shows how well you\'re rewarded for downside risk.',
  volatility: 'How much your portfolio bounces around. Lower volatility = smoother ride, higher = more ups and downs.',
  maxDrawdown: 'The worst peak-to-trough drop your portfolio experienced. A -20% drawdown means you lost 20% from a high point before recovering.',
  beta: 'How much your portfolio moves with the market. Beta of 1.0 = moves like the market. Below 1 = less volatile, above 1 = more volatile.',
} as const;

/** Metric display with info tooltip for retail investors */
function MetricWithTooltip({ 
  label, 
  value, 
  tooltipKey 
}: { 
  label: string; 
  value: string; 
  tooltipKey: keyof typeof METRIC_TOOLTIPS 
}) {
  return (
    <div>
      <div className="flex items-center gap-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="h-3 w-3 text-muted-foreground/60 cursor-help" />
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-[280px]">
            <p>{METRIC_TOOLTIPS[tooltipKey]}</p>
          </TooltipContent>
        </Tooltip>
      </div>
      <p className="font-semibold">{value}</p>
    </div>
  );
}

/** Skeleton for Risk Summary card - matches layout to prevent CLS */
function RiskSummarySkeleton() {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Skeleton className="h-5 w-20" />
        <Skeleton className="h-4 w-16" />
      </div>
      <Skeleton className="h-5 w-full" />
      <div className="space-y-1">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-4 w-2/3" />
      </div>
      <div className="rounded-md bg-muted/50 p-3">
        <Skeleton className="h-3 w-28 mb-2" />
        <div className="space-y-1">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
        </div>
      </div>
    </div>
  );
}

/** Skeleton for Performance Snapshot card */
function PerformanceSnapshotSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-3">
      {[...Array(6)].map((_, i) => (
        <div key={i}>
          <Skeleton className="h-3 w-12 mb-1" />
          <Skeleton className="h-5 w-16" />
        </div>
      ))}
    </div>
  );
}

/** Skeleton for Diversification & Market card */
function DiversificationSkeleton() {
  return (
    <div className="space-y-3">
      <div>
        <Skeleton className="h-3 w-24 mb-1" />
        <Skeleton className="h-5 w-8" />
      </div>
      <div>
        <Skeleton className="h-3 w-32 mb-1" />
        <Skeleton className="h-5 w-20" />
      </div>
      <div className="space-y-1">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-4/5" />
      </div>
      <div className="rounded-md bg-muted/50 p-3">
        <Skeleton className="h-3 w-24 mb-2" />
        <Skeleton className="h-5 w-32 mb-1" />
        <Skeleton className="h-3 w-full" />
      </div>
    </div>
  );
}

/** Skeleton for stat card (Total Value, Gain/Loss, Risk Alerts) */
function StatCardSkeleton() {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-3">
          <Skeleton className="h-9 w-9 rounded-lg" />
          <div className="flex-1">
            <Skeleton className="h-3 w-20 mb-2" />
            <Skeleton className="h-6 w-24" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/** Skeleton for AI Portfolio Analysis card */
function AIAnalysisSkeleton() {
  return (
    <div className="space-y-4">
      {/* Health badge and headline */}
      <div className="flex items-start gap-3">
        <Skeleton className="h-6 w-16 rounded-full" />
        <Skeleton className="h-5 w-3/4" />
      </div>
      
      {/* Insights section */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-20" />
        <div className="space-y-1">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-4/5" />
        </div>
      </div>
      
      {/* Actions section */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-24" />
        <div className="space-y-1">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </div>
      
      {/* Risks section */}
      <div className="space-y-2">
        <Skeleton className="h-4 w-16" />
        <div className="space-y-1">
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      </div>
    </div>
  );
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
  'var(--chart-6)',
  'var(--chart-7)',
  'var(--chart-8)',
];

/** Format ISO date string to localized date/time */
function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

const deferStateUpdate = (callback: () => void) => {
  Promise.resolve().then(callback);
};

const mergeAnalyticsResults = (
  previous: PortfolioAnalyticsResponse | null,
  next: PortfolioAnalyticsResponse
): PortfolioAnalyticsResponse => {
  if (!previous) return next;
  const resultMap = new Map<string, typeof next.results[number]>();
  for (const result of previous.results) {
    resultMap.set(result.tool, result);
  }
  for (const result of next.results) {
    resultMap.set(result.tool, result);
  }
  return {
    ...previous,
    ...next,
    results: Array.from(resultMap.values()),
  };
};

export function PortfolioPage() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<PortfolioDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stockInfoMap, setStockInfoMap] = useState<Record<string, StockInfo | null>>({});
  const [riskAnalytics, setRiskAnalytics] = useState<PortfolioRiskAnalyticsResponse | null>(null);
  const [riskAnalyticsError, setRiskAnalyticsError] = useState<string | null>(null);
  const [isRiskAnalyticsLoading, setIsRiskAnalyticsLoading] = useState(false);
  const [toolAnalytics, setToolAnalytics] = useState<PortfolioAnalyticsResponse | null>(null);
  const [toolAnalyticsError, setToolAnalyticsError] = useState<string | null>(null);
  const [toolAnalyticsJob, setToolAnalyticsJob] = useState<PortfolioAnalyticsJob | null>(null);
  const [isToolAnalyticsLoading, setIsToolAnalyticsLoading] = useState(false);
  const [allocationAmount, setAllocationAmount] = useState('1000');
  const [allocationMethod, setAllocationMethod] = useState('auto');  // auto = cross-validate and select best
  const [strategyComboboxOpen, setStrategyComboboxOpen] = useState(false);
  const [allocationResult, setAllocationResult] = useState<PortfolioAllocationRecommendation | null>(null);
  const [allocationError, setAllocationError] = useState<string | null>(null);
  const [isAllocationLoading, setIsAllocationLoading] = useState(false);
  
  // Sparkline data for holdings
  const [sparklineData, setSparklineData] = useState<Record<string, import('@/services/api').HoldingSparklineData>>({});

  // Portfolio dialog
  const [portfolioDialogOpen, setPortfolioDialogOpen] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState<Portfolio | null>(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [portfolioCurrency, setPortfolioCurrency] = useState('USD');

  // Holding dialog
  const [holdingDialogOpen, setHoldingDialogOpen] = useState(false);
  const [editingHolding, setEditingHolding] = useState<{ symbol: string; quantity: number; avg_cost?: number | null } | null>(null);
  const [holdingValidationError, setHoldingValidationError] = useState<string | null>(null);
  
  // Bulk import dialog
  const [bulkImportOpen, setBulkImportOpen] = useState(false);
  const [holdingSymbol, setHoldingSymbol] = useState('');
  const [holdingQty, setHoldingQty] = useState('');
  const [holdingAvgCost, setHoldingAvgCost] = useState('');

  const loadPortfolios = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getPortfolios();
      setPortfolios(data);
      if (!selectedId && data.length > 0) {
        setSelectedId(data[0].id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolios');
    } finally {
      setIsLoading(false);
    }
  }, [selectedId]);

  const loadDetail = useCallback(async (portfolioId: number) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getPortfolioDetail(portfolioId);
      setDetail(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolio details');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadRiskAnalytics = useCallback(async (portfolioId: number) => {
    deferStateUpdate(() => setIsRiskAnalyticsLoading(true));
    setRiskAnalyticsError(null);
    try {
      const data = await getPortfolioRiskAnalytics(portfolioId);
      setRiskAnalytics(data);
    } catch (err) {
      setRiskAnalyticsError(err instanceof Error ? err.message : 'Failed to load portfolio insights');
      setRiskAnalytics(null);
    } finally {
      setIsRiskAnalyticsLoading(false);
    }
  }, []);

  const loadToolAnalytics = useCallback(
    async (portfolioId: number, tools: string[], forceRefresh = false) => {
      deferStateUpdate(() => setIsToolAnalyticsLoading(true));
      setToolAnalyticsError(null);
      try {
        const data = await runPortfolioAnalytics(portfolioId, {
          tools,
          force_refresh: forceRefresh,
        });
        setToolAnalytics((previous) => mergeAnalyticsResults(previous, data));
        if (data.job_id) {
          setToolAnalyticsJob({
            job_id: data.job_id,
            portfolio_id: portfolioId,
            status: data.job_status || 'pending',
            tools: data.scheduled_tools.length ? data.scheduled_tools : tools,
            results_count: 0,
            created_at: new Date().toISOString(),
          });
        } else {
          setToolAnalyticsJob(null);
        }
      } catch (err) {
        setToolAnalyticsError(err instanceof Error ? err.message : 'Failed to run analytics tools');
      } finally {
        setIsToolAnalyticsLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    loadPortfolios();
  }, [loadPortfolios]);

  useEffect(() => {
    if (selectedId) {
      loadDetail(selectedId);
    }
  }, [selectedId, loadDetail]);

  useEffect(() => {
    if (!selectedId) {
      deferStateUpdate(() => {
        setRiskAnalytics(null);
        setToolAnalytics(null);
        setToolAnalyticsJob(null);
        setAllocationResult(null);
      });
      return;
    }
    if (!detail?.holdings || detail.holdings.length === 0) {
      return;
    }
    const timeout = setTimeout(() => {
      loadRiskAnalytics(selectedId);
      loadToolAnalytics(selectedId, ['quantstats', 'pyfolio']);
    }, 0);
    return () => clearTimeout(timeout);
  }, [selectedId, detail?.holdings, loadRiskAnalytics, loadToolAnalytics]);

  useEffect(() => {
    if (!detail?.holdings || detail.holdings.length === 0) {
      deferStateUpdate(() => setStockInfoMap({}));
      return;
    }
    let isActive = true;
    const symbols = detail.holdings.map((holding) => holding.symbol);
    const timeout = setTimeout(() => {
      Promise.all(
        symbols.map(async (symbol) => {
          try {
            const info = await getStockInfo(symbol);
            return [symbol, info] as const;
          } catch {
            return [symbol, null] as const;
          }
        })
      ).then((entries) => {
        if (!isActive) return;
        const nextMap: Record<string, StockInfo | null> = {};
        for (const [symbol, info] of entries) {
          nextMap[symbol] = info;
        }
        setStockInfoMap(nextMap);
      });
    }, 0);
    return () => {
      isActive = false;
      clearTimeout(timeout);
    };
  }, [detail?.holdings]);

  // Fetch sparkline data for holdings
  useEffect(() => {
    if (!selectedId || !detail?.holdings || detail.holdings.length === 0) {
      setSparklineData({});
      return;
    }
    let isActive = true;
    const symbols = detail.holdings.map((h) => h.symbol);
    
    import('@/services/api').then(({ getHoldingsSparklines }) => {
      getHoldingsSparklines(selectedId, symbols, 180)
        .then((response) => {
          if (isActive) {
            setSparklineData(response.sparklines);
          }
        })
        .catch((err) => {
          console.error('Failed to load sparkline data:', err);
        });
    });
    
    return () => {
      isActive = false;
    };
  }, [selectedId, detail?.holdings]);

  useEffect(() => {
    if (!selectedId || !toolAnalyticsJob?.job_id) return;
    if (toolAnalyticsJob.status === 'completed' || toolAnalyticsJob.status === 'failed') {
      return;
    }
    const timeout = setTimeout(async () => {
      try {
        const status = await getPortfolioAnalyticsJob(selectedId, toolAnalyticsJob.job_id);
        setToolAnalyticsJob(status);
        if (status.status === 'completed') {
          await loadToolAnalytics(selectedId, status.tools);
        }
      } catch (err) {
        setToolAnalyticsError(err instanceof Error ? err.message : 'Failed to load analytics status');
      }
    }, 10000);
    return () => clearTimeout(timeout);
  }, [selectedId, toolAnalyticsJob, loadToolAnalytics]);

  const selectedPortfolio = useMemo(
    () => portfolios.find((p) => p.id === selectedId) || null,
    [portfolios, selectedId]
  );

  const holdingSnapshots = useMemo(() => {
    if (!detail?.holdings) return [];
    return detail.holdings.map((holding) => {
      const info = stockInfoMap[holding.symbol];
      const currentPrice = info?.current_price ?? null;
      const valuationPrice = currentPrice ?? holding.avg_cost ?? 0;
      const marketValue = valuationPrice * holding.quantity;
      const costBasis = (holding.avg_cost ?? 0) * holding.quantity;
      const unrealized = holding.avg_cost && currentPrice
        ? (currentPrice - holding.avg_cost) * holding.quantity
        : null;
      const unrealizedPct = holding.avg_cost
        && currentPrice
        ? ((currentPrice - holding.avg_cost) / holding.avg_cost) * 100
        : null;
      // ETFs typically have no sector/country - show "Diversified" instead of "Unknown"
      const isLikelyETF = !info?.sector && !info?.country;
      return {
        ...holding,
        name: info?.name ?? null,
        sector: info?.sector ?? (isLikelyETF ? 'Diversified' : 'Unknown'),
        country: info?.country ?? (isLikelyETF ? 'Global' : 'Unknown'),
        currentPrice,
        marketValue,
        costBasis,
        unrealized,
        unrealizedPct,
      };
    });
  }, [detail, stockInfoMap]);

  const holdingSnapshotMap = useMemo(() => {
    return new Map(holdingSnapshots.map((holding) => [holding.symbol, holding]));
  }, [holdingSnapshots]);

  // Calculate portfolio stats
  const portfolioStats = useMemo(() => {
    if (!detail?.holdings) {
      return {
        totalValue: 0,
        totalCost: 0,
        gainLoss: 0,
        gainLossPercent: 0,
        investedValue: 0,
      };
    }

    const investedValue = holdingSnapshots.reduce((sum, holding) => sum + holding.marketValue, 0);
    const totalCost = holdingSnapshots.reduce((sum, holding) => sum + holding.costBasis, 0);
    const totalValue = investedValue;
    const gainLoss = totalValue - totalCost;
    const gainLossPercent = totalCost > 0 ? (gainLoss / totalCost) * 100 : 0;

    return {
      totalValue,
      totalCost,
      gainLoss,
      gainLossPercent,
      investedValue,
    };
  }, [detail, holdingSnapshots]);

  const missingAvgCost = useMemo(
    () => detail?.holdings.filter((holding) => !holding.avg_cost) ?? [],
    [detail]
  );

  const sectorAllocation = useMemo(() => {
    if (!holdingSnapshots.length) return [];
    const totals = new Map<string, number>();
    let total = 0;
    for (const holding of holdingSnapshots) {
      if (!holding.marketValue) continue;
      totals.set(holding.sector, (totals.get(holding.sector) || 0) + holding.marketValue);
      total += holding.marketValue;
    }
    return Array.from(totals.entries())
      .map(([name, value]) => ({
        name,
        value,
        percent: total > 0 ? (value / total) * 100 : 0,
      }))
      .sort((a, b) => b.value - a.value);
  }, [holdingSnapshots]);

  const countryAllocation = useMemo(() => {
    if (!holdingSnapshots.length) return [];
    const totals = new Map<string, number>();
    let total = 0;
    for (const holding of holdingSnapshots) {
      if (!holding.marketValue) continue;
      totals.set(holding.country, (totals.get(holding.country) || 0) + holding.marketValue);
      total += holding.marketValue;
    }
    return Array.from(totals.entries())
      .map(([name, value]) => ({
        name,
        value,
        percent: total > 0 ? (value / total) * 100 : 0,
      }))
      .sort((a, b) => b.value - a.value);
  }, [holdingSnapshots]);

  const topHoldings = useMemo(() => {
    return [...holdingSnapshots]
      .filter((holding) => holding.marketValue > 0)
      .sort((a, b) => b.marketValue - a.marketValue)
      .slice(0, 8);
  }, [holdingSnapshots]);

  const riskContributionData = useMemo(() => {
    const contributions = riskAnalytics?.raw.risk_contributions;
    if (!contributions) return [];
    return Object.entries(contributions)
      .map(([symbol, value]) => ({ symbol, value: value * 100 }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
  }, [riskAnalytics]);

  const riskHighlights = riskAnalytics?.risk_highlights ?? [];

  const performanceMetrics = useMemo(() => {
    const quantstats = toolAnalytics?.results.find((result) => result.tool === 'quantstats');
    const pyfolio = toolAnalytics?.results.find((result) => result.tool === 'pyfolio');
    const data = (quantstats?.data ?? pyfolio?.data) as Record<string, number> | undefined;
    return {
      cagr: data?.cagr ?? null,
      sharpe: data?.sharpe ?? null,
      sortino: data?.sortino ?? null,
      volatility: data?.volatility ?? null,
      maxDrawdown: data?.max_drawdown ?? null,
      beta: data?.beta ?? null,
    };
  }, [toolAnalytics]);

  const allocationWeightData = useMemo(() => {
    if (!allocationResult) return [];
    const entries = Object.entries(allocationResult.target_weights || {});
    return entries
      .map(([symbol, target]) => ({
        symbol,
        target,
        current: allocationResult.current_weights?.[symbol] ?? 0,
      }))
      .sort((a, b) => b.target - a.target)
      .slice(0, 8);
  }, [allocationResult]);

  function openCreateDialog() {
    setEditingPortfolio(null);
    setPortfolioName('');
    setPortfolioCurrency('USD');
    setPortfolioDialogOpen(true);
  }

  function openEditDialog() {
    if (!selectedPortfolio) return;
    setEditingPortfolio(selectedPortfolio);
    setPortfolioName(selectedPortfolio.name);
    setPortfolioCurrency(selectedPortfolio.base_currency);
    setPortfolioDialogOpen(true);
  }

  async function handleSavePortfolio() {
    if (!portfolioName.trim()) return;
    try {
      if (editingPortfolio) {
        await updatePortfolio(editingPortfolio.id, {
          name: portfolioName.trim(),
          base_currency: portfolioCurrency.trim(),
        });
      } else {
        const created = await createPortfolio({
          name: portfolioName.trim(),
          base_currency: portfolioCurrency.trim(),
        });
        setSelectedId(created.id);
      }
      await loadPortfolios();
      setPortfolioDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save portfolio');
    }
  }

  async function handleDeletePortfolio() {
    if (!selectedPortfolio) return;
    if (!confirm('Are you sure you want to delete this portfolio?')) return;
    try {
      await deletePortfolio(selectedPortfolio.id);
      setSelectedId(null);
      setDetail(null);
      await loadPortfolios();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete portfolio');
    }
  }

  function openAddHoldingDialog() {
    setEditingHolding(null);
    setHoldingSymbol('');
    setHoldingQty('');
    setHoldingAvgCost('');
    setHoldingValidationError(null);
    setHoldingDialogOpen(true);
  }

  function openEditHoldingDialog(holding: { symbol: string; quantity: number; avg_cost?: number | null }) {
    setEditingHolding(holding);
    setHoldingSymbol(holding.symbol);
    setHoldingQty(String(holding.quantity));
    setHoldingAvgCost(holding.avg_cost ? String(holding.avg_cost) : '');
    setHoldingValidationError(null);
    setHoldingDialogOpen(true);
  }

  async function handleSaveHolding() {
    if (!selectedId || !holdingSymbol.trim() || !holdingQty) return;
    if (!holdingAvgCost || Number(holdingAvgCost) <= 0) {
      setHoldingValidationError('Average cost is required for portfolio analysis.');
      return;
    }
    try {
      await upsertHolding(selectedId, {
        symbol: holdingSymbol.trim().toUpperCase(),
        quantity: Number(holdingQty),
        avg_cost: Number(holdingAvgCost),
      });
      await loadDetail(selectedId);
      setHoldingDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save holding');
    }
  }

  async function handleDeleteHolding(symbol: string) {
    if (!selectedId) return;
    try {
      await deleteHolding(selectedId, symbol);
      await loadDetail(selectedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete holding');
    }
  }

  async function handleGenerateAllocation() {
    if (!selectedId) return;
    const inflow = Number(allocationAmount);
    if (!inflow || inflow <= 0) {
      setAllocationError('Enter a valid investment amount.');
      return;
    }
    setAllocationError(null);
    setAllocationResult(null);
    deferStateUpdate(() => setIsAllocationLoading(true));
    try {
      const data = await getPortfolioAllocationRecommendation(selectedId, {
        inflow_eur: inflow,
        method: allocationMethod,
      });
      setAllocationResult(data);
    } catch (err) {
      setAllocationError(err instanceof Error ? err.message : 'Failed to generate recommendations');
    } finally {
      setIsAllocationLoading(false);
    }
  }

  const formatCurrency = (value: number) => {
    const safeValue = Number.isFinite(value) ? value : 0;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: selectedPortfolio?.base_currency || 'USD',
      minimumFractionDigits: 2,
    }).format(safeValue);
  };

  const isHoldingValid = Boolean(
    holdingSymbol.trim() &&
      holdingQty &&
      Number(holdingQty) > 0 &&
      holdingAvgCost &&
      Number(holdingAvgCost) > 0
  );

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Portfolio</h1>
          <p className="text-muted-foreground">Track your holdings</p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            value={selectedId ? String(selectedId) : ''}
            onValueChange={(value) => setSelectedId(Number(value))}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select portfolio" />
            </SelectTrigger>
            <SelectContent>
              {portfolios.map((portfolio) => (
                <SelectItem key={portfolio.id} value={String(portfolio.id)}>
                  {portfolio.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={openCreateDialog}>
                <Plus className="h-4 w-4 mr-2" />
                New Portfolio
              </DropdownMenuItem>
              <DropdownMenuItem onClick={openEditDialog} disabled={!selectedPortfolio}>
                <Pencil className="h-4 w-4 mr-2" />
                Edit Portfolio
              </DropdownMenuItem>
              <DropdownMenuItem 
                onClick={handleDeletePortfolio} 
                disabled={!selectedPortfolio}
                className="text-destructive focus:text-destructive"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Portfolio
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-destructive/10 p-4 text-destructive text-sm">
          {error}
        </div>
      )}

      {selectedPortfolio && missingAvgCost.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-900/80">
          <div className="flex items-center gap-2 font-semibold text-amber-700">
            <AlertTriangle className="h-4 w-4" />
            Missing average cost on {missingAvgCost.length} holding{missingAvgCost.length > 1 ? 's' : ''}
          </div>
          <p className="mt-1 text-xs text-amber-800/80">
            Add an average price to unlock accurate portfolio analytics and allocation insights.
          </p>
        </div>
      )}

      {/* Stats Cards */}
      {isLoading && (
        <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <StatCardSkeleton />
          <StatCardSkeleton />
          <StatCardSkeleton />
        </div>
      )}
      {selectedPortfolio && !isLoading && (
        <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Wallet className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Value</p>
                  <p className="text-xl font-semibold">{formatCurrency(portfolioStats.totalValue)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className={`rounded-lg p-2 ${portfolioStats.gainLoss >= 0 ? 'bg-success/10' : 'bg-danger/10'}`}>
                  {portfolioStats.gainLoss >= 0 
                    ? <TrendingUp className="h-5 w-5 text-success" />
                    : <TrendingDown className="h-5 w-5 text-danger" />
                  }
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Gain/Loss</p>
                  <p className={`text-xl font-semibold ${portfolioStats.gainLoss >= 0 ? 'text-success' : 'text-danger'}`}>
                    {portfolioStats.gainLoss >= 0 ? '+' : ''}{formatCurrency(portfolioStats.gainLoss)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {portfolioStats.gainLoss >= 0 ? '+' : ''}{portfolioStats.gainLossPercent.toFixed(1)}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className={`rounded-lg p-2 ${riskHighlights.length > 0 ? 'bg-amber-500/10' : 'bg-success/10'}`}>
                  <AlertTriangle className={`h-5 w-5 ${riskHighlights.length > 0 ? 'text-amber-500' : 'text-success'}`} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm text-muted-foreground">Risk Alerts</p>
                  {isRiskAnalyticsLoading ? (
                    <p className="text-sm text-muted-foreground">Analyzing...</p>
                  ) : riskHighlights.length > 0 ? (
                    <>
                      <p className="text-xl font-semibold text-amber-500">{riskHighlights.length} issue{riskHighlights.length !== 1 ? 's' : ''}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        {riskHighlights.slice(0, 2).map(h => h.symbol).join(', ')}{riskHighlights.length > 2 ? ` +${riskHighlights.length - 2} more` : ''}
                      </p>
                    </>
                  ) : riskAnalytics ? (
                    <p className="text-xl font-semibold text-success">All clear</p>
                  ) : (
                    <p className="text-sm text-muted-foreground">Not analyzed</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* AI Portfolio Analysis */}
      {isLoading && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Brain className="h-4 w-4 text-primary" />
              AI Portfolio Analysis
            </CardTitle>
            <CardDescription>Professional insights from our AI Portfolio Advisor</CardDescription>
          </CardHeader>
          <CardContent>
            <AIAnalysisSkeleton />
          </CardContent>
        </Card>
      )}
      {selectedPortfolio && !isLoading && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Brain className="h-4 w-4 text-primary" />
              AI Portfolio Analysis
            </CardTitle>
            <CardDescription className="flex items-center gap-2">
              Professional insights from our AI Portfolio Advisor
              {selectedPortfolio.ai_analysis_at && (
                <span className="text-xs">
                  (Updated {formatDate(selectedPortfolio.ai_analysis_at)})
                </span>
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help flex-shrink-0" />
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-[280px]">
                  <p className="text-xs">
                    AI analysis runs automatically when your portfolio changes. 
                    It evaluates risk, diversification, and provides actionable recommendations.
                  </p>
                </TooltipContent>
              </Tooltip>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AIPortfolioAnalysis summary={selectedPortfolio.ai_analysis_summary} />
          </CardContent>
        </Card>
      )}

      {selectedPortfolio && (
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Target className="h-4 w-4 text-primary" />
                Risk Summary
              </CardTitle>
              <CardDescription>Plain-English portfolio diagnostics</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {isRiskAnalyticsLoading && <RiskSummarySkeleton />}
              {!isRiskAnalyticsLoading && riskAnalytics && (
                <>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{riskAnalytics.summary.risk_label}</Badge>
                    <span className="text-xs text-muted-foreground">
                      Score {riskAnalytics.summary.risk_score}/10
                    </span>
                  </div>
                  <p className="text-sm font-medium">{riskAnalytics.summary.headline}</p>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <p>{riskAnalytics.risk.volatility_explanation}</p>
                    <p>{riskAnalytics.risk.bad_day_loss}</p>
                    <p>{riskAnalytics.risk.crash_loss}</p>
                    <p>{riskAnalytics.risk.worst_ever}</p>
                  </div>
                  {riskAnalytics.action_items.length > 0 && (
                    <div className="rounded-md bg-muted/50 p-3">
                      <p className="text-xs font-semibold text-muted-foreground">Actionable next steps</p>
                      <ul className="mt-2 space-y-1 text-xs">
                        {riskAnalytics.action_items.map((item) => (
                          <li key={item}>• {item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
              {!isRiskAnalyticsLoading && !riskAnalytics && (
                <p className="text-sm text-muted-foreground">Run analytics to see risk insights.</p>
              )}
              {riskAnalyticsError && (
                <p className="text-xs text-destructive">{riskAnalyticsError}</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-primary" />
                Performance Snapshot
              </CardTitle>
              <CardDescription>Key stats from Quantstats/Pyfolio</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {isToolAnalyticsLoading && !toolAnalytics && <PerformanceSnapshotSkeleton />}
              {!isToolAnalyticsLoading && (
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <MetricWithTooltip
                    label="CAGR"
                    tooltipKey="cagr"
                    value={performanceMetrics.cagr !== null
                      ? `${(performanceMetrics.cagr * 100).toFixed(1)}%`
                      : '—'}
                  />
                  <MetricWithTooltip
                    label="Sharpe"
                    tooltipKey="sharpe"
                    value={performanceMetrics.sharpe !== null
                      ? performanceMetrics.sharpe.toFixed(2)
                      : '—'}
                  />
                  <MetricWithTooltip
                    label="Sortino"
                    tooltipKey="sortino"
                    value={performanceMetrics.sortino !== null
                      ? performanceMetrics.sortino.toFixed(2)
                      : '—'}
                  />
                  <MetricWithTooltip
                    label="Volatility"
                    tooltipKey="volatility"
                    value={performanceMetrics.volatility !== null
                      ? `${(performanceMetrics.volatility * 100).toFixed(1)}%`
                      : '—'}
                  />
                  <MetricWithTooltip
                    label="Max Drawdown"
                    tooltipKey="maxDrawdown"
                    value={performanceMetrics.maxDrawdown !== null
                      ? `${(performanceMetrics.maxDrawdown * 100).toFixed(1)}%`
                      : '—'}
                  />
                  <MetricWithTooltip
                    label="Beta"
                    tooltipKey="beta"
                    value={performanceMetrics.beta !== null ? performanceMetrics.beta.toFixed(2) : '—'}
                  />
                </div>
              )}
              {toolAnalyticsError && (
                <p className="text-xs text-destructive">{toolAnalyticsError}</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Sparkles className="h-4 w-4 text-primary" />
                Diversification & Market
              </CardTitle>
              <CardDescription>How balanced your portfolio is</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {isRiskAnalyticsLoading && <DiversificationSkeleton />}
              {!isRiskAnalyticsLoading && riskAnalytics && (
                <>
                  <div>
                    <p className="text-xs text-muted-foreground">Effective positions</p>
                    <p className="font-semibold">{riskAnalytics.diversification.effective_positions}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Diversification quality</p>
                    <p className="font-semibold">{riskAnalytics.diversification.quality}</p>
                  </div>
                  {riskAnalytics.diversification.warnings.length > 0 && (
                    <ul className="space-y-1 text-xs text-muted-foreground">
                      {riskAnalytics.diversification.warnings.map((warning) => (
                        <li key={warning}>• {warning}</li>
                      ))}
                    </ul>
                  )}
                  <div className="rounded-md bg-muted/50 p-3">
                    <p className="text-xs font-semibold text-muted-foreground">Market regime</p>
                    <p className="text-sm font-medium">{riskAnalytics.market.current_regime}</p>
                    <p className="text-xs text-muted-foreground mt-1">{riskAnalytics.market.explanation}</p>
                  </div>
                </>
              )}
              {!isRiskAnalyticsLoading && !riskAnalytics && (
                <p className="text-sm text-muted-foreground">Portfolio regime insights appear here.</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {selectedPortfolio && (
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <PieChartIcon className="h-4 w-4 text-primary" />
                Sector Allocation
              </CardTitle>
              <CardDescription>Invested exposure by sector</CardDescription>
            </CardHeader>
            <CardContent>
              {sectorAllocation.length > 0 ? (
                <div className="flex flex-col gap-4 md:flex-row md:items-center">
                  <ChartContainer
                    config={{ value: { label: 'Allocation', color: 'var(--chart-1)' } }}
                    className="h-[200px] w-full md:h-[220px] md:w-[220px] md:flex-shrink-0"
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <ChartTooltip
                          content={
                            <ChartTooltipContent
                              nameKey="name"
                              formatter={(value, name, item) => {
                                const payload = item?.payload as { percent?: number };
                                const pct = payload?.percent ?? 0;
                                return [`${formatCurrency(Number(value))} (${pct.toFixed(1)}%)`, name];
                              }}
                            />
                          }
                        />
                        <Pie
                          data={sectorAllocation}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={50}
                          outerRadius={80}
                        >
                          {sectorAllocation.map((entry, index) => (
                            <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                  <div className="flex-1 space-y-1.5 text-xs">
                    {sectorAllocation.map((entry, index) => (
                      <div key={entry.name} className="flex items-center gap-2">
                        <div 
                          className="h-3 w-3 rounded-sm flex-shrink-0" 
                          style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                        />
                        <span className="font-medium flex-1 truncate">{entry.name}</span>
                        <span className="text-muted-foreground tabular-nums">
                          {entry.percent.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No sector data available yet.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <PieChartIcon className="h-4 w-4 text-primary" />
                Country Allocation
              </CardTitle>
              <CardDescription>Invested exposure by country</CardDescription>
            </CardHeader>
            <CardContent>
              {countryAllocation.length > 0 ? (
                <div className="flex flex-col gap-4 md:flex-row md:items-center">
                  <ChartContainer
                    config={{ value: { label: 'Allocation', color: 'var(--chart-2)' } }}
                    className="h-[200px] w-full md:h-[220px] md:w-[220px] md:flex-shrink-0"
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <ChartTooltip
                          content={
                            <ChartTooltipContent
                              nameKey="name"
                              formatter={(value, name, item) => {
                                const payload = item?.payload as { percent?: number };
                                const pct = payload?.percent ?? 0;
                                return [`${formatCurrency(Number(value))} (${pct.toFixed(1)}%)`, name];
                              }}
                            />
                          }
                        />
                        <Pie
                          data={countryAllocation}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={50}
                          outerRadius={80}
                        >
                          {countryAllocation.map((entry, index) => (
                            <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                  <div className="flex-1 space-y-1.5 text-xs">
                    {countryAllocation.map((entry, index) => (
                      <div key={entry.name} className="flex items-center gap-2">
                        <div 
                          className="h-3 w-3 rounded-sm flex-shrink-0" 
                          style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                        />
                        <span className="font-medium flex-1 truncate">{entry.name}</span>
                        <span className="text-muted-foreground tabular-nums">
                          {entry.percent.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No country data available yet.</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {selectedPortfolio && (
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <BarChart3 className="h-4 w-4 text-primary" />
                Largest Positions
              </CardTitle>
              <CardDescription>Top holdings by market value</CardDescription>
            </CardHeader>
            <CardContent>
              {topHoldings.length > 0 ? (
                <ChartContainer
                  config={{ value: { label: 'Value', color: 'var(--chart-1)' } }}
                  className="h-[200px] w-full sm:h-[240px]"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topHoldings.map((holding) => ({
                      symbol: holding.symbol,
                      value: holding.marketValue,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="symbol" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={50} />
                      <YAxis tickFormatter={(value) => `${Math.round(value / 1000)}k`} width={45} />
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            formatter={(value, name) => [formatCurrency(Number(value)), name]}
                          />
                        }
                      />
                      <Bar dataKey="value" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              ) : (
                <p className="text-sm text-muted-foreground">Add holdings to see position sizing.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Target className="h-4 w-4 text-primary" />
                Risk Concentration
              </CardTitle>
              <CardDescription>Where most of your risk sits</CardDescription>
            </CardHeader>
            <CardContent>
              {riskContributionData.length > 0 ? (
                <ChartContainer
                  config={{ value: { label: 'Risk %', color: 'var(--chart-2)' } }}
                  className="h-[200px] w-full sm:h-[240px]"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={riskContributionData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="symbol" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={50} />
                      <YAxis tickFormatter={(value) => `${value.toFixed(0)}%`} width={40} />
                      <ChartTooltip
                        content={
                          <ChartTooltipContent
                            formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
                          />
                        }
                      />
                      <Bar dataKey="value" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Risk contribution appears after portfolio analytics runs.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {selectedPortfolio && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              What Should I Buy Next?
            </CardTitle>
            <CardDescription className="flex items-start gap-1.5">
              <span>Enter how much you want to invest and we'll calculate the optimal trades to rebalance your portfolio.</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help flex-shrink-0 mt-0.5" />
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-[320px]">
                  <p className="font-medium mb-1">How it works:</p>
                  <ol className="text-xs space-y-1 list-decimal list-inside">
                    <li>Enter the amount you want to invest</li>
                    <li>Choose an optimization strategy</li>
                    <li>Get specific buy/sell amounts to improve your portfolio</li>
                  </ol>
                  <p className="text-xs mt-2 text-muted-foreground">
                    Trades are calculated to reduce risk and improve diversification based on your current holdings.
                  </p>
                </TooltipContent>
              </Tooltip>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-[1fr,1fr,auto]">
              <div className="space-y-1">
                <Label>Investment Amount ({selectedPortfolio?.base_currency || 'USD'})</Label>
                <Input
                  type="number"
                  value={allocationAmount}
                  onChange={(event) => setAllocationAmount(event.target.value)}
                  placeholder="1000"
                />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-1">
                  <Label>Strategy</Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3 w-3 text-muted-foreground/60 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[320px]">
                      <ul className="text-xs space-y-1">
                        <li><strong>Auto:</strong> Cross-validates all methods, picks best Sharpe</li>
                        <li><strong>Max Sharpe/Sortino:</strong> Best risk-adjusted returns</li>
                        <li><strong>Min Variance/CVaR:</strong> Minimize volatility or tail risk</li>
                        <li><strong>Risk/CVaR Parity:</strong> Equal risk contribution per asset</li>
                        <li><strong>HRP/HERC:</strong> Hierarchical clustering-based allocation</li>
                        <li><strong>Max Diversification:</strong> Spread across uncorrelated assets</li>
                      </ul>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Popover open={strategyComboboxOpen} onOpenChange={setStrategyComboboxOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={strategyComboboxOpen}
                      className="w-full justify-between font-normal"
                    >
                      {OPTIMIZATION_STRATEGIES.find((s) => s.value === allocationMethod)?.label || 'Select strategy...'}
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[280px] p-0" align="start">
                    <Command>
                      <CommandInput placeholder="Search strategies..." />
                      <CommandList>
                        <CommandEmpty>No strategy found.</CommandEmpty>
                        <CommandGroup>
                          {OPTIMIZATION_STRATEGIES.map((strategy) => (
                            <CommandItem
                              key={strategy.value}
                              value={strategy.value}
                              onSelect={(currentValue) => {
                                setAllocationMethod(currentValue);
                                setStrategyComboboxOpen(false);
                              }}
                            >
                              <Check
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  allocationMethod === strategy.value ? "opacity-100" : "opacity-0"
                                )}
                              />
                              <div className="flex flex-col">
                                <span>{strategy.label}</span>
                                <span className="text-xs text-muted-foreground">{strategy.description}</span>
                              </div>
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </div>
              <div className="flex items-end">
                <Button onClick={handleGenerateAllocation} disabled={isAllocationLoading}>
                  {isAllocationLoading ? 'Calculating…' : 'Calculate Trades'}
                </Button>
              </div>
            </div>
            {allocationError && (
              <p className="text-xs text-destructive">{allocationError}</p>
            )}
            {allocationResult ? (
              <div className="space-y-4">
                <div className="rounded-lg border bg-muted/30 p-3 text-sm">
                  <p className="font-medium">{allocationResult.explanation}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {allocationResult.risk_improvement} • Confidence: {allocationResult.confidence}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Portfolio: {formatCurrency(allocationResult.portfolio_value_eur)} + {formatCurrency(allocationResult.inflow_eur)} new = {formatCurrency(allocationResult.portfolio_value_eur + allocationResult.inflow_eur)} total
                  </p>
                </div>
                <div className="grid gap-3 sm:grid-cols-3 text-sm">
                  <div>
                    <p className="text-xs text-muted-foreground">Current Volatility</p>
                    <p className="font-semibold">
                      {(allocationResult.current_risk.volatility * 100).toFixed(1)}% • {allocationResult.current_risk.label}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Target Volatility</p>
                    <p className="font-semibold">
                      {(allocationResult.optimal_risk.volatility * 100).toFixed(1)}% • {allocationResult.optimal_risk.label}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Diversification Ratio</p>
                    <p className="font-semibold">
                      {allocationResult.optimal_risk.diversification_ratio.toFixed(2)}
                    </p>
                  </div>
                </div>
                {allocationResult.trades.length > 0 ? (
                  <div className="space-y-2">
                    <p className="text-sm font-semibold">Suggested Trades</p>
                    <div className="space-y-2">
                      {allocationResult.trades.map((trade) => (
                        <div key={`${trade.symbol}-${trade.action}`} className="rounded-md border p-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Badge variant={trade.action === 'BUY' ? 'default' : 'destructive'}>
                                {trade.action}
                              </Badge>
                              <span className="font-semibold">{trade.symbol}</span>
                            </div>
                            <span className="font-semibold">{formatCurrency(trade.amount_eur)}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{trade.reason}</p>
                          <p className="text-xs text-muted-foreground">
                            {trade.current_weight_pct.toFixed(1)}% → {trade.target_weight_pct.toFixed(1)}%
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No trades needed for this allocation.</p>
                )}
                {allocationWeightData.length > 0 && (
                  <div>
                    <p className="text-sm font-semibold mb-2">Current vs Target Weights</p>
                    <ChartContainer
                      config={{
                        current: { label: 'Current', color: 'var(--chart-3)' },
                        target: { label: 'Target', color: 'var(--chart-1)' },
                      }}
                      className="h-[180px] w-full sm:h-[220px]"
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={allocationWeightData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                          <XAxis dataKey="symbol" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={50} />
                          <YAxis tickFormatter={(value) => `${value.toFixed(0)}%`} width={40} />
                          <ChartTooltip
                            content={
                              <ChartTooltipContent
                                formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
                              />
                            }
                          />
                          <Bar dataKey="current" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="target" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </div>
                )}
                {allocationResult.warnings.length > 0 && (
                  <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-800">
                    {allocationResult.warnings.map((warning) => (
                      <p key={warning}>• {warning}</p>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Enter your investment amount and choose a strategy to get optimized trade recommendations.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Holdings */}
      {selectedPortfolio && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <div>
              <CardTitle>Holdings</CardTitle>
              <CardDescription>Your stock positions</CardDescription>
            </div>
            <Button onClick={openAddHoldingDialog} size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Holding
            </Button>
          </CardHeader>
          <CardContent>
            {detail?.holdings && detail.holdings.length > 0 ? (
              <motion.div 
                className="space-y-2"
                variants={container}
                initial="hidden"
                animate="show"
              >
                <AnimatePresence mode="popLayout">
                  {detail.holdings.map((holding) => {
                    const snapshot = holdingSnapshotMap.get(holding.symbol);
                    const unrealizedPct = snapshot?.unrealizedPct ?? null;
                    const marketValue = snapshot?.marketValue || 0;
                    return (
                      <motion.div
                        key={holding.symbol}
                        variants={item}
                        layout
                        className="flex items-center justify-between rounded-lg border p-4 hover:bg-muted/50 transition-colors"
                      >
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
                          <div>
                            <div className="font-mono font-semibold text-lg">{holding.symbol}</div>
                            {snapshot?.name && (
                              <div className="text-xs text-muted-foreground">
                                {snapshot.name}
                              </div>
                            )}
                          </div>
                          <Badge variant="secondary">{holding.quantity} shares</Badge>
                          {holding.avg_cost ? (
                            <span className="text-sm text-muted-foreground">
                              Avg {formatCurrency(holding.avg_cost)}
                            </span>
                          ) : (
                            <Badge variant="outline" className="border-amber-500/40 text-amber-700">
                              Avg cost required
                            </Badge>
                          )}
                          {snapshot?.currentPrice ? (
                            <span className="text-sm text-muted-foreground">
                              Now {formatCurrency(snapshot.currentPrice)}
                            </span>
                          ) : null}
                          {unrealizedPct !== null && (
                            <Badge variant={unrealizedPct >= 0 ? 'default' : 'destructive'}>
                              {unrealizedPct >= 0 ? '+' : ''}
                              {unrealizedPct.toFixed(1)}%
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-4">
                          {/* Mini sparkline chart */}
                          <div className="hidden sm:block">
                            <HoldingSparkline 
                              data={sparklineData[holding.symbol] ?? null} 
                              width={100}
                              height={36}
                            />
                          </div>
                          <div className="text-right">
                            <div className="font-semibold">
                              {formatCurrency(marketValue)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {portfolioStats.investedValue > 0
                                ? `${((marketValue / portfolioStats.investedValue) * 100).toFixed(1)}%`
                                : '0%'}
                            </div>
                          </div>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => openEditHoldingDialog(holding)}>
                                <Pencil className="h-4 w-4 mr-2" />
                                Edit
                              </DropdownMenuItem>
                              <DropdownMenuItem 
                                onClick={() => handleDeleteHolding(holding.symbol)}
                                className="text-destructive focus:text-destructive"
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Remove
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </motion.div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Wallet className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No holdings yet</p>
                <p className="text-sm">Add your first stock position to get started</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Empty state */}
      {!selectedPortfolio && !isLoading && (
        <Card>
          <CardContent className="py-12 text-center">
            <Wallet className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <h3 className="font-semibold mb-2">No Portfolio Selected</h3>
            <p className="text-muted-foreground mb-4">Create a portfolio to start tracking your holdings</p>
            <Button onClick={openCreateDialog}>
              <Plus className="h-4 w-4 mr-2" />
              Create Portfolio
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Portfolio Dialog */}
      <Dialog open={portfolioDialogOpen} onOpenChange={setPortfolioDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingPortfolio ? 'Edit Portfolio' : 'New Portfolio'}</DialogTitle>
            <DialogDescription>
              {editingPortfolio ? 'Update your portfolio details' : 'Create a new portfolio to track your investments'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Portfolio Name</Label>
              <Input 
                value={portfolioName} 
                onChange={(e) => setPortfolioName(e.target.value)}
                placeholder="My Portfolio"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Currency</Label>
                <Select value={portfolioCurrency} onValueChange={setPortfolioCurrency}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="USD">USD</SelectItem>
                    <SelectItem value="EUR">EUR</SelectItem>
                    <SelectItem value="GBP">GBP</SelectItem>
                    <SelectItem value="JPY">JPY</SelectItem>
                    <SelectItem value="CAD">CAD</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPortfolioDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSavePortfolio}>
              {editingPortfolio ? 'Save Changes' : 'Create Portfolio'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Holding Dialog */}
      <Dialog
        open={holdingDialogOpen}
        onOpenChange={(open) => {
          setHoldingDialogOpen(open);
          if (!open) setHoldingValidationError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingHolding ? 'Edit Holding' : 'Add Holding'}</DialogTitle>
            <DialogDescription>
              {editingHolding ? 'Update your position' : 'Add a new stock position to your portfolio'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Symbol</Label>
              <Input 
                value={holdingSymbol} 
                onChange={(e) => setHoldingSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                disabled={!!editingHolding}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Quantity</Label>
                <Input 
                  type="number"
                  value={holdingQty} 
                  onChange={(e) => setHoldingQty(e.target.value)}
                  placeholder="100"
                />
              </div>
              <div>
                <Label>Avg Cost</Label>
                <Input 
                  type="number"
                  step="0.01"
                  value={holdingAvgCost} 
                  onChange={(e) => {
                    setHoldingAvgCost(e.target.value);
                    if (holdingValidationError) setHoldingValidationError(null);
                  }}
                  placeholder="150.00"
                />
                {!holdingAvgCost && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Required for portfolio analytics and allocation insights.
                  </p>
                )}
              </div>
            </div>
            {holdingValidationError && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive">
                {holdingValidationError}
              </div>
            )}
          </div>
          <DialogFooter className="flex-col sm:flex-row gap-2">
            {!editingHolding && (
              <Button 
                variant="secondary" 
                onClick={() => {
                  setHoldingDialogOpen(false);
                  setBulkImportOpen(true);
                }}
                className="w-full sm:w-auto sm:mr-auto"
              >
                <Upload className="h-4 w-4 mr-2" />
                Import Bulk
              </Button>
            )}
            <div className="flex gap-2 w-full sm:w-auto">
              <Button variant="outline" onClick={() => setHoldingDialogOpen(false)} className="flex-1 sm:flex-none">
                Cancel
              </Button>
              <Button onClick={handleSaveHolding} className="flex-1 sm:flex-none" disabled={!isHoldingValid}>
                {editingHolding ? 'Save Changes' : 'Add Holding'}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Bulk Import Modal */}
      {selectedId && (
        <BulkImportModal
          open={bulkImportOpen}
          onOpenChange={setBulkImportOpen}
          portfolioId={selectedId}
          onImportComplete={() => loadDetail(selectedId)}
        />
      )}
    </div>
  );
}
