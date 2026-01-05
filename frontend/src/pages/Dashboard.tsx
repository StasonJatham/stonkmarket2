import { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  getStockChart, 
  getStockInfo,
  getBenchmarkChart,
  mergeChartData,
  aggregatePortfolioPerformance,
  getDipCard,
  quantToStockCardData,
} from '@/services/api';
import { useBenchmarks, useBatchCharts } from '@/features/market-data/api/queries';
import type { 
  DipStock, 
  ChartDataPoint, 
  StockInfo, 
  BenchmarkType,
  ComparisonChartData,
  AggregatedPerformance,
  DipCard,
  StockCardData,
} from '@/services/api';
import { useQuantRecommendations } from '@/features/quant-engine/api/queries';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
import { StockCardV2 } from '@/components/cards/StockCardV2';
import { StockDetailsPanel } from '@/components/StockDetailsPanel';
import { BenchmarkSelector } from '@/components/BenchmarkSelector';
import { PortfolioChart } from '@/components/PortfolioChart';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Drawer, DrawerContent, DrawerTitle } from '@/components/ui/drawer';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  Search, 
  TrendingDown, 
  LayoutGrid, 
  List,
  SlidersHorizontal,
  X,
  BarChart3,
  AlertCircle
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

type ViewMode = 'grid' | 'list';
type SortBy = 'chance' | 'utility' | 'return' | 'depth' | 'name';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.3 }
  },
};

export function Dashboard() {
  // Local state for UI toggles that were previously in DipContext
  const [showAllStocks, setShowAllStocks] = useState(false);
  // Inflow amount for quant engine - could be exposed in UI later
  const inflow = 1000;
  
  // Use quant engine for primary stock ranking (sorted by marginal utility)
  // TanStack Query replaces the old QuantContext with proper caching
  const quantQuery = useQuantRecommendations(inflow, 40);
  
  // Extract data from query - the select transform gives us 'stocks' already
  // Plain derivation - React Compiler optimizes this automatically
  const recommendations = quantQuery.data?.recommendations ?? [];
  const quantStocks = quantQuery.data?.stocks ?? [];
  const asOfDate = quantQuery.data?.as_of_date ?? null;
  const marketMessage = quantQuery.data?.market_message ?? null;
  const portfolioStats = quantQuery.data ? {
    expectedReturn: quantQuery.data.expected_portfolio_return,
    expectedRisk: quantQuery.data.expected_portfolio_risk,
    totalTrades: quantQuery.data.total_trades,
    transactionCostEur: quantQuery.data.total_transaction_cost_eur,
  } : null;
  const isLoadingQuant = quantQuery.isLoading;
  const quantError = quantQuery.error?.message ?? null;
  
  // Use quant stocks as primary data source
  const stocks = quantStocks;
  const isLoadingRanking = isLoadingQuant;
  const lastUpdated = asOfDate;
  const error = quantError;
  
  // Convert recommendations to StockCardData for enhanced card display
  // Plain derivation - React Compiler optimizes this automatically
  const stockCardDataMap = (() => {
    const map = new Map<string, StockCardData>();
    recommendations.forEach(rec => {
      map.set(rec.ticker, quantToStockCardData(rec));
    });
    return map;
  })();
  
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedStock, setSelectedStock] = useState<DipStock | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [chartPeriod, setChartPeriod] = useState(365);
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [isLoadingInfo, setIsLoadingInfo] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [sortBy, setSortBy] = useState<SortBy>('chance');
  const [isMobileDetailOpen, setIsMobileDetailOpen] = useState(false);
  
  // Benchmark data via TanStack Query - map to BenchmarkConfig type
  const benchmarksQuery = useBenchmarks();
  const availableBenchmarks = (benchmarksQuery.data ?? []).map(b => ({
    id: b.id ?? b.symbol, // Use symbol as fallback for id
    symbol: b.symbol,
    name: b.symbol, // Use symbol as name since API only returns id/symbol
    description: `${b.symbol} benchmark`, // Generate description
  }));
  
  // Benchmark state
  const [benchmark, setBenchmark] = useState<BenchmarkType>(null);
  const [benchmarkData, setBenchmarkData] = useState<ChartDataPoint[]>([]);
  const [isLoadingBenchmark, setIsLoadingBenchmark] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonChartData[]>([]);
  const [aggregatedData, setAggregatedData] = useState<AggregatedPerformance[]>([]);
  const [showPortfolioChart, setShowPortfolioChart] = useState(false);
  
  // AI Analysis state
  const [aiData, setAiData] = useState<{
    ai_rating: DipCard['ai_rating'];
    ai_reasoning: string | null;
    domain_analysis: string | null;
    domain_context?: string | null;
    domain_adjustment?: number | null;
    domain_adjustment_reason?: string | null;
    domain_risk_level?: string | null;
    domain_risk_factors?: string[] | null;
    domain_recovery_days?: number | null;
    domain_warnings?: string[] | null;
    volatility_regime?: string | null;
    volatility_percentile?: number | null;
    vs_sector_performance?: number | null;
    sector?: string | null;
  } | null>(null);
  const [isLoadingAi, setIsLoadingAi] = useState(false);
  
  // Infinite scroll state
  const [visibleCount, setVisibleCount] = useState(20);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // SEO - Dynamic meta tags based on selected stock
  useSEO({
    title: selectedStock 
      ? `${selectedStock.symbol} Stock Analysis - Quant Ranked`
      : 'Smart Stock Allocation Dashboard',
    description: selectedStock
      ? `Analyze ${selectedStock.symbol} with quantitative risk-adjusted metrics. Expected returns and portfolio optimization powered by mean-variance analysis.`
      : 'Quantitative stock allocation using mean-variance optimization. Risk-adjusted expected returns with statistical analysis.',
    keywords: selectedStock
      ? `${selectedStock.symbol}, stock analysis, portfolio optimization, expected return, ${selectedStock.symbol} price`
      : 'quantitative investing, portfolio optimization, mean-variance, risk-adjusted returns, stock analysis',
    canonical: '/',
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'Dashboard', url: '/' },
    ]),
  });

  // Sync URL params to state on mount
  useEffect(() => {
    const urlSearch = searchParams.get('search');
    const urlSort = searchParams.get('sort') as SortBy | null;
    const urlView = searchParams.get('view') as ViewMode | null;
    const urlShowAll = searchParams.get('showAll');
    
    if (urlSearch !== null) setSearchQuery(urlSearch);
    if (urlSort && ['chance', 'utility', 'return', 'depth', 'name'].includes(urlSort)) setSortBy(urlSort);
    if (urlView && ['grid', 'list'].includes(urlView)) setViewMode(urlView);
    if (urlShowAll === 'true') setShowAllStocks(true);
  }, [searchParams, setShowAllStocks]);

  // Update URL when filters change (debounced to avoid too many updates)
  function updateUrlParams(updates: Record<string, string | null>) {
    const newParams = new URLSearchParams(searchParams);
    
    Object.entries(updates).forEach(([key, value]) => {
      if (value === null || value === '' || value === 'false') {
        newParams.delete(key);
      } else {
        newParams.set(key, value);
      }
    });
    
    // Only update if params actually changed
    const currentStr = searchParams.toString();
    const newStr = newParams.toString();
    if (currentStr !== newStr) {
      setSearchParams(newParams, { replace: true });
    }
  }

  // Sync state changes to URL
  useEffect(() => {
    updateUrlParams({
      search: searchQuery || null,
      sort: sortBy !== 'chance' ? sortBy : null,
      view: viewMode !== 'grid' ? viewMode : null,
      showAll: showAllStocks ? 'true' : null,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery, sortBy, viewMode, showAllStocks]);

  // Handle URL query param for stock selection (from ticker click)
  useEffect(() => {
    const stockSymbol = searchParams.get('stock');
    if (stockSymbol && stocks.length > 0) {
      // First check if stock is in current view
      const stock = stocks.find(s => s.symbol.toUpperCase() === stockSymbol.toUpperCase());
      
      // If not found and not showing all stocks, switch to all stocks view
      if (!stock && !showAllStocks) {
        setShowAllStocks(true);
        // Stock will be found after showAllStocks triggers a re-fetch
        return;
      }
      
      if (stock) {
        setSelectedStock(stock);
        // Keep the stock param in URL for sharing
        updateUrlParams({ stock: stock.symbol });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, stocks, showAllStocks, setShowAllStocks]);

  // Filtered and sorted stocks - React Compiler handles memoization
  const filteredStocks = (() => {
    let result = [...stocks];

    // Filter by search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (s) =>
          s.symbol.toLowerCase().includes(query) ||
          s.name?.toLowerCase().includes(query) ||
          s.sector?.toLowerCase().includes(query)
      );
    }

    // Sort
    result.sort((a, b) => {
      switch (sortBy) {
        case 'chance':
          // Sort by best_chance_score from quant engine (or dip_score as fallback)
          return (b.dip_score ?? 0) - (a.dip_score ?? 0);
        case 'utility':
          // dip_score now contains marginal_utility from quant engine
          return (b.dip_score ?? 0) - (a.dip_score ?? 0);
        case 'return':
          // recovery_potential now contains mu_hat (expected return)
          return (b.recovery_potential ?? 0) - (a.recovery_potential ?? 0);
        case 'depth':
          return b.depth - a.depth;
        case 'name':
          return (a.name || a.symbol).localeCompare(b.name || b.symbol);
        default:
          return 0;
      }
    });

    // If a stock is selected via URL param (e.g. from ticker click), move it to the top
    const stockParam = searchParams.get('stock');
    if (stockParam) {
      const paramUpper = stockParam.toUpperCase();
      const selectedIndex = result.findIndex(s => s.symbol.toUpperCase() === paramUpper);
      if (selectedIndex > 0) {
        const [selected] = result.splice(selectedIndex, 1);
        result.unshift(selected);
      }
    }

    return result;
  })();

  // Reset visible count when search/filter changes
  useEffect(() => {
    setVisibleCount(20);
  }, [searchQuery, sortBy, showAllStocks]);

  // Visible stocks for pagination - plain derivation
  const visibleStocks = filteredStocks.slice(0, visibleCount);

  // Fetch mini charts for visible cards via TanStack Query
  const symbolsForCards = visibleStocks.map(s => s.symbol);
  const cardChartsQuery = useBatchCharts(symbolsForCards, 90);
  const cardCharts = cardChartsQuery.data ?? {};

  // Infinite scroll observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && visibleCount < filteredStocks.length) {
          setVisibleCount((prev) => Math.min(prev + 20, filteredStocks.length));
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [visibleCount, filteredStocks.length]);

  // Calculate optimal chart period to show high and dip
  function calculateOptimalPeriod(stock: DipStock): number {
    const daysSinceDip = stock.days_since_dip || 90;
    // Add buffer to show context before the dip (at least 20% more time)
    const requiredDays = Math.ceil(daysSinceDip * 1.3);
    
    // Find smallest period that covers the required days
    const periods = [30, 90, 180, 365];
    for (const period of periods) {
      if (period >= requiredDays) {
        return period;
      }
    }
    return 365; // Default to 1 year if nothing fits
  }

  // Helper functions for loading stock data - defined before useEffect that uses them
  async function loadStockInfo(symbol: string) {
    setIsLoadingInfo(true);
    try {
      const info = await getStockInfo(symbol);
      setStockInfo(info);
    } catch (err) {
      console.error('Failed to load stock info:', err);
      setStockInfo(null);
    } finally {
      setIsLoadingInfo(false);
    }
  }

  async function loadAiData(symbol: string) {
    setIsLoadingAi(true);
    try {
      // First check if we have domain analysis from the recommendations
      const cardData = stockCardDataMap.get(symbol);
      const card = await getDipCard(symbol);
      setAiData({
        ai_rating: card.ai_rating,
        ai_reasoning: card.ai_reasoning,
        // Use domain_analysis from recommendations (quant data) if available
        domain_analysis: cardData?.domain_analysis || null,
        // Pass through all domain-specific analysis fields
        domain_context: cardData?.domain_context || null,
        domain_adjustment: cardData?.domain_adjustment ?? null,
        domain_adjustment_reason: cardData?.domain_adjustment_reason || null,
        domain_risk_level: cardData?.domain_risk_level || null,
        domain_risk_factors: cardData?.domain_risk_factors || null,
        domain_recovery_days: cardData?.domain_recovery_days ?? null,
        domain_warnings: cardData?.domain_warnings || null,
        volatility_regime: cardData?.volatility_regime || null,
        volatility_percentile: cardData?.volatility_percentile ?? null,
        vs_sector_performance: cardData?.vs_sector_performance ?? null,
        sector: cardData?.sector || null,
      });
    } catch (err) {
      console.error('Failed to load AI data:', err);
      setAiData(null);
    } finally {
      setIsLoadingAi(false);
    }
  }

  // Load chart and info when stock selected
  useEffect(() => {
    if (selectedStock) {
      // Auto-select optimal period based on dip timeline
      const optimalPeriod = calculateOptimalPeriod(selectedStock);
      setChartPeriod(optimalPeriod);
      
      loadStockInfo(selectedStock.symbol);
      loadAiData(selectedStock.symbol);
    } else {
      setAiData(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStock]);

  // Consolidated chart loading - load both stock and benchmark chart together
  useEffect(() => {
    async function loadAllChartData() {
      if (!selectedStock) {
        setChartData([]);
        setBenchmarkData([]);
        setComparisonData([]);
        return;
      }

      // Deferred loading: only show spinner after 150ms to avoid flash for fast responses
      const loadingTimer = setTimeout(() => setIsLoadingChart(true), 150);
      
      try {
        const stockChartData = await getStockChart(selectedStock.symbol, chartPeriod);
        clearTimeout(loadingTimer);
        setIsLoadingChart(false);
        setChartData(stockChartData);

        // If benchmark is selected, load it and merge
        if (benchmark) {
          setIsLoadingBenchmark(true);
          try {
            const benchData = await getBenchmarkChart(benchmark, chartPeriod);
            setBenchmarkData(benchData);
            
            // Merge data for comparison chart
            if (stockChartData.length > 0 && benchData.length > 0) {
              const merged = mergeChartData(stockChartData, benchData);
              setComparisonData(merged);
            } else {
              setComparisonData([]);
            }
          } catch (err) {
            console.error('Failed to load benchmark data:', err);
            setBenchmarkData([]);
            setComparisonData([]);
          } finally {
            setIsLoadingBenchmark(false);
          }
        } else {
          setBenchmarkData([]);
          setComparisonData([]);
        }
      } catch (err) {
        clearTimeout(loadingTimer);
        console.error('Failed to load chart:', err);
        setChartData([]);
        setComparisonData([]);
        setIsLoadingChart(false);
      }
    }

    loadAllChartData();
  }, [selectedStock, chartPeriod, benchmark]);

  async function calculateAggregatedPerformance() {
    if (!benchmark || stocks.length === 0) return;
    try {
      // Load chart data for all stocks in parallel (limit to first 20)
      const stocksToLoad = stocks.slice(0, 20);
      const chartPromises = stocksToLoad.map(s => 
        getStockChart(s.symbol, chartPeriod)
          .then(data => ({ symbol: s.symbol, data }))
          .catch(() => ({ symbol: s.symbol, data: [] as ChartDataPoint[] }))
      );
      const allCharts = await Promise.all(chartPromises);
      
      // Convert to Map format
      const stocksMap = new Map<string, ChartDataPoint[]>();
      allCharts.forEach(({ symbol, data }) => {
        if (data.length > 0) {
          stocksMap.set(symbol, data);
        }
      });
      
      if (stocksMap.size > 0) {
        const aggregated = aggregatePortfolioPerformance(stocksMap, benchmarkData);
        setAggregatedData(aggregated);
      }
    } catch (err) {
      console.error('Failed to calculate aggregated performance:', err);
    }
  }

  // Calculate aggregated portfolio performance when stocks or benchmark changes
  useEffect(() => {
    if (stocks.length > 0 && benchmarkData.length > 0 && benchmark) {
      calculateAggregatedPerformance();
    } else {
      setAggregatedData([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stocks.length, benchmarkData.length, benchmark, chartPeriod]);

  function handleStockSelect(stock: DipStock) {
    setSelectedStock(stock);
    // Only open mobile drawer on small screens
    if (window.innerWidth < 1024) {
      setIsMobileDetailOpen(true);
    }
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-6"
        >
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Smart Allocation</h1>
            <p className="text-muted-foreground mt-1">
              Stocks ranked by expected risk-adjusted return
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground hidden sm:flex">
            {!isLoadingRanking && (
              <span className="flex items-center gap-1">
                <TrendingDown className="h-3 w-3" />
                {stocks.length} analyzed
              </span>
            )}
            {portfolioStats && (
              <span className="flex items-center gap-1">
                E[r]: {(portfolioStats.expectedReturn * 100).toFixed(1)}%
              </span>
            )}
            {lastUpdated && (
              <span>
                As of {new Date(lastUpdated).toLocaleDateString()}
              </span>
            )}
          </div>
        </motion.div>
      </section>

      {/* Error Alert */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-danger/10 text-danger p-4 rounded-xl border border-danger/20"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Market Message Alert */}
      {marketMessage && !error && (
        <Alert variant={marketMessage.includes('No certified') ? 'destructive' : 'default'}>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{marketMessage}</AlertDescription>
        </Alert>
      )}

      {/* Filters & View Toggle */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-3 sm:items-center"
      >
        {/* Search */}
        <div className="relative flex-1 sm:max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search stocks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7"
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Benchmark Selector */}
          <BenchmarkSelector 
            value={benchmark} 
            onChange={setBenchmark}
            customBenchmarks={availableBenchmarks}
          />

          {/* Portfolio Chart Toggle */}
          {benchmark && (
            <Button
              variant={showPortfolioChart ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setShowPortfolioChart(!showPortfolioChart)}
              className="h-9 gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Portfolio</span>
            </Button>
          )}

          {/* Sort Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-9">
                <SlidersHorizontal className="h-4 w-4 mr-2" />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setSortBy('chance')}>
                <Badge variant={sortBy === 'chance' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'chance' && '✓'}
                </Badge>
                Best Chance
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('utility')}>
                <Badge variant={sortBy === 'utility' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'utility' && '✓'}
                </Badge>
                Marginal Utility
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('return')}>
                <Badge variant={sortBy === 'return' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'return' && '✓'}
                </Badge>
                Expected Return
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('depth')}>
                <Badge variant={sortBy === 'depth' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'depth' && '✓'}
                </Badge>
                Dip Depth
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('name')}>
                <Badge variant={sortBy === 'name' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'name' && '✓'}
                </Badge>
                Name
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Show All Toggle */}
          <div className="flex items-center gap-2 border rounded-lg px-3 py-1.5">
            <Switch
              id="show-all"
              checked={showAllStocks}
              onCheckedChange={setShowAllStocks}
            />
            <Label htmlFor="show-all" className="text-sm cursor-pointer whitespace-nowrap">
              Show all
            </Label>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center border rounded-lg p-0.5">
            <Button
              variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('grid')}
              aria-label="Grid view"
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('list')}
              aria-label="List view"
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Mobile stock count */}
        <div className="flex sm:hidden items-center gap-2 text-xs text-muted-foreground">
          {!isLoadingRanking && (
            <>
              <TrendingDown className="h-3 w-3" />
              <span>
                {filteredStocks.length} ranked
                {searchQuery && ` • "${searchQuery}"`}
              </span>
            </>
          )}
        </div>
      </motion.div>

      {/* Portfolio Chart Section */}
      <AnimatePresence>
        {showPortfolioChart && benchmark && aggregatedData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <PortfolioChart
              data={aggregatedData}
              benchmark={benchmark}
              stockCount={Math.min(stocks.length, 20)}
              isLoading={isLoadingBenchmark}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        {/* Stock Cards */}
        <div>
          {isLoadingRanking ? (
            <div className={
              viewMode === 'grid'
                ? 'grid gap-4 sm:grid-cols-2 xl:grid-cols-3'
                : 'space-y-3'
            }>
              {[...Array(6)].map((_, i) => (
                <Skeleton key={i} className="h-48 rounded-xl" />
              ))}
            </div>
          ) : filteredStocks.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12 text-muted-foreground"
            >
              <TrendingDown className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p>No stocks match your search</p>
            </motion.div>
          ) : (
            <>
              <motion.div
                variants={container}
                initial="hidden"
                animate="show"
                className={
                  viewMode === 'grid'
                    ? 'grid gap-4 sm:grid-cols-2 xl:grid-cols-3'
                    : 'space-y-3'
                }
              >
                {visibleStocks.map((stock) => {
                  const cardData = stockCardDataMap.get(stock.symbol);
                  const miniChartData = cardCharts[stock.symbol];
                  return (
                    <motion.div 
                      key={stock.symbol} 
                      variants={item}
                      initial="hidden"
                      animate="show"
                    >
                      <StockCardV2
                        stock={cardData || {
                          symbol: stock.symbol,
                          name: stock.name,
                          sector: stock.sector,
                          last_price: stock.last_price,
                          change_percent: stock.change_percent,
                          high_52w: stock.high_52w,
                          low_52w: stock.low_52w,
                          market_cap: stock.market_cap,
                          depth: stock.depth,
                          days_since_dip: stock.days_since_dip,
                          dip_bucket: null,
                        }}
                        chartData={miniChartData}
                        isSelected={selectedStock?.symbol === stock.symbol}
                        compact={viewMode === 'list'}
                        onClick={() => handleStockSelect(stock)}
                      />
                    </motion.div>
                  );
                })}
              </motion.div>
              
              {/* Load more trigger for infinite scroll */}
              {visibleCount < filteredStocks.length && (
                <div
                  ref={loadMoreRef}
                  className="flex justify-center py-8"
                >
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    Loading more stocks...
                  </div>
                </div>
              )}
              
              {/* Show count info */}
              {visibleCount >= filteredStocks.length && filteredStocks.length > 20 && (
                <div className="text-center py-4 text-sm text-muted-foreground">
                  Showing all {filteredStocks.length} stocks
                </div>
              )}
            </>
          )}
        </div>

        {/* Desktop Details Panel */}
        <div className="hidden lg:block sticky top-16 h-[calc(100vh-5rem)]">
          <StockDetailsPanel
            stock={selectedStock}
            chartData={chartData}
            stockInfo={stockInfo}
            chartPeriod={chartPeriod}
            onPeriodChange={setChartPeriod}
            isLoadingChart={isLoadingChart}
            isLoadingInfo={isLoadingInfo}
            onClose={() => setSelectedStock(null)}
            benchmark={benchmark}
            comparisonData={comparisonData}
            isLoadingBenchmark={isLoadingBenchmark}
            aiData={aiData}
            isLoadingAi={isLoadingAi}
          />
        </div>
      </div>

      {/* Mobile Details Drawer - swipe down to close */}
      <Drawer open={isMobileDetailOpen && !!selectedStock} onOpenChange={setIsMobileDetailOpen}>
        <DrawerContent className="h-[85dvh] max-h-[calc(100dvh-env(safe-area-inset-top)-2rem)] p-0">
          <DrawerTitle className="sr-only">Stock Details</DrawerTitle>
          <div className="h-full overflow-hidden pb-safe">
            <StockDetailsPanel
              stock={selectedStock}
              chartData={chartData}
              stockInfo={stockInfo}
              chartPeriod={chartPeriod}
              onPeriodChange={setChartPeriod}
              isLoadingChart={isLoadingChart}
              isLoadingInfo={isLoadingInfo}
              onClose={() => setIsMobileDetailOpen(false)}
              benchmark={benchmark}
              comparisonData={comparisonData}
              isLoadingBenchmark={isLoadingBenchmark}
              aiData={aiData}
              isLoadingAi={isLoadingAi}
            />
          </div>
        </DrawerContent>
      </Drawer>
    </div>
  );
}
