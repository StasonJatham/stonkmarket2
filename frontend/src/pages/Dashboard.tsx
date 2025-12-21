import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  getRanking, 
  getStockChart, 
  getStockInfo,
  getBenchmarkChart,
  mergeChartData,
  aggregatePortfolioPerformance,
} from '@/services/api';
import type { 
  DipStock, 
  ChartDataPoint, 
  StockInfo, 
  BenchmarkType,
  ComparisonChartData,
  AggregatedPerformance,
} from '@/services/api';
import { FeaturedSlider } from '@/components/FeaturedSlider';
import { StockCard } from '@/components/StockCard';
import { StockDetailsPanel } from '@/components/StockDetailsPanel';
import { BenchmarkSelector } from '@/components/BenchmarkSelector';
import { PortfolioChart } from '@/components/PortfolioChart';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Sheet, SheetContent } from '@/components/ui/sheet';
import { 
  RefreshCw, 
  Search, 
  TrendingDown, 
  LayoutGrid, 
  List,
  SlidersHorizontal,
  X,
  BarChart3
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
type SortBy = 'score' | 'depth' | 'recovery' | 'name';

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

export function Dashboard() {
  const [stocks, setStocks] = useState<DipStock[]>([]);
  const [selectedStock, setSelectedStock] = useState<DipStock | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [chartPeriod, setChartPeriod] = useState(365);
  const [isLoadingRanking, setIsLoadingRanking] = useState(true);
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [isLoadingInfo, setIsLoadingInfo] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [sortBy, setSortBy] = useState<SortBy>('score');
  const [isMobileDetailOpen, setIsMobileDetailOpen] = useState(false);
  
  // Benchmark state
  const [benchmark, setBenchmark] = useState<BenchmarkType>(null);
  const [benchmarkData, setBenchmarkData] = useState<ChartDataPoint[]>([]);
  const [isLoadingBenchmark, setIsLoadingBenchmark] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonChartData[]>([]);
  const [aggregatedData, setAggregatedData] = useState<AggregatedPerformance[]>([]);
  const [showPortfolioChart, setShowPortfolioChart] = useState(false);

  // Load ranking on mount
  useEffect(() => {
    loadRanking();
  }, []);

  // Memoized filtered and sorted stocks
  const filteredStocks = useMemo(() => {
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
        case 'score':
          return (b.dip_score ?? 0) - (a.dip_score ?? 0);
        case 'depth':
          return b.depth - a.depth;
        case 'recovery':
          return (b.recovery_potential ?? 0) - (a.recovery_potential ?? 0);
        case 'name':
          return (a.name || a.symbol).localeCompare(b.name || b.symbol);
        default:
          return 0;
      }
    });

    return result;
  }, [stocks, searchQuery, sortBy]);

  // Memoized top five stocks for featured slider
  const topFiveStocks = useMemo(() => stocks.slice(0, 5), [stocks]);

  // Load chart and info when stock selected
  useEffect(() => {
    if (selectedStock) {
      loadChart(selectedStock.symbol);
      loadStockInfo(selectedStock.symbol);
    }
  }, [selectedStock?.symbol]);

  // Reload chart when period changes
  useEffect(() => {
    if (selectedStock) {
      loadChart(selectedStock.symbol);
    }
  }, [chartPeriod]);

  // Load benchmark data when benchmark changes
  useEffect(() => {
    if (benchmark) {
      loadBenchmarkData();
    } else {
      setBenchmarkData([]);
      setComparisonData([]);
      setAggregatedData([]);
    }
  }, [benchmark, chartPeriod]);

  // Merge stock and benchmark data for comparison
  useEffect(() => {
    if (chartData.length > 0 && benchmarkData.length > 0 && benchmark) {
      const merged = mergeChartData(chartData, benchmarkData);
      setComparisonData(merged);
    } else {
      setComparisonData([]);
    }
  }, [chartData, benchmarkData, benchmark]);

  // Calculate aggregated portfolio performance when stocks or benchmark changes
  useEffect(() => {
    if (stocks.length > 0 && benchmarkData.length > 0 && benchmark) {
      calculateAggregatedPerformance();
    } else {
      setAggregatedData([]);
    }
  }, [stocks, benchmarkData, benchmark]);

  async function loadBenchmarkData() {
    if (!benchmark) return;
    setIsLoadingBenchmark(true);
    try {
      const data = await getBenchmarkChart(benchmark, chartPeriod);
      setBenchmarkData(data);
    } catch (err) {
      console.error('Failed to load benchmark data:', err);
      setBenchmarkData([]);
    } finally {
      setIsLoadingBenchmark(false);
    }
  }

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

  async function loadRanking() {
    setIsLoadingRanking(true);
    setError(null);
    try {
      const response = await getRanking();
      setStocks(response.ranking);
      setLastUpdated(response.last_updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ranking');
    } finally {
      setIsLoadingRanking(false);
    }
  }

  async function loadChart(symbol: string) {
    setIsLoadingChart(true);
    try {
      const data = await getStockChart(symbol, chartPeriod);
      setChartData(data);
    } catch (err) {
      console.error('Failed to load chart:', err);
      setChartData([]);
    } finally {
      setIsLoadingChart(false);
    }
  }

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

  const handleStockSelect = useCallback((stock: DipStock) => {
    setSelectedStock(stock);
    setIsMobileDetailOpen(true);
  }, []);

  const handleSliderSelect = useCallback((symbol: string) => {
    const stock = stocks.find(s => s.symbol === symbol);
    if (stock) {
      setSelectedStock(stock);
    }
  }, [stocks]);

  return (
    <div className="space-y-8">
      {/* Hero Section with Featured Slider */}
      <section>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-6"
        >
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Market Dips</h1>
            <p className="text-muted-foreground mt-1">
              Discover stocks with the best recovery potential
            </p>
          </div>
          {lastUpdated && (
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              onClick={loadRanking}
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              <span className="hidden sm:inline">
                Updated {new Date(lastUpdated).toLocaleTimeString()}
              </span>
            </Button>
          )}
        </motion.div>

        {/* Featured Dips Slider */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <FeaturedSlider
            stocks={topFiveStocks}
            isLoading={isLoadingRanking}
            onSelectStock={handleSliderSelect}
          />
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

      {/* Filters & View Toggle */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search stocks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7"
              onClick={() => setSearchQuery('')}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Benchmark Selector */}
          <BenchmarkSelector 
            value={benchmark} 
            onChange={setBenchmark} 
          />

          {/* Portfolio Chart Toggle */}
          {benchmark && (
            <Button
              variant={showPortfolioChart ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setShowPortfolioChart(!showPortfolioChart)}
              className="gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Portfolio</span>
            </Button>
          )}

          {/* Sort Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <SlidersHorizontal className="h-4 w-4 mr-2" />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setSortBy('score')}>
                <Badge variant={sortBy === 'score' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'score' && '✓'}
                </Badge>
                Dip Score
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('depth')}>
                <Badge variant={sortBy === 'depth' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'depth' && '✓'}
                </Badge>
                Dip Depth
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('recovery')}>
                <Badge variant={sortBy === 'recovery' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'recovery' && '✓'}
                </Badge>
                Recovery Potential
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setSortBy('name')}>
                <Badge variant={sortBy === 'name' ? 'default' : 'outline'} className="mr-2">
                  {sortBy === 'name' && '✓'}
                </Badge>
                Name
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* View Mode Toggle */}
          <div className="flex items-center border rounded-lg p-1">
            <Button
              variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('grid')}
            >
              <LayoutGrid className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-8 w-8"
              onClick={() => setViewMode('list')}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </motion.div>

      {/* Stock Count */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <TrendingDown className="h-4 w-4" />
        {isLoadingRanking ? (
          <Skeleton className="h-4 w-32" />
        ) : (
          <span>
            {filteredStocks.length} {filteredStocks.length === 1 ? 'stock' : 'stocks'} in dip
            {searchQuery && ` matching "${searchQuery}"`}
          </span>
        )}
      </div>

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
              {filteredStocks.map((stock) => (
                <motion.div key={stock.symbol} variants={item}>
                  <StockCard
                    stock={stock}
                    isSelected={selectedStock?.symbol === stock.symbol}
                    onClick={() => handleStockSelect(stock)}
                  />
                </motion.div>
              ))}
            </motion.div>
          )}
        </div>

        {/* Desktop Details Panel */}
        <div className="hidden lg:block sticky top-24 h-[calc(100vh-8rem)]">
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
          />
        </div>
      </div>

      {/* Mobile Details Sheet */}
      <Sheet open={isMobileDetailOpen && !!selectedStock} onOpenChange={setIsMobileDetailOpen}>
        <SheetContent side="bottom" className="h-[85vh] p-0">
          <div className="h-full pt-4">
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
            />
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
