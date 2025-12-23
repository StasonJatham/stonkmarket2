import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  getDipFinderSignals, 
  getSymbols,
  type DipSignal, 
  type Symbol,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip';
import { 
  TrendingDown, 
  AlertTriangle,
  Filter,
  Search,
  BarChart3,
  Zap,
  ChevronLeft,
  ChevronRight,
  HelpCircle,
} from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
import { useIsMobile } from '@/hooks/useIsMobile';
import { SignalDetailsSheet } from '@/components/dipfinder/SignalDetailsSheet';
import { DipFinderMethodology, getDipTypeBadge } from '@/components/dipfinder/DipFinderMethodology';
import { StockLogo } from '@/components/StockLogo';

const WINDOWS = [
  { value: 7, label: '7 days', desc: 'Short-term dips (1 week lookback)' },
  { value: 30, label: '30 days', desc: 'Medium-term dips (1 month lookback)' },
  { value: 100, label: '100 days', desc: 'Long-term dips (quarter lookback)' },
  { value: 365, label: '365 days', desc: 'Annual dips (1 year lookback)' },
];

const ITEMS_PER_PAGE = 20;

// Score badge with colorblind-safe colors
function getScoreBadgeClass(score: number, colorblindMode: boolean): string {
  if (score >= 70) {
    return colorblindMode ? 'bg-blue-500/20 text-blue-500' : 'bg-success/20 text-success';
  }
  if (score >= 50) {
    return colorblindMode ? 'bg-purple-500/20 text-purple-500' : 'bg-chart-4/20 text-chart-4';
  }
  if (score >= 30) {
    return colorblindMode ? 'bg-orange-500/20 text-orange-500' : 'bg-chart-2/20 text-chart-2';
  }
  return 'bg-muted text-muted-foreground';
}

// Progress bar with colorblind-safe colors
function getProgressColor(colorblindMode: boolean, type: 'quality' | 'stability'): string {
  if (colorblindMode) {
    return type === 'quality' ? 'bg-blue-500' : 'bg-purple-500';
  }
  return type === 'quality' ? 'bg-success' : 'bg-chart-2';
}

function formatPercent(value: number, decimals = 1): string {
  const pct = value * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(decimals)}%`;
}

function formatScore(value: number): string {
  return value.toFixed(0);
}

// Mobile card component
function SignalCard({ 
  signal, 
  colorblindMode, 
  onClick 
}: { 
  signal: DipSignal; 
  colorblindMode: boolean;
  onClick: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="bg-card border rounded-lg p-4 cursor-pointer hover:border-primary/50 transition-colors"
      onClick={onClick}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-bold text-lg">{signal.ticker}</span>
          {getDipTypeBadge(signal.dip_class, colorblindMode)}
        </div>
        <div className={`font-bold text-xl px-3 py-1 rounded ${getScoreBadgeClass(signal.final_score, colorblindMode)}`}>
          {formatScore(signal.final_score)}
        </div>
      </div>
      
      {/* Stats grid - 2 rows */}
      <div className="grid grid-cols-3 gap-3 text-sm">
        {/* Row 1 */}
        <div>
          <span className="text-muted-foreground text-xs">Dip</span>
          <p className={colorblindMode ? 'font-medium text-orange-500' : 'font-medium text-danger'}>
            {formatPercent(-signal.dip_stock)}
          </p>
        </div>
        <div>
          <span className="text-muted-foreground text-xs">Excess</span>
          <p className={signal.excess_dip > 0 
            ? (colorblindMode ? 'text-blue-500' : 'text-success') 
            : 'text-muted-foreground'
          }>
            {formatPercent(signal.excess_dip)}
          </p>
        </div>
        <div>
          <span className="text-muted-foreground text-xs">Days</span>
          <p className="font-medium">
            {signal.persist_days > 0 ? signal.persist_days : '—'}
          </p>
        </div>
        
        {/* Row 2 - Quality & Stability */}
        <div className="col-span-3 flex gap-4 mt-1">
          <div className="flex-1">
            <span className="text-muted-foreground text-xs">Quality</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                <div 
                  className={`h-full ${getProgressColor(colorblindMode, 'quality')} transition-all`}
                  style={{ width: `${Math.min(signal.quality_score, 100)}%` }}
                />
              </div>
              <span className="text-xs w-8">{formatScore(signal.quality_score)}</span>
            </div>
          </div>
          <div className="flex-1">
            <span className="text-muted-foreground text-xs">Stability</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                <div 
                  className={`h-full ${getProgressColor(colorblindMode, 'stability')} transition-all`}
                  style={{ width: `${Math.min(signal.stability_score, 100)}%` }}
                />
              </div>
              <span className="text-xs w-8">{formatScore(signal.stability_score)}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export function DipFinderPage() {
  const { colorblindMode } = useTheme();
  const isMobile = useIsMobile();

  // SEO for DipFinder page (protected but still good to have)
  useSEO({
    title: 'Dip Signals - Advanced Stock Analysis',
    description: 'Advanced dip detection signals for stocks. Filter by timeframe, analyze recovery potential, and identify buying opportunities.',
    keywords: 'dip signals, stock analysis, dip detection, recovery potential, stock scanner',
    canonical: '/signals',
    noindex: true, // Protected page - don't index
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'Dip Signals', url: '/signals' },
    ]),
  });

  const [signals, setSignals] = useState<DipSignal[]>([]);
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [window, setWindow] = useState(30);
  const [searchQuery, setSearchQuery] = useState('');
  const [dipTypeFilter, setDipTypeFilter] = useState<string>('all');
  const [showActiveDips, setShowActiveDips] = useState(true);
  
  // Pagination
  const [page, setPage] = useState(0);
  
  // Selected signal for sheet
  const [selectedSignal, setSelectedSignal] = useState<DipSignal | null>(null);

  // Load symbols on mount
  useEffect(() => {
    async function loadSymbols() {
      try {
        const data = await getSymbols();
        setSymbols(data);
      } catch (err) {
        console.error('Failed to load symbols:', err);
      }
    }
    loadSymbols();
  }, []);

  // Load signals
  const loadSignals = useCallback(async () => {
    if (symbols.length === 0) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const tickers = symbols.map(s => s.symbol);
      const response = await getDipFinderSignals(tickers, { 
        window, 
        includeFactors: true 
      });
      setSignals(response.signals);
      setPage(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load signals');
    } finally {
      setIsLoading(false);
    }
  }, [symbols, window]);

  useEffect(() => {
    if (symbols.length > 0) {
      loadSignals();
    }
  }, [symbols, window, loadSignals]);

  // Filtered and sorted signals
  const filteredSignals = useMemo(() => {
    let result = [...signals];
    
    if (searchQuery) {
      const query = searchQuery.toUpperCase();
      result = result.filter(s => s.ticker.includes(query));
    }
    
    if (dipTypeFilter !== 'all') {
      result = result.filter(s => s.dip_class === dipTypeFilter);
    }
    
    if (showActiveDips) {
      result = result.filter(s => s.dip_stock >= 0.10);
    }
    
    result.sort((a, b) => b.final_score - a.final_score);
    
    return result;
  }, [signals, searchQuery, dipTypeFilter, showActiveDips]);

  // Paginated signals
  const paginatedSignals = useMemo(() => {
    const start = page * ITEMS_PER_PAGE;
    return filteredSignals.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredSignals, page]);
  
  const totalPages = Math.ceil(filteredSignals.length / ITEMS_PER_PAGE);

  // Stats
  const stats = useMemo(() => {
    const displaySignals = showActiveDips ? filteredSignals : signals;
    const stockSpecific = displaySignals.filter(s => s.dip_class === 'STOCK_SPECIFIC').length;
    const mixed = displaySignals.filter(s => s.dip_class === 'MIXED').length;
    const marketDip = displaySignals.filter(s => s.dip_class === 'MARKET_DIP').length;
    const avgScore = displaySignals.length > 0 
      ? displaySignals.reduce((sum, s) => sum + s.final_score, 0) / displaySignals.length 
      : 0;
    return { stockSpecific, mixed, marketDip, avgScore };
  }, [signals, filteredSignals, showActiveDips]);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <BarChart3 className="h-8 w-8" />
          DipFinder Signals
        </h1>
        <p className="text-muted-foreground mt-1">
          AI-powered dip detection with quality and stability scoring
        </p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <TrendingDown className={`w-5 h-5 ${colorblindMode ? 'text-blue-500' : 'text-success'}`} />
              <span className="text-sm text-muted-foreground">Stock-Specific</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.stockSpecific}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Zap className={`w-5 h-5 ${colorblindMode ? 'text-purple-500' : 'text-chart-2'}`} />
              <span className="text-sm text-muted-foreground">Mixed Dips</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.mixed}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className={`w-5 h-5 ${colorblindMode ? 'text-orange-500' : 'text-chart-4'}`} />
              <span className="text-sm text-muted-foreground">Market-Wide</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.marketDip}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              <span className="text-sm text-muted-foreground">Avg Score</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.avgScore.toFixed(1)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters - Clean layout */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
            {/* Search */}
            <div className="relative flex-1 w-full sm:max-w-xs">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search ticker..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            
            {/* Filter controls */}
            <div className="flex items-center gap-4 flex-wrap">
              {/* Lookback period */}
              <Select value={window.toString()} onValueChange={(v) => setWindow(Number(v))}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Period" />
                </SelectTrigger>
                <SelectContent>
                  {WINDOWS.map(w => (
                    <SelectItem key={w.value} value={w.value.toString()}>
                      {w.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              {/* Dip type */}
              <Select value={dipTypeFilter} onValueChange={setDipTypeFilter}>
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="Dip Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="STOCK_SPECIFIC">Stock-Specific</SelectItem>
                  <SelectItem value="MIXED">Mixed</SelectItem>
                  <SelectItem value="MARKET_DIP">Market Dip</SelectItem>
                </SelectContent>
              </Select>

              {/* Active dips toggle with Switch */}
              <div className="flex items-center gap-2">
                <Switch
                  id="active-dips"
                  checked={showActiveDips}
                  onCheckedChange={setShowActiveDips}
                />
                <Label htmlFor="active-dips" className="text-sm cursor-pointer whitespace-nowrap">
                  {showActiveDips ? 'Active Dips' : 'All Stocks'}
                </Label>
              </div>
            </div>
          </div>
          
          {/* Results count */}
          <p className="text-xs text-muted-foreground mt-3">
            {showActiveDips 
              ? `Showing stocks with ≥10% dip from ${window}-day peak`
              : 'Showing all stocks regardless of dip status'
            } • {filteredSignals.length} results
          </p>
        </CardContent>
      </Card>

      {/* Error state */}
      {error && (
        <Card className="mb-6 border-danger/50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-danger">
              <AlertTriangle className="w-5 h-5" />
              <p>{error}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Signals Table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle>Signals ({filteredSignals.length})</CardTitle>
            <Badge variant="outline">{window}-day lookback</Badge>
          </div>
          {/* Score Legend */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2 flex-wrap">
            <span className="font-medium">Score:</span>
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${colorblindMode ? 'bg-blue-500' : 'bg-success'}`} /> 70+ Strong
            </span>
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${colorblindMode ? 'bg-purple-500' : 'bg-chart-4'}`} /> 50-69 Good
            </span>
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${colorblindMode ? 'bg-orange-500' : 'bg-chart-2'}`} /> 30-49 Mod
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-muted-foreground" /> &lt;30 Weak
            </span>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : filteredSignals.length === 0 ? (
            <div className="text-center py-12">
              <Filter className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                {showActiveDips 
                  ? 'No active dips match your filters. Try showing all stocks.'
                  : 'No signals match your filters'
                }
              </p>
            </div>
          ) : isMobile ? (
            // Mobile: Card layout
            <div className="space-y-3">
              <AnimatePresence mode="popLayout">
                {paginatedSignals.map((signal) => (
                  <SignalCard 
                    key={signal.ticker} 
                    signal={signal} 
                    colorblindMode={colorblindMode}
                    onClick={() => setSelectedSignal(signal)}
                  />
                ))}
              </AnimatePresence>
            </div>
          ) : (
            // Desktop: Table layout
            <TooltipProvider>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Ticker</TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Dip %
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>How much the stock has dropped from its recent high.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Excess Dip
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Stock's dip minus market's dip. Higher means the stock fell more than the market.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Type
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Dip classification based on magnitude: Big Dip (&gt;25%), Normal (10-25%), or None.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        Days
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Consecutive days in dip territory (&gt;10% below peak).</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Quality
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Company fundamentals score based on profitability, dividend history, and P/E ratio.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Stability
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Price stability score based on 52-week volatility. Lower volatility = higher stability.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                    <TableHead>
                      <div className="flex items-center gap-1">
                        Score
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-[200px]">
                            <p>Overall dip opportunity score combining excess dip, quality, and stability metrics.</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence mode="popLayout">
                    {paginatedSignals.map((signal) => (
                      <motion.tr
                        key={signal.ticker}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="hover:bg-muted/50 transition-colors border-b cursor-pointer"
                        onClick={() => setSelectedSignal(signal)}
                      >
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <StockLogo symbol={signal.ticker} size="xs" />
                            <span className="font-semibold">{signal.ticker}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className={colorblindMode ? 'text-orange-500 font-medium' : 'text-danger font-medium'}>
                            {formatPercent(-signal.dip_stock)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className={signal.excess_dip > 0 
                            ? (colorblindMode ? 'text-blue-500' : 'text-success') 
                            : 'text-muted-foreground'
                          }>
                            {formatPercent(signal.excess_dip)}
                          </span>
                        </TableCell>
                        <TableCell>{getDipTypeBadge(signal.dip_class, colorblindMode)}</TableCell>
                        <TableCell className="text-center">
                          {signal.persist_days > 0 ? (
                            <span className="font-medium">{signal.persist_days}</span>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-muted rounded-full h-2 overflow-hidden">
                              <div 
                                className={`h-full ${getProgressColor(colorblindMode, 'quality')} transition-all`}
                                style={{ width: `${Math.min(signal.quality_score, 100)}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground w-8">
                              {formatScore(signal.quality_score)}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-muted rounded-full h-2 overflow-hidden">
                              <div 
                                className={`h-full ${getProgressColor(colorblindMode, 'stability')} transition-all`}
                                style={{ width: `${Math.min(signal.stability_score, 100)}%` }}
                              />
                            </div>
                            <span className="text-xs text-muted-foreground w-8">
                              {formatScore(signal.stability_score)}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className={`font-bold text-lg px-2 py-1 rounded inline-block ${getScoreBadgeClass(signal.final_score, colorblindMode)}`}>
                            {formatScore(signal.final_score)}
                          </div>
                        </TableCell>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </TableBody>
              </Table>
            </TooltipProvider>
          )}
          
          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t">
              <p className="text-sm text-muted-foreground">
                {page * ITEMS_PER_PAGE + 1}-{Math.min((page + 1) * ITEMS_PER_PAGE, filteredSignals.length)} of {filteredSignals.length}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  disabled={page === 0}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-muted-foreground px-2">
                  {page + 1} / {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Methodology Section */}
      <DipFinderMethodology colorblindMode={colorblindMode} />

      {/* Signal Details Sheet */}
      <SignalDetailsSheet
        signal={selectedSignal}
        isOpen={!!selectedSignal}
        onClose={() => setSelectedSignal(null)}
        colorblindMode={colorblindMode}
      />
    </div>
  );
}
