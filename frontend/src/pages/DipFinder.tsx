import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  getDipFinderSignals, 
  getSymbols,
  type DipSignal, 
  type Symbol 
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { 
  RefreshCw, 
  TrendingDown, 
  AlertTriangle,
  Filter,
  Search,
  BarChart3,
  Zap,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff
} from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';
import { useIsMobile } from '@/hooks/useIsMobile';

const WINDOWS = [
  { value: 7, label: '7 days', desc: 'Short-term dips (1 week lookback)' },
  { value: 30, label: '30 days', desc: 'Medium-term dips (1 month lookback)' },
  { value: 100, label: '100 days', desc: 'Long-term dips (quarter lookback)' },
  { value: 365, label: '365 days', desc: 'Annual dips (1 year lookback)' },
];

const ITEMS_PER_PAGE = 20;

// Dip type badge - shows whether dip is stock-specific, mixed, or market-wide
function getDipTypeBadge(dipType: string, colorblindMode: boolean) {
  const config: Record<string, { color: string; colorblindColor: string; label: string; desc: string }> = {
    STOCK_SPECIFIC: { 
      color: 'bg-success/20 text-success border-success/30', 
      colorblindColor: 'bg-blue-500/20 text-blue-500 border-blue-500/30',
      label: 'Stock',
      desc: 'This dip is specific to this stock, not market-wide' 
    },
    MIXED: { 
      color: 'bg-chart-2/20 text-chart-2 border-chart-2/30', 
      colorblindColor: 'bg-purple-500/20 text-purple-500 border-purple-500/30',
      label: 'Mixed',
      desc: 'Part stock-specific, part market-driven dip' 
    },
    MARKET_DIP: { 
      color: 'bg-chart-4/20 text-chart-4 border-chart-4/30', 
      colorblindColor: 'bg-orange-500/20 text-orange-500 border-orange-500/30',
      label: 'Market',
      desc: 'This dip is primarily driven by overall market decline' 
    },
  };
  const item = config[dipType] || { color: '', colorblindColor: '', label: dipType, desc: '' };
  return (
    <Badge variant="outline" className={`${colorblindMode ? item.colorblindColor : item.color} font-medium`} title={item.desc}>
      {item.label}
    </Badge>
  );
}

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
  // value is a fraction (e.g., 0.047 = 4.7%), multiply by 100 to get percentage
  const pct = value * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(decimals)}%`;
}

function formatScore(value: number): string {
  return value.toFixed(0);
}

// Mobile card component
function SignalCard({ signal, colorblindMode }: { signal: DipSignal; colorblindMode: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="bg-card border rounded-lg p-4"
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
      setPage(0); // Reset to first page when data changes
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
    
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toUpperCase();
      result = result.filter(s => s.ticker.includes(query));
    }
    
    // Dip type filter
    if (dipTypeFilter !== 'all') {
      result = result.filter(s => s.dip_class === dipTypeFilter);
    }
    
    // Active dips filter (dip >= 10%)
    if (showActiveDips) {
      result = result.filter(s => s.dip_stock >= 0.10);
    }
    
    // Sort by final score descending
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

  const windowInfo = WINDOWS.find(w => w.value === window);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <BarChart3 className="h-8 w-8" />
            DipFinder Signals
          </h1>
          <p className="text-muted-foreground mt-1">
            AI-powered dip detection with quality and stability scoring
          </p>
        </div>
        <Button onClick={loadSignals} disabled={isLoading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
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
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-2 cursor-help">
                    <BarChart3 className="w-5 h-5 text-primary" />
                    <span className="text-sm text-muted-foreground">Avg Score</span>
                    <HelpCircle className="w-3 h-3 text-muted-foreground" />
                  </div>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                  <p className="font-medium mb-1">Buying Opportunity Score (0-100)</p>
                  <p className="text-muted-foreground">Higher = better dip opportunity. Based on dip magnitude (40pts), rarity percentile (25pts), vs typical behavior (20pts), persistence (10pts), and classification (5pts).</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <p className="text-3xl font-bold mt-2">{stats.avgScore.toFixed(1)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4">
            {/* Row 1: Search and main filters */}
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <Label htmlFor="search" className="sr-only">Search</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    id="search"
                    placeholder="Search by ticker..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              
              <div className="flex gap-3 flex-wrap">
                {/* Window selector with tooltip */}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="w-36">
                        <Label htmlFor="window" className="sr-only">Lookback Period</Label>
                        <Select value={window.toString()} onValueChange={(v) => setWindow(Number(v))}>
                          <SelectTrigger>
                            <SelectValue placeholder="Period" />
                          </SelectTrigger>
                          <SelectContent className="bg-popover text-popover-foreground border">
                            {WINDOWS.map(w => (
                              <SelectItem key={w.value} value={w.value.toString()}>
                                {w.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                      <p className="font-medium">Lookback Period</p>
                      <p className="text-muted-foreground">{windowInfo?.desc}</p>
                      <p className="text-muted-foreground mt-1">Dips are measured from the peak price within this window.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <div className="w-36">
                  <Label htmlFor="dipType" className="sr-only">Dip Type</Label>
                  <Select value={dipTypeFilter} onValueChange={setDipTypeFilter}>
                    <SelectTrigger>
                      <SelectValue placeholder="Dip Type" />
                    </SelectTrigger>
                    <SelectContent className="bg-popover text-popover-foreground border">
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="STOCK_SPECIFIC">Stock-Specific</SelectItem>
                      <SelectItem value="MIXED">Mixed</SelectItem>
                      <SelectItem value="MARKET_DIP">Market Dip</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                {/* Active dips toggle */}
                <Button
                  variant={showActiveDips ? 'default' : 'outline'}
                  size="default"
                  onClick={() => setShowActiveDips(!showActiveDips)}
                  className="gap-2"
                >
                  {showActiveDips ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                  {showActiveDips ? 'Active Dips' : 'All Stocks'}
                </Button>
              </div>
            </div>
            
            {/* Info text about active dips */}
            <p className="text-xs text-muted-foreground">
              {showActiveDips 
                ? `Showing stocks with ≥10% dip from ${window}-day peak. `
                : `Showing all stocks regardless of dip status. `
              }
              {filteredSignals.length} results
            </p>
          </div>
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

      {/* Signals */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Signals ({filteredSignals.length})</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="cursor-help">
                    {window}-day lookback
                    <HelpCircle className="w-3 h-3 ml-1" />
                  </Badge>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                  <p>{windowInfo?.desc}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </CardTitle>
          {/* Score Legend - colorblind aware */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2 flex-wrap">
            <span className="font-medium">Score Guide:</span>
            <span className="flex items-center gap-1">
              <span className={`w-3 h-3 rounded ${colorblindMode ? 'bg-blue-500/30' : 'bg-success/20'}`} /> 70+ Strong
            </span>
            <span className="flex items-center gap-1">
              <span className={`w-3 h-3 rounded ${colorblindMode ? 'bg-purple-500/30' : 'bg-chart-4/20'}`} /> 50-69 Good
            </span>
            <span className="flex items-center gap-1">
              <span className={`w-3 h-3 rounded ${colorblindMode ? 'bg-orange-500/30' : 'bg-chart-2/20'}`} /> 30-49 Moderate
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-muted" /> &lt;30 Weak
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
                  <SignalCard key={signal.ticker} signal={signal} colorblindMode={colorblindMode} />
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
                    <TableHead>Dip %</TableHead>
                    <TableHead>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Excess Dip
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          How much deeper than typical market dip. Positive = stock fell more than market.
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                    <TableHead>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Dip Type
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          <p><b>Stock:</b> Dip is specific to this stock</p>
                          <p><b>Mixed:</b> Part stock, part market-driven</p>
                          <p><b>Market:</b> Primarily market-wide decline</p>
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                    <TableHead className="text-center">
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Days
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          Consecutive days the stock has been in dip territory. "—" means just started.
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                    <TableHead>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Quality
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          Company quality score (0-100) based on profitability, balance sheet, cash flow, and growth.
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                    <TableHead>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Stability
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          Price stability score (0-100). Higher = more stable, lower volatility/beta.
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                    <TableHead>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1 cursor-help">
                          Score
                          <HelpCircle className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                          <p className="font-medium mb-1">Buying Opportunity Score (0-100)</p>
                          <p>Higher = better dip. Combines dip magnitude, rarity, persistence, and classification.</p>
                        </TooltipContent>
                      </Tooltip>
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
                        className="hover:bg-muted/50 transition-colors border-b"
                      >
                        <TableCell className="font-semibold">{signal.ticker}</TableCell>
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
                            <span className="text-muted-foreground" title="Dip just started or threshold not met">—</span>
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
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <div className={`font-bold text-lg px-2 py-1 rounded cursor-help ${getScoreBadgeClass(signal.final_score, colorblindMode)}`}>
                                {formatScore(signal.final_score)}
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs text-xs bg-popover text-popover-foreground border">
                              <p className="font-medium mb-1">
                                {signal.final_score >= 70 ? '🔥 Strong Opportunity' :
                                 signal.final_score >= 50 ? '👍 Good Opportunity' :
                                 signal.final_score >= 30 ? '🤔 Moderate Opportunity' :
                                 '😐 Weak Opportunity'}
                              </p>
                              <p className="text-muted-foreground">
                                Score: {formatScore(signal.final_score)}/100
                              </p>
                            </TooltipContent>
                          </Tooltip>
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
                Showing {page * ITEMS_PER_PAGE + 1}-{Math.min((page + 1) * ITEMS_PER_PAGE, filteredSignals.length)} of {filteredSignals.length}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  disabled={page === 0}
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground px-2">
                  Page {page + 1} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
