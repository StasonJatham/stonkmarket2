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
  HelpCircle
} from 'lucide-react';

const WINDOWS = [7, 30, 100, 365];

// Dip type badge - shows whether dip is stock-specific, mixed, or market-wide
function getDipTypeBadge(dipType: string) {
  const config: Record<string, { color: string; label: string; desc: string }> = {
    STOCK_SPECIFIC: { 
      color: 'bg-success/20 text-success border-success/30', 
      label: 'Stock',
      desc: 'This dip is specific to this stock, not market-wide' 
    },
    MIXED: { 
      color: 'bg-chart-2/20 text-chart-2 border-chart-2/30', 
      label: 'Mixed',
      desc: 'Part stock-specific, part market-driven dip' 
    },
    MARKET_DIP: { 
      color: 'bg-chart-4/20 text-chart-4 border-chart-4/30', 
      label: 'Market',
      desc: 'This dip is primarily driven by overall market decline' 
    },
  };
  const item = config[dipType] || { color: '', label: dipType, desc: '' };
  return (
    <Badge variant="outline" className={`${item.color} font-medium`} title={item.desc}>
      {item.label}
    </Badge>
  );
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

export function DipFinderPage() {
  const [signals, setSignals] = useState<DipSignal[]>([]);
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [window, setWindow] = useState(30);
  const [searchQuery, setSearchQuery] = useState('');
  const [dipTypeFilter, setDipTypeFilter] = useState<string>('all');

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
    
    // Sort by final score descending
    result.sort((a, b) => b.final_score - a.final_score);
    
    return result;
  }, [signals, searchQuery, dipTypeFilter]);

  // Stats
  const stats = useMemo(() => {
    const stockSpecific = signals.filter(s => s.dip_class === 'STOCK_SPECIFIC').length;
    const mixed = signals.filter(s => s.dip_class === 'MIXED').length;
    const marketDip = signals.filter(s => s.dip_class === 'MARKET_DIP').length;
    const avgScore = signals.length > 0 
      ? signals.reduce((sum, s) => sum + s.final_score, 0) / signals.length 
      : 0;
    return { stockSpecific, mixed, marketDip, avgScore };
  }, [signals]);

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
              <TrendingDown className="w-5 h-5 text-success" />
              <span className="text-sm text-muted-foreground">Stock-Specific</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.stockSpecific}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-chart-2" />
              <span className="text-sm text-muted-foreground">Mixed Dips</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.mixed}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-chart-4" />
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
                <TooltipContent className="max-w-xs text-xs">
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
            
            <div className="flex gap-4">
              <div className="w-32">
                <Label htmlFor="window" className="sr-only">Window</Label>
                <Select value={window.toString()} onValueChange={(v) => setWindow(Number(v))}>
                  <SelectTrigger>
                    <SelectValue placeholder="Window" />
                  </SelectTrigger>
                  <SelectContent>
                    {WINDOWS.map(w => (
                      <SelectItem key={w} value={w.toString()}>
                        {w} days
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="w-36">
                <Label htmlFor="dipType" className="sr-only">Dip Type</Label>
                <Select value={dipTypeFilter} onValueChange={setDipTypeFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Dip Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="STOCK_SPECIFIC">Stock-Specific</SelectItem>
                    <SelectItem value="MIXED">Mixed</SelectItem>
                    <SelectItem value="MARKET_DIP">Market Dip</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
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

      {/* Signals table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Signals ({filteredSignals.length})</span>
            <Badge variant="outline">{window}-day window</Badge>
          </CardTitle>
          {/* Score Legend */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground mt-2 flex-wrap">
            <span className="font-medium">Score Guide:</span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-success/20" /> 70+ Strong
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-chart-4/20" /> 50-69 Good
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-chart-2/20" /> 30-49 Moderate
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
              <p className="text-muted-foreground">No signals match your filters</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
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
                        <TooltipContent className="max-w-xs text-xs">
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
                        <TooltipContent className="max-w-xs text-xs">
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
                        <TooltipContent className="max-w-xs text-xs">
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
                        <TooltipContent className="max-w-xs text-xs">
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
                        <TooltipContent className="max-w-xs text-xs">
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
                        <TooltipContent className="max-w-xs text-xs">
                          <p className="font-medium mb-1">Buying Opportunity Score (0-100)</p>
                          <p>Higher = better dip. Combines dip magnitude, rarity, persistence, and classification.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence mode="popLayout">
                    {filteredSignals.map((signal) => (
                      <motion.tr
                        key={signal.ticker}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="hover:bg-muted/50 transition-colors border-b"
                      >
                        <TableCell className="font-semibold">{signal.ticker}</TableCell>
                        <TableCell>
                          <span className="text-danger font-medium">
                            {formatPercent(-signal.dip_stock)}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className={signal.excess_dip > 0 ? 'text-success' : 'text-muted-foreground'}>
                            {formatPercent(signal.excess_dip)}
                          </span>
                        </TableCell>
                        <TableCell>{getDipTypeBadge(signal.dip_class)}</TableCell>
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
                                className="h-full bg-success transition-all"
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
                                className="h-full bg-chart-2 transition-all"
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
                              <div className={`font-bold text-lg px-2 py-1 rounded cursor-help ${
                                signal.final_score >= 70 ? 'bg-success/20 text-success' :
                                signal.final_score >= 50 ? 'bg-chart-4/20 text-chart-4' :
                                signal.final_score >= 30 ? 'bg-chart-2/20 text-chart-2' :
                                'bg-muted text-muted-foreground'
                              }`}>
                                {formatScore(signal.final_score)}
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs text-xs">
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
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
