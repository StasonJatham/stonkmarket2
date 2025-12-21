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
import { 
  RefreshCw, 
  TrendingDown, 
  AlertTriangle,
  CheckCircle,
  Filter,
  Search,
  BarChart3,
  Zap
} from 'lucide-react';

const WINDOWS = [7, 30, 100, 365];

// Alert level badge colors
function getAlertBadge(level: string) {
  const variants: Record<string, { class: string; icon: React.ReactNode }> = {
    HIGH: { 
      class: 'bg-success/20 text-success border-success/30', 
      icon: <Zap className="w-3 h-3" /> 
    },
    MEDIUM: { 
      class: 'bg-chart-4/20 text-chart-4 border-chart-4/30', 
      icon: <AlertTriangle className="w-3 h-3" /> 
    },
    LOW: { 
      class: 'bg-muted text-muted-foreground border-border', 
      icon: <CheckCircle className="w-3 h-3" /> 
    },
    NONE: { 
      class: 'bg-muted/50 text-muted-foreground border-border/50', 
      icon: null 
    },
  };
  const variant = variants[level] || variants.NONE;
  return (
    <Badge variant="outline" className={`${variant.class} font-medium gap-1`}>
      {variant.icon}
      {level}
    </Badge>
  );
}

// Dip class badge
function getDipClassBadge(dipClass: string) {
  const colors: Record<string, string> = {
    UNIQUE: 'bg-success/20 text-success border-success/30',
    RELATIVE: 'bg-chart-2/20 text-chart-2 border-chart-2/30',
    MARKET_WIDE: 'bg-chart-4/20 text-chart-4 border-chart-4/30',
    MICRO: 'bg-muted text-muted-foreground border-border',
  };
  return (
    <Badge variant="outline" className={`${colors[dipClass] || ''} font-medium`}>
      {dipClass.replace('_', ' ')}
    </Badge>
  );
}

function formatPercent(value: number, decimals = 1): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
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
  const [alertFilter, setAlertFilter] = useState<string>('all');
  const [dipClassFilter, setDipClassFilter] = useState<string>('all');

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
    
    // Alert level filter
    if (alertFilter !== 'all') {
      result = result.filter(s => s.alert_level === alertFilter);
    }
    
    // Dip class filter
    if (dipClassFilter !== 'all') {
      result = result.filter(s => s.dip_class === dipClassFilter);
    }
    
    // Sort by final score descending
    result.sort((a, b) => b.final_score - a.final_score);
    
    return result;
  }, [signals, searchQuery, alertFilter, dipClassFilter]);

  // Stats
  const stats = useMemo(() => {
    const high = signals.filter(s => s.alert_level === 'HIGH').length;
    const medium = signals.filter(s => s.alert_level === 'MEDIUM').length;
    const unique = signals.filter(s => s.dip_class === 'UNIQUE').length;
    const avgScore = signals.length > 0 
      ? signals.reduce((sum, s) => sum + s.final_score, 0) / signals.length 
      : 0;
    return { high, medium, unique, avgScore };
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
              <Zap className="w-5 h-5 text-success" />
              <span className="text-sm text-muted-foreground">High Alerts</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.high}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-chart-4" />
              <span className="text-sm text-muted-foreground">Medium Alerts</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.medium}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-success" />
              <span className="text-sm text-muted-foreground">Unique Dips</span>
            </div>
            <p className="text-3xl font-bold mt-2">{stats.unique}</p>
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
              
              <div className="w-32">
                <Label htmlFor="alert" className="sr-only">Alert Level</Label>
                <Select value={alertFilter} onValueChange={setAlertFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Alert" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Alerts</SelectItem>
                    <SelectItem value="HIGH">High</SelectItem>
                    <SelectItem value="MEDIUM">Medium</SelectItem>
                    <SelectItem value="LOW">Low</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="w-36">
                <Label htmlFor="class" className="sr-only">Dip Class</Label>
                <Select value={dipClassFilter} onValueChange={setDipClassFilter}>
                  <SelectTrigger>
                    <SelectValue placeholder="Class" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Classes</SelectItem>
                    <SelectItem value="UNIQUE">Unique</SelectItem>
                    <SelectItem value="RELATIVE">Relative</SelectItem>
                    <SelectItem value="MARKET_WIDE">Market Wide</SelectItem>
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
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Ticker</TableHead>
                    <TableHead>Dip %</TableHead>
                    <TableHead>Excess Dip</TableHead>
                    <TableHead>Class</TableHead>
                    <TableHead className="text-center">Days</TableHead>
                    <TableHead>Quality</TableHead>
                    <TableHead>Stability</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Alert</TableHead>
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
                        <TableCell>{getDipClassBadge(signal.dip_class)}</TableCell>
                        <TableCell className="text-center">{signal.persist_days}</TableCell>
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
                          <span className="font-bold text-lg">
                            {formatScore(signal.final_score)}
                          </span>
                        </TableCell>
                        <TableCell>{getAlertBadge(signal.alert_level)}</TableCell>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
