import { useCallback, useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  createPortfolio,
  getPortfolioDetail,
  getPortfolios,
  updatePortfolio,
  deletePortfolio,
  upsertHolding,
  deleteHolding,
  type Portfolio,
  type PortfolioDetail,
} from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
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
  DollarSign,
  Upload,
} from 'lucide-react';
import { BulkImportModal } from '@/components/BulkImportModal';

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

export function PortfolioPage() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<PortfolioDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Portfolio dialog
  const [portfolioDialogOpen, setPortfolioDialogOpen] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState<Portfolio | null>(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [portfolioCurrency, setPortfolioCurrency] = useState('USD');
  const [portfolioCash, setPortfolioCash] = useState('0');

  // Holding dialog
  const [holdingDialogOpen, setHoldingDialogOpen] = useState(false);
  const [editingHolding, setEditingHolding] = useState<{ symbol: string; quantity: number; avg_cost?: number | null } | null>(null);
  
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

  useEffect(() => {
    loadPortfolios();
  }, [loadPortfolios]);

  useEffect(() => {
    if (selectedId) {
      loadDetail(selectedId);
    }
  }, [selectedId, loadDetail]);

  const selectedPortfolio = useMemo(
    () => portfolios.find((p) => p.id === selectedId) || null,
    [portfolios, selectedId]
  );

  // Calculate portfolio stats
  const portfolioStats = useMemo(() => {
    if (!detail?.holdings) return { totalValue: 0, totalCost: 0, gainLoss: 0, gainLossPercent: 0 };
    
    let totalValue = 0;
    let totalCost = 0;
    
    for (const h of detail.holdings) {
      // We don't have current prices here, so use avg_cost * quantity as value estimate
      const value = h.quantity * (h.avg_cost || 0);
      totalValue += value;
      totalCost += h.quantity * (h.avg_cost || 0);
    }
    
    // Add cash
    totalValue += detail.cash_balance || 0;
    
    const gainLoss = totalValue - totalCost;
    const gainLossPercent = totalCost > 0 ? (gainLoss / totalCost) * 100 : 0;
    
    return { totalValue, totalCost, gainLoss, gainLossPercent };
  }, [detail]);

  function openCreateDialog() {
    setEditingPortfolio(null);
    setPortfolioName('');
    setPortfolioCurrency('USD');
    setPortfolioCash('0');
    setPortfolioDialogOpen(true);
  }

  function openEditDialog() {
    if (!selectedPortfolio) return;
    setEditingPortfolio(selectedPortfolio);
    setPortfolioName(selectedPortfolio.name);
    setPortfolioCurrency(selectedPortfolio.base_currency);
    setPortfolioCash(String(selectedPortfolio.cash_balance ?? 0));
    setPortfolioDialogOpen(true);
  }

  async function handleSavePortfolio() {
    if (!portfolioName.trim()) return;
    try {
      if (editingPortfolio) {
        await updatePortfolio(editingPortfolio.id, {
          name: portfolioName.trim(),
          base_currency: portfolioCurrency.trim(),
          cash_balance: Number(portfolioCash || 0),
        });
      } else {
        const created = await createPortfolio({
          name: portfolioName.trim(),
          base_currency: portfolioCurrency.trim(),
          cash_balance: Number(portfolioCash || 0),
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
    setHoldingDialogOpen(true);
  }

  function openEditHoldingDialog(holding: { symbol: string; quantity: number; avg_cost?: number | null }) {
    setEditingHolding(holding);
    setHoldingSymbol(holding.symbol);
    setHoldingQty(String(holding.quantity));
    setHoldingAvgCost(holding.avg_cost ? String(holding.avg_cost) : '');
    setHoldingDialogOpen(true);
  }

  async function handleSaveHolding() {
    if (!selectedId || !holdingSymbol.trim() || !holdingQty) return;
    try {
      await upsertHolding(selectedId, {
        symbol: holdingSymbol.trim().toUpperCase(),
        quantity: Number(holdingQty),
        avg_cost: holdingAvgCost ? Number(holdingAvgCost) : undefined,
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

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: selectedPortfolio?.base_currency || 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
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

      {/* Stats Cards */}
      {selectedPortfolio && (
        <div className="grid gap-4 sm:grid-cols-3">
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
                <div className="rounded-lg bg-muted p-2">
                  <DollarSign className="h-5 w-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Cash Balance</p>
                  <p className="text-xl font-semibold">{formatCurrency(detail?.cash_balance || 0)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <div className={`rounded-lg p-2 ${portfolioStats.gainLoss >= 0 ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                  {portfolioStats.gainLoss >= 0 
                    ? <TrendingUp className="h-5 w-5 text-green-600" />
                    : <TrendingDown className="h-5 w-5 text-red-600" />
                  }
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Gain/Loss</p>
                  <p className={`text-xl font-semibold ${portfolioStats.gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {portfolioStats.gainLoss >= 0 ? '+' : ''}{formatCurrency(portfolioStats.gainLoss)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
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
                  {detail.holdings.map((holding) => (
                    <motion.div
                      key={holding.symbol}
                      variants={item}
                      layout
                      className="flex items-center justify-between rounded-lg border p-4 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <div className="font-mono font-semibold text-lg">{holding.symbol}</div>
                        <Badge variant="secondary">{holding.quantity} shares</Badge>
                        {holding.avg_cost && (
                          <span className="text-sm text-muted-foreground">
                            @ {formatCurrency(holding.avg_cost)}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">
                          {formatCurrency(holding.quantity * (holding.avg_cost || 0))}
                        </span>
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
                  ))}
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
              <div>
                <Label>Cash Balance</Label>
                <Input 
                  type="number"
                  value={portfolioCash} 
                  onChange={(e) => setPortfolioCash(e.target.value)}
                  placeholder="0.00"
                />
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
      <Dialog open={holdingDialogOpen} onOpenChange={setHoldingDialogOpen}>
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
                <Label>Avg Cost (optional)</Label>
                <Input 
                  type="number"
                  step="0.01"
                  value={holdingAvgCost} 
                  onChange={(e) => setHoldingAvgCost(e.target.value)}
                  placeholder="150.00"
                />
              </div>
            </div>
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
              <Button onClick={handleSaveHolding} className="flex-1 sm:flex-none">
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
