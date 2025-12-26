import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  createPortfolio,
  getPortfolioDetail,
  getPortfolios,
  updatePortfolio,
  deletePortfolio,
  upsertHolding,
  deleteHolding,
  addTransaction,
  deleteTransaction,
  runPortfolioAnalytics,
  type Portfolio,
  type PortfolioDetail,
  type PortfolioAnalyticsResponse,
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Plus, Trash2, RefreshCw } from 'lucide-react';

const DEFAULT_TOOLS = [
  'quantstats',
  'skfolio',
  'arch',
  'prophet',
  'vectorbt',
  'pandas_ta',
  'talipp',
  'mlfinlab',
  'alphalens',
  'pyfolio',
  'finquant',
  'gluonts',
  'pyflux',
  'lppls',
  'eiten',
];

export function PortfolioPage() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<PortfolioDetail | null>(null);
  const [analytics, setAnalytics] = useState<PortfolioAnalyticsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [portfolioDialogOpen, setPortfolioDialogOpen] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState<Portfolio | null>(null);
  const [portfolioName, setPortfolioName] = useState('');
  const [portfolioCurrency, setPortfolioCurrency] = useState('USD');
  const [portfolioCash, setPortfolioCash] = useState('0');

  const [holdingSymbol, setHoldingSymbol] = useState('');
  const [holdingQty, setHoldingQty] = useState('');
  const [holdingAvgCost, setHoldingAvgCost] = useState('');
  const [holdingTargetWeight, setHoldingTargetWeight] = useState('');

  const [txnSymbol, setTxnSymbol] = useState('');
  const [txnSide, setTxnSide] = useState('buy');
  const [txnQty, setTxnQty] = useState('');
  const [txnPrice, setTxnPrice] = useState('');
  const [txnFees, setTxnFees] = useState('');
  const [txnDate, setTxnDate] = useState(new Date().toISOString().slice(0, 10));
  const [txnNotes, setTxnNotes] = useState('');

  const [selectedTools, setSelectedTools] = useState<string[]>(DEFAULT_TOOLS);
  const [analysisSymbol, setAnalysisSymbol] = useState('');
  const [analysisFast, setAnalysisFast] = useState('20');
  const [analysisSlow, setAnalysisSlow] = useState('50');

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
    try {
      await deletePortfolio(selectedPortfolio.id);
      setSelectedId(null);
      setDetail(null);
      await loadPortfolios();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete portfolio');
    }
  }

  async function handleAddHolding() {
    if (!selectedId || !holdingSymbol.trim() || !holdingQty) return;
    try {
      await upsertHolding(selectedId, {
        symbol: holdingSymbol.trim().toUpperCase(),
        quantity: Number(holdingQty),
        avg_cost: holdingAvgCost ? Number(holdingAvgCost) : undefined,
        target_weight: holdingTargetWeight ? Number(holdingTargetWeight) : undefined,
      });
      await loadDetail(selectedId);
      setHoldingSymbol('');
      setHoldingQty('');
      setHoldingAvgCost('');
      setHoldingTargetWeight('');
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

  async function handleAddTransaction() {
    if (!selectedId || !txnSymbol.trim() || !txnDate) return;
    try {
      await addTransaction(selectedId, {
        symbol: txnSymbol.trim().toUpperCase(),
        side: txnSide,
        quantity: txnQty ? Number(txnQty) : undefined,
        price: txnPrice ? Number(txnPrice) : undefined,
        fees: txnFees ? Number(txnFees) : undefined,
        trade_date: txnDate,
        notes: txnNotes || undefined,
      });
      await loadDetail(selectedId);
      setTxnSymbol('');
      setTxnQty('');
      setTxnPrice('');
      setTxnFees('');
      setTxnNotes('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add transaction');
    }
  }

  async function handleDeleteTransaction(transactionId: number) {
    if (!selectedId) return;
    try {
      await deleteTransaction(selectedId, transactionId);
      await loadDetail(selectedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete transaction');
    }
  }

  async function handleRunAnalytics() {
    if (!selectedId) return;
    setIsLoading(true);
    setError(null);
    try {
      const params: Record<string, unknown> = {};
      if (analysisSymbol.trim()) {
        params.symbol = analysisSymbol.trim().toUpperCase();
      }
      if (analysisFast) {
        params.fast = Number(analysisFast);
      }
      if (analysisSlow) {
        params.slow = Number(analysisSlow);
      }
      const result = await runPortfolioAnalytics(selectedId, {
        tools: selectedTools,
        params: Object.keys(params).length > 0 ? params : undefined,
      });
      setAnalytics(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run analytics');
    } finally {
      setIsLoading(false);
    }
  }

  function toggleTool(tool: string) {
    setSelectedTools((prev) =>
      prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool]
    );
  }

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <CardTitle>Portfolio Manager</CardTitle>
              <CardDescription>Track holdings, trades, and analytics in one place.</CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={openCreateDialog}>
                <Plus className="mr-2 h-4 w-4" />
                New Portfolio
              </Button>
              <Button variant="outline" onClick={openEditDialog} disabled={!selectedPortfolio}>
                Edit
              </Button>
              <Button variant="ghost" onClick={handleDeletePortfolio} disabled={!selectedPortfolio}>
                <Trash2 className="mr-2 h-4 w-4" />
                Archive
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="flex-1">
              <Label>Portfolio</Label>
              <Select
                value={selectedId ? String(selectedId) : ''}
                onValueChange={(value) => setSelectedId(Number(value))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a portfolio" />
                </SelectTrigger>
                <SelectContent>
                  {portfolios.map((portfolio) => (
                    <SelectItem key={portfolio.id} value={String(portfolio.id)}>
                      {portfolio.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label>Cash Balance</Label>
              <Input value={detail?.cash_balance ?? 0} readOnly />
            </div>
          </div>
          {error && <p className="text-sm text-destructive mt-3">{error}</p>}
        </CardContent>
      </Card>

      <Tabs defaultValue="holdings" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="holdings">Holdings</TabsTrigger>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="holdings">
          <Card>
            <CardHeader>
              <CardTitle>Holdings</CardTitle>
              <CardDescription>Add or update positions in your portfolio.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-4">
                <div>
                  <Label>Symbol</Label>
                  <Input value={holdingSymbol} onChange={(e) => setHoldingSymbol(e.target.value)} />
                </div>
                <div>
                  <Label>Quantity</Label>
                  <Input value={holdingQty} onChange={(e) => setHoldingQty(e.target.value)} />
                </div>
                <div>
                  <Label>Avg Cost</Label>
                  <Input value={holdingAvgCost} onChange={(e) => setHoldingAvgCost(e.target.value)} />
                </div>
                <div>
                  <Label>Target Weight</Label>
                  <Input value={holdingTargetWeight} onChange={(e) => setHoldingTargetWeight(e.target.value)} />
                </div>
              </div>
              <Button onClick={handleAddHolding} disabled={!selectedId}>
                Save Holding
              </Button>

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>Avg Cost</TableHead>
                    <TableHead>Target Weight</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {detail?.holdings?.map((holding) => (
                    <TableRow key={holding.id}>
                      <TableCell className="font-medium">{holding.symbol}</TableCell>
                      <TableCell>{holding.quantity}</TableCell>
                      <TableCell>{holding.avg_cost ?? '-'}</TableCell>
                      <TableCell>{holding.target_weight ?? '-'}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteHolding(holding.symbol)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {detail?.holdings.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={5} className="text-muted-foreground">
                        No holdings yet.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="transactions">
          <Card>
            <CardHeader>
              <CardTitle>Transactions</CardTitle>
              <CardDescription>Log trades to track cost basis and performance.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-6">
                <div>
                  <Label>Symbol</Label>
                  <Input value={txnSymbol} onChange={(e) => setTxnSymbol(e.target.value)} />
                </div>
                <div>
                  <Label>Side</Label>
                  <Select value={txnSide} onValueChange={setTxnSide}>
                    <SelectTrigger>
                      <SelectValue placeholder="Side" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="buy">Buy</SelectItem>
                      <SelectItem value="sell">Sell</SelectItem>
                      <SelectItem value="dividend">Dividend</SelectItem>
                      <SelectItem value="split">Split</SelectItem>
                      <SelectItem value="deposit">Deposit</SelectItem>
                      <SelectItem value="withdrawal">Withdrawal</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Qty</Label>
                  <Input value={txnQty} onChange={(e) => setTxnQty(e.target.value)} />
                </div>
                <div>
                  <Label>Price</Label>
                  <Input value={txnPrice} onChange={(e) => setTxnPrice(e.target.value)} />
                </div>
                <div>
                  <Label>Fees</Label>
                  <Input value={txnFees} onChange={(e) => setTxnFees(e.target.value)} />
                </div>
                <div>
                  <Label>Date</Label>
                  <Input type="date" value={txnDate} onChange={(e) => setTxnDate(e.target.value)} />
                </div>
              </div>
              <div>
                <Label>Notes</Label>
                <Input value={txnNotes} onChange={(e) => setTxnNotes(e.target.value)} />
              </div>
              <Button onClick={handleAddTransaction} disabled={!selectedId}>
                Add Transaction
              </Button>

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Qty</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Fees</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {detail?.transactions?.map((txn) => (
                    <TableRow key={txn.id}>
                      <TableCell>{txn.trade_date}</TableCell>
                      <TableCell className="font-medium">{txn.symbol}</TableCell>
                      <TableCell>{txn.side}</TableCell>
                      <TableCell>{txn.quantity ?? '-'}</TableCell>
                      <TableCell>{txn.price ?? '-'}</TableCell>
                      <TableCell>{txn.fees ?? '-'}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteTransaction(txn.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                  {detail?.transactions.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} className="text-muted-foreground">
                        No transactions yet.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics">
          <Card>
            <CardHeader>
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <CardTitle>Analytics</CardTitle>
                  <CardDescription>Run quant tools and review outputs.</CardDescription>
                </div>
                <Button onClick={handleRunAnalytics} disabled={!selectedId || isLoading}>
                  <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  Run Analytics
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-3 md:grid-cols-3">
                {DEFAULT_TOOLS.map((tool) => (
                  <div key={tool} className="flex items-center justify-between rounded-md border p-3">
                    <span className="text-sm">{tool}</span>
                    <Switch
                      checked={selectedTools.includes(tool)}
                      onCheckedChange={() => toggleTool(tool)}
                    />
                  </div>
                ))}
              </div>

              <div className="grid gap-3 md:grid-cols-3">
                <div>
                  <Label>Analysis Symbol (for indicator tools)</Label>
                  <Input value={analysisSymbol} onChange={(e) => setAnalysisSymbol(e.target.value)} />
                </div>
                <div>
                  <Label>VectorBT Fast Window</Label>
                  <Input value={analysisFast} onChange={(e) => setAnalysisFast(e.target.value)} />
                </div>
                <div>
                  <Label>VectorBT Slow Window</Label>
                  <Input value={analysisSlow} onChange={(e) => setAnalysisSlow(e.target.value)} />
                </div>
              </div>

              {analytics?.scheduled_tools?.length ? (
                <div className="rounded-md border p-3 text-sm space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">Batch analytics queued</Badge>
                    {analytics.job_status && (
                      <span className="text-muted-foreground">Status: {analytics.job_status}</span>
                    )}
                  </div>
                  <div className="text-muted-foreground">
                    Tools: {analytics.scheduled_tools.join(', ')}
                  </div>
                  {analytics.job_id && (
                    <div className="text-xs text-muted-foreground">Job ID: {analytics.job_id}</div>
                  )}
                </div>
              ) : null}

              <div className="space-y-3">
                {analytics?.results.map((result) => (
                  <Card key={result.tool}>
                    <CardHeader>
                      <CardTitle className="text-base capitalize">{result.tool}</CardTitle>
                      <CardDescription>
                        Status: {result.status}
                        {result.source ? ` • ${result.source}` : ''}
                        {result.generated_at
                          ? ` • ${new Date(result.generated_at).toLocaleString()}`
                          : ''}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {result.warnings.length > 0 && (
                        <div className="text-sm text-muted-foreground mb-2">
                          Warnings: {result.warnings.join(', ')}
                        </div>
                      )}
                      <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(result.data, null, 2)}</pre>
                    </CardContent>
                  </Card>
                ))}
                {!analytics && (
                  <div className="text-sm text-muted-foreground">
                    Run analytics to see results.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Dialog open={portfolioDialogOpen} onOpenChange={setPortfolioDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingPortfolio ? 'Edit Portfolio' : 'New Portfolio'}</DialogTitle>
            <DialogDescription>
              Set a name, currency, and optional cash balance.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Name</Label>
              <Input value={portfolioName} onChange={(e) => setPortfolioName(e.target.value)} />
            </div>
            <div>
              <Label>Base Currency</Label>
              <Input value={portfolioCurrency} onChange={(e) => setPortfolioCurrency(e.target.value)} />
            </div>
            <div>
              <Label>Cash Balance</Label>
              <Input value={portfolioCash} onChange={(e) => setPortfolioCash(e.target.value)} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPortfolioDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSavePortfolio}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
