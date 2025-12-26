import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Plus, 
  Pencil, 
  Trash2, 
  Loader2, 
  CheckCircle, 
  XCircle, 
  Search,
  TrendingDown
} from 'lucide-react';
import { 
  getSymbols, 
  createSymbol, 
  updateSymbol, 
  deleteSymbol,
  getCronJobs,
  getTaskStatus,
  validateSymbol,
  type Symbol,
  type TaskStatus,
} from '@/services/api';

interface SymbolManagerProps {
  onError?: (error: string) => void;
}

export function SymbolManager({ onError }: SymbolManagerProps) {
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [nextIngestRun, setNextIngestRun] = useState<string | null>(null);
  const [taskStatuses, setTaskStatuses] = useState<Record<string, TaskStatus>>({});
  
  // Add/Edit dialog
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingSymbol, setEditingSymbol] = useState<Symbol | null>(null);
  const [formSymbol, setFormSymbol] = useState('');
  const [formMinDipPct, setFormMinDipPct] = useState('15');
  const [formMinDays, setFormMinDays] = useState('5');
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{ valid: boolean; name?: string; error?: string } | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  
  // Delete dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingSymbol, setDeletingSymbol] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const loadSymbols = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getSymbols();
      setSymbols(data);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to load symbols');
    } finally {
      setIsLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    loadSymbols();
  }, [loadSymbols]);

  useEffect(() => {
    getCronJobs()
      .then((jobs) => {
        const ingest = jobs.find((job) => job.name === 'initial_data_ingest');
        setNextIngestRun(ingest?.next_run || null);
      })
      .catch(() => {});
  }, []);

  // Auto-poll when any symbol is in 'fetching' state
  useEffect(() => {
    const hasFetching = symbols.some(
      (symbol) => symbol.fetch_status === 'fetching' || symbol.fetch_status === 'pending'
    );
    if (!hasFetching) return;

    const pollInterval = setInterval(() => {
      // Silently reload without showing loading state, skip cache
      getSymbols(true).then(setSymbols).catch(() => {});
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [symbols]);

  const activeTaskIds = useMemo(
    () =>
      symbols
        .filter(
          (symbol) =>
            symbol.task_id &&
            (symbol.fetch_status === 'pending' || symbol.fetch_status === 'fetching')
        )
        .map((symbol) => symbol.task_id as string),
    [symbols]
  );

  useEffect(() => {
    if (activeTaskIds.length === 0) return;
    let cancelled = false;

    const pollTaskStatuses = async () => {
      const updates = await Promise.all(
        activeTaskIds.map(async (taskId) => {
          try {
            return await getTaskStatus(taskId);
          } catch {
            return null;
          }
        })
      );

      if (cancelled) return;

      setTaskStatuses((prev) => {
        const next = { ...prev };
        updates.forEach((status) => {
          if (status) {
            next[status.task_id] = status;
          }
        });
        return next;
      });

      const hasFinished = updates.some(
        (status) =>
          status && ['SUCCESS', 'FAILURE', 'REVOKED'].includes(status.status)
      );
      if (hasFinished) {
        getSymbols(true).then(setSymbols).catch(() => {});
      }
    };

    pollTaskStatuses();
    const pollInterval = setInterval(pollTaskStatuses, 2000);

    return () => {
      cancelled = true;
      clearInterval(pollInterval);
    };
  }, [activeTaskIds]);

  async function handleValidateSymbol() {
    if (!formSymbol.trim()) return;
    setIsValidating(true);
    setValidationResult(null);
    try {
      const result = await validateSymbol(formSymbol.trim().toUpperCase());
      setValidationResult(result);
    } catch {
      setValidationResult({ valid: false, error: 'Validation failed' });
    } finally {
      setIsValidating(false);
    }
  }

  async function handleSave() {
    if (!formSymbol.trim()) return;
    
    setIsSaving(true);
    try {
      // Backend expects min_dip_pct as decimal (0.0 to 1.0), not percentage
      const data = {
        symbol: formSymbol.trim().toUpperCase(),
        min_dip_pct: (parseFloat(formMinDipPct) || 15) / 100,
        min_days: parseInt(formMinDays) || 5,
      };
      
      if (editingSymbol) {
        await updateSymbol(editingSymbol.symbol, {
          min_dip_pct: data.min_dip_pct,
          min_days: data.min_days,
        });
      } else {
        await createSymbol(data);
      }
      
      await loadSymbols();
      closeDialog();
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to save symbol');
    } finally {
      setIsSaving(false);
    }
  }

  async function handleDelete() {
    if (!deletingSymbol) return;
    
    setIsDeleting(true);
    try {
      await deleteSymbol(deletingSymbol);
      await loadSymbols();
      setDeleteDialogOpen(false);
      setDeletingSymbol(null);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to delete symbol');
    } finally {
      setIsDeleting(false);
    }
  }

  function openAddDialog() {
    setEditingSymbol(null);
    setFormSymbol('');
    setFormMinDipPct('15');
    setFormMinDays('5');
    setValidationResult(null);
    setDialogOpen(true);
  }

  function openEditDialog(symbol: Symbol) {
    setEditingSymbol(symbol);
    setFormSymbol(symbol.symbol);
    // Convert decimal back to percentage for display
    setFormMinDipPct((symbol.min_dip_pct * 100).toString());
    setFormMinDays(symbol.min_days.toString());
    setValidationResult({ valid: true, name: symbol.symbol });
    setDialogOpen(true);
  }

  function closeDialog() {
    setDialogOpen(false);
    setEditingSymbol(null);
    setValidationResult(null);
  }

  const filteredSymbols = symbols.filter(s => 
    s.symbol.toLowerCase().includes(search.toLowerCase())
  );
  const hasPending = symbols.some(
    (s) => s.fetch_status === 'fetching' || s.fetch_status === 'pending'
  );
  const ingestHint = nextIngestRun
    ? `Next ingest run: ${new Date(nextIngestRun).toLocaleString()}`
    : 'Ingest queue active';

  const getTaskBadge = (taskId?: string | null) => {
    if (!taskId) return null;
    const status = taskStatuses[taskId]?.status;
    if (!status) {
      return (
        <Badge variant="outline" className="text-muted-foreground" title={taskId}>
          Queued
        </Badge>
      );
    }
    switch (status) {
      case 'PENDING':
      case 'RECEIVED':
        return (
          <Badge variant="outline" className="text-muted-foreground" title={taskId}>
            Queued
          </Badge>
        );
      case 'STARTED':
        return (
          <Badge variant="secondary" className="text-primary" title={taskId}>
            Running
          </Badge>
        );
      case 'RETRY':
        return (
          <Badge variant="outline" className="text-muted-foreground" title={taskId}>
            Retrying
          </Badge>
        );
      case 'FAILURE':
        return (
          <Badge variant="destructive" title={taskId}>
            Failed
          </Badge>
        );
      default:
        return null;
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingDown className="h-5 w-5" />
              Tracked Symbols
            </CardTitle>
            <CardDescription>
              Manage which stocks are tracked for dip detection
            </CardDescription>
          </div>
          <Button onClick={openAddDialog}>
            <Plus className="h-4 w-4 mr-2" />
            Add Symbol
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Search */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search symbols..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        {hasPending && (
          <div className="text-xs text-muted-foreground mb-3">
            {ingestHint}
          </div>
        )}

        {/* Table */}
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : filteredSymbols.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            {search ? 'No symbols match your search' : 'No symbols configured'}
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Min Dip %</TableHead>
                <TableHead>Min Days</TableHead>
                <TableHead className="w-[100px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <AnimatePresence>
                {filteredSymbols.map((symbol) => (
                  <motion.tr
                    key={symbol.symbol}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="border-b"
                  >
                    <TableCell>
                      <Badge variant="outline" className="font-mono">
                        {symbol.symbol}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {symbol.name || '—'}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {symbol.fetch_status === 'fetching' ? (
                          <Badge variant="secondary" className="animate-pulse">
                            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                            Fetching...
                          </Badge>
                        ) : symbol.fetch_status === 'error' ? (
                          <Badge variant="destructive" title={symbol.fetch_error || 'Unknown error'}>
                            <XCircle className="h-3 w-3 mr-1" />
                            Error
                          </Badge>
                        ) : symbol.fetch_status === 'fetched' ? (
                          <Badge variant="default" className="bg-success">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Ready
                          </Badge>
                        ) : (
                          <Badge variant="outline">
                            Pending
                          </Badge>
                        )}
                        {(symbol.fetch_status === 'pending' || symbol.fetch_status === 'fetching') &&
                          getTaskBadge(symbol.task_id)}
                      </div>
                    </TableCell>
                    <TableCell>{(symbol.min_dip_pct * 100).toFixed(0)}%</TableCell>
                    <TableCell>{symbol.min_days} days</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => openEditDialog(symbol)}
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => {
                            setDeletingSymbol(symbol.symbol);
                            setDeleteDialogOpen(true);
                          }}
                        >
                          <Trash2 className="h-4 w-4 text-danger" />
                        </Button>
                      </div>
                    </TableCell>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </TableBody>
          </Table>
        )}

        {/* Add/Edit Dialog */}
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>
                {editingSymbol ? 'Edit Symbol' : 'Add Symbol'}
              </DialogTitle>
              <DialogDescription>
                {editingSymbol 
                  ? 'Update the dip detection settings for this symbol'
                  : 'Add a new stock symbol to track. Symbol will be validated against Yahoo Finance.'
                }
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              {/* Symbol Input */}
              <div className="space-y-2">
                <Label>Symbol</Label>
                <div className="flex gap-2">
                  <Input
                    value={formSymbol}
                    onChange={(e) => {
                      setFormSymbol(e.target.value.toUpperCase());
                      setValidationResult(null);
                    }}
                    placeholder="AAPL"
                    className="font-mono"
                    disabled={!!editingSymbol}
                  />
                  {!editingSymbol && (
                    <Button
                      variant="outline"
                      onClick={handleValidateSymbol}
                      disabled={isValidating || !formSymbol.trim()}
                    >
                      {isValidating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        'Validate'
                      )}
                    </Button>
                  )}
                </div>
                {validationResult && (
                  <div className={`flex items-center gap-2 text-sm ${validationResult.valid ? 'text-success' : 'text-danger'}`}>
                    {validationResult.valid ? (
                      <>
                        <CheckCircle className="h-4 w-4" />
                        Valid: {validationResult.name}
                      </>
                    ) : (
                      <>
                        <XCircle className="h-4 w-4" />
                        {validationResult.error}
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* Min Dip % */}
              <div className="space-y-2">
                <Label>Minimum Dip %</Label>
                <Input
                  type="number"
                  value={formMinDipPct}
                  onChange={(e) => setFormMinDipPct(e.target.value)}
                  min="1"
                  max="100"
                />
                <p className="text-xs text-muted-foreground">
                  Only detect dips larger than this percentage
                </p>
              </div>

              {/* Min Days */}
              <div className="space-y-2">
                <Label>Minimum Days</Label>
                <Input
                  type="number"
                  value={formMinDays}
                  onChange={(e) => setFormMinDays(e.target.value)}
                  min="1"
                  max="365"
                />
                <p className="text-xs text-muted-foreground">
                  Only detect dips lasting at least this many days
                </p>
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={closeDialog}>
                Cancel
              </Button>
              <Button
                onClick={handleSave}
                disabled={
                  isSaving || 
                  !formSymbol.trim() || 
                  (!editingSymbol && !validationResult?.valid)
                }
              >
                {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                {editingSymbol ? 'Save Changes' : 'Add Symbol'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete Symbol</DialogTitle>
              <DialogDescription>
                Are you sure you want to delete <strong>{deletingSymbol}</strong>? 
                This will remove all tracking data for this symbol.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
                {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}
