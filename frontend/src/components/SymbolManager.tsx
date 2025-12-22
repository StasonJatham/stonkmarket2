import { useState, useEffect } from 'react';
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
  validateSymbol,
  type Symbol 
} from '@/services/api';

interface SymbolManagerProps {
  onError?: (error: string) => void;
}

export function SymbolManager({ onError }: SymbolManagerProps) {
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  
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

  useEffect(() => {
    loadSymbols();
  }, []);

  async function loadSymbols() {
    setIsLoading(true);
    try {
      const data = await getSymbols();
      setSymbols(data);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to load symbols');
    } finally {
      setIsLoading(false);
    }
  }

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
