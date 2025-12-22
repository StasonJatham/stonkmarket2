import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getAllSuggestions,
  approveSuggestion,
  rejectSuggestion,
  updateSuggestion,
  refreshSuggestionData,
  retrySuggestionFetch,
  getRuntimeSettings,
  updateRuntimeSettings,
  type Suggestion,
  type SuggestionStatus,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Slider } from '@/components/ui/slider';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Check,
  X,
  Lightbulb,
  ThumbsUp,
  Clock,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Loader2,
  AlertCircle,
  Pencil,
  Settings,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

function getStatusBadge(status: SuggestionStatus) {
  switch (status) {
    case 'approved':
      return (
        <Badge className="bg-success/20 text-success border-success/30 hover:bg-success/30">
          <Check className="h-3 w-3 mr-1" />
          Approved
        </Badge>
      );
    case 'rejected':
      return (
        <Badge variant="destructive" className="bg-danger/20 text-danger border-danger/30">
          <X className="h-3 w-3 mr-1" />
          Rejected
        </Badge>
      );
    default:
      return (
        <Badge variant="outline" className="bg-chart-4/20 text-chart-4 border-chart-4/30">
          <Clock className="h-3 w-3 mr-1" />
          Pending
        </Badge>
      );
  }
}

function getFetchStatusBadge(fetchStatus: string | null) {
  switch (fetchStatus) {
    case 'fetched':
      return (
        <Badge variant="outline" className="bg-success/10 text-success border-success/30">
          <CheckCircle2 className="h-3 w-3 mr-1" />
          Ready
        </Badge>
      );
    case 'rate_limited':
      return (
        <Badge variant="outline" className="bg-chart-4/10 text-chart-4 border-chart-4/30">
          <AlertTriangle className="h-3 w-3 mr-1" />
          Rate Limited
        </Badge>
      );
    case 'error':
      return (
        <Badge variant="outline" className="bg-danger/10 text-danger border-danger/30">
          <XCircle className="h-3 w-3 mr-1" />
          Error
        </Badge>
      );
    case 'invalid':
      return (
        <Badge variant="outline" className="bg-danger/10 text-danger border-danger/30">
          <XCircle className="h-3 w-3 mr-1" />
          Invalid
        </Badge>
      );
    case 'pending':
    default:
      return (
        <Badge variant="outline" className="bg-muted text-muted-foreground">
          <Clock className="h-3 w-3 mr-1" />
          Pending
        </Badge>
      );
  }
}

export function SuggestionManager() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<SuggestionStatus | 'all'>('pending');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  
  // Action states
  const [actionLoading, setActionLoading] = useState<number | null>(null);
  const [rejectDialogOpen, setRejectDialogOpen] = useState(false);
  const [rejectingId, setRejectingId] = useState<number | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  
  // Edit dialog states
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingSymbol, setEditingSymbol] = useState('');
  const [newSymbol, setNewSymbol] = useState('');

  // Auto-approve settings
  const [autoApproveVotes, setAutoApproveVotes] = useState(10);
  const [savedAutoApproveVotes, setSavedAutoApproveVotes] = useState(10);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pageSize = 15;

  const loadSuggestions = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const status = statusFilter === 'all' ? undefined : statusFilter;
      const response = await getAllSuggestions(status, page, pageSize);
      setSuggestions(response.items);
      setTotal(response.total);
      setTotalPages(Math.ceil(response.total / pageSize));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load suggestions');
    } finally {
      setIsLoading(false);
    }
  }, [statusFilter, page]);

  // Load runtime settings on mount
  useEffect(() => {
    async function loadSettings() {
      try {
        const settings = await getRuntimeSettings();
        setAutoApproveVotes(settings.auto_approve_votes);
        setSavedAutoApproveVotes(settings.auto_approve_votes);
      } catch (err) {
        console.error('Failed to load settings:', err);
      }
    }
    loadSettings();
  }, []);

  // Auto-save when slider changes (debounced)
  const handleAutoApproveChange = (value: number) => {
    setAutoApproveVotes(value);
    
    // Clear existing timeout
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    
    // Debounce save by 800ms
    saveTimeoutRef.current = setTimeout(async () => {
      if (value !== savedAutoApproveVotes) {
        setIsSavingSettings(true);
        try {
          await updateRuntimeSettings({ auto_approve_votes: value });
          setSavedAutoApproveVotes(value);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to save settings');
          // Revert on error
          setAutoApproveVotes(savedAutoApproveVotes);
        } finally {
          setIsSavingSettings(false);
        }
      }
    }, 800);
  };

  useEffect(() => {
    loadSuggestions();
  }, [loadSuggestions]);

  // Reset page when filter changes
  useEffect(() => {
    setPage(1);
  }, [statusFilter]);

  async function handleApprove(id: number) {
    setActionLoading(id);
    try {
      await approveSuggestion(id);
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve suggestion');
    } finally {
      setActionLoading(null);
    }
  }

  function openRejectDialog(id: number) {
    setRejectingId(id);
    setRejectReason('');
    setRejectDialogOpen(true);
  }

  async function handleReject() {
    if (!rejectingId) return;
    
    setActionLoading(rejectingId);
    setRejectDialogOpen(false);
    try {
      await rejectSuggestion(rejectingId, rejectReason);
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject suggestion');
    } finally {
      setActionLoading(null);
      setRejectingId(null);
      setRejectReason('');
    }
  }

  function openEditDialog(id: number, symbol: string) {
    setEditingId(id);
    setEditingSymbol(symbol);
    setNewSymbol(symbol);
    setEditDialogOpen(true);
  }

  async function handleEdit() {
    if (!editingId || !newSymbol.trim()) return;
    
    setActionLoading(editingId);
    setEditDialogOpen(false);
    try {
      await updateSuggestion(editingId, newSymbol.trim());
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update suggestion');
    } finally {
      setActionLoading(null);
      setEditingId(null);
      setEditingSymbol('');
      setNewSymbol('');
    }
  }

  async function handleRefreshData(id: number) {
    setActionLoading(id);
    try {
      await refreshSuggestionData(id);
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setActionLoading(null);
    }
  }

  async function handleRetryFetch(id: number) {
    setActionLoading(id);
    try {
      await retrySuggestionFetch(id);
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry fetch');
    } finally {
      setActionLoading(null);
    }
  }

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleDateString();
  }

  return (
    <TooltipProvider>
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5" />
              Stock Suggestions
            </CardTitle>
            <CardDescription>
              Review and manage community-suggested stocks
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Select 
              value={statusFilter} 
              onValueChange={(v) => setStatusFilter(v as SuggestionStatus | 'all')}
            >
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="approved">Approved</SelectItem>
                <SelectItem value="rejected">Rejected</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="icon" onClick={loadSuggestions} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Auto-Approve Settings */}
        <div className="mb-6 p-4 rounded-lg border bg-muted/30">
          <div className="flex items-center gap-2 mb-3">
            <Settings className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Auto-Approve Settings</span>
            {isSavingSettings && (
              <Loader2 className="h-3 w-3 animate-spin text-muted-foreground ml-auto" />
            )}
            {!isSavingSettings && autoApproveVotes !== savedAutoApproveVotes && (
              <span className="text-xs text-muted-foreground ml-auto">Saving...</span>
            )}
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm text-muted-foreground">Votes to auto-approve</Label>
              <Badge variant="secondary">{autoApproveVotes} votes</Badge>
            </div>
            <Slider
              value={[autoApproveVotes]}
              onValueChange={([v]) => handleAutoApproveChange(v)}
              min={3}
              max={50}
              step={1}
            />
            <p className="text-xs text-muted-foreground">
              Suggestions with this many votes will be automatically approved for tracking
            </p>
          </div>
        </div>
        
        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2 bg-danger/10 text-danger p-3 rounded-lg mb-4"
            >
              <AlertCircle className="h-4 w-4" />
              <span className="flex-1 text-sm">{error}</span>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setError(null)}>
                <X className="h-4 w-4" />
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Stats */}
        <div className="flex items-center gap-4 mb-4 text-sm text-muted-foreground">
          <span>Total: <strong className="text-foreground">{total}</strong></span>
          {statusFilter === 'pending' && (
            <Badge variant="outline">
              {total} awaiting review
            </Badge>
          )}
        </div>

        {/* Table */}
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : suggestions.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Lightbulb className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p>No suggestions found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Sector</TableHead>
                  <TableHead className="text-center">Votes</TableHead>
                  <TableHead>Data</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Submitted</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <AnimatePresence mode="popLayout">
                  {suggestions.map((suggestion) => (
                    <motion.tr
                      key={suggestion.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="border-b"
                    >
                      <TableCell className="font-semibold">{suggestion.symbol}</TableCell>
                      <TableCell className="text-muted-foreground max-w-[200px] truncate" title={suggestion.name || undefined}>
                        {suggestion.name || '—'}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {suggestion.sector || '—'}
                      </TableCell>
                      <TableCell className="text-center">
                        <div className="flex items-center justify-center gap-1">
                          <ThumbsUp className="h-3 w-3 text-chart-4" />
                          <span className="font-medium">{suggestion.vote_count}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {suggestion.fetch_error ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                {getFetchStatusBadge(suggestion.fetch_status)}
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{suggestion.fetch_error}</p>
                              </TooltipContent>
                            </Tooltip>
                          ) : (
                            getFetchStatusBadge(suggestion.fetch_status)
                          )}
                          {(suggestion.fetch_status === 'rate_limited' || suggestion.fetch_status === 'error' || suggestion.fetch_status === 'pending') && (
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={() => handleRetryFetch(suggestion.id)}
                              disabled={actionLoading === suggestion.id}
                              title="Retry fetching stock data"
                            >
                              {actionLoading === suggestion.id ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <RefreshCw className="h-3 w-3" />
                              )}
                            </Button>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>{getStatusBadge(suggestion.status)}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDate(suggestion.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        {suggestion.status === 'pending' ? (
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-8"
                              onClick={() => openEditDialog(suggestion.id, suggestion.symbol)}
                              disabled={actionLoading === suggestion.id}
                              title="Edit symbol"
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-8 text-success hover:bg-success/10"
                              onClick={() => handleApprove(suggestion.id)}
                              disabled={actionLoading === suggestion.id}
                            >
                              {actionLoading === suggestion.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <>
                                  <Check className="h-4 w-4 mr-1" />
                                  Approve
                                </>
                              )}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-8 text-danger hover:bg-danger/10"
                              onClick={() => openRejectDialog(suggestion.id)}
                              disabled={actionLoading === suggestion.id}
                            >
                              <X className="h-4 w-4 mr-1" />
                              Reject
                            </Button>
                          </div>
                        ) : suggestion.status === 'approved' ? (
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-8"
                              onClick={() => handleRefreshData(suggestion.id)}
                              disabled={actionLoading === suggestion.id}
                              title="Fetch latest data and regenerate AI content"
                            >
                              {actionLoading === suggestion.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <>
                                  <RefreshCw className="h-4 w-4 mr-1" />
                                  Refresh Data
                                </>
                              )}
                            </Button>
                          </div>
                        ) : (
                          <span className="text-sm text-muted-foreground">
                            {suggestion.approved_by ? `by user #${suggestion.approved_by}` : '—'}
                          </span>
                        )}
                      </TableCell>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </TableBody>
            </Table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <span className="text-sm text-muted-foreground">
              Page {page} of {totalPages}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>

      {/* Reject Dialog */}
      <Dialog open={rejectDialogOpen} onOpenChange={setRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Suggestion</DialogTitle>
            <DialogDescription>
              Optionally provide a reason for rejecting this suggestion.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label htmlFor="reason" className="sr-only">Reason</Label>
            <Textarea
              id="reason"
              placeholder="Reason for rejection (optional)"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              rows={3}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleReject}>
              Reject
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Symbol Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Symbol</DialogTitle>
            <DialogDescription>
              Change the stock symbol for this suggestion. Useful for German stocks (.F vs .DE) or typos.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 space-y-4">
            <div>
              <Label htmlFor="current-symbol" className="text-sm text-muted-foreground">
                Current Symbol
              </Label>
              <p className="font-semibold">{editingSymbol}</p>
            </div>
            <div>
              <Label htmlFor="new-symbol">New Symbol</Label>
              <Input
                id="new-symbol"
                placeholder="Enter new symbol"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                className="mt-1.5 font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Examples: AAPL, BMW.DE, VOW3.DE
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleEdit}
              disabled={!newSymbol.trim() || newSymbol === editingSymbol}
            >
              <Pencil className="h-4 w-4 mr-1" />
              Update Symbol
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
    </TooltipProvider>
  );
}
