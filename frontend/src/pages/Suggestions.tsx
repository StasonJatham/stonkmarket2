import { useState, useEffect, useCallback } from 'react';
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { 
  Lightbulb, 
  ThumbsUp, 
  Loader2, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  X
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
import { 
  suggestStock, 
  voteForSuggestion, 
  getTopSuggestions,
  getAllSuggestions,
  approveSuggestion,
  rejectSuggestion,
  validateSymbol,
  type TopSuggestion,
  type Suggestion,
  type SuggestionStatus
} from '@/services/api';

export function SuggestionsPage() {
  const { isAdmin } = useAuth();

  // SEO for suggestions page
  useSEO({
    title: 'Stock Suggestions - Vote for New Stocks',
    description: 'Suggest new stocks to track on StonkMarket or vote for existing suggestions. Community-driven stock discovery for dip tracking.',
    keywords: 'stock suggestions, community voting, stock discovery, dip tracking, suggest stocks',
    canonical: '/suggest',
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'Suggestions', url: '/suggest' },
    ]),
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold tracking-tight">Stock Suggestions</h1>
        <p className="text-muted-foreground">
          Suggest new stocks to track or vote for existing suggestions
        </p>
      </motion.div>

      {isAdmin ? (
        <AdminSuggestionsView />
      ) : (
        <GuestSuggestionsView />
      )}
    </div>
  );
}

// Guest view - suggest and vote
function GuestSuggestionsView() {
  const [topSuggestions, setTopSuggestions] = useState<TopSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasLoaded, setHasLoaded] = useState(false);
  
  // Suggest form
  const [symbol, setSymbol] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<{ valid: boolean; name?: string; error?: string } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<{ success: boolean; message: string } | null>(null);
  
  // Voting
  const [votingSymbol, setVotingSymbol] = useState<string | null>(null);
  const [voteSuccess, setVoteSuccess] = useState<string | null>(null);

  const loadSuggestions = useCallback(async () => {
    if (!hasLoaded) setIsLoading(true);
    try {
      const data = await getTopSuggestions(20);
      setTopSuggestions(data);
      setHasLoaded(true);
    } catch (err) {
      console.error('Failed to load suggestions:', err);
    } finally {
      setIsLoading(false);
    }
  }, [hasLoaded]);

  useEffect(() => {
    loadSuggestions();
  }, [loadSuggestions]);

  async function handleValidate() {
    if (!symbol.trim()) return;
    setIsValidating(true);
    setValidationResult(null);
    try {
      const result = await validateSymbol(symbol.trim());
      setValidationResult(result);
    } catch {
      setValidationResult({ valid: false, error: 'Validation failed' });
    } finally {
      setIsValidating(false);
    }
  }

  async function handleSubmit() {
    if (!symbol.trim()) return;
    
    setIsSubmitting(true);
    setSubmitResult(null);
    try {
      const result = await suggestStock(symbol.trim().toUpperCase());
      setSubmitResult({ success: true, message: result.message });
      setSymbol('');
      setValidationResult(null);
      await loadSuggestions();
    } catch (err) {
      setSubmitResult({ 
        success: false, 
        message: err instanceof Error ? err.message : 'Failed to submit suggestion' 
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleVote(sym: string) {
    setVotingSymbol(sym);
    try {
      await voteForSuggestion(sym);
      setVoteSuccess(sym);
      await loadSuggestions();
      setTimeout(() => setVoteSuccess(null), 2000);
    } catch (err) {
      console.error('Vote failed:', err);
    } finally {
      setVotingSymbol(null);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Suggest Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5" />
              Suggest a Stock
            </CardTitle>
            <CardDescription>
              Request a new stock to be tracked. Use Yahoo Finance symbol format.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Stock Symbol</Label>
              <div className="flex gap-2">
                <Input
                  value={symbol}
                  onChange={(e) => {
                    setSymbol(e.target.value.toUpperCase());
                    setValidationResult(null);
                    setSubmitResult(null);
                  }}
                  placeholder="AAPL, MSFT, BTC-USD..."
                  className="font-mono h-9"
                />
                <Button
                  variant="outline"
                  onClick={handleValidate}
                  disabled={isValidating || !symbol.trim()}
                  className="h-9"
                >
                  {isValidating ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Check'}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Examples: AAPL (US), 7203.T (Japan), BTC-USD (Crypto), ^GSPC (Index)
              </p>
            </div>

            <AnimatePresence>
              {validationResult && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className={`flex items-center gap-2 p-3 rounded-lg ${validationResult.valid ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}
                >
                  {validationResult.valid ? (
                    <>
                      <CheckCircle className="h-4 w-4" />
                      <span>Valid: {validationResult.name}</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-4 w-4" />
                      <span>{validationResult.error}</span>
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {submitResult && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className={`flex items-center gap-2 p-3 rounded-lg ${submitResult.success ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}
                >
                  {submitResult.success ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
                  <span>{submitResult.message}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <Button
              onClick={handleSubmit}
              disabled={isSubmitting || !symbol.trim()}
              className="w-full"
            >
              {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              <Lightbulb className="h-4 w-4 mr-2" />
              Submit Suggestion
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Top Suggestions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Top Requested
            </CardTitle>
            <CardDescription>
              Vote for stocks you want to see tracked
            </CardDescription>
          </CardHeader>
          <CardContent className="min-h-[200px]">
            {isLoading && !hasLoaded ? (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : topSuggestions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Lightbulb className="h-12 w-12 mx-auto mb-2 opacity-20" />
                <p>No suggestions yet. Be the first!</p>
              </div>
            ) : (
              <ScrollArea className="h-[400px]">
                <div className="space-y-2">
                  {topSuggestions.map((suggestion, index) => (
                    <motion.div
                      key={suggestion.symbol}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.03 }}
                      className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <Badge variant="outline" className="font-mono">
                          {suggestion.symbol}
                        </Badge>
                        <div>
                          <div className="font-medium text-sm">{suggestion.name || suggestion.symbol}</div>
                          {suggestion.sector && (
                            <div className="text-xs text-muted-foreground">{suggestion.sector}</div>
                          )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">
                        {suggestion.vote_count} votes
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleVote(suggestion.symbol)}
                        disabled={votingSymbol === suggestion.symbol}
                      >
                        {votingSymbol === suggestion.symbol ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : voteSuccess === suggestion.symbol ? (
                          <CheckCircle className="h-4 w-4 text-success" />
                        ) : (
                          <ThumbsUp className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
      </motion.div>
    </div>
  );
}

// Admin view - review and manage suggestions
function AdminSuggestionsView() {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<SuggestionStatus>('pending');
  const [page, setPage] = useState(1);
  const pageSize = 20;
  
  // Reject dialog
  const [rejectDialogOpen, setRejectDialogOpen] = useState(false);
  const [rejectingId, setRejectingId] = useState<number | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  const [isRejecting, setIsRejecting] = useState(false);
  
  // Approve
  const [approvingId, setApprovingId] = useState<number | null>(null);
  
  const [error, setError] = useState<string | null>(null);

  const loadSuggestions = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getAllSuggestions(activeTab, page, pageSize);
      setSuggestions(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load suggestions');
    } finally {
      setIsLoading(false);
    }
  }, [activeTab, page]);

  useEffect(() => {
    loadSuggestions();
  }, [loadSuggestions]);

  async function handleApprove(id: number) {
    setApprovingId(id);
    try {
      await approveSuggestion(id);
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve');
    } finally {
      setApprovingId(null);
    }
  }

  async function handleReject() {
    if (!rejectingId || !rejectReason.trim()) return;
    
    setIsRejecting(true);
    try {
      await rejectSuggestion(rejectingId, rejectReason);
      setRejectDialogOpen(false);
      setRejectingId(null);
      setRejectReason('');
      await loadSuggestions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject');
    } finally {
      setIsRejecting(false);
    }
  }

  function getStatusBadge(status: SuggestionStatus) {
    switch (status) {
      case 'approved':
        return <Badge className="bg-success/20 text-success"><CheckCircle className="h-3 w-3 mr-1" />Approved</Badge>;
      case 'rejected':
        return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Rejected</Badge>;
      default:
        return <Badge variant="secondary"><Clock className="h-3 w-3 mr-1" />Pending</Badge>;
    }
  }

  const totalPages = Math.ceil(total / pageSize);

  return (
    <>
      {/* Error Alert */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-2 bg-danger/10 text-danger p-4 rounded-xl border border-danger/20"
          >
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <Button 
              variant="ghost" 
              size="icon"
              className="h-6 w-6"
              onClick={() => setError(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </motion.div>
        )}
      </AnimatePresence>

      <Tabs value={activeTab} onValueChange={(v) => { setActiveTab(v as SuggestionStatus); setPage(1); }}>
        <TabsList className="grid w-full max-w-md grid-cols-3">
          <TabsTrigger value="pending">
            <Clock className="h-4 w-4 mr-2" />
            Pending
          </TabsTrigger>
          <TabsTrigger value="approved">
            <CheckCircle className="h-4 w-4 mr-2" />
            Approved
          </TabsTrigger>
          <TabsTrigger value="rejected">
            <XCircle className="h-4 w-4 mr-2" />
            Rejected
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>
                  {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Suggestions
                </CardTitle>
                <CardDescription>
                  {total} total suggestions
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-2">
                    {[...Array(5)].map((_, i) => (
                      <Skeleton key={i} className="h-16 w-full" />
                    ))}
                  </div>
                ) : suggestions.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No {activeTab} suggestions
                  </div>
                ) : (
                  <>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Name</TableHead>
                        <TableHead>Votes</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Date</TableHead>
                        {activeTab === 'pending' && <TableHead>Actions</TableHead>}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {suggestions.map((s) => (
                        <TableRow key={s.id}>
                          <TableCell>
                            <Badge variant="outline" className="font-mono">
                              {s.symbol}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div>
                              <div className="font-medium">{s.name || '—'}</div>
                              {s.sector && (
                                <div className="text-xs text-muted-foreground">{s.sector}</div>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>{s.vote_count}</TableCell>
                          <TableCell>{getStatusBadge(s.status)}</TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {new Date(s.created_at).toLocaleDateString()}
                          </TableCell>
                          {activeTab === 'pending' && (
                            <TableCell>
                              <div className="flex gap-1">
                                <Button
                                  size="sm"
                                  onClick={() => handleApprove(s.id)}
                                  disabled={approvingId === s.id}
                                >
                                  {approvingId === s.id ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <CheckCircle className="h-4 w-4" />
                                  )}
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={() => {
                                    setRejectingId(s.id);
                                    setRejectDialogOpen(true);
                                  }}
                                >
                                  <XCircle className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between pt-4 border-t mt-4">
                      <p className="text-sm text-muted-foreground">
                        Page {page} of {totalPages}
                      </p>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setPage(p => p - 1)}
                          disabled={page === 1}
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setPage(p => p + 1)}
                          disabled={page >= totalPages}
                        >
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
          </motion.div>
        </TabsContent>
      </Tabs>

      {/* Reject Dialog */}
      <Dialog open={rejectDialogOpen} onOpenChange={setRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Suggestion</DialogTitle>
            <DialogDescription>
              Provide a reason for rejecting this suggestion.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label>Reason</Label>
            <Textarea
              value={rejectReason}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setRejectReason(e.target.value)}
              placeholder="e.g., Symbol not available, duplicate, etc."
              className="mt-2"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={!rejectReason.trim() || isRejecting}
            >
              {isRejecting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Reject
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
