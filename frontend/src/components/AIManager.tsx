import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getDipCardsPaged,
  getDipCard,
  refreshAiAnalysis,
  refreshAiField,
  regenerateSymbolAiSummary,
  getTaskStatuses,
  type DipCard,
  type AiFieldType,
  type TaskStatus,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  RefreshCw,
  Loader2,
  AlertCircle,
  X,
  Sparkles,
  Brain,
  MessageSquare,
  Eye,
  FileText,
} from 'lucide-react';
import { BatchJobsPanel } from '@/components/BatchJobsPanel';
import { DataTableControls } from '@/components/ui/data-table-controls';

export function AIManager() {
  const [dipCards, setDipCards] = useState<DipCard[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regeneratingSymbol, setRegeneratingSymbol] = useState<string | null>(null);
  const [regeneratingField, setRegeneratingField] = useState<AiFieldType | null>(null);
  const [summaryTasks, setSummaryTasks] = useState<Record<string, string>>({});
  
  // Pagination and search state
  const [searchValue, setSearchValue] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalCards, setTotalCards] = useState(0);
  
  // Detail dialog
  const [selectedCard, setSelectedCard] = useState<DipCard | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);

  const loadCards = useCallback(async (skipCache = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getDipCardsPaged({
        includeAi: true,
        limit: pageSize,
        offset: (currentPage - 1) * pageSize,
        search: debouncedSearch.trim() || undefined,
        skipCache,
      });
      setDipCards(response.cards);
      setTotalCards(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dip cards');
    } finally {
      setIsLoading(false);
    }
  }, [currentPage, debouncedSearch, pageSize]);

  useEffect(() => {
    loadCards();
  }, [loadCards]);

  useEffect(() => {
    const handle = setTimeout(() => {
      setDebouncedSearch(searchValue);
    }, 300);
    return () => clearTimeout(handle);
  }, [searchValue]);

  const pendingTasks = useMemo(() => {
    const tasks: { symbol: string; taskId: string; type: 'summary' | 'swipe' }[] = [];
    dipCards.forEach((card) => {
      if (card.ai_pending && card.ai_task_id) {
        tasks.push({ symbol: card.symbol, taskId: card.ai_task_id, type: 'swipe' });
      }
    });
    Object.entries(summaryTasks).forEach(([symbol, taskId]) => {
      if (taskId) {
        tasks.push({ symbol, taskId, type: 'summary' });
      }
    });
    return tasks;
  }, [dipCards, summaryTasks]);

  useEffect(() => {
    if (pendingTasks.length === 0) return;
    let cancelled = false;

    const pollTasks = async () => {
      let results: { task: (typeof pendingTasks)[number]; status: TaskStatus | null }[] = [];
      try {
        const statuses = await getTaskStatuses(pendingTasks.map((task) => task.taskId));
        const statusMap = new Map(statuses.map((status) => [status.task_id, status]));
        results = pendingTasks.map((task) => ({
          task,
          status: statusMap.get(task.taskId) || null,
        }));
      } catch {
        results = pendingTasks.map((task) => ({ task, status: null }));
      }

      if (cancelled) return;

      const summaryDone = new Set<string>();
      const swipeDone = new Set<string>();
      const failedSymbols = new Set<string>();

      results.forEach(({ task, status }) => {
        if (!status) return;
        const normalized = status.status.toUpperCase();
        if (normalized === 'SUCCESS' || normalized === 'FAILURE') {
          if (task.type === 'summary') {
            summaryDone.add(task.symbol);
          } else {
            swipeDone.add(task.symbol);
          }
          if (normalized === 'FAILURE') {
            failedSymbols.add(task.symbol);
          }
        }
      });

      const refreshSymbols = new Set<string>([...summaryDone, ...swipeDone]);
      if (refreshSymbols.size > 0) {
        const refreshedCards = await Promise.all(
          Array.from(refreshSymbols).map(async (symbol) => {
            try {
              return await getDipCard(symbol);
            } catch {
              return null;
            }
          })
        );

        if (!cancelled) {
          setDipCards(prev =>
            prev.map(card => {
              if (!refreshSymbols.has(card.symbol)) {
                return card;
              }
              const refreshed = refreshedCards.find(r => r?.symbol === card.symbol);
              if (refreshed) {
                return refreshed;
              }
              return { ...card, ai_pending: false, ai_task_id: null };
            })
          );
          if (selectedCard && refreshSymbols.has(selectedCard.symbol)) {
            const refreshed = refreshedCards.find(r => r?.symbol === selectedCard.symbol);
            if (refreshed) {
              setSelectedCard(refreshed);
            } else {
              setSelectedCard({
                ...selectedCard,
                ai_pending: false,
                ai_task_id: null,
              });
            }
          }
        }
      }

      if (summaryDone.size > 0) {
        setSummaryTasks(prev => {
          const next = { ...prev };
          summaryDone.forEach(symbol => {
            delete next[symbol];
          });
          return next;
        });
      }

      if (failedSymbols.size > 0) {
        setError(`AI task failed for ${Array.from(failedSymbols).join(', ')}`);
      }
    };

    pollTasks();
    const interval = setInterval(pollTasks, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [pendingTasks, selectedCard]);

  async function handleRegenerateAI(symbol: string) {
    setRegeneratingSymbol(symbol);
    try {
      const [swipeResult, summaryResult] = await Promise.allSettled([
        refreshAiAnalysis(symbol),
        regenerateSymbolAiSummary(symbol),
      ]);

      if (swipeResult.status === 'fulfilled') {
        const updatedCard = swipeResult.value;
        setDipCards(prev =>
          prev.map(card => card.symbol === symbol ? updatedCard : card)
        );
        if (selectedCard?.symbol === symbol) {
          setSelectedCard(updatedCard);
        }
      } else {
        setError(`Failed to regenerate AI analysis for ${symbol}`);
      }

      if (summaryResult.status === 'fulfilled') {
        const taskId = summaryResult.value.task_id;
        if (taskId) {
          setSummaryTasks(prev => ({ ...prev, [symbol]: taskId }));
        }
      } else {
        setError(`Failed to regenerate AI summary for ${symbol}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to regenerate AI for ${symbol}`);
    } finally {
      setRegeneratingSymbol(null);
    }
  }

  function openDetailDialog(card: DipCard) {
    setSelectedCard(card);
    setDetailDialogOpen(true);
  }

  async function handleRegenerateField(symbol: string, field: AiFieldType) {
    setRegeneratingField(field);
    try {
      if (field === 'summary') {
        const summaryResult = await regenerateSymbolAiSummary(symbol);
        const taskId = summaryResult.task_id;
        if (taskId) {
          setSummaryTasks(prev => ({ ...prev, [symbol]: taskId }));
        }
        return;
      }

      const updatedCard = await refreshAiField(symbol, field);
      setDipCards(prev =>
        prev.map(card => card.symbol === symbol ? updatedCard : card)
      );
      if (selectedCard?.symbol === symbol) {
        setSelectedCard(updatedCard);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to regenerate ${field} for ${symbol}`);
    } finally {
      setRegeneratingField(null);
    }
  }

  // Count cards with/without AI on current page
  const cardsWithAI = dipCards.filter(c => c.swipe_bio).length;
  const cardsWithRating = dipCards.filter(c => c.ai_rating).length;
  const summaryQueued = selectedCard ? !!summaryTasks[selectedCard.symbol] : false;
  const swipeQueued = Boolean(selectedCard?.ai_pending && selectedCard?.ai_task_id);

  const totalPages = Math.ceil(totalCards / pageSize);

  // Reset page when search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearch, pageSize]);

  useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(totalCards / pageSize));
    if (currentPage > maxPage) {
      setCurrentPage(1);
    }
  }, [currentPage, pageSize, totalCards]);

  return (
    <div className="space-y-6">
      {/* Batch Jobs Panel */}
      <BatchJobsPanel />
      
      {/* AI Content Manager */}
      <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Content Manager
            </CardTitle>
            <CardDescription>
              View and regenerate AI-generated content for stocks in dips
            </CardDescription>
          </div>
          <Button variant="outline" onClick={() => loadCards(true)} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
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
        <div className="flex items-center gap-4 mb-4 text-sm">
          <Badge variant="outline" className="bg-chart-4/10">
            <MessageSquare className="h-3 w-3 mr-1" />
            {cardsWithAI} with bio (page)
          </Badge>
          <Badge variant="outline" className="bg-success/10">
            <Sparkles className="h-3 w-3 mr-1" />
            {cardsWithRating} with rating (page)
          </Badge>
        </div>

        {/* Search and Pagination Controls */}
        <DataTableControls
          searchValue={searchValue}
          onSearchChange={setSearchValue}
          searchPlaceholder="Search symbols, names, ratings..."
          currentPage={currentPage}
          totalPages={totalPages}
          totalItems={totalCards}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
          itemName="stocks"
        />

        {/* Cards */}
        {isLoading ? (
          <div className="space-y-3 mt-4">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : dipCards.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Brain className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p>{searchValue ? 'No stocks match your search' : 'No stocks currently in dips'}</p>
          </div>
        ) : (
          <div className="space-y-4 mt-4">
            <AnimatePresence mode="popLayout">
              {dipCards.map((card) => {
                const queued = card.ai_pending || summaryTasks[card.symbol];
                return (
                  <motion.div
                    key={card.symbol}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="rounded-lg border bg-card p-4"
                  >
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm font-semibold">{card.symbol}</span>
                          <Badge variant="destructive" className="font-mono">
                            -{card.dip_pct.toFixed(1)}%
                          </Badge>
                          {card.ai_rating && (
                            <Badge variant="outline" className="uppercase text-xs">
                              {card.ai_rating.replace('_', ' ')}
                            </Badge>
                          )}
                          {queued && (
                            <Badge variant="outline" className="text-muted-foreground">
                              Queued
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {card.name || '—'} {card.sector ? `· ${card.sector}` : ''}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => openDetailDialog(card)}
                        >
                          <Eye className="h-4 w-4 mr-2" />
                          View
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleRegenerateAI(card.symbol)}
                          disabled={regeneratingSymbol === card.symbol || card.ai_pending || !!summaryTasks[card.symbol]}
                        >
                          {regeneratingSymbol === card.symbol ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <Sparkles className="h-4 w-4 mr-2" />
                              Regenerate
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                    <div className="mt-3 text-xs text-muted-foreground">
                      {card.swipe_bio ? (
                        <p className="line-clamp-2">
                          {card.swipe_bio.replace(/^"|"$/g, '')}
                        </p>
                      ) : (
                        <span className="italic">No swipe bio generated yet.</span>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </CardContent>

      {/* Detail Dialog */}
      <Dialog open={detailDialogOpen} onOpenChange={setDetailDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Content: {selectedCard?.symbol}
            </DialogTitle>
            <DialogDescription>
              {selectedCard?.name || selectedCard?.symbol} - Currently {selectedCard?.dip_pct.toFixed(1)}% below peak
            </DialogDescription>
          </DialogHeader>
          
          {selectedCard && (
            <ScrollArea className="max-h-[60vh]">
              <div className="space-y-6 py-4">
                {/* Swipe Bio Section */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-chart-4" />
                      Swipe Bio
                    </h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRegenerateField(selectedCard.symbol, 'bio')}
                      disabled={regeneratingField === 'bio' || swipeQueued}
                    >
                      {regeneratingField === 'bio' || swipeQueued ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3 w-3" />
                      )}
                    </Button>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-4">
                    {selectedCard.swipe_bio ? (
                      <p className="text-sm leading-relaxed">
                        {selectedCard.swipe_bio.replace(/^"|"$/g, '')}
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        No bio available. Click refresh to generate.
                      </p>
                    )}
                  </div>
                </div>

                {/* AI Summary Section */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <FileText className="h-4 w-4 text-chart-4" />
                      AI Summary
                    </h4>
                    <div className="flex items-center gap-2">
                      {summaryQueued && (
                        <Badge variant="outline" className="text-muted-foreground">
                          Queued
                        </Badge>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRegenerateField(selectedCard.symbol, 'summary')}
                        disabled={regeneratingField === 'summary' || summaryQueued}
                      >
                        {regeneratingField === 'summary' || summaryQueued ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <RefreshCw className="h-3 w-3" />
                        )}
                      </Button>
                    </div>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-4">
                    {selectedCard.summary_ai ? (
                      <p className="text-sm leading-relaxed">
                        {selectedCard.summary_ai}
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        No AI summary available. Click refresh to generate from company description.
                      </p>
                    )}
                  </div>
                </div>

                {/* Stock Info */}
                <div>
                  <h4 className="text-sm font-medium mb-2">Stock Info</h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-muted/30 rounded-lg p-3">
                      <span className="text-muted-foreground">Current Price</span>
                      <p className="font-semibold">${selectedCard.current_price.toFixed(2)}</p>
                    </div>
                    <div className="bg-muted/30 rounded-lg p-3">
                      <span className="text-muted-foreground">52W High</span>
                      <p className="font-semibold">${selectedCard.ref_high.toFixed(2)}</p>
                    </div>
                    <div className="bg-muted/30 rounded-lg p-3">
                      <span className="text-muted-foreground">Dip %</span>
                      <p className="font-semibold text-danger">-{selectedCard.dip_pct.toFixed(1)}%</p>
                    </div>
                    <div className="bg-muted/30 rounded-lg p-3">
                      <span className="text-muted-foreground">Sector</span>
                      <p className="font-semibold">{selectedCard.sector || '—'}</p>
                    </div>
                  </div>
                </div>

                {/* Regenerate All Button */}
                <div className="flex justify-end pt-4 border-t">
                  <Button
                    onClick={() => handleRegenerateAI(selectedCard.symbol)}
                    disabled={
                      regeneratingSymbol === selectedCard.symbol
                      || regeneratingField !== null
                      || swipeQueued
                      || summaryQueued
                    }
                  >
                    {regeneratingSymbol === selectedCard.symbol ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Regenerating All...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Regenerate All
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </ScrollArea>
          )}
        </DialogContent>
      </Dialog>
    </Card>
    </div>
  );
}
