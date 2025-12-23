import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getDipCards,
  refreshAiAnalysis,
  refreshAiField,
  type DipCard,
  type AiFieldType,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
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
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  FileText,
} from 'lucide-react';

function getRatingBadge(rating: string | null) {
  if (!rating) {
    return (
      <Badge variant="outline" className="text-muted-foreground">
        <Minus className="h-3 w-3 mr-1" />
        No Rating
      </Badge>
    );
  }
  
  switch (rating) {
    case 'strong_buy':
      return (
        <Badge className="bg-success/20 text-success border-success/30">
          <TrendingUp className="h-3 w-3 mr-1" />
          Strong Buy
        </Badge>
      );
    case 'buy':
      return (
        <Badge className="bg-success/10 text-success border-success/20">
          <TrendingUp className="h-3 w-3 mr-1" />
          Buy
        </Badge>
      );
    case 'hold':
      return (
        <Badge variant="outline">
          <Minus className="h-3 w-3 mr-1" />
          Hold
        </Badge>
      );
    case 'sell':
      return (
        <Badge className="bg-danger/10 text-danger border-danger/20">
          <TrendingDown className="h-3 w-3 mr-1" />
          Sell
        </Badge>
      );
    case 'strong_sell':
      return (
        <Badge className="bg-danger/20 text-danger border-danger/30">
          <TrendingDown className="h-3 w-3 mr-1" />
          Strong Sell
        </Badge>
      );
    default:
      return (
        <Badge variant="outline">{rating}</Badge>
      );
  }
}

export function AIManager() {
  const [dipCards, setDipCards] = useState<DipCard[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regeneratingSymbol, setRegeneratingSymbol] = useState<string | null>(null);
  const [regeneratingField, setRegeneratingField] = useState<AiFieldType | null>(null);
  
  // Detail dialog
  const [selectedCard, setSelectedCard] = useState<DipCard | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);

  const loadCards = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getDipCards(true);
      setDipCards(response.cards);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dip cards');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCards();
  }, [loadCards]);

  async function handleRegenerateAI(symbol: string) {
    setRegeneratingSymbol(symbol);
    try {
      const updatedCard = await refreshAiAnalysis(symbol);
      // Update the card in the list
      setDipCards(prev => 
        prev.map(card => card.symbol === symbol ? updatedCard : card)
      );
      // If we were viewing this card's details, update it
      if (selectedCard?.symbol === symbol) {
        setSelectedCard(updatedCard);
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
      const updatedCard = await refreshAiField(symbol, field);
      // Update the card in the list
      setDipCards(prev => 
        prev.map(card => card.symbol === symbol ? updatedCard : card)
      );
      // If we were viewing this card's details, update it
      if (selectedCard?.symbol === symbol) {
        setSelectedCard(updatedCard);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to regenerate ${field} for ${symbol}`);
    } finally {
      setRegeneratingField(null);
    }
  }

  // Count cards with/without AI
  const cardsWithAI = dipCards.filter(c => c.swipe_bio).length;
  const cardsWithRating = dipCards.filter(c => c.ai_rating).length;

  return (
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
          <Button variant="outline" onClick={loadCards} disabled={isLoading}>
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
          <span className="text-muted-foreground">
            Total: <strong className="text-foreground">{dipCards.length}</strong> stocks
          </span>
          <Badge variant="outline" className="bg-chart-4/10">
            <MessageSquare className="h-3 w-3 mr-1" />
            {cardsWithAI} with bio
          </Badge>
          <Badge variant="outline" className="bg-success/10">
            <Sparkles className="h-3 w-3 mr-1" />
            {cardsWithRating} with rating
          </Badge>
        </div>

        {/* Table */}
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : dipCards.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Brain className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p>No stocks currently in dips</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Dip %</TableHead>
                  <TableHead>AI Rating</TableHead>
                  <TableHead>Swipe Bio</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <AnimatePresence mode="popLayout">
                  {dipCards.map((card) => (
                    <motion.tr
                      key={card.symbol}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="border-b"
                    >
                      <TableCell className="font-semibold">{card.symbol}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {card.name || '—'}
                      </TableCell>
                      <TableCell>
                        <Badge variant="destructive" className="font-mono">
                          -{card.dip_pct.toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {getRatingBadge(card.ai_rating)}
                      </TableCell>
                      <TableCell className="max-w-[200px]">
                        {card.swipe_bio ? (
                          <span className="text-sm text-muted-foreground line-clamp-1">
                            {card.swipe_bio.replace(/^"|"$/g, '')}
                          </span>
                        ) : (
                          <span className="text-sm text-muted-foreground italic">Not generated</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8"
                            onClick={() => openDetailDialog(card)}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            View
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8"
                            onClick={() => handleRegenerateAI(card.symbol)}
                            disabled={regeneratingSymbol === card.symbol}
                          >
                            {regeneratingSymbol === card.symbol ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <>
                                <Sparkles className="h-4 w-4 mr-1" />
                                Regenerate
                              </>
                            )}
                          </Button>
                        </div>
                      </TableCell>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </TableBody>
            </Table>
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
                {/* Rating Section */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-chart-4" />
                      AI Rating (Serious Analysis)
                    </h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRegenerateField(selectedCard.symbol, 'rating')}
                      disabled={regeneratingField === 'rating'}
                    >
                      {regeneratingField === 'rating' ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3 w-3" />
                      )}
                    </Button>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      {getRatingBadge(selectedCard.ai_rating)}
                      {selectedCard.ai_confidence && (
                        <Badge variant="outline">
                          Confidence: {selectedCard.ai_confidence}/10
                        </Badge>
                      )}
                    </div>
                    {selectedCard.ai_reasoning ? (
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {selectedCard.ai_reasoning}
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground italic">
                        No analysis available. Click refresh to generate.
                      </p>
                    )}
                  </div>
                </div>

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
                      disabled={regeneratingField === 'bio'}
                    >
                      {regeneratingField === 'bio' ? (
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
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRegenerateField(selectedCard.symbol, 'summary')}
                      disabled={regeneratingField === 'summary'}
                    >
                      {regeneratingField === 'summary' ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3 w-3" />
                      )}
                    </Button>
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
                    disabled={regeneratingSymbol === selectedCard.symbol || regeneratingField !== null}
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
  );
}
