import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Lightbulb,
  Search,
  Loader2,
  CheckCircle,
  AlertCircle,
  Building2,
  Sparkles,
  Send,
} from 'lucide-react';
import { searchSymbols, validateSymbol, suggestStock, type SymbolSearchResult } from '@/services/api';
import { useTheme } from '@/context/ThemeContext';
import { StockLogo } from '@/components/StockLogo';

interface StockPreview {
  symbol: string;
  name: string;
  sector?: string;
  summary?: string;
}

interface SuggestStockDialogProps {
  variant?: 'default' | 'outline' | 'ghost' | 'secondary';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  showLabel?: boolean;
}

export function SuggestStockDialog({ 
  variant = 'outline', 
  size = 'sm', 
  className = '',
  showLabel = false,
}: SuggestStockDialogProps = {}) {
  const { getActiveColors } = useTheme();
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SymbolSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [stockPreview, setStockPreview] = useState<StockPreview | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const colors = getActiveColors();

  // Debounced search
  const handleSearch = useCallback(async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const results = await searchSymbols(query, 8);
      setSearchResults(results);
    } catch {
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (searchQuery.length >= 2) {
      searchTimeoutRef.current = setTimeout(() => {
        handleSearch(searchQuery);
      }, 300);
    } else {
      setSearchResults([]);
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery, handleSearch]);

  const resetState = () => {
    setSearchQuery('');
    setSearchResults([]);
    setStockPreview(null);
    setValidationError(null);
    setIsSubmitting(false);
    setSubmitSuccess(false);
    setSubmitMessage(null);
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen) {
      resetState();
      setTimeout(() => inputRef.current?.focus(), 100);
    }
    setOpen(newOpen);
  };

  const handleSelectStock = async (result: SymbolSearchResult) => {
    setSearchQuery('');
    setSearchResults([]);
    setIsValidating(true);
    setValidationError(null);

    try {
      const validation = await validateSymbol(result.symbol);
      if (validation.valid && validation.name) {
        setStockPreview({
          symbol: validation.symbol,
          name: validation.name,
          sector: validation.sector,
          summary: validation.summary,
        });
      } else {
        setStockPreview({
          symbol: result.symbol,
          name: result.name || result.symbol,
          sector: result.sector || undefined,
        });
      }
    } catch {
      setStockPreview({
        symbol: result.symbol,
        name: result.name || result.symbol,
        sector: result.sector || undefined,
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleSubmit = async () => {
    if (!stockPreview) return;

    setIsSubmitting(true);
    setSubmitMessage(null);

    try {
      const result = await suggestStock(stockPreview.symbol);
      setSubmitSuccess(true);
      setSubmitMessage(result.message);
      
      setTimeout(() => {
        setOpen(false);
        resetState();
      }, 2000);
    } catch (err) {
      setSubmitMessage(err instanceof Error ? err.message : 'Failed to submit suggestion');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant={variant} size={size} className={`gap-2 ${className}`}>
          <Lightbulb className="h-4 w-4" />
          <span className={showLabel ? '' : 'hidden sm:inline'}>Suggest</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-primary" />
            Suggest a Stock
          </DialogTitle>
          <DialogDescription>
            Search for a stock by name or symbol
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait">
          {submitSuccess ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="py-8 text-center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 20, delay: 0.1 }}
              >
                <CheckCircle className="h-16 w-16 mx-auto" style={{ color: colors.up }} />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <h3 className="text-lg font-semibold mt-4">{submitMessage}</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Thanks for your suggestion!
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="mt-4"
              >
                <Sparkles className="h-6 w-6 mx-auto text-primary animate-pulse" />
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              {!stockPreview ? (
                <div className="space-y-3">
                  <Label>Search for a Stock</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      ref={inputRef}
                      placeholder="Type company name or symbol..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                    {isSearching && (
                      <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin" />
                    )}
                  </div>
                  
                  {/* Search Results */}
                  {searchResults.length > 0 && (
                    <ScrollArea className="h-[250px] rounded-md border">
                      <div className="p-2 space-y-1">
                        {searchResults.map((result) => (
                          <button
                            key={result.symbol}
                            onClick={() => handleSelectStock(result)}
                            className="w-full p-3 rounded-lg hover:bg-muted transition-colors flex items-center gap-3 text-left"
                          >
                            <StockLogo symbol={result.symbol} size="sm" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold">{result.symbol}</span>
                                {result.quote_type && (
                                  <Badge variant="secondary" className="text-[10px] px-1 py-0">
                                    {result.quote_type}
                                  </Badge>
                                )}
                              </div>
                              <p className="text-sm text-muted-foreground truncate">
                                {result.name || 'Unknown'}
                              </p>
                            </div>
                            {result.sector && (
                              <Badge variant="outline" className="text-xs shrink-0">
                                {result.sector}
                              </Badge>
                            )}
                          </button>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                  
                  {searchQuery.length >= 2 && searchResults.length === 0 && !isSearching && (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No stocks found. Try a different search term.
                    </p>
                  )}
                  
                  {searchQuery.length < 2 && (
                    <p className="text-xs text-muted-foreground">
                      Search by company name (e.g., "Apple") or ticker symbol (e.g., "AAPL")
                    </p>
                  )}
                </div>
              ) : (
                <>
                  {isValidating ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    </div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <Card className="bg-muted/50">
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            <StockLogo symbol={stockPreview.symbol} size="lg" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-lg">{stockPreview.symbol}</span>
                                <CheckCircle className="h-4 w-4" style={{ color: colors.up }} />
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {stockPreview.name}
                              </p>
                              {stockPreview.sector && (
                                <div className="flex items-center gap-1 mt-2">
                                  <Building2 className="h-3 w-3 text-muted-foreground" />
                                  <Badge variant="secondary" className="text-xs">
                                    {stockPreview.sector}
                                  </Badge>
                                </div>
                              )}
                              {stockPreview.summary && (
                                <p className="text-xs text-muted-foreground mt-3 line-clamp-3">
                                  {stockPreview.summary}
                                </p>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                      <Button
                        onClick={() => setStockPreview(null)}
                        variant="ghost"
                        size="sm"
                        className="mt-2 w-full"
                      >
                        Choose a different stock
                      </Button>
                    </motion.div>
                  )}
                </>
              )}

              <AnimatePresence>
                {validationError && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-2 text-sm"
                    style={{ color: colors.down }}
                  >
                    <AlertCircle className="h-4 w-4" />
                    {validationError}
                  </motion.div>
                )}
              </AnimatePresence>

              <AnimatePresence>
                {submitMessage && !submitSuccess && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-2 text-sm"
                    style={{ color: colors.down }}
                  >
                    <AlertCircle className="h-4 w-4" />
                    {submitMessage}
                  </motion.div>
                )}
              </AnimatePresence>

              {stockPreview && !isValidating && (
                <>
                  <Separator />
                  <Button
                    onClick={handleSubmit}
                    disabled={!stockPreview || isSubmitting}
                    className="w-full gap-2"
                  >
                    {isSubmitting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                    {isSubmitting ? 'Submitting...' : 'Submit Suggestion'}
                  </Button>
                  <p className="text-xs text-muted-foreground text-center">
                    Your suggestion will be reviewed and may be added to tracking
                  </p>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}
