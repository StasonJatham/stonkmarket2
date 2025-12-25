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
  Globe,
} from 'lucide-react';
import { 
  searchSymbols, 
  searchStoredSuggestions, 
  validateSymbol, 
  suggestStock, 
  type SymbolSearchResult,
  type StoredSearchResult,
} from '@/services/api';
import { useTheme } from '@/context/ThemeContext';
import { StockLogo } from '@/components/StockLogo';

// Combined result type for unified display
interface SearchResult {
  symbol: string;
  name: string | null;
  sector?: string | null;
  source: 'tracked' | 'suggestion' | 'yfinance';
  quote_type?: string | null;
  vote_count?: number | null;
}

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
  const [storedResults, setStoredResults] = useState<StoredSearchResult[]>([]);
  const [yfinanceResults, setYfinanceResults] = useState<SymbolSearchResult[]>([]);
  const [isSearchingStored, setIsSearchingStored] = useState(false);
  const [isSearchingYfinance, setIsSearchingYfinance] = useState(false);
  const [hasSearchedYfinance, setHasSearchedYfinance] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [stockPreview, setStockPreview] = useState<StockPreview | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const colors = getActiveColors();

  // Debounced search for stored/cached results only
  const handleStoredSearch = useCallback(async (query: string) => {
    if (query.length < 1) {
      setStoredResults([]);
      return;
    }

    setIsSearchingStored(true);
    try {
      const results = await searchStoredSuggestions(query, 8);
      setStoredResults(results);
    } catch {
      setStoredResults([]);
    } finally {
      setIsSearchingStored(false);
    }
  }, []);

  // Manual yfinance search (triggered by button)
  const handleYfinanceSearch = useCallback(async () => {
    if (searchQuery.length < 2) return;
    
    setIsSearchingYfinance(true);
    setHasSearchedYfinance(true);
    try {
      const results = await searchSymbols(searchQuery, 8);
      setYfinanceResults(results);
    } catch {
      setYfinanceResults([]);
    } finally {
      setIsSearchingYfinance(false);
    }
  }, [searchQuery]);

  // Debounce stored search
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Reset yfinance results when query changes
    setYfinanceResults([]);
    setHasSearchedYfinance(false);

    if (searchQuery.length >= 1) {
      searchTimeoutRef.current = setTimeout(() => {
        handleStoredSearch(searchQuery);
      }, 150); // Fast debounce for cached results
    } else {
      setStoredResults([]);
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery, handleStoredSearch]);

  const resetState = () => {
    setSearchQuery('');
    setStoredResults([]);
    setYfinanceResults([]);
    setHasSearchedYfinance(false);
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

  // Handle selection from either stored or yfinance results
  const handleSelectStock = async (result: SearchResult) => {
    setSearchQuery('');
    setStoredResults([]);
    setYfinanceResults([]);
    setHasSearchedYfinance(false);
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

  // Combine and dedupe results for display
  const combinedResults: SearchResult[] = (() => {
    const seen = new Set<string>();
    const results: SearchResult[] = [];
    
    // Add stored results first (tracked and suggestions)
    for (const r of storedResults) {
      if (!seen.has(r.symbol)) {
        seen.add(r.symbol);
        results.push({
          symbol: r.symbol,
          name: r.name,
          sector: r.sector,
          source: r.source,
          vote_count: r.vote_count,
        });
      }
    }
    
    // Add yfinance results (excluding already shown)
    for (const r of yfinanceResults) {
      if (!seen.has(r.symbol)) {
        seen.add(r.symbol);
        results.push({
          symbol: r.symbol,
          name: r.name,
          sector: r.sector,
          source: 'yfinance',
          quote_type: r.quote_type,
        });
      }
    }
    
    return results;
  })();

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
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        ref={inputRef}
                        placeholder="Type company name or symbol..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && searchQuery.length >= 2) {
                            handleYfinanceSearch();
                          }
                        }}
                        className="pl-10"
                      />
                      {isSearchingStored && (
                        <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin" />
                      )}
                    </div>
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={handleYfinanceSearch}
                      disabled={searchQuery.length < 2 || isSearchingYfinance}
                      title="Search Yahoo Finance for more results"
                    >
                      {isSearchingYfinance ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Globe className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  
                  {/* Search Results */}
                  {combinedResults.length > 0 && (
                    <ScrollArea className="h-[250px] rounded-md border">
                      <div className="p-2 space-y-1">
                        {combinedResults.map((result) => (
                          <button
                            key={result.symbol}
                            onClick={() => handleSelectStock(result)}
                            className="w-full p-3 rounded-lg hover:bg-muted transition-colors flex items-center gap-3 text-left"
                          >
                            <StockLogo symbol={result.symbol} size="sm" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold">{result.symbol}</span>
                                {result.source === 'tracked' && (
                                  <Badge variant="default" className="text-[10px] px-1 py-0 bg-success/20 text-success">
                                    Tracked
                                  </Badge>
                                )}
                                {result.source === 'suggestion' && (
                                  <Badge variant="secondary" className="text-[10px] px-1 py-0">
                                    Suggested {result.vote_count ? `(${result.vote_count} votes)` : ''}
                                  </Badge>
                                )}
                                {result.source === 'yfinance' && result.quote_type && (
                                  <Badge variant="outline" className="text-[10px] px-1 py-0">
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
                  
                  {/* No stored results - prompt for yfinance search */}
                  {searchQuery.length >= 2 && storedResults.length === 0 && !isSearchingStored && !hasSearchedYfinance && (
                    <div className="text-center py-4 space-y-2">
                      <p className="text-sm text-muted-foreground">
                        Not in our database yet.
                      </p>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleYfinanceSearch}
                        disabled={isSearchingYfinance}
                        className="gap-2"
                      >
                        {isSearchingYfinance ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Globe className="h-4 w-4" />
                        )}
                        Search Yahoo Finance
                      </Button>
                    </div>
                  )}
                  
                  {/* No yfinance results either */}
                  {searchQuery.length >= 2 && combinedResults.length === 0 && hasSearchedYfinance && !isSearchingYfinance && (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No stocks found. Try a different search term.
                    </p>
                  )}
                  
                  {searchQuery.length < 1 && (
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
