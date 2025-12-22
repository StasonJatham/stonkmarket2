import { useState } from 'react';
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
import { validateSymbol, suggestStock } from '@/services/api';

interface StockPreview {
  symbol: string;
  name: string;
  sector?: string;
}

export function SuggestStockDialog() {
  const [open, setOpen] = useState(false);
  const [symbol, setSymbol] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [stockPreview, setStockPreview] = useState<StockPreview | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  // Rate limit state - can be populated from API response headers
  const [rateLimit] = useState<{ remaining: number; resetIn: number } | null>(null);

  const resetState = () => {
    setSymbol('');
    setStockPreview(null);
    setValidationError(null);
    setIsSubmitting(false);
    setSubmitSuccess(false);
    setSubmitMessage(null);
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen) {
      resetState();
    }
    setOpen(newOpen);
  };

  const handleValidate = async () => {
    if (!symbol.trim()) return;

    setIsValidating(true);
    setValidationError(null);
    setStockPreview(null);

    try {
      const result = await validateSymbol(symbol.trim().toUpperCase());
      if (result.valid && result.name) {
        setStockPreview({
          symbol: symbol.trim().toUpperCase(),
          name: result.name,
          sector: (result as { sector?: string }).sector,
        });
      } else {
        setValidationError(result.error || 'Symbol not found');
      }
    } catch (err) {
      setValidationError('Failed to validate symbol');
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
      
      // Auto-close after success animation
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && symbol.trim() && !isValidating && !stockPreview) {
      handleValidate();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Lightbulb className="h-4 w-4" />
          <span className="hidden sm:inline">Suggest</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-primary" />
            Suggest a Stock
          </DialogTitle>
          <DialogDescription>
            Enter a stock symbol to suggest it for tracking
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
                <CheckCircle className="h-16 w-16 mx-auto text-success" />
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
              {/* Symbol Input */}
              <div className="space-y-2">
                <Label htmlFor="symbol">Stock Symbol</Label>
                <div className="flex gap-2">
                  <Input
                    id="symbol"
                    placeholder="e.g., AAPL, MSFT, TSLA"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    onKeyDown={handleKeyDown}
                    disabled={isValidating || !!stockPreview}
                    className="flex-1 uppercase"
                    maxLength={10}
                  />
                  {!stockPreview ? (
                    <Button
                      onClick={handleValidate}
                      disabled={!symbol.trim() || isValidating}
                      variant="secondary"
                    >
                      {isValidating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Search className="h-4 w-4" />
                      )}
                    </Button>
                  ) : (
                    <Button
                      onClick={() => {
                        setStockPreview(null);
                        setSymbol('');
                      }}
                      variant="ghost"
                      size="sm"
                    >
                      Change
                    </Button>
                  )}
                </div>
              </div>

              {/* Validation Error */}
              <AnimatePresence>
                {validationError && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-2 text-sm text-destructive"
                  >
                    <AlertCircle className="h-4 w-4" />
                    {validationError}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Stock Preview Card */}
              <AnimatePresence>
                {stockPreview && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <Card className="bg-muted/50">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-lg">{stockPreview.symbol}</span>
                              <CheckCircle className="h-4 w-4 text-success" />
                            </div>
                            <p className="text-sm text-muted-foreground truncate">
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
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit Error */}
              <AnimatePresence>
                {submitMessage && !submitSuccess && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex items-center gap-2 text-sm text-destructive"
                  >
                    <AlertCircle className="h-4 w-4" />
                    {submitMessage}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Rate Limit Info */}
              {rateLimit && (
                <div className="text-xs text-muted-foreground text-center">
                  {rateLimit.remaining} suggestions remaining • Resets in {rateLimit.resetIn}s
                </div>
              )}

              <Separator />

              {/* Submit Button */}
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
            </motion.div>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}
