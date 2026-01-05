/**
 * Bulk Import Modal - Import portfolio positions from screenshots
 * 
 * Features:
 * - Drag & drop image upload (PNG, JPEG, WebP, GIF)
 * - AI-powered position extraction
 * - Editable table for review
 * - Batch import with progress
 */

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileImage,
  Loader2,
  Check,
  X,
  AlertTriangle,
  Trash2,
  Edit2,
  Sparkles,
  CheckCircle,
  XCircle,
  AlertCircle,
  SkipForward,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
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
import { Checkbox } from '@/components/ui/checkbox';
import { cn } from '@/lib/utils';
import {
  extractPositionsFromImage,
  bulkImportPositions,
  type ExtractedPosition,
  type ImageExtractionResponse,
  type BulkImportResponse,
  type ExtractionConfidence,
} from '@/services/api';


// =============================================================================
// Types
// =============================================================================

type ImportState = 
  | 'idle'           // Initial state, showing dropzone
  | 'uploading'      // File is being uploaded
  | 'analyzing'      // AI is analyzing the image (backend optimizes first)
  | 'results'        // Showing extracted positions for review
  | 'importing'      // Importing positions to portfolio
  | 'complete';      // Import finished, showing summary

interface EditablePosition extends ExtractedPosition {
  id: string;  // Unique ID for React key
  isEditing: boolean;
  errors: Record<string, string>;
}


// =============================================================================
// Props
// =============================================================================

interface BulkImportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  portfolioId: number;
  onImportComplete: () => void;
}


// =============================================================================
// Helper Components
// =============================================================================

function ConfidenceBadge({ confidence }: { confidence: ExtractionConfidence }) {
  const variants = {
    high: { color: 'text-green-600', bg: 'bg-green-100', label: 'High' },
    medium: { color: 'text-yellow-600', bg: 'bg-yellow-100', label: 'Med' },
    low: { color: 'text-red-600', bg: 'bg-red-100', label: 'Low' },
  };
  
  const variant = variants[confidence];
  
  return (
    <span className={cn('text-xs px-1.5 py-0.5 rounded', variant.bg, variant.color)}>
      {variant.label}
    </span>
  );
}


function DropZone({
  onFileDrop,
  isLoading,
  loadingMessage,
}: {
  onFileDrop: (file: File) => void;
  isLoading: boolean;
  loadingMessage?: string;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('image/') || file.name.toLowerCase().endsWith('.heic') || file.name.toLowerCase().endsWith('.heif'))) {
      onFileDrop(file);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onFileDrop(file);
    }
  }

  return (
    <div
      className={cn(
        'relative border-2 border-dashed rounded-lg p-8 transition-colors cursor-pointer',
        isDragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50',
        isLoading && 'pointer-events-none opacity-50'
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp,image/gif,image/heic,image/heif,.heic,.heif"
        className="hidden"
        onChange={handleFileSelect}
        disabled={isLoading}
      />
      
      <div className="flex flex-col items-center gap-4 text-center">
        {isLoading ? (
          <>
            <Loader2 className="h-12 w-12 text-muted-foreground animate-spin" />
            <div>
              <p className="font-medium">{loadingMessage || 'Processing image...'}</p>
              <p className="text-sm text-muted-foreground">This may take a moment</p>
            </div>
          </>
        ) : (
          <>
            <div className="relative">
              <FileImage className="h-12 w-12 text-muted-foreground" />
              <Sparkles className="h-5 w-5 text-primary absolute -top-1 -right-1" />
            </div>
            <div>
              <p className="font-medium">Drop your portfolio screenshot here</p>
              <p className="text-sm text-muted-foreground">or click to browse</p>
            </div>
            <div className="flex gap-2 text-xs text-muted-foreground">
              <span>PNG</span>
              <span>•</span>
              <span>JPEG</span>
              <span>•</span>
              <span>HEIC</span>
              <span>•</span>
              <span>WebP</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}


function PositionRow({
  position,
  onUpdate,
  onRemove,
  onToggleSkip,
}: {
  position: EditablePosition;
  onUpdate: (id: string, updates: Partial<EditablePosition>) => void;
  onRemove: (id: string) => void;
  onToggleSkip: (id: string) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editSymbol, setEditSymbol] = useState(position.symbol || '');
  const [editQty, setEditQty] = useState(String(position.quantity || ''));
  const [editCost, setEditCost] = useState(String(position.avg_cost || ''));

  const handleSave = () => {
    const updates: Partial<EditablePosition> = {
      symbol: editSymbol.toUpperCase() || null,
      quantity: editQty ? parseFloat(editQty) : null,
      avg_cost: editCost ? parseFloat(editCost) : null,
      errors: {},
    };
    
    // Validate
    const errors: Record<string, string> = {};
    if (!updates.symbol) {
      errors.symbol = 'Required';
    }
    if (!updates.quantity || updates.quantity <= 0) {
      errors.quantity = 'Required';
    }
    
    if (Object.keys(errors).length > 0) {
      onUpdate(position.id, { errors });
      return;
    }
    
    onUpdate(position.id, updates);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditSymbol(position.symbol || '');
    setEditQty(String(position.quantity || ''));
    setEditCost(String(position.avg_cost || ''));
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <TableRow className="bg-muted/50">
        <TableCell className="w-8">
          <Checkbox checked={!position.skip} disabled />
        </TableCell>
        <TableCell>
          <Input
            value={editSymbol}
            onChange={(e) => setEditSymbol(e.target.value.toUpperCase())}
            placeholder="AAPL"
            className={cn('h-8 w-24', position.errors?.symbol && 'border-red-500')}
          />
        </TableCell>
        <TableCell>
          <span className="text-sm text-muted-foreground truncate max-w-[150px] block">
            {position.name || '-'}
          </span>
        </TableCell>
        <TableCell>
          <Input
            type="number"
            value={editQty}
            onChange={(e) => setEditQty(e.target.value)}
            placeholder="100"
            className={cn('h-8 w-20', position.errors?.quantity && 'border-red-500')}
          />
        </TableCell>
        <TableCell>
          <Input
            type="number"
            value={editCost}
            onChange={(e) => setEditCost(e.target.value)}
            placeholder="150.00"
            className="h-8 w-24"
          />
        </TableCell>
        <TableCell>
          <span className="text-sm">{position.currency}</span>
        </TableCell>
        <TableCell>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" className="h-7 px-2" onClick={handleSave}>
              <Check className="h-4 w-4" />
            </Button>
            <Button size="sm" variant="ghost" className="h-7 px-2" onClick={handleCancel}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </TableCell>
      </TableRow>
    );
  }

  return (
    <TableRow className={cn(position.skip && 'opacity-50')}>
      <TableCell className="w-8">
        <Checkbox
          checked={!position.skip}
          onCheckedChange={() => onToggleSkip(position.id)}
        />
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <span className={cn('font-mono font-medium', !position.symbol && 'text-red-500')}>
            {position.symbol || 'Missing'}
          </span>
          <ConfidenceBadge confidence={position.confidence} />
        </div>
        {position.notes && (
          <span className="text-xs text-yellow-600 block">{position.notes}</span>
        )}
      </TableCell>
      <TableCell>
        <span className="text-sm text-muted-foreground truncate max-w-[150px] block">
          {position.name || '-'}
        </span>
      </TableCell>
      <TableCell>
        <span className={cn(!position.quantity && 'text-red-500')}>
          {position.quantity ?? 'Missing'}
        </span>
      </TableCell>
      <TableCell>
        <span className="text-sm">
          {position.avg_cost?.toFixed(2) || '-'}
        </span>
      </TableCell>
      <TableCell>
        <span className="text-sm">{position.currency}</span>
      </TableCell>
      <TableCell>
        <div className="flex gap-1">
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => setIsEditing(true)}
            disabled={position.skip}
          >
            <Edit2 className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-destructive hover:text-destructive"
            onClick={() => onRemove(position.id)}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}


function ImportSummary({ response }: { response: BulkImportResponse }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4 text-center">
        <div className="p-4 rounded-lg bg-green-50 border border-green-200">
          <CheckCircle className="h-6 w-6 text-green-600 mx-auto mb-1" />
          <div className="text-2xl font-bold text-green-600">{response.created}</div>
          <div className="text-xs text-green-700">Created</div>
        </div>
        <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
          <Edit2 className="h-6 w-6 text-blue-600 mx-auto mb-1" />
          <div className="text-2xl font-bold text-blue-600">{response.updated}</div>
          <div className="text-xs text-blue-700">Updated</div>
        </div>
        <div className="p-4 rounded-lg bg-yellow-50 border border-yellow-200">
          <SkipForward className="h-6 w-6 text-yellow-600 mx-auto mb-1" />
          <div className="text-2xl font-bold text-yellow-600">{response.skipped}</div>
          <div className="text-xs text-yellow-700">Skipped</div>
        </div>
        <div className="p-4 rounded-lg bg-red-50 border border-red-200">
          <XCircle className="h-6 w-6 text-red-600 mx-auto mb-1" />
          <div className="text-2xl font-bold text-red-600">{response.failed}</div>
          <div className="text-xs text-red-700">Failed</div>
        </div>
      </div>
      
      {response.failed > 0 && (
        <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg">
          <strong>Errors:</strong>
          <ul className="mt-1 list-disc list-inside">
            {response.results
              .filter(r => r.status === 'failed')
              .map((r, i) => (
                <li key={i}>{r.symbol}: {r.message}</li>
              ))}
          </ul>
        </div>
      )}
    </div>
  );
}


// =============================================================================
// Main Component
// =============================================================================

export function BulkImportModal({
  open,
  onOpenChange,
  portfolioId,
  onImportComplete,
}: BulkImportModalProps) {
  const [state, setState] = useState<ImportState>('idle');
  const [positions, setPositions] = useState<EditablePosition[]>([]);
  const [extractionResult, setExtractionResult] = useState<ImageExtractionResponse | null>(null);
  const [importResult, setImportResult] = useState<BulkImportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Reset state when modal closes
  function handleOpenChange(newOpen: boolean) {
    if (!newOpen) {
      // Reset after animation
      setTimeout(() => {
        setState('idle');
        setPositions([]);
        setExtractionResult(null);
        setImportResult(null);
        setError(null);
      }, 200);
    }
    onOpenChange(newOpen);
  }

  // Handle file upload
  async function handleFileDrop(file: File) {
    setError(null);
    setState('analyzing');
    
    try {
      // Backend handles image optimization (HEIC conversion, resize, compression)
      const result = await extractPositionsFromImage(portfolioId, file);
      
      setExtractionResult(result);
      
      if (!result.success) {
        setError(result.error_message || 'Failed to extract positions');
        setState('idle');
        return;
      }
      
      // Convert to editable positions
      const editable: EditablePosition[] = result.positions.map((pos, idx) => ({
        ...pos,
        id: `pos-${idx}-${Date.now()}`,
        isEditing: false,
        errors: {},
      }));
      
      setPositions(editable);
      setState('results');
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setState('idle');
    }
  }

  // Update a position
  function handleUpdatePosition(id: string, updates: Partial<EditablePosition>) {
    setPositions(prev => prev.map(p => 
      p.id === id ? { ...p, ...updates } : p
    ));
  }

  // Remove a position
  function handleRemovePosition(id: string) {
    setPositions(prev => prev.filter(p => p.id !== id));
  }

  // Toggle skip
  function handleToggleSkip(id: string) {
    setPositions(prev => prev.map(p => 
      p.id === id ? { ...p, skip: !p.skip } : p
    ));
  }

  // Import positions
  async function handleImport() {
    // Filter valid, non-skipped positions
    const toImport = positions.filter(p => 
      !p.skip && p.symbol && p.quantity && p.quantity > 0
    );
    
    if (toImport.length === 0) {
      setError('No valid positions to import');
      return;
    }
    
    setState('importing');
    setError(null);
    
    try {
      const result = await bulkImportPositions(portfolioId, {
        positions: toImport.map(p => ({
          symbol: p.symbol!,
          quantity: p.quantity!,
          avg_cost: p.avg_cost || undefined,
          currency: p.currency,
        })),
        skip_duplicates: true,
      });
      
      setImportResult(result);
      setState('complete');
      
      if (result.created > 0 || result.updated > 0) {
        onImportComplete();
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Import failed');
      setState('results');
    }
  }

  // Count valid positions
  const validCount = positions.filter(p => 
    !p.skip && p.symbol && p.quantity && p.quantity > 0
  ).length;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Import Portfolio from Screenshot
          </DialogTitle>
          <DialogDescription>
            Upload a screenshot of your portfolio from any broker app.
            AI will extract your positions for review.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-auto py-4">
          <AnimatePresence mode="wait">
            {/* Idle / Upload State */}
            {(state === 'idle' || state === 'uploading' || state === 'analyzing') && (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <DropZone
                  onFileDrop={handleFileDrop}
                  isLoading={state === 'uploading' || state === 'analyzing'}
                  loadingMessage={
                    state === 'analyzing' ? 'AI is extracting positions...' :
                    'Uploading...'
                  }
                />
                
                {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm">{error}</span>
                  </div>
                )}
              </motion.div>
            )}

            {/* Results State */}
            {state === 'results' && (
              <motion.div
                key="results"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                {/* Extraction info */}
                {extractionResult && (
                  <div className="flex flex-wrap gap-2 text-sm">
                    {extractionResult.detected_broker && (
                      <Badge variant="secondary">
                        Broker: {extractionResult.detected_broker}
                      </Badge>
                    )}
                    {extractionResult.currency_hint && (
                      <Badge variant="secondary">
                        Currency: {extractionResult.currency_hint}
                      </Badge>
                    )}
                    {extractionResult.processing_time_ms && (
                      <Badge variant="outline">
                        Processed in {(extractionResult.processing_time_ms / 1000).toFixed(1)}s
                      </Badge>
                    )}
                  </div>
                )}

                {/* Warnings */}
                {extractionResult?.warnings && extractionResult.warnings.length > 0 && (
                  <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-center gap-2 text-yellow-700 mb-1">
                      <AlertCircle className="h-4 w-4" />
                      <span className="font-medium text-sm">Notes</span>
                    </div>
                    <ul className="text-xs text-yellow-600 list-disc list-inside">
                      {extractionResult.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Positions table */}
                {positions.length > 0 ? (
                  <div className="border rounded-lg overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-8"></TableHead>
                          <TableHead>Symbol</TableHead>
                          <TableHead>Name</TableHead>
                          <TableHead>Qty</TableHead>
                          <TableHead>Avg Cost</TableHead>
                          <TableHead>Currency</TableHead>
                          <TableHead className="w-20"></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {positions.map(pos => (
                          <PositionRow
                            key={pos.id}
                            position={pos}
                            onUpdate={handleUpdatePosition}
                            onRemove={handleRemovePosition}
                            onToggleSkip={handleToggleSkip}
                          />
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No positions extracted. Try a clearer image.
                  </div>
                )}

                {error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-sm">{error}</span>
                  </div>
                )}
              </motion.div>
            )}

            {/* Importing State */}
            {state === 'importing' && (
              <motion.div
                key="importing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-12"
              >
                <Loader2 className="h-12 w-12 text-primary animate-spin mb-4" />
                <p className="font-medium">Importing positions...</p>
                <p className="text-sm text-muted-foreground">This may take a moment</p>
              </motion.div>
            )}

            {/* Complete State */}
            {state === 'complete' && importResult && (
              <motion.div
                key="complete"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
              >
                <ImportSummary response={importResult} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <DialogFooter className="gap-2">
          {state === 'idle' && (
            <Button variant="outline" onClick={() => handleOpenChange(false)}>
              Cancel
            </Button>
          )}
          
          {state === 'results' && (
            <>
              <Button variant="outline" onClick={() => setState('idle')}>
                Upload Different Image
              </Button>
              <Button
                onClick={handleImport}
                disabled={validCount === 0}
              >
                Import {validCount} Position{validCount !== 1 ? 's' : ''}
              </Button>
            </>
          )}
          
          {state === 'complete' && (
            <Button onClick={() => handleOpenChange(false)}>
              Done
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
