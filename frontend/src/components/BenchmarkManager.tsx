import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { BenchmarkConfig } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { 
  Plus, 
  Trash2, 
  Edit2, 
  LineChart,
  GripVertical
} from 'lucide-react';

interface BenchmarkFormProps {
  formId: string;
  setFormId: (v: string) => void;
  formSymbol: string;
  setFormSymbol: (v: string) => void;
  formName: string;
  setFormName: (v: string) => void;
  formDescription: string;
  setFormDescription: (v: string) => void;
  onSubmit: () => void;
  submitLabel: string;
}

function BenchmarkForm({
  formId, setFormId,
  formSymbol, setFormSymbol,
  formName, setFormName,
  formDescription, setFormDescription,
  onSubmit, submitLabel
}: BenchmarkFormProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>ID</Label>
        <Input
          value={formId}
          onChange={(e) => setFormId(e.target.value.toUpperCase())}
          placeholder="SP500, NASDAQ, DAX..."
          className="font-mono"
        />
        <p className="text-xs text-muted-foreground">Unique identifier, no spaces</p>
      </div>
      
      <div className="space-y-2">
        <Label>Symbol</Label>
        <Input
          value={formSymbol}
          onChange={(e) => setFormSymbol(e.target.value.toUpperCase())}
          placeholder="^GSPC, ^IXIC, ^GDAXI..."
          className="font-mono"
        />
        <p className="text-xs text-muted-foreground">Yahoo Finance ticker symbol</p>
      </div>

      <div className="space-y-2">
        <Label>Display Name</Label>
        <Input
          value={formName}
          onChange={(e) => setFormName(e.target.value)}
          placeholder="S&P 500"
        />
      </div>

      <div className="space-y-2">
        <Label>Description (optional)</Label>
        <Input
          value={formDescription}
          onChange={(e) => setFormDescription(e.target.value)}
          placeholder="US Large Cap Index"
        />
      </div>

      <DialogFooter>
        <Button type="button" onClick={onSubmit} disabled={!formId.trim() || !formSymbol.trim() || !formName.trim()}>
          {submitLabel}
        </Button>
      </DialogFooter>
    </div>
  );
}

interface BenchmarkManagerProps {
  benchmarks: BenchmarkConfig[];
  onChange: (benchmarks: BenchmarkConfig[]) => void;
  onSave?: (benchmarks: BenchmarkConfig[]) => Promise<void>;
}

export function BenchmarkManager({ benchmarks, onChange, onSave }: BenchmarkManagerProps) {
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  
  // Form state
  const [formId, setFormId] = useState('');
  const [formSymbol, setFormSymbol] = useState('');
  const [formName, setFormName] = useState('');
  const [formDescription, setFormDescription] = useState('');

  function resetForm() {
    setFormId('');
    setFormSymbol('');
    setFormName('');
    setFormDescription('');
  }

  function openEdit(index: number) {
    const benchmark = benchmarks[index];
    setFormId(benchmark.id);
    setFormSymbol(benchmark.symbol);
    setFormName(benchmark.name);
    setFormDescription(benchmark.description || '');
    setEditingIndex(index);
  }

  async function handleAdd() {
    if (!formId.trim() || !formSymbol.trim() || !formName.trim()) return;
    
    const newBenchmark: BenchmarkConfig = {
      id: formId.trim().toUpperCase().replace(/\s+/g, '_'),
      symbol: formSymbol.trim().toUpperCase(),
      name: formName.trim(),
      description: formDescription.trim() || null,
    };

    // Check for duplicate ID
    if (benchmarks.some(b => b.id === newBenchmark.id)) {
      return; // Could add error handling
    }

    const newBenchmarks = [...benchmarks, newBenchmark];
    onChange(newBenchmarks);
    
    // Save immediately if onSave is provided
    if (onSave) {
      await onSave(newBenchmarks);
    }
    
    resetForm();
    setIsAddOpen(false);
  }

  async function handleEdit() {
    if (editingIndex === null) return;
    if (!formId.trim() || !formSymbol.trim() || !formName.trim()) return;

    const updated: BenchmarkConfig = {
      id: formId.trim().toUpperCase().replace(/\s+/g, '_'),
      symbol: formSymbol.trim().toUpperCase(),
      name: formName.trim(),
      description: formDescription.trim() || null,
    };

    const newBenchmarks = [...benchmarks];
    newBenchmarks[editingIndex] = updated;
    onChange(newBenchmarks);
    
    // Save immediately if onSave is provided
    if (onSave) {
      await onSave(newBenchmarks);
    }
    
    resetForm();
    setEditingIndex(null);
  }

  async function handleDelete(index: number) {
    const newBenchmarks = benchmarks.filter((_, i) => i !== index);
    onChange(newBenchmarks);
    
    // Save immediately if onSave is provided
    if (onSave) {
      await onSave(newBenchmarks);
    }
  }

  const formProps = {
    formId, setFormId,
    formSymbol, setFormSymbol,
    formName, setFormName,
    formDescription, setFormDescription,
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <LineChart className="h-5 w-5 text-primary" />
              Benchmark Indices
            </CardTitle>
            <CardDescription>
              Configure benchmark indices for stock comparison charts.
            </CardDescription>
          </div>
          
          <Dialog open={isAddOpen} onOpenChange={(open) => { setIsAddOpen(open); if (!open) resetForm(); }}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Add Benchmark
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Benchmark</DialogTitle>
                <DialogDescription>
                  Add a new benchmark index for comparison.
                </DialogDescription>
              </DialogHeader>
              <BenchmarkForm {...formProps} onSubmit={handleAdd} submitLabel="Add Benchmark" />
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        {benchmarks.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <LineChart className="h-10 w-10 mb-2 opacity-20" />
            <p>No benchmarks configured</p>
            <p className="text-sm">Add a benchmark to enable comparison charts</p>
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence>
              {benchmarks.map((benchmark, index) => (
                <motion.div
                  key={benchmark.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors"
                >
                  <GripVertical className="h-4 w-4 text-muted-foreground/50 shrink-0" />
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{benchmark.name}</span>
                      <Badge variant="outline" className="font-mono text-xs">
                        {benchmark.symbol}
                      </Badge>
                    </div>
                    {benchmark.description && (
                      <p className="text-sm text-muted-foreground truncate">
                        {benchmark.description}
                      </p>
                    )}
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <Dialog 
                      open={editingIndex === index} 
                      onOpenChange={(open) => { 
                        if (open) openEdit(index); 
                        else { setEditingIndex(null); resetForm(); }
                      }}
                    >
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <Edit2 className="h-4 w-4" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Edit Benchmark</DialogTitle>
                          <DialogDescription>
                            Update benchmark configuration.
                          </DialogDescription>
                        </DialogHeader>
                        <BenchmarkForm {...formProps} onSubmit={handleEdit} submitLabel="Save Changes" />
                      </DialogContent>
                    </Dialog>
                    
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-8 w-8 text-danger hover:text-danger hover:bg-danger/10"
                      onClick={() => handleDelete(index)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
