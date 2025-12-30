import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SectorETFConfig } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
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
import { 
  Plus, 
  Trash2, 
  Edit2, 
  Building2
} from 'lucide-react';
import { ConfigForm, type ConfigField } from '@/components/ConfigForm';

interface SectorETFManagerProps {
  sectorEtfs: SectorETFConfig[];
  onChange: (sectorEtfs: SectorETFConfig[]) => void;
  onSave?: (sectorEtfs: SectorETFConfig[]) => Promise<void>;
}

export function SectorETFManager({ sectorEtfs, onChange, onSave }: SectorETFManagerProps) {
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  
  // Form state
  const [formSector, setFormSector] = useState('');
  const [formSymbol, setFormSymbol] = useState('');
  const [formName, setFormName] = useState('');

  function resetForm() {
    setFormSector('');
    setFormSymbol('');
    setFormName('');
  }

  function openEdit(index: number) {
    const etf = sectorEtfs[index];
    setFormSector(etf.sector);
    setFormSymbol(etf.symbol);
    setFormName(etf.name);
    setEditingIndex(index);
  }

  async function handleAdd() {
    const newEtf: SectorETFConfig = {
      sector: formSector.trim(),
      symbol: formSymbol.trim().toUpperCase(),
      name: formName.trim(),
    };
    
    // Check if sector already exists
    const existingIndex = sectorEtfs.findIndex(
      e => e.sector.toLowerCase() === newEtf.sector.toLowerCase()
    );
    
    let updated: SectorETFConfig[];
    if (existingIndex >= 0) {
      // Update existing
      updated = [...sectorEtfs];
      updated[existingIndex] = newEtf;
    } else {
      // Add new
      updated = [...sectorEtfs, newEtf];
    }
    
    if (onSave) {
      setIsSaving(true);
      try {
        await onSave(updated);
        onChange(updated);
      } finally {
        setIsSaving(false);
      }
    } else {
      onChange(updated);
    }
    
    resetForm();
    setIsAddOpen(false);
  }

  async function handleEdit() {
    if (editingIndex === null) return;
    
    const updated = [...sectorEtfs];
    updated[editingIndex] = {
      sector: formSector.trim(),
      symbol: formSymbol.trim().toUpperCase(),
      name: formName.trim(),
    };
    
    if (onSave) {
      setIsSaving(true);
      try {
        await onSave(updated);
        onChange(updated);
      } finally {
        setIsSaving(false);
      }
    } else {
      onChange(updated);
    }
    
    resetForm();
    setEditingIndex(null);
  }

  async function handleDelete(index: number) {
    const updated = sectorEtfs.filter((_, i) => i !== index);
    
    if (onSave) {
      setIsSaving(true);
      try {
        await onSave(updated);
        onChange(updated);
      } finally {
        setIsSaving(false);
      }
    } else {
      onChange(updated);
    }
  }

  const formFields: ConfigField[] = [
    {
      id: 'sector-name',
      label: 'Sector Name',
      value: formSector,
      onChange: setFormSector,
      placeholder: 'Technology, Healthcare, Financials...',
      helperText: 'Match the sector name from Yahoo Finance',
    },
    {
      id: 'sector-etf-symbol',
      label: 'ETF Symbol',
      value: formSymbol,
      onChange: setFormSymbol,
      placeholder: 'XLK, XLV, XLF...',
      helperText: 'Sector ETF ticker symbol',
      inputProps: { className: 'font-mono' },
      transform: (value) => value.toUpperCase(),
    },
    {
      id: 'sector-etf-name',
      label: 'Display Name',
      value: formName,
      onChange: setFormName,
      placeholder: 'Technology Select Sector SPDR',
    },
  ];
  const isFormValid = Boolean(
    formSector.trim() && formSymbol.trim() && formName.trim()
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Building2 className="h-5 w-5" />
              Sector ETF Mappings
            </CardTitle>
            <CardDescription>
              Map sectors to their benchmark ETFs for performance comparison
            </CardDescription>
          </div>
          <Dialog open={isAddOpen} onOpenChange={(open) => {
            setIsAddOpen(open);
            if (!open) resetForm();
          }}>
            <Button onClick={() => setIsAddOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Add Sector ETF
            </Button>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Sector ETF Mapping</DialogTitle>
                <DialogDescription>
                  Add a new sector-to-ETF mapping for performance benchmarking.
                </DialogDescription>
              </DialogHeader>
              <ConfigForm
                fields={formFields}
                onSubmit={handleAdd}
                submitLabel="Add Sector ETF"
                isSubmitting={isSaving}
                isValid={isFormValid}
              />
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        {sectorEtfs.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Building2 className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p>No sector ETF mappings configured</p>
            <p className="text-sm mt-1">Add sector ETFs to enable performance comparison</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Sector</TableHead>
                  <TableHead>ETF Symbol</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead className="w-[100px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <AnimatePresence>
                  {sectorEtfs.map((etf, index) => (
                    <motion.tr
                      key={etf.sector}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="border-b"
                    >
                      <TableCell className="font-medium">{etf.sector}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono">
                          {etf.symbol}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {etf.name}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => openEdit(index)}
                          >
                            <Edit2 className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-destructive hover:text-destructive"
                            onClick={() => handleDelete(index)}
                          >
                            <Trash2 className="h-4 w-4" />
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

        {/* Edit Dialog */}
        <Dialog 
          open={editingIndex !== null} 
          onOpenChange={(open) => {
            if (!open) {
              setEditingIndex(null);
              resetForm();
            }
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Sector ETF Mapping</DialogTitle>
              <DialogDescription>
                Update the ETF mapping for this sector.
              </DialogDescription>
            </DialogHeader>
            <ConfigForm
              fields={formFields}
              onSubmit={handleEdit}
              submitLabel="Save Changes"
              isSubmitting={isSaving}
              isValid={isFormValid}
            />
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}
