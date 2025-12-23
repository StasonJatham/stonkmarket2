import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
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
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Key, 
  Plus, 
  Trash2, 
  Eye, 
  Loader2, 
  Copy, 
  CheckCircle,
  Shield
} from 'lucide-react';
import { 
  listApiKeys, 
  createApiKey, 
  revealApiKey, 
  deleteApiKey, 
  getMFAStatus,
  type ApiKeyInfo 
} from '@/services/api';

interface ApiKeyManagerProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function ApiKeyManager({ onError, onSuccess }: ApiKeyManagerProps) {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [mfaEnabled, setMfaEnabled] = useState(false);
  
  // Add dialog
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [keyName, setKeyName] = useState('OPENAI_API_KEY');
  const [keyValue, setKeyValue] = useState('');
  const [mfaCode, setMfaCode] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  
  // Reveal dialog
  const [revealDialogOpen, setRevealDialogOpen] = useState(false);
  const [revealingKey, setRevealingKey] = useState<string | null>(null);
  const [revealMfaCode, setRevealMfaCode] = useState('');
  const [revealedValue, setRevealedValue] = useState<string | null>(null);
  const [isRevealing, setIsRevealing] = useState(false);
  
  // Delete dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [deleteMfaCode, setDeleteMfaCode] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  
  const [copied, setCopied] = useState(false);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [keysData, mfaStatus] = await Promise.all([
        listApiKeys(),
        getMFAStatus(),
      ]);
      setKeys(keysData.keys);
      setMfaEnabled(mfaStatus.enabled);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to load API keys');
    } finally {
      setIsLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  async function handleAdd() {
    if (!keyName || !keyValue || mfaCode.length !== 6) return;
    
    setIsSaving(true);
    try {
      await createApiKey(keyName, keyValue, mfaCode);
      await loadData();
      setAddDialogOpen(false);
      setKeyName('OPENAI_API_KEY');
      setKeyValue('');
      setMfaCode('');
      onSuccess?.('API key saved successfully');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to save API key');
    } finally {
      setIsSaving(false);
    }
  }

  async function handleReveal() {
    if (!revealingKey || revealMfaCode.length !== 6) return;
    
    setIsRevealing(true);
    try {
      const result = await revealApiKey(revealingKey, revealMfaCode);
      setRevealedValue(result.api_key);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to reveal API key');
    } finally {
      setIsRevealing(false);
    }
  }

  async function handleDelete() {
    if (!deletingKey || deleteMfaCode.length !== 6) return;
    
    setIsDeleting(true);
    try {
      await deleteApiKey(deletingKey, deleteMfaCode);
      await loadData();
      setDeleteDialogOpen(false);
      setDeletingKey(null);
      setDeleteMfaCode('');
      onSuccess?.('API key deleted');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to delete API key');
    } finally {
      setIsDeleting(false);
    }
  }

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function closeRevealDialog() {
    setRevealDialogOpen(false);
    setRevealingKey(null);
    setRevealMfaCode('');
    setRevealedValue(null);
  }

  const openAiKey = keys.find(k => k.key_name === 'OPENAI_API_KEY');

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-72" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  // MFA required warning
  if (!mfaEnabled) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Key className="h-5 w-5" />
              Service Keys
            </CardTitle>
            <CardDescription>
              Configure API keys for external services (OpenAI for AI features)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <motion.div 
              className="flex items-center gap-3 p-4 bg-warning/10 text-warning rounded-lg"
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1 }}
            >
              <Shield className="h-5 w-5 shrink-0" />
              <div>
                <div className="font-medium">MFA Required</div>
                <div className="text-sm opacity-80">
                  You must enable two-factor authentication before managing API keys.
                </div>
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                Service Keys
              </CardTitle>
              <CardDescription>
                Configure API keys for backend services (OpenAI for AI features). MFA required.
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* OpenAI Key Status */}
          <motion.div 
            className="p-4 bg-muted/50 rounded-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-background rounded-lg">
                <Key className="h-5 w-5" />
              </div>
              <div>
                <div className="font-medium">OpenAI API Key</div>
                <div className="text-sm text-muted-foreground">
                  {openAiKey ? (
                    <>
                      <span className="font-mono">{openAiKey.key_hint}</span>
                      <span className="ml-2 text-xs">
                        Updated {new Date(openAiKey.updated_at).toLocaleDateString()}
                      </span>
                    </>
                  ) : (
                    'Not configured'
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {openAiKey ? (
                <>
                  <Badge variant="default" className="bg-success/20 text-success">
                    Configured
                  </Badge>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setKeyName('OPENAI_API_KEY');
                      setKeyValue('');
                      setAddDialogOpen(true);
                    }}
                  >
                    Update
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setRevealingKey('OPENAI_API_KEY');
                      setRevealDialogOpen(true);
                    }}
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setDeletingKey('OPENAI_API_KEY');
                      setDeleteDialogOpen(true);
                    }}
                  >
                    <Trash2 className="h-4 w-4 text-danger" />
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  onClick={() => {
                    setKeyName('OPENAI_API_KEY');
                    setKeyValue('');
                    setAddDialogOpen(true);
                  }}
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Configure
                </Button>
              )}
            </div>
          </div>
        </motion.div>
      </CardContent>

      {/* Add/Update Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {openAiKey && keyName === 'OPENAI_API_KEY' ? 'Update OpenAI API Key' : 'Configure API Key'}
            </DialogTitle>
            <DialogDescription>
              {openAiKey && keyName === 'OPENAI_API_KEY'
                ? 'Enter the new API key value. This will replace the existing key.'
                : 'Store an API key securely. You\'ll need your MFA code to save it.'
              }
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {keyName !== 'OPENAI_API_KEY' && (
              <div className="space-y-2">
                <Label>Key Name</Label>
                <Input
                  value={keyName}
                  onChange={(e) => setKeyName(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ''))}
                  placeholder="OPENAI_API_KEY"
                  className="font-mono"
                />
              </div>
            )}
            <div className="space-y-2">
              <Label>API Key Value</Label>
              <Input
                value={keyValue}
                onChange={(e) => setKeyValue(e.target.value)}
                placeholder="sk-..."
                type="password"
              />
            </div>
            <div className="space-y-2">
              <Label>MFA Code</Label>
              <Input
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="000000"
                className="font-mono text-center tracking-widest"
                maxLength={6}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAdd}
              disabled={!keyName || !keyValue || mfaCode.length !== 6 || isSaving}
            >
              {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reveal Dialog */}
      <Dialog open={revealDialogOpen} onOpenChange={(open) => !open && closeRevealDialog()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reveal API Key</DialogTitle>
            <DialogDescription>
              Enter your MFA code to view the full API key for <strong>{revealingKey}</strong>.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {revealedValue ? (
              <div className="space-y-2">
                <Label>API Key</Label>
                <div className="flex gap-2">
                  <Input
                    value={revealedValue}
                    readOnly
                    className="font-mono text-sm"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(revealedValue)}
                  >
                    {copied ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <Label>MFA Code</Label>
                <Input
                  value={revealMfaCode}
                  onChange={(e) => setRevealMfaCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                  placeholder="000000"
                  className="font-mono text-center tracking-widest"
                  maxLength={6}
                />
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeRevealDialog}>
              Close
            </Button>
            {!revealedValue && (
              <Button
                onClick={handleReveal}
                disabled={revealMfaCode.length !== 6 || isRevealing}
              >
                {isRevealing && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                <Eye className="h-4 w-4 mr-2" />
                Reveal
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete API Key</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{deletingKey}</strong>? 
              Enter your MFA code to confirm.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label>MFA Code</Label>
            <Input
              value={deleteMfaCode}
              onChange={(e) => setDeleteMfaCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              className="font-mono text-center tracking-widest mt-2"
              maxLength={6}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMfaCode.length !== 6 || isDeleting}
            >
              {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
    </motion.div>
  );
}
