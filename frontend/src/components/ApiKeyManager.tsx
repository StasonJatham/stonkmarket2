import { useState } from 'react';
import { motion } from 'framer-motion';
import { useQueryClient } from '@tanstack/react-query';
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
  Shield,
  AlertCircle
} from 'lucide-react';
import { 
  useApiKeys,
  useCreateApiKey,
  useRevealApiKey,
  useDeleteApiKey,
  useMFAStatus,
  useMFASession,
  useSystemStatus,
} from '@/features/admin/api/queries';

interface ApiKeyManagerProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function ApiKeyManager({ onSuccess }: ApiKeyManagerProps) {
  const queryClient = useQueryClient();
  
  // TanStack Query for data fetching
  const keysQuery = useApiKeys();
  const mfaStatusQuery = useMFAStatus();
  const mfaSessionQuery = useMFASession();
  const systemStatusQuery = useSystemStatus();
  
  // Mutations
  const createMutation = useCreateApiKey();
  const revealMutation = useRevealApiKey();
  const deleteMutation = useDeleteApiKey();
  
  // Derived state from queries - plain derivation, React Compiler optimizes this
  const keys = keysQuery.data ?? [];
  const isLoading = keysQuery.isLoading || mfaStatusQuery.isLoading || systemStatusQuery.isLoading;
  const mfaEnabled = mfaStatusQuery.data?.enabled ?? false;
  const hasMfaSession = mfaSessionQuery.data?.has_session ?? false;
  
  // Check if Logo.dev is configured via env (configured but not in database)
  const logoDevDbKey = keys.find(k => k.key_name === 'logo_dev_public_key');
  const logoDevEnvConfigured = (systemStatusQuery.data?.logo_dev_configured ?? false) && !logoDevDbKey;
  
  // Add dialog
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [keyName, setKeyName] = useState('OPENAI_API_KEY');
  const [keyValue, setKeyValue] = useState('');
  const [mfaCode, setMfaCode] = useState('');
  const [addError, setAddError] = useState<string | null>(null);
  
  // Reveal dialog
  const [revealDialogOpen, setRevealDialogOpen] = useState(false);
  const [revealingKey, setRevealingKey] = useState<string | null>(null);
  const [revealMfaCode, setRevealMfaCode] = useState('');
  const [revealedValue, setRevealedValue] = useState<string | null>(null);
  const [revealError, setRevealError] = useState<string | null>(null);
  
  // Delete dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [deleteMfaCode, setDeleteMfaCode] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);
  
  const [copied, setCopied] = useState(false);

  async function handleAdd(codeOverride?: string) {
    const code = codeOverride ?? mfaCode;
    // If we have a session, code can be empty; otherwise need 6 digits
    if (!keyName || !keyValue || (!hasMfaSession && code.length !== 6)) return;
    
    setAddError(null);
    try {
      await createMutation.mutateAsync({ key_name: keyName, key_value: keyValue, mfa_code: code || undefined });
      // Invalidate MFA session query to refresh session state
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfaSession'] });
      setAddDialogOpen(false);
      setKeyName('OPENAI_API_KEY');
      setKeyValue('');
      setMfaCode('');
      setAddError(null);
      onSuccess?.('API key saved successfully');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save API key';
      setAddError(message);
      setMfaCode('');
    }
  }

  async function handleReveal(codeOverride?: string) {
    const code = codeOverride ?? revealMfaCode;
    // If we have a session, code can be empty; otherwise need 6 digits
    if (!revealingKey || (!hasMfaSession && code.length !== 6)) return;
    
    setRevealError(null);
    try {
      const result = await revealMutation.mutateAsync({ key_name: revealingKey, mfa_code: code || undefined });
      setRevealedValue(result.key_value);
      // Invalidate MFA session query to refresh session state
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfaSession'] });
      setRevealError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to reveal API key';
      setRevealError(message);
      setRevealMfaCode('');
    }
  }

  async function handleDelete(codeOverride?: string) {
    const code = codeOverride ?? deleteMfaCode;
    // If we have a session, code can be empty; otherwise need 6 digits
    if (!deletingKey || (!hasMfaSession && code.length !== 6)) return;
    
    setDeleteError(null);
    try {
      await deleteMutation.mutateAsync({ key_name: deletingKey, mfa_code: code || undefined });
      // Invalidate MFA session query to refresh session state
      queryClient.invalidateQueries({ queryKey: ['auth', 'mfaSession'] });
      setDeleteDialogOpen(false);
      setDeletingKey(null);
      setDeleteMfaCode('');
      setDeleteError(null);
      onSuccess?.('API key deleted');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete API key';
      setDeleteError(message);
      setDeleteMfaCode('');
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
    setRevealError(null);
  }

  const openAiKey = keys.find(k => k.key_name === 'OPENAI_API_KEY');
  const logoDevKey = keys.find(k => k.key_name === 'logo_dev_public_key');

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
                Configure API keys for backend services (OpenAI for AI features, Logo.dev for company logos). MFA required.
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

          {/* Logo.dev Key Status */}
          <motion.div 
            className="p-4 bg-muted/50 rounded-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.15 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-background rounded-lg">
                  <Key className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-medium">Logo.dev API Key</div>
                  <div className="text-sm text-muted-foreground">
                    {logoDevKey ? (
                      <>
                        <span className="font-mono">{logoDevKey.key_hint}</span>
                        <span className="ml-2 text-xs">
                          Updated {new Date(logoDevKey.updated_at).toLocaleDateString()}
                        </span>
                      </>
                    ) : logoDevEnvConfigured ? (
                      'Configured via environment variable'
                    ) : (
                      'Not configured – company logos will show fallback icons'
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {logoDevKey ? (
                  <>
                    <Badge variant="default" className="bg-success/20 text-success">
                      Configured
                    </Badge>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setKeyName('logo_dev_public_key');
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
                        setRevealingKey('logo_dev_public_key');
                        setRevealDialogOpen(true);
                      }}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setDeletingKey('logo_dev_public_key');
                        setDeleteDialogOpen(true);
                      }}
                    >
                      <Trash2 className="h-4 w-4 text-danger" />
                    </Button>
                  </>
                ) : logoDevEnvConfigured ? (
                  <Badge variant="default" className="bg-success/20 text-success">
                    Via Environment
                  </Badge>
                ) : (
                  <Button
                    size="sm"
                    onClick={() => {
                      setKeyName('logo_dev_public_key');
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
      <Dialog open={addDialogOpen} onOpenChange={(open) => {
        setAddDialogOpen(open);
        if (!open) setAddError(null);
      }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {keyName === 'OPENAI_API_KEY' && openAiKey 
                ? 'Update OpenAI API Key' 
                : keyName === 'logo_dev_public_key' && logoDevKey
                  ? 'Update Logo.dev API Key'
                  : keyName === 'logo_dev_public_key'
                    ? 'Configure Logo.dev API Key'
                    : keyName === 'OPENAI_API_KEY'
                      ? 'Configure OpenAI API Key'
                      : 'Configure API Key'
              }
            </DialogTitle>
            <DialogDescription>
              {keyName === 'OPENAI_API_KEY'
                ? openAiKey 
                  ? 'Enter the new API key value. This will replace the existing key.'
                  : 'Configure OpenAI API key for AI enrichment features.'
                : keyName === 'logo_dev_public_key'
                  ? logoDevKey
                    ? 'Enter the new API key value. This will replace the existing key.'
                    : 'Configure Logo.dev public key for company logos. Get a free key at logo.dev'
                  : 'Store an API key securely. You\'ll need your MFA code to save it.'
              }
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {keyName !== 'OPENAI_API_KEY' && keyName !== 'logo_dev_public_key' && (
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
                placeholder={keyName === 'logo_dev_public_key' ? 'pk_...' : 'sk-...'}
                type="password"
              />
            </div>
            {!hasMfaSession && (
              <div className="space-y-2">
                <Label>MFA Code</Label>
                <Input
                  value={mfaCode}
                  onChange={(e) => {
                    const code = e.target.value.replace(/\D/g, '').slice(0, 6);
                    setMfaCode(code);
                    if (code.length === 6 && keyName && keyValue) {
                      handleAdd(code);
                    }
                  }}
                  placeholder="000000"
                  className="font-mono text-center tracking-widest"
                  maxLength={6}
                  autoComplete="one-time-code"
                />
                {addError && (
                  <div className="flex items-center gap-2 text-sm text-destructive mt-2">
                    <AlertCircle className="h-4 w-4" />
                    {addError}
                  </div>
                )}
              </div>
            )}
            {hasMfaSession && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground p-2 bg-muted/50 rounded">
                <Shield className="h-4 w-4" />
                MFA verified - session active
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => handleAdd()}
              disabled={!keyName || !keyValue || (!hasMfaSession && mfaCode.length !== 6) || createMutation.isPending}
            >
              {createMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
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
              {hasMfaSession 
                ? <>View the full API key for <strong>{revealingKey}</strong>.</>
                : <>Enter your MFA code to view the full API key for <strong>{revealingKey}</strong>.</>
              }
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
            ) : hasMfaSession ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground p-2 bg-muted/50 rounded">
                <Shield className="h-4 w-4" />
                MFA verified - session active
              </div>
            ) : (
              <div className="space-y-2">
                <Label>MFA Code</Label>
                <Input
                  value={revealMfaCode}
                  onChange={(e) => {
                    const code = e.target.value.replace(/\D/g, '').slice(0, 6);
                    setRevealMfaCode(code);
                    if (code.length === 6 && revealingKey) {
                      handleReveal(code);
                    }
                  }}
                  placeholder="000000"
                  className="font-mono text-center tracking-widest"
                  maxLength={6}
                  autoComplete="one-time-code"
                />
                {revealError && (
                  <div className="flex items-center gap-2 text-sm text-destructive mt-2">
                    <AlertCircle className="h-4 w-4" />
                    {revealError}
                  </div>
                )}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeRevealDialog}>
              Close
            </Button>
            {!revealedValue && (
              <Button
                onClick={() => handleReveal()}
                disabled={(!hasMfaSession && revealMfaCode.length !== 6) || revealMutation.isPending}
              >
                {revealMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                <Eye className="h-4 w-4 mr-2" />
                Reveal
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={(open) => {
        setDeleteDialogOpen(open);
        if (!open) setDeleteError(null);
      }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete API Key</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{deletingKey}</strong>? 
              {!hasMfaSession && ' Enter your MFA code to confirm.'}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            {hasMfaSession ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground p-2 bg-muted/50 rounded">
                <Shield className="h-4 w-4" />
                MFA verified - session active
              </div>
            ) : (
              <>
                <Label>MFA Code</Label>
                <Input
                  value={deleteMfaCode}
                  onChange={(e) => {
                    const code = e.target.value.replace(/\D/g, '').slice(0, 6);
                    setDeleteMfaCode(code);
                    if (code.length === 6 && deletingKey) {
                      handleDelete(code);
                    }
                  }}
                  placeholder="000000"
                  className="font-mono text-center tracking-widest mt-2"
                  maxLength={6}
                  autoComplete="one-time-code"
                />
                {deleteError && (
                  <div className="flex items-center gap-2 text-sm text-destructive mt-2">
                    <AlertCircle className="h-4 w-4" />
                    {deleteError}
                  </div>
                )}
              </>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => handleDelete()}
              disabled={(!hasMfaSession && deleteMfaCode.length !== 6) || deleteMutation.isPending}
            >
              {deleteMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
    </motion.div>
  );
}