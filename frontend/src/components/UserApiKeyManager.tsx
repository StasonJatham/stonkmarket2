import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { Switch } from '@/components/ui/switch';
import { 
  Key, 
  Plus, 
  Trash2, 
  Loader2, 
  Copy, 
  CheckCircle,
  XCircle,
  AlertCircle,
  Users,
  Activity,
  Power,
  PowerOff,
  Edit
} from 'lucide-react';
import { 
  listUserApiKeys, 
  createUserApiKey, 
  getUserApiKeyStats,
  updateUserApiKey,
  deactivateUserApiKey,
  reactivateUserApiKey,
  deleteUserApiKey,
  type UserApiKey,
  type UserApiKeyStats,
  type CreateUserKeyRequest,
  type CreateUserKeyResponse
} from '@/services/api';

interface UserApiKeyManagerProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function UserApiKeyManager({ onError, onSuccess }: UserApiKeyManagerProps) {
  const [keys, setKeys] = useState<UserApiKey[]>([]);
  const [stats, setStats] = useState<UserApiKeyStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showInactive, setShowInactive] = useState(false);
  
  // Create dialog
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [newKeyDescription, setNewKeyDescription] = useState('');
  const [newKeyVoteWeight, setNewKeyVoteWeight] = useState(10);
  const [newKeyRateLimitBypass, setNewKeyRateLimitBypass] = useState(true);
  const [newKeyExpiresDays, setNewKeyExpiresDays] = useState<number | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  
  // New key display
  const [newKeyResult, setNewKeyResult] = useState<CreateUserKeyResponse | null>(null);
  const [copied, setCopied] = useState(false);
  
  // Edit dialog
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<UserApiKey | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editVoteWeight, setEditVoteWeight] = useState(10);
  const [editRateLimitBypass, setEditRateLimitBypass] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  
  // Delete dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingKeyId, setDeletingKeyId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  
  // Toggle loading
  const [togglingId, setTogglingId] = useState<number | null>(null);

  async function loadData() {
    setIsLoading(true);
    try {
      const [keysData, statsData] = await Promise.all([
        listUserApiKeys(!showInactive),
        getUserApiKeyStats(),
      ]);
      setKeys(keysData);
      setStats(statsData);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to load user API keys');
    } finally {
      setIsLoading(false);
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: loadData defined in component scope
  useEffect(() => {
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showInactive]);

  async function handleCreate() {
    if (!newKeyName.trim()) return;
    
    setIsCreating(true);
    try {
      const request: CreateUserKeyRequest = {
        name: newKeyName,
        description: newKeyDescription || undefined,
        vote_weight: newKeyVoteWeight,
        rate_limit_bypass: newKeyRateLimitBypass,
        expires_days: newKeyExpiresDays || undefined,
      };
      const result = await createUserApiKey(request);
      setNewKeyResult(result);
      await loadData();
      onSuccess?.('API key created successfully');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to create API key');
    } finally {
      setIsCreating(false);
    }
  }

  async function handleToggleActive(key: UserApiKey) {
    setTogglingId(key.id);
    try {
      if (key.is_active) {
        await deactivateUserApiKey(key.id);
        onSuccess?.('Key deactivated');
      } else {
        await reactivateUserApiKey(key.id);
        onSuccess?.('Key reactivated');
      }
      await loadData();
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to toggle key status');
    } finally {
      setTogglingId(null);
    }
  }

  async function handleSaveEdit() {
    if (!editingKey) return;
    
    setIsSaving(true);
    try {
      await updateUserApiKey(editingKey.id, {
        name: editName,
        description: editDescription,
        vote_weight: editVoteWeight,
        rate_limit_bypass: editRateLimitBypass,
      });
      setEditDialogOpen(false);
      setEditingKey(null);
      await loadData();
      onSuccess?.('Key updated successfully');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to update key');
    } finally {
      setIsSaving(false);
    }
  }

  async function handleDelete() {
    if (!deletingKeyId) return;
    
    setIsDeleting(true);
    try {
      await deleteUserApiKey(deletingKeyId);
      setDeleteDialogOpen(false);
      setDeletingKeyId(null);
      await loadData();
      onSuccess?.('Key deleted');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to delete key');
    } finally {
      setIsDeleting(false);
    }
  }

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function openEditDialog(key: UserApiKey) {
    setEditingKey(key);
    setEditName(key.name);
    setEditDescription(key.description || '');
    setEditVoteWeight(key.vote_weight);
    setEditRateLimitBypass(key.rate_limit_bypass);
    setEditDialogOpen(true);
  }

  function closeCreateDialog() {
    setCreateDialogOpen(false);
    setNewKeyName('');
    setNewKeyDescription('');
    setNewKeyVoteWeight(10);
    setNewKeyRateLimitBypass(true);
    setNewKeyExpiresDays(null);
    setNewKeyResult(null);
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-72" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-48 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-6"
    >
      {/* Stats */}
      {stats && (
        <div className="grid gap-4 grid-cols-2 sm:grid-cols-4">
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center gap-2">
                <Users className="h-4 w-4 text-muted-foreground" />
                <span className="text-2xl font-bold">{stats.total_keys}</span>
              </div>
              <p className="text-xs text-muted-foreground">Total Keys</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-success" />
                <span className="text-2xl font-bold">{stats.active_keys}</span>
              </div>
              <p className="text-xs text-muted-foreground">Active</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="text-2xl font-bold">{stats.total_usage.toLocaleString()}</span>
              </div>
              <p className="text-xs text-muted-foreground">Total Requests</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-warning" />
                <span className="text-2xl font-bold">{stats.expired_keys}</span>
              </div>
              <p className="text-xs text-muted-foreground">Expired</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Card */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                User API Keys
              </CardTitle>
              <CardDescription>
                Generate keys for external apps/users to access the stonkmarket API
              </CardDescription>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Switch
                  id="show-inactive"
                  checked={showInactive}
                  onCheckedChange={setShowInactive}
                />
                <Label htmlFor="show-inactive" className="text-sm text-muted-foreground">
                  Show inactive
                </Label>
              </div>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Key
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {keys.length === 0 ? (
            <motion.div 
              className="text-center py-12 text-muted-foreground"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <Key className="h-12 w-12 mx-auto mb-4 opacity-20" />
              <p className="font-medium">No API keys yet</p>
              <p className="text-sm">Create your first key to give external access to the API</p>
            </motion.div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Key Prefix</TableHead>
                    <TableHead>Weight</TableHead>
                    <TableHead>Usage</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence mode="popLayout">
                    {keys.map((key) => (
                      <motion.tr
                        key={key.id}
                        layout
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="border-b"
                      >
                        <TableCell>
                          <div>
                            <div className="font-medium">{key.name}</div>
                            {key.description && (
                              <div className="text-xs text-muted-foreground">{key.description}</div>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <code className="text-xs bg-muted px-2 py-1 rounded">
                            {key.key_prefix}...
                          </code>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{key.vote_weight}x</Badge>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            {key.usage_count.toLocaleString()}
                            {key.last_used_at && (
                              <div className="text-xs text-muted-foreground">
                                Last: {new Date(key.last_used_at).toLocaleDateString()}
                              </div>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          {key.is_active ? (
                            <Badge className="bg-success/20 text-success">Active</Badge>
                          ) : (
                            <Badge variant="secondary">Inactive</Badge>
                          )}
                          {key.expires_at && new Date(key.expires_at) < new Date() && (
                            <Badge variant="destructive" className="ml-1">Expired</Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => openEditDialog(key)}
                              title="Edit"
                            >
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleToggleActive(key)}
                              disabled={togglingId === key.id}
                              title={key.is_active ? 'Deactivate' : 'Activate'}
                            >
                              {togglingId === key.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : key.is_active ? (
                                <PowerOff className="h-4 w-4" />
                              ) : (
                                <Power className="h-4 w-4" />
                              )}
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => {
                                setDeletingKeyId(key.id);
                                setDeleteDialogOpen(true);
                              }}
                              title="Delete"
                            >
                              <Trash2 className="h-4 w-4 text-danger" />
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
      </Card>

      {/* Create Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={(open) => !open && closeCreateDialog()}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create User API Key</DialogTitle>
            <DialogDescription>
              {newKeyResult 
                ? 'Save this key now - you won\'t be able to see it again!'
                : 'Generate a new API key for external access'
              }
            </DialogDescription>
          </DialogHeader>
          
          {newKeyResult ? (
            <div className="space-y-4">
              <div className="flex items-start gap-2 p-3 bg-warning/10 text-warning rounded-lg">
                <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
                <div className="text-sm">
                  <div className="font-medium">Save this key!</div>
                  <div className="opacity-80">{newKeyResult.warning}</div>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label>API Key</Label>
                <div className="flex gap-2">
                  <code className="flex-1 p-3 bg-muted rounded-lg text-sm font-mono break-all">
                    {newKeyResult.key}
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(newKeyResult.key)}
                  >
                    {copied ? <CheckCircle className="h-4 w-4 text-success" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              
              <DialogFooter>
                <Button onClick={closeCreateDialog}>
                  Done
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Name *</Label>
                <Input
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="e.g., Mobile App, Partner Integration"
                />
              </div>
              
              <div className="space-y-2">
                <Label>Description</Label>
                <Input
                  value={newKeyDescription}
                  onChange={(e) => setNewKeyDescription(e.target.value)}
                  placeholder="Optional notes about this key"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Vote Weight</Label>
                  <Input
                    type="number"
                    min={1}
                    max={100}
                    value={newKeyVoteWeight}
                    onChange={(e) => setNewKeyVoteWeight(parseInt(e.target.value) || 10)}
                  />
                  <p className="text-xs text-muted-foreground">Multiplier for votes (1-100)</p>
                </div>
                
                <div className="space-y-2">
                  <Label>Expires In (days)</Label>
                  <Input
                    type="number"
                    min={1}
                    max={365}
                    value={newKeyExpiresDays || ''}
                    onChange={(e) => setNewKeyExpiresDays(e.target.value ? parseInt(e.target.value) : null)}
                    placeholder="Never"
                  />
                  <p className="text-xs text-muted-foreground">Leave empty for no expiry</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
                <Switch
                  id="rate-limit-bypass"
                  checked={newKeyRateLimitBypass}
                  onCheckedChange={setNewKeyRateLimitBypass}
                />
                <div>
                  <Label htmlFor="rate-limit-bypass" className="font-medium">Bypass Rate Limits</Label>
                  <p className="text-xs text-muted-foreground">Allow unlimited API requests</p>
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={closeCreateDialog}>
                  Cancel
                </Button>
                <Button onClick={handleCreate} disabled={!newKeyName.trim() || isCreating}>
                  {isCreating && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Create Key
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Edit API Key</DialogTitle>
            <DialogDescription>
              Update key settings. The key value cannot be changed.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Description</Label>
              <Input
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Vote Weight</Label>
              <Input
                type="number"
                min={1}
                max={100}
                value={editVoteWeight}
                onChange={(e) => setEditVoteWeight(parseInt(e.target.value) || 10)}
              />
            </div>
            
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <Switch
                id="edit-rate-limit-bypass"
                checked={editRateLimitBypass}
                onCheckedChange={setEditRateLimitBypass}
              />
              <Label htmlFor="edit-rate-limit-bypass">Bypass Rate Limits</Label>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveEdit} disabled={isSaving}>
              {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete API Key</DialogTitle>
            <DialogDescription>
              This will permanently delete the API key. Any applications using this key will lose access immediately.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
              {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
