/**
 * Channels Panel - Manage notification delivery channels.
 * 
 * Supports: Discord, Telegram, Email, Slack, Pushover, Ntfy, Webhook
 */

import { useState } from 'react';
import { 
  Plus, 
  Trash2, 
  Edit, 
  TestTube2, 
  Loader2, 
  CheckCircle, 
  XCircle,
  Radio,
  AlertCircle,
  Send,
  MessageSquare,
  Mail,
  Webhook,
  Bell,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Skeleton } from '@/components/ui/skeleton';
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

import {
  useNotificationChannels,
  useCreateChannel,
  useUpdateChannel,
  useDeleteChannel,
  useTestChannel,
  type NotificationChannel,
  type ChannelType,
} from '@/features/notifications';

const CHANNEL_TYPE_INFO: Record<ChannelType, { label: string; icon: typeof Radio; placeholder: string }> = {
  discord: { 
    label: 'Discord', 
    icon: MessageSquare,
    placeholder: 'discord://webhook_id/webhook_token',
  },
  telegram: { 
    label: 'Telegram', 
    icon: Send,
    placeholder: 'tgram://bot_token/chat_id',
  },
  email: { 
    label: 'Email', 
    icon: Mail,
    placeholder: 'mailto://user:pass@gmail.com',
  },
  slack: { 
    label: 'Slack', 
    icon: MessageSquare,
    placeholder: 'slack://token_a/token_b/token_c',
  },
  pushover: { 
    label: 'Pushover', 
    icon: Bell,
    placeholder: 'pover://user@token',
  },
  ntfy: { 
    label: 'ntfy', 
    icon: Bell,
    placeholder: 'ntfy://topic',
  },
  webhook: { 
    label: 'Webhook', 
    icon: Webhook,
    placeholder: 'json://hostname/path',
  },
};

export function ChannelsPanel() {
  const channelsQuery = useNotificationChannels();
  const createMutation = useCreateChannel();
  const updateMutation = useUpdateChannel();
  const deleteMutation = useDeleteChannel();
  const testMutation = useTestChannel();

  // Dialog states
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editingChannel, setEditingChannel] = useState<NotificationChannel | null>(null);
  const [deletingChannelId, setDeletingChannelId] = useState<number | null>(null);

  // Form state
  const [formName, setFormName] = useState('');
  const [formType, setFormType] = useState<ChannelType>('discord');
  const [formUrl, setFormUrl] = useState('');
  const [formActive, setFormActive] = useState(true);

  // Feedback
  const [testingId, setTestingId] = useState<number | null>(null);
  const [testResult, setTestResult] = useState<{ channelId: number; success: boolean; message: string } | null>(null);

  const resetForm = () => {
    setFormName('');
    setFormType('discord');
    setFormUrl('');
    setFormActive(true);
  };

  const handleCreate = async () => {
    try {
      await createMutation.mutateAsync({
        name: formName,
        channel_type: formType,
        apprise_url: formUrl,
      });
      setCreateDialogOpen(false);
      resetForm();
    } catch {
      // Error handled by mutation
    }
  };

  const handleEdit = (channel: NotificationChannel) => {
    setEditingChannel(channel);
    setFormName(channel.name);
    setFormType(channel.channel_type);
    setFormUrl(''); // Don't show existing URL for security
    setFormActive(channel.is_active);
    setEditDialogOpen(true);
  };

  const handleUpdate = async () => {
    if (!editingChannel) return;
    try {
      await updateMutation.mutateAsync({
        channelId: editingChannel.id,
        data: {
          name: formName,
          apprise_url: formUrl || undefined, // Only update if provided
          is_active: formActive,
        },
      });
      setEditDialogOpen(false);
      setEditingChannel(null);
      resetForm();
    } catch {
      // Error handled by mutation
    }
  };

  const handleDelete = async () => {
    if (deletingChannelId === null) return;
    try {
      await deleteMutation.mutateAsync(deletingChannelId);
      setDeleteDialogOpen(false);
      setDeletingChannelId(null);
    } catch {
      // Error handled by mutation
    }
  };

  const handleTest = async (channelId: number) => {
    setTestingId(channelId);
    setTestResult(null);
    try {
      const result = await testMutation.mutateAsync(channelId);
      setTestResult({ channelId, success: result.success, message: result.message });
    } catch (error) {
      setTestResult({ 
        channelId, 
        success: false, 
        message: error instanceof Error ? error.message : 'Test failed',
      });
    } finally {
      setTestingId(null);
    }
  };

  const openDeleteDialog = (channelId: number) => {
    setDeletingChannelId(channelId);
    setDeleteDialogOpen(true);
  };

  if (channelsQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-9 w-32" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const channels = channelsQuery.data ?? [];

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Radio className="h-5 w-5" />
                Notification Channels
              </CardTitle>
              <CardDescription>
                Connect to Discord, Telegram, Email, and more for alerts
              </CardDescription>
            </div>
            <Button onClick={() => setCreateDialogOpen(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              Add Channel
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {channels.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Radio className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p className="font-medium">No channels configured</p>
              <p className="text-sm mt-1">Add a channel to start receiving notifications</p>
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {channels.map((channel) => {
                    const typeInfo = CHANNEL_TYPE_INFO[channel.channel_type];
                    const Icon = typeInfo?.icon ?? Radio;
                    const result = testResult?.channelId === channel.id ? testResult : null;
                    
                    return (
                      <TableRow key={channel.id}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            <Icon className="h-4 w-4 text-muted-foreground" />
                            {channel.name}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{typeInfo?.label ?? channel.channel_type}</Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {channel.is_active ? (
                              <Badge variant="default" className="gap-1">
                                <CheckCircle className="h-3 w-3" />
                                Active
                              </Badge>
                            ) : (
                              <Badge variant="secondary" className="gap-1">
                                <XCircle className="h-3 w-3" />
                                Inactive
                              </Badge>
                            )}
                            {channel.is_verified && (
                              <Badge variant="outline" className="gap-1 text-green-600 border-green-200">
                                <CheckCircle className="h-3 w-3" />
                                Verified
                              </Badge>
                            )}
                            {channel.error_count > 0 && (
                              <Tooltip>
                                <TooltipTrigger>
                                  <Badge variant="destructive" className="gap-1">
                                    <AlertCircle className="h-3 w-3" />
                                    {channel.error_count} errors
                                  </Badge>
                                </TooltipTrigger>
                                <TooltipContent>
                                  {channel.last_error ?? 'Recent delivery errors'}
                                </TooltipContent>
                              </Tooltip>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleTest(channel.id)}
                                  disabled={testingId === channel.id}
                                >
                                  {testingId === channel.id ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : result ? (
                                    result.success ? (
                                      <CheckCircle className="h-4 w-4 text-green-600" />
                                    ) : (
                                      <XCircle className="h-4 w-4 text-red-600" />
                                    )
                                  ) : (
                                    <TestTube2 className="h-4 w-4" />
                                  )}
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                {result ? result.message : 'Send test notification'}
                              </TooltipContent>
                            </Tooltip>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleEdit(channel)}
                            >
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => openDeleteDialog(channel.id)}
                            >
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Create Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Notification Channel</DialogTitle>
            <DialogDescription>
              Connect a new channel to receive alerts
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="channel-name">Name</Label>
              <Input
                id="channel-name"
                placeholder="My Discord Server"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Channel Type</Label>
              <Select value={formType} onValueChange={(v) => setFormType(v as ChannelType)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(CHANNEL_TYPE_INFO).map(([key, info]) => {
                    const Icon = info.icon;
                    return (
                      <SelectItem key={key} value={key}>
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4" />
                          {info.label}
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="apprise-url">Apprise URL</Label>
              <Input
                id="apprise-url"
                type="password"
                placeholder={CHANNEL_TYPE_INFO[formType]?.placeholder ?? 'Enter Apprise URL'}
                value={formUrl}
                onChange={(e) => setFormUrl(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                See <a href="https://github.com/caronc/apprise/wiki" target="_blank" rel="noopener noreferrer" className="underline">Apprise documentation</a> for URL format
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleCreate} 
              disabled={!formName || !formUrl || createMutation.isPending}
            >
              {createMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Add Channel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Channel</DialogTitle>
            <DialogDescription>
              Update channel settings
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">Name</Label>
              <Input
                id="edit-name"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-url">New Apprise URL (optional)</Label>
              <Input
                id="edit-url"
                type="password"
                placeholder="Leave empty to keep current URL"
                value={formUrl}
                onChange={(e) => setFormUrl(e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="edit-active">Active</Label>
              <Switch
                id="edit-active"
                checked={formActive}
                onCheckedChange={setFormActive}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleUpdate} 
              disabled={!formName || updateMutation.isPending}
            >
              {updateMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Channel</DialogTitle>
            <DialogDescription>
              Are you sure? This will also delete all rules using this channel.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={handleDelete} 
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Delete Channel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
