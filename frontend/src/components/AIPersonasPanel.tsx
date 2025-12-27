import { useState, useEffect, useCallback, useRef } from 'react';
import {
  getAIPersonas,
  updateAIPersona,
  uploadAIPersonaAvatar,
  deleteAIPersonaAvatar,
  type AIPersona,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  RefreshCw,
  Upload,
  Trash2,
  User,
  Edit,
  Save,
  X,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

export function AIPersonasPanel() {
  const [personas, setPersonas] = useState<AIPersona[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [editingPersona, setEditingPersona] = useState<AIPersona | null>(null);
  const [editForm, setEditForm] = useState<{
    name: string;
    description: string;
    philosophy: string;
  }>({ name: '', description: '', philosophy: '' });
  const [uploadingKey, setUploadingKey] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pendingUploadKey = useRef<string | null>(null);

  const loadPersonas = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getAIPersonas(false); // Get all, not just active
      setPersonas(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load personas');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPersonas();
  }, [loadPersonas]);

  const handleToggleActive = async (persona: AIPersona) => {
    try {
      await updateAIPersona(persona.key, { is_active: !persona.is_active });
      setSuccess(`${persona.name} ${persona.is_active ? 'disabled' : 'enabled'}`);
      setTimeout(() => setSuccess(null), 3000);
      loadPersonas();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update persona');
    }
  };

  const handleEditClick = (persona: AIPersona) => {
    setEditingPersona(persona);
    setEditForm({
      name: persona.name,
      description: persona.description || '',
      philosophy: persona.philosophy || '',
    });
  };

  const handleSaveEdit = async () => {
    if (!editingPersona) return;
    try {
      await updateAIPersona(editingPersona.key, {
        name: editForm.name,
        description: editForm.description || undefined,
        philosophy: editForm.philosophy || undefined,
      });
      setSuccess(`${editForm.name} updated successfully`);
      setTimeout(() => setSuccess(null), 3000);
      setEditingPersona(null);
      loadPersonas();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update persona');
    }
  };

  const handleUploadClick = (personaKey: string) => {
    pendingUploadKey.current = personaKey;
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    const personaKey = pendingUploadKey.current;
    if (!file || !personaKey) return;

    setUploadingKey(personaKey);
    setError(null);
    try {
      await uploadAIPersonaAvatar(personaKey, file, 128);
      setSuccess('Avatar uploaded successfully');
      setTimeout(() => setSuccess(null), 3000);
      loadPersonas();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload avatar');
    } finally {
      setUploadingKey(null);
      pendingUploadKey.current = null;
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDeleteAvatar = async (persona: AIPersona) => {
    if (!confirm(`Remove avatar for ${persona.name}?`)) return;
    try {
      await deleteAIPersonaAvatar(persona.key);
      setSuccess('Avatar removed');
      setTimeout(() => setSuccess(null), 3000);
      loadPersonas();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove avatar');
    }
  };

  const activeCount = personas.filter(p => p.is_active).length;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              AI Personas
            </CardTitle>
            <CardDescription>
              Manage AI investment analyst personas and their avatars
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{activeCount} active</Badge>
            <Button variant="outline" size="sm" onClick={loadPersonas} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {success && (
          <Alert className="bg-success/10 border-success/30 text-success">
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {personas.map((persona) => (
            <div
              key={persona.key}
              className={`p-4 rounded-lg border transition-colors ${
                persona.is_active 
                  ? 'bg-card border-border' 
                  : 'bg-muted/30 border-muted opacity-60'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="relative">
                  <Avatar className="h-12 w-12">
                    {persona.avatar_url ? (
                      <AvatarImage 
                        src={persona.avatar_url} 
                        alt={persona.name}
                      />
                    ) : null}
                    <AvatarFallback className="bg-primary/10 text-primary">
                      {persona.name.split(' ').map(n => n[0]).join('')}
                    </AvatarFallback>
                  </Avatar>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="absolute -bottom-1 -right-1 h-6 w-6 rounded-full bg-background border shadow-sm"
                    onClick={() => handleUploadClick(persona.key)}
                    disabled={uploadingKey === persona.key}
                  >
                    {uploadingKey === persona.key ? (
                      <RefreshCw className="h-3 w-3 animate-spin" />
                    ) : (
                      <Upload className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="font-medium truncate">{persona.name}</h4>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6 shrink-0"
                      onClick={() => handleEditClick(persona)}
                    >
                      <Edit className="h-3 w-3" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground truncate">
                    {persona.key}
                  </p>
                  {persona.description && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {persona.description}
                    </p>
                  )}
                </div>
              </div>
              
              <div className="flex items-center justify-between mt-3 pt-3 border-t">
                <div className="flex items-center gap-2">
                  <Switch
                    checked={persona.is_active}
                    onCheckedChange={() => handleToggleActive(persona)}
                    className="data-[state=checked]:bg-success"
                  />
                  <span className="text-xs text-muted-foreground">
                    {persona.is_active ? 'Active' : 'Disabled'}
                  </span>
                </div>
                
                {persona.has_avatar && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive"
                    onClick={() => handleDeleteAvatar(persona)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>

        {personas.length === 0 && !isLoading && (
          <p className="text-center text-muted-foreground py-8">
            No AI personas configured
          </p>
        )}
      </CardContent>

      {/* Edit Dialog */}
      <Dialog open={!!editingPersona} onOpenChange={(open) => !open && setEditingPersona(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit {editingPersona?.name}</DialogTitle>
            <DialogDescription>
              Update persona details and investment philosophy
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="name">Display Name</Label>
              <Input
                id="name"
                value={editForm.name}
                onChange={(e) => setEditForm(f => ({ ...f, name: e.target.value }))}
                placeholder="Warren Buffett"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={editForm.description}
                onChange={(e) => setEditForm(f => ({ ...f, description: e.target.value }))}
                placeholder="The Oracle of Omaha..."
                rows={2}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="philosophy">Investment Philosophy</Label>
              <Textarea
                id="philosophy"
                value={editForm.philosophy}
                onChange={(e) => setEditForm(f => ({ ...f, philosophy: e.target.value }))}
                placeholder="Value investing, long-term holdings, competitive moats..."
                rows={3}
              />
            </div>
          </div>
          
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setEditingPersona(null)}>
              <X className="h-4 w-4 mr-1" />
              Cancel
            </Button>
            <Button onClick={handleSaveEdit}>
              <Save className="h-4 w-4 mr-1" />
              Save Changes
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
