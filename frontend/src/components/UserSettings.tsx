import { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { User, Loader2, CheckCircle } from 'lucide-react';
import { updateCredentials } from '@/services/api';
import { useAuth } from '@/context/AuthContext';

interface UserSettingsProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function UserSettings({ onError, onSuccess }: UserSettingsProps) {
  const { user } = useAuth();
  
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [newUsername, setNewUsername] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const passwordsMatch = newPassword === confirmPassword;
  const canSave = currentPassword && newPassword && passwordsMatch && newPassword.length >= 8;

  async function handleSave() {
    if (!canSave) return;
    
    setIsSaving(true);
    try {
      await updateCredentials({
        current_password: currentPassword,
        new_password: newPassword,
        new_username: newUsername || undefined,
      });
      
      setSaved(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setNewUsername('');
      onSuccess?.('Credentials updated successfully');
      
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to update credentials');
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            Account Settings
          </CardTitle>
          <CardDescription>
            Update your username and password
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Current user info */}
          <motion.div 
            className="p-4 bg-muted/50 rounded-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            <div className="text-sm text-muted-foreground">Current username</div>
            <div className="font-medium">{user?.username}</div>
          </motion.div>

          {/* Username change (optional) */}
          <div className="space-y-2">
            <Label>New Username (optional)</Label>
            <Input
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              placeholder="Leave blank to keep current"
            />
          </div>

        {/* Current password */}
        <div className="space-y-2">
          <Label>Current Password *</Label>
          <Input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            placeholder="Enter your current password"
          />
        </div>

        {/* New password */}
        <div className="space-y-2">
          <Label>New Password *</Label>
          <Input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="Enter new password (min 8 chars)"
          />
        </div>

        {/* Confirm password */}
        <div className="space-y-2">
          <Label>Confirm New Password *</Label>
          <Input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm new password"
          />
          {confirmPassword && !passwordsMatch && (
            <p className="text-xs text-danger">Passwords do not match</p>
          )}
        </div>

        <Button
          onClick={handleSave}
          disabled={!canSave || isSaving}
          className="w-full"
        >
          {isSaving ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : saved ? (
            <CheckCircle className="h-4 w-4 mr-2" />
          ) : null}
          {saved ? 'Saved!' : 'Update Credentials'}
        </Button>
      </CardContent>
    </Card>
    </motion.div>
  );
}
