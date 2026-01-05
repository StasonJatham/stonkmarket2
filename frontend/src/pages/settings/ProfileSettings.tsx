import { useState, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { User, Calendar, Loader2, CheckCircle, Camera, Upload } from 'lucide-react';
import { updateCredentials } from '@/services/api';

export function ProfileSettings() {
  const { user } = useAuth();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [newUsername, setNewUsername] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  
  // Avatar upload state
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [isUploadingAvatar, setIsUploadingAvatar] = useState(false);

  const handleSaveUsername = async () => {
    if (!newUsername.trim() || !currentPassword) return;
    
    setIsSaving(true);
    setError('');
    setSuccess('');
    
    try {
      await updateCredentials({
        current_password: currentPassword,
        new_password: currentPassword,
        new_username: newUsername,
      });
      setSuccess('Username updated successfully');
      setNewUsername('');
      setCurrentPassword('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update username');
    } finally {
      setIsSaving(false);
    }
  };

  const handleAvatarClick = () => {
    fileInputRef.current?.click();
  };

  const handleAvatarChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Validate file size (max 2MB)
    if (file.size > 2 * 1024 * 1024) {
      setError('Image must be less than 2MB');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setAvatarPreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Upload avatar
    setIsUploadingAvatar(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('avatar', file);
      
      // TODO: Implement avatar upload endpoint
      // await uploadAvatar(formData);
      
      // For now, just show the preview
      setSuccess('Avatar preview updated. Upload will be available soon.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload avatar');
      setAvatarPreview(null);
    } finally {
      setIsUploadingAvatar(false);
    }
  };

  const initials = user?.username?.slice(0, 2).toUpperCase() || 'U';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Profile</h2>
        <p className="text-sm text-muted-foreground">
          Manage your public profile information
        </p>
      </div>

      {/* Profile Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            Profile Overview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center gap-6">
            {/* Avatar with upload */}
            <div className="relative group">
              <Avatar 
                className="h-24 w-24 cursor-pointer transition-opacity group-hover:opacity-80"
                onClick={handleAvatarClick}
              >
                <AvatarImage src={avatarPreview || undefined} alt={user?.username} />
                <AvatarFallback className="text-2xl">{initials}</AvatarFallback>
              </Avatar>
              
              {/* Upload overlay */}
              <div 
                className="absolute inset-0 flex items-center justify-center rounded-full bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                onClick={handleAvatarClick}
              >
                {isUploadingAvatar ? (
                  <Loader2 className="h-6 w-6 text-white animate-spin" />
                ) : (
                  <Camera className="h-6 w-6 text-white" />
                )}
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleAvatarChange}
              />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-xl">{user?.username}</h3>
                <Badge variant="outline" className="text-xs">
                  {user?.is_admin ? 'Admin' : 'User'}
                </Badge>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Calendar className="h-4 w-4" />
                <span>Member since {new Date().getFullYear()}</span>
              </div>
              <Button 
                variant="outline" 
                size="sm" 
                className="gap-2 mt-2"
                onClick={handleAvatarClick}
              >
                <Upload className="h-4 w-4" />
                Change Avatar
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Change Username */}
      <Card>
        <CardHeader>
          <CardTitle>Change Username</CardTitle>
          <CardDescription>
            Update your display username. This will change how others see you.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {success && (
            <div className="flex items-center gap-2 text-sm text-success p-3 bg-success/10 rounded-md">
              <CheckCircle className="h-4 w-4" />
              {success}
            </div>
          )}
          {error && (
            <div className="text-sm text-destructive p-3 bg-destructive/10 rounded-md">
              {error}
            </div>
          )}
          
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label>New Username</Label>
              <Input
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="Enter new username"
              />
            </div>
            <div className="space-y-2">
              <Label>Current Password</Label>
              <Input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                placeholder="Confirm with password"
              />
            </div>
          </div>
          
          <Button 
            onClick={handleSaveUsername}
            disabled={isSaving || !newUsername.trim() || !currentPassword}
          >
            {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Save Username
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default ProfileSettings;
