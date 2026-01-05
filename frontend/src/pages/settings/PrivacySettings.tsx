import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Eye, Download, Trash2, AlertTriangle, Loader2, CheckCircle } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

export function PrivacySettings() {
  const { user, logout } = useAuth();
  
  // Privacy settings state
  const [isPublicProfile, setIsPublicProfile] = useState(false);
  const [showWatchlist, setShowWatchlist] = useState(false);
  const [showVoteHistory, setShowVoteHistory] = useState(false);
  
  // Export state
  const [isExporting, setIsExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  
  // Delete account state
  const [deleteConfirmation, setDeleteConfirmation] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  const handleExportData = async () => {
    setIsExporting(true);
    setExportSuccess(false);
    
    try {
      // TODO: Implement data export endpoint
      // const data = await exportUserData();
      // downloadJSON(data, `stonkmarket-export-${Date.now()}.json`);
      
      // Simulate export for now
      await new Promise(resolve => setTimeout(resolve, 1500));
      setExportSuccess(true);
    } catch {
      // Handle error
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (deleteConfirmation !== user?.username) return;
    
    setIsDeleting(true);
    setDeleteError('');
    
    try {
      // TODO: Implement account deletion endpoint
      // await deleteAccount();
      
      // Simulate deletion for now
      await new Promise(resolve => setTimeout(resolve, 1000));
      logout();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete account');
      setIsDeleting(false);
    }
  };

  const handlePrivacyToggle = (setting: 'profile' | 'watchlist' | 'votes', value: boolean) => {
    // Update local state immediately (optimistic)
    switch (setting) {
      case 'profile':
        setIsPublicProfile(value);
        break;
      case 'watchlist':
        setShowWatchlist(value);
        break;
      case 'votes':
        setShowVoteHistory(value);
        break;
    }
    
    // TODO: Persist to API
    // await updatePrivacySettings({ [setting]: value });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Privacy</h2>
        <p className="text-sm text-muted-foreground">
          Control your profile visibility and data
        </p>
      </div>

      {/* Profile Visibility */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Profile Visibility
          </CardTitle>
          <CardDescription>
            Control who can see your profile and activity
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Public Profile</Label>
              <p className="text-sm text-muted-foreground">
                Allow others to view your profile at /u/{user?.username}
              </p>
            </div>
            <Switch 
              checked={isPublicProfile}
              onCheckedChange={(checked) => handlePrivacyToggle('profile', checked)}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Show Watchlist</Label>
              <p className="text-sm text-muted-foreground">
                Display your watched stocks on your public profile
              </p>
            </div>
            <Switch 
              checked={showWatchlist}
              onCheckedChange={(checked) => handlePrivacyToggle('watchlist', checked)}
              disabled={!isPublicProfile}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Show Vote History</Label>
              <p className="text-sm text-muted-foreground">
                Display your dip votes on your public profile
              </p>
            </div>
            <Switch 
              checked={showVoteHistory}
              onCheckedChange={(checked) => handlePrivacyToggle('votes', checked)}
              disabled={!isPublicProfile}
            />
          </div>
        </CardContent>
      </Card>

      {/* Data Management */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Data Management
          </CardTitle>
          <CardDescription>
            Export or delete your data
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <div className="font-medium">Export Your Data</div>
              <div className="text-sm text-muted-foreground">
                Download a copy of all your data including watchlists, votes, and settings
              </div>
            </div>
            <Button 
              variant="outline" 
              className="gap-2"
              onClick={handleExportData}
              disabled={isExporting}
            >
              {isExporting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : exportSuccess ? (
                <CheckCircle className="h-4 w-4 text-success" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              {isExporting ? 'Exporting...' : exportSuccess ? 'Exported!' : 'Export'}
            </Button>
          </div>

          <div className="flex items-center justify-between p-4 border border-destructive/50 rounded-lg bg-destructive/5">
            <div>
              <div className="font-medium text-destructive">Delete Account</div>
              <div className="text-sm text-muted-foreground">
                Permanently delete your account and all associated data
              </div>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive" className="gap-2">
                  <Trash2 className="h-4 w-4" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                    Delete Account
                  </AlertDialogTitle>
                  <AlertDialogDescription className="space-y-4">
                    <p>
                      This action cannot be undone. This will permanently delete your
                      account and remove all your data from our servers including:
                    </p>
                    <ul className="list-disc list-inside space-y-1">
                      <li>Your profile and settings</li>
                      <li>All watchlists and portfolios</li>
                      <li>Vote history and preferences</li>
                      <li>API keys and connected accounts</li>
                    </ul>
                    
                    <div className="pt-4 space-y-2">
                      <Label>
                        Type <span className="font-bold text-foreground">{user?.username}</span> to confirm
                      </Label>
                      <Input
                        value={deleteConfirmation}
                        onChange={(e) => setDeleteConfirmation(e.target.value)}
                        placeholder="Enter your username"
                      />
                    </div>
                    
                    {deleteError && (
                      <p className="text-destructive text-sm">{deleteError}</p>
                    )}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel onClick={() => setDeleteConfirmation('')}>
                    Cancel
                  </AlertDialogCancel>
                  <AlertDialogAction 
                    onClick={handleDeleteAccount}
                    disabled={deleteConfirmation !== user?.username || isDeleting}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                    Yes, delete my account
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
      </Card>

      <div className="text-sm text-muted-foreground p-3 bg-muted/50 rounded-lg">
        <strong>Note:</strong> Changes to privacy settings may take a few minutes to 
        propagate across the platform.
      </div>
    </div>
  );
}

export default PrivacySettings;
