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
  Shield, 
  ShieldCheck, 
  ShieldOff, 
  Loader2, 
  Copy, 
  CheckCircle,
  Key,
  AlertTriangle
} from 'lucide-react';
import { 
  getMFAStatus, 
  setupMFA, 
  verifyMFA, 
  disableMFA, 
  regenerateBackupCodes,
  type MFAStatus,
  type MFASetupResponse 
} from '@/services/api';

interface MFASetupProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export function MFASetup({ onError, onSuccess }: MFASetupProps) {
  const [status, setStatus] = useState<MFAStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  // Setup flow
  const [setupData, setSetupData] = useState<MFASetupResponse | null>(null);
  const [isSettingUp, setIsSettingUp] = useState(false);
  const [verifyCode, setVerifyCode] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [backupCodes, setBackupCodes] = useState<string[] | null>(null);
  
  // Disable flow
  const [disableDialogOpen, setDisableDialogOpen] = useState(false);
  const [disableCode, setDisableCode] = useState('');
  const [isDisabling, setIsDisabling] = useState(false);
  
  // Regenerate backup codes
  const [regenDialogOpen, setRegenDialogOpen] = useState(false);
  const [regenCode, setRegenCode] = useState('');
  const [isRegenerating, setIsRegenerating] = useState(false);
  
  const [copiedSecret, setCopiedSecret] = useState(false);
  const [copiedCodes, setCopiedCodes] = useState(false);

  const loadStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getMFAStatus();
      setStatus(data);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to load MFA status');
    } finally {
      setIsLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  async function handleStartSetup() {
    setIsSettingUp(true);
    try {
      const data = await setupMFA();
      setSetupData(data);
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Failed to start MFA setup');
    } finally {
      setIsSettingUp(false);
    }
  }

  async function handleVerify() {
    if (verifyCode.length !== 6) return;
    
    setIsVerifying(true);
    try {
      const result = await verifyMFA(verifyCode);
      setBackupCodes(result.backup_codes);
      await loadStatus();
      onSuccess?.('MFA enabled successfully!');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Invalid verification code');
    } finally {
      setIsVerifying(false);
    }
  }

  async function handleDisable() {
    if (disableCode.length !== 6) return;
    
    setIsDisabling(true);
    try {
      await disableMFA(disableCode);
      await loadStatus();
      setDisableDialogOpen(false);
      setDisableCode('');
      setSetupData(null);
      setBackupCodes(null);
      onSuccess?.('MFA disabled');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Invalid code');
    } finally {
      setIsDisabling(false);
    }
  }

  async function handleRegenerate() {
    if (regenCode.length !== 6) return;
    
    setIsRegenerating(true);
    try {
      const result = await regenerateBackupCodes(regenCode);
      setBackupCodes(result.backup_codes);
      setRegenDialogOpen(false);
      setRegenCode('');
      await loadStatus();
      onSuccess?.('Backup codes regenerated');
    } catch (err) {
      onError?.(err instanceof Error ? err.message : 'Invalid code');
    } finally {
      setIsRegenerating(false);
    }
  }

  function copyToClipboard(text: string, type: 'secret' | 'codes') {
    navigator.clipboard.writeText(text);
    if (type === 'secret') {
      setCopiedSecret(true);
      setTimeout(() => setCopiedSecret(false), 2000);
    } else {
      setCopiedCodes(true);
      setTimeout(() => setCopiedCodes(false), 2000);
    }
  }

  function handleComplete() {
    setSetupData(null);
    setBackupCodes(null);
    setVerifyCode('');
  }

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

  // Show backup codes after successful setup
  if (backupCodes) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-success">
              <CheckCircle className="h-5 w-5" />
              MFA Enabled Successfully!
            </CardTitle>
            <CardDescription>
              Save these backup codes in a secure location. Each code can only be used once.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-2 p-4 bg-muted rounded-lg font-mono text-sm">
              {backupCodes.map((code, i) => (
                <motion.div 
                  key={i} 
                  className="p-2 bg-background rounded text-center"
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  {code}
                </motion.div>
              ))}
            </div>
          
            <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => copyToClipboard(backupCodes.join('\n'), 'codes')}
              className="flex-1"
            >
              {copiedCodes ? (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4 mr-2" />
                  Copy All Codes
                </>
              )}
            </Button>
            <Button onClick={handleComplete} className="flex-1">
              Done
            </Button>
          </div>
          
          <div className="flex items-center gap-2 p-3 bg-warning/10 text-warning rounded-lg text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>These codes will not be shown again. Store them safely!</span>
          </div>
        </CardContent>
      </Card>
      </motion.div>
    );
  }

  // Setup flow in progress
  if (setupData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Set Up MFA
          </CardTitle>
          <CardDescription>
            Scan the QR code with your authenticator app (Google Authenticator, Authy, etc.)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* QR Code */}
          <div className="flex justify-center">
            <div className="p-4 bg-white rounded-lg">
              <img 
                src={`data:image/png;base64,${setupData.qr_code_base64}`} 
                alt="MFA QR Code"
                className="w-48 h-48"
              />
            </div>
          </div>
          
          {/* Manual entry */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">
              Can't scan? Enter this code manually:
            </Label>
            <div className="flex gap-2">
              <Input
                value={setupData.secret}
                readOnly
                className="font-mono text-xs"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => copyToClipboard(setupData.secret, 'secret')}
              >
                {copiedSecret ? <CheckCircle className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
          </div>
          
          {/* Verification */}
          <div className="space-y-2">
            <Label>Enter the 6-digit code from your app</Label>
            <div className="flex gap-2">
              <Input
                value={verifyCode}
                onChange={(e) => setVerifyCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="000000"
                className="font-mono text-center text-lg tracking-widest"
                maxLength={6}
              />
              <Button
                onClick={handleVerify}
                disabled={verifyCode.length !== 6 || isVerifying}
              >
                {isVerifying ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Verify'
                )}
              </Button>
            </div>
          </div>
          
          <Button variant="ghost" onClick={() => setSetupData(null)} className="w-full">
            Cancel Setup
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Normal status view
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {status?.enabled ? (
            <ShieldCheck className="h-5 w-5 text-success" />
          ) : (
            <ShieldOff className="h-5 w-5 text-muted-foreground" />
          )}
          Two-Factor Authentication
        </CardTitle>
        <CardDescription>
          Add an extra layer of security to your account
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
          <div className="flex items-center gap-3">
            <Shield className="h-8 w-8 text-muted-foreground" />
            <div>
              <div className="font-medium">MFA Status</div>
              <div className="text-sm text-muted-foreground">
                {status?.enabled 
                  ? `${status.backup_codes_remaining || 0} backup codes remaining`
                  : 'Not configured'
                }
              </div>
            </div>
          </div>
          <Badge variant={status?.enabled ? 'default' : 'secondary'}>
            {status?.enabled ? 'Enabled' : 'Disabled'}
          </Badge>
        </div>

        {status?.enabled ? (
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setRegenDialogOpen(true)}
              className="flex-1"
            >
              <Key className="h-4 w-4 mr-2" />
              New Backup Codes
            </Button>
            <Button
              variant="destructive"
              onClick={() => setDisableDialogOpen(true)}
              className="flex-1"
            >
              <ShieldOff className="h-4 w-4 mr-2" />
              Disable MFA
            </Button>
          </div>
        ) : (
          <Button onClick={handleStartSetup} disabled={isSettingUp} className="w-full">
            {isSettingUp && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            <Shield className="h-4 w-4 mr-2" />
            Enable MFA
          </Button>
        )}
      </CardContent>

      {/* Disable Dialog */}
      <Dialog open={disableDialogOpen} onOpenChange={setDisableDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Disable MFA</DialogTitle>
            <DialogDescription>
              Enter your current MFA code or a backup code to disable two-factor authentication.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label>Verification Code</Label>
            <Input
              value={disableCode}
              onChange={(e) => setDisableCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              className="font-mono text-center text-lg tracking-widest mt-2"
              maxLength={6}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDisableDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDisable}
              disabled={disableCode.length !== 6 || isDisabling}
            >
              {isDisabling && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Disable MFA
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Regenerate Codes Dialog */}
      <Dialog open={regenDialogOpen} onOpenChange={setRegenDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Regenerate Backup Codes</DialogTitle>
            <DialogDescription>
              This will invalidate all existing backup codes and generate new ones.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label>Enter your MFA code</Label>
            <Input
              value={regenCode}
              onChange={(e) => setRegenCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="000000"
              className="font-mono text-center text-lg tracking-widest mt-2"
              maxLength={6}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRegenDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleRegenerate}
              disabled={regenCode.length !== 6 || isRegenerating}
            >
              {isRegenerating && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Generate New Codes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
