import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { MFASetup } from '@/components/MFASetup';
import { UserSettings } from '@/components/UserSettings';
import { Shield } from 'lucide-react';

export function SecuritySettings() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Security</h2>
        <p className="text-sm text-muted-foreground">
          Manage your password and two-factor authentication
        </p>
      </div>

      {/* Password Change - reuse existing component */}
      <UserSettings />

      {/* MFA Setup - reuse existing component */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Two-Factor Authentication
          </CardTitle>
          <CardDescription>
            Add an extra layer of security to your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <MFASetup />
        </CardContent>
      </Card>
    </div>
  );
}

export default SecuritySettings;
