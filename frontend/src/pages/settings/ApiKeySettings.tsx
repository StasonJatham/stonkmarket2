import { UserApiKeyManager } from '@/components/UserApiKeyManager';

export function ApiKeySettings() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">API Keys</h2>
        <p className="text-sm text-muted-foreground">
          Manage API keys for programmatic access
        </p>
      </div>

      <UserApiKeyManager />
    </div>
  );
}

export default ApiKeySettings;
