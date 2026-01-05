import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Link2, Github } from 'lucide-react';
import { apiGet } from '@/lib/api-client';

// Google icon component
function GoogleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  );
}

interface OAuthProvidersResponse {
  providers: string[];
}

async function getOAuthProviders(): Promise<string[]> {
  const data = await apiGet<OAuthProvidersResponse>('/oauth/providers');
  return data.providers;
}

const PROVIDER_CONFIG: Record<string, {
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}> = {
  google: {
    name: 'Google',
    icon: GoogleIcon,
    description: 'Sign in with your Google account',
  },
  github: {
    name: 'GitHub',
    icon: Github,
    description: 'Sign in with your GitHub account',
  },
};

function handleOAuthConnect(provider: string) {
  // Redirect to OAuth flow
  window.location.href = `/api/oauth/${provider}`;
}

export function ConnectionSettings() {
  const { data: providers, isLoading, error } = useQuery({
    queryKey: ['oauth', 'providers'],
    queryFn: getOAuthProviders,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Connected Accounts</h2>
        <p className="text-sm text-muted-foreground">
          Manage your connected third-party accounts for seamless sign-in
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Link2 className="h-5 w-5" />
            OAuth Connections
          </CardTitle>
          <CardDescription>
            Connect your accounts from these providers to enable single sign-on
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : error ? (
            <div className="text-center py-4 text-muted-foreground">
              <p>Unable to load OAuth providers</p>
            </div>
          ) : providers && providers.length > 0 ? (
            <div className="space-y-4">
              {providers.map((providerKey) => {
                const config = PROVIDER_CONFIG[providerKey];
                if (!config) return null;

                const Icon = config.icon;

                return (
                  <div
                    key={providerKey}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center gap-4">
                      <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center">
                        <Icon className="h-5 w-5" />
                      </div>
                      <div>
                        <div className="font-medium">{config.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {config.description}
                        </div>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => handleOAuthConnect(providerKey)}
                    >
                      Connect
                    </Button>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Link2 className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="font-medium">No OAuth providers configured</p>
              <p className="text-sm mt-1">
                OAuth connections are not available at this time
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default ConnectionSettings;
