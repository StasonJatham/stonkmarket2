import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Bell, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';

export function NotificationsSettings() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Notifications</h2>
        <p className="text-sm text-muted-foreground">
          Manage your alert preferences and channels
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Center
          </CardTitle>
          <CardDescription>
            Configure your alerts and notification channels
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-4">
            Your notification preferences are managed in the Notifications page where you can:
          </p>
          <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 mb-6">
            <li>Set up Telegram bot integration for instant alerts</li>
            <li>Configure price alert rules for specific stocks</li>
            <li>Manage dip detection notifications</li>
            <li>Control email notification preferences</li>
          </ul>
          <Button asChild className="gap-2">
            <Link to="/notifications">
              <ExternalLink className="h-4 w-4" />
              Go to Notifications
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default NotificationsSettings;
