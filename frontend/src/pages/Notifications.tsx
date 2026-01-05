/**
 * Notification Settings Page
 * 
 * Allows users to manage notification channels (Discord, Telegram, etc.),
 * create rules for alerts, and view notification history.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, Radio, ListChecks, History } from 'lucide-react';
import { useSEO } from '@/lib/seo';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  useNotificationSummary, 
} from '@/features/notifications';
import { ChannelsPanel } from '@/components/notifications/ChannelsPanel';
import { RulesPanel } from '@/components/notifications/RulesPanel';
import { HistoryPanel } from '@/components/notifications/HistoryPanel';

export function NotificationsPage() {
  useSEO({
    title: 'Notification Settings',
    description: 'Configure price alerts, dip notifications, and portfolio monitoring.',
    noindex: true,
  });

  const [activeTab, setActiveTab] = useState<string>('channels');
  const summaryQuery = useNotificationSummary();

  return (
    <div className="container mx-auto max-w-6xl px-4 py-6 space-y-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Bell className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Notifications</h1>
            <p className="text-sm text-muted-foreground">
              Configure alerts for price movements, dips, and portfolio changes
            </p>
          </div>
        </div>

        {/* Summary Stats */}
        {summaryQuery.isLoading ? (
          <div className="flex gap-2">
            <Skeleton className="h-6 w-20" />
            <Skeleton className="h-6 w-20" />
            <Skeleton className="h-6 w-20" />
          </div>
        ) : summaryQuery.data ? (
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="gap-1.5">
              <Radio className="h-3 w-3" />
              {summaryQuery.data.active_channels} / {summaryQuery.data.total_channels} channels
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <ListChecks className="h-3 w-3" />
              {summaryQuery.data.active_rules} / {summaryQuery.data.total_rules} rules
            </Badge>
            <Badge variant="secondary" className="gap-1.5">
              <History className="h-3 w-3" />
              {summaryQuery.data.notifications_today} today
            </Badge>
          </div>
        ) : null}
      </motion.div>

      {/* Error State */}
      {summaryQuery.error && (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load notification data: {summaryQuery.error.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-flex">
          <TabsTrigger value="channels" className="gap-2">
            <Radio className="h-4 w-4" />
            <span className="hidden sm:inline">Channels</span>
          </TabsTrigger>
          <TabsTrigger value="rules" className="gap-2">
            <ListChecks className="h-4 w-4" />
            <span className="hidden sm:inline">Rules</span>
          </TabsTrigger>
          <TabsTrigger value="history" className="gap-2">
            <History className="h-4 w-4" />
            <span className="hidden sm:inline">History</span>
          </TabsTrigger>
        </TabsList>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            <TabsContent value="channels" className="mt-6">
              <ChannelsPanel />
            </TabsContent>

            <TabsContent value="rules" className="mt-6">
              <RulesPanel />
            </TabsContent>

            <TabsContent value="history" className="mt-6">
              <HistoryPanel />
            </TabsContent>
          </motion.div>
        </AnimatePresence>
      </Tabs>
    </div>
  );
}

export default NotificationsPage;
