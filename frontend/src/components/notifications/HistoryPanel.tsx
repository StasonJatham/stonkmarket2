/**
 * History Panel - View notification delivery history.
 */

import { useState } from 'react';
import { format, formatDistanceToNow } from 'date-fns';
import { 
  History, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  TrendingDown,
  Zap,
  BarChart3,
  Brain,
  Briefcase,
  Eye,
  Target,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  useNotificationHistory,
  useTriggerTypes,
  type NotificationStatus,
} from '@/features/notifications';

const STATUS_CONFIG: Record<NotificationStatus, { icon: typeof CheckCircle; className: string; label: string }> = {
  sent: { icon: CheckCircle, className: 'text-green-600', label: 'Sent' },
  pending: { icon: Clock, className: 'text-yellow-600', label: 'Pending' },
  failed: { icon: XCircle, className: 'text-red-600', label: 'Failed' },
  skipped: { icon: AlertCircle, className: 'text-gray-500', label: 'Skipped' },
};

const CATEGORY_ICONS: Record<string, typeof TrendingDown> = {
  'Price & Dip': TrendingDown,
  'Signals': Zap,
  'Fundamentals': BarChart3,
  'AI Analysis': Brain,
  'Portfolio': Briefcase,
  'Watchlist': Eye,
};

export function HistoryPanel() {
  const [page, setPage] = useState(1);
  const pageSize = 20;
  
  const historyQuery = useNotificationHistory({ page, pageSize });
  const triggerTypesQuery = useTriggerTypes();

  const handleRefresh = () => {
    historyQuery.refetch();
  };

  if (historyQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-9 w-24" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const history = historyQuery.data;
  const notifications = history?.notifications ?? [];
  const total = history?.total ?? 0;
  const totalPages = Math.ceil(total / pageSize);

  const getTriggerInfo = (triggerType: string) => {
    return triggerTypesQuery.data?.find((t) => t.type === triggerType);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Notification History
            </CardTitle>
            <CardDescription>
              Recent notification deliveries ({total} total)
            </CardDescription>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleRefresh}
            disabled={historyQuery.isFetching}
            className="gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${historyQuery.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {historyQuery.error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>
              Failed to load history: {historyQuery.error.message}
            </AlertDescription>
          </Alert>
        )}

        {notifications.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <History className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="font-medium">No notifications yet</p>
            <p className="text-sm mt-1">Notifications will appear here once rules trigger</p>
          </div>
        ) : (
          <>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[140px]">Time</TableHead>
                    <TableHead>Notification</TableHead>
                    <TableHead>Channel</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {notifications.map((notification) => {
                    const triggerInfo = getTriggerInfo(notification.trigger_type);
                    const CategoryIcon = CATEGORY_ICONS[triggerInfo?.category ?? ''] ?? Target;
                    const statusConfig = STATUS_CONFIG[notification.status] ?? STATUS_CONFIG.pending;
                    const StatusIcon = statusConfig.icon;
                    
                    return (
                      <TableRow key={notification.id}>
                        <TableCell className="text-sm">
                          <Tooltip>
                            <TooltipTrigger className="text-left">
                              <div>
                                <div className="font-medium">
                                  {formatDistanceToNow(new Date(notification.triggered_at), { addSuffix: true })}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {format(new Date(notification.triggered_at), 'MMM d, HH:mm')}
                                </div>
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              {format(new Date(notification.triggered_at), 'PPpp')}
                            </TooltipContent>
                          </Tooltip>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-start gap-2">
                            <CategoryIcon className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
                            <div className="min-w-0">
                              <div className="font-medium truncate">
                                {notification.title}
                              </div>
                              <div className="text-sm text-muted-foreground line-clamp-1">
                                {notification.body}
                              </div>
                              {notification.trigger_symbol && (
                                <Badge variant="outline" className="mt-1">
                                  {notification.trigger_symbol}
                                </Badge>
                              )}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            {notification.channel_name ?? (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Tooltip>
                            <TooltipTrigger>
                              <Badge 
                                variant={notification.status === 'sent' ? 'default' : 'outline'}
                                className={`gap-1 ${statusConfig.className}`}
                              >
                                <StatusIcon className="h-3 w-3" />
                                {statusConfig.label}
                              </Badge>
                            </TooltipTrigger>
                            {notification.error_message && (
                              <TooltipContent>
                                {notification.error_message}
                              </TooltipContent>
                            )}
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-4">
                <div className="text-sm text-muted-foreground">
                  Page {page} of {totalPages}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page - 1)}
                    disabled={page <= 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page + 1)}
                    disabled={page >= totalPages}
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
