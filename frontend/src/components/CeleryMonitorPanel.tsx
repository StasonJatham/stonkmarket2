import { useCallback, useEffect, useState } from 'react';
import {
  getCelerySnapshot,
  type CeleryBrokerInfo,
  type CeleryQueueInfo,
  type CeleryWorkerInfo,
  type CeleryWorkersResponse,
} from '@/services/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  AlertCircle,
  CheckCircle,
  Database,
  Layers,
  RefreshCw,
  Server,
  XCircle,
  Activity,
} from 'lucide-react';

function getActiveCount(info: CeleryWorkerInfo): number | null {
  if (Array.isArray(info.active)) return info.active.length;
  if (typeof info.active === 'number') return info.active;
  return null;
}

function getProcessedCount(info: CeleryWorkerInfo): number | null {
  if (typeof info.processed === 'number') return info.processed;
  if (info.total && typeof info.total === 'object') {
    return Object.values(info.total).reduce((sum, value) => {
      if (typeof value === 'number') return sum + value;
      return sum;
    }, 0);
  }
  return null;
}

function getConcurrency(info: CeleryWorkerInfo): number | null {
  if (!info.pool || typeof info.pool !== 'object') return null;
  const pool = info.pool as Record<string, unknown>;
  const value = pool['max-concurrency'] ?? pool['max_concurrency'] ?? pool['maxConcurrency'];
  return typeof value === 'number' ? value : null;
}

function formatUptime(seconds?: number): string {
  if (!seconds) return '—';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 24) {
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
  }
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

interface BrokerCardProps {
  broker: CeleryBrokerInfo | null;
}

function BrokerCard({ broker }: BrokerCardProps) {
  if (!broker) {
    return (
      <div className="text-sm text-muted-foreground">No broker data available.</div>
    );
  }

  const isConnected = broker.connected === true;
  const transport = (broker.transport as string) || 'unknown';
  const redisVersion = (broker.redis_version as string) || '';
  const clients = (broker.connected_clients as number) || 0;
  const memory = (broker.used_memory_human as string) || '';
  const uptime = broker.uptime_in_seconds as number | undefined;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground">Status</span>
        <span className="flex items-center gap-1.5 mt-1">
          {isConnected ? (
            <>
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Connected</span>
            </>
          ) : (
            <>
              <XCircle className="h-4 w-4 text-destructive" />
              <span className="text-sm font-medium">Disconnected</span>
            </>
          )}
        </span>
      </div>
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground">Transport</span>
        <span className="text-sm font-medium mt-1 capitalize">{transport}</span>
      </div>
      {redisVersion && (
        <div className="flex flex-col">
          <span className="text-xs text-muted-foreground">Version</span>
          <span className="text-sm font-medium mt-1">{redisVersion}</span>
        </div>
      )}
      <div className="flex flex-col">
        <span className="text-xs text-muted-foreground">Clients</span>
        <span className="text-sm font-medium mt-1">{clients}</span>
      </div>
      {memory && (
        <div className="flex flex-col">
          <span className="text-xs text-muted-foreground">Memory</span>
          <span className="text-sm font-medium mt-1">{memory}</span>
        </div>
      )}
      {uptime !== undefined && (
        <div className="flex flex-col">
          <span className="text-xs text-muted-foreground">Uptime</span>
          <span className="text-sm font-medium mt-1">{formatUptime(uptime)}</span>
        </div>
      )}
    </div>
  );
}

export function CeleryMonitorPanel() {
  const [workers, setWorkers] = useState<CeleryWorkersResponse>({});
  const [queues, setQueues] = useState<CeleryQueueInfo[]>([]);
  const [broker, setBroker] = useState<CeleryBrokerInfo | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStats = useCallback(async (showRefreshIndicator = false) => {
    if (showRefreshIndicator) {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const snapshot = await getCelerySnapshot(10); // Only need minimal tasks for health check
      setWorkers(snapshot.workers);
      setQueues(snapshot.queues);
      setBroker(snapshot.broker);
    } catch (err) {
      console.error('Celery snapshot error:', err);
      setError('Failed to connect to Celery monitoring');
    }

    setIsInitialLoad(false);
    setIsRefreshing(false);
  }, []);

  useEffect(() => {
    const initialTimeout = setTimeout(() => loadStats(), 0);
    const interval = setInterval(() => loadStats(false), 15000);
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [loadStats]);

  const handleManualRefresh = useCallback(() => {
    loadStats(true);
  }, [loadStats]);

  const workerEntries = Object.entries(workers);
  const totalProcessed = workerEntries.reduce((sum, [, info]) => {
    const count = getProcessedCount(info);
    return sum + (count || 0);
  }, 0);

  return (
    <Card className="mt-6">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              System Health
            </CardTitle>
            <CardDescription>
              Celery workers, queues, and broker status
            </CardDescription>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5">
                <Server className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">{workerEntries.length}</span>
                <span className="text-muted-foreground">workers</span>
              </span>
              <span className="flex items-center gap-1.5">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">{totalProcessed}</span>
                <span className="text-muted-foreground">processed</span>
              </span>
            </div>
            <Button variant="outline" size="sm" onClick={handleManualRefresh} disabled={isRefreshing}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <div className="flex items-center gap-2 bg-destructive/10 text-destructive p-3 rounded-lg">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {isInitialLoad ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : (
          <>
            {/* Workers Section */}
            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Server className="h-4 w-4 text-muted-foreground" />
                Workers
                <Badge variant="outline" className="text-muted-foreground">
                  {workerEntries.length}
                </Badge>
              </div>
              {workerEntries.length === 0 ? (
                <div className="flex items-center gap-2 p-4 bg-muted/30 rounded-lg">
                  <AlertCircle className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">
                    No workers connected. Check that Celery workers are running.
                  </span>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Worker</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead className="text-right">Active</TableHead>
                        <TableHead className="text-right">Processed</TableHead>
                        <TableHead className="text-right">Concurrency</TableHead>
                        <TableHead className="text-right">Uptime</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {workerEntries.map(([name, info]) => {
                        const active = getActiveCount(info);
                        const processed = getProcessedCount(info);
                        const concurrency = getConcurrency(info);
                        const uptime = info.uptime;
                        return (
                          <TableRow key={name}>
                            <TableCell className="font-medium font-mono text-xs">
                              {name.split('@')[0]}
                              <span className="text-muted-foreground">@{name.split('@')[1]?.slice(0, 8)}</span>
                            </TableCell>
                            <TableCell>
                              <Badge variant="default" className="bg-green-600">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                online
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right tabular-nums">
                              {active !== null ? (
                                <span className={active > 0 ? 'text-amber-500 font-medium' : ''}>
                                  {active}
                                </span>
                              ) : '—'}
                            </TableCell>
                            <TableCell className="text-right tabular-nums">{processed ?? '—'}</TableCell>
                            <TableCell className="text-right tabular-nums">{concurrency ?? '—'}</TableCell>
                            <TableCell className="text-right text-muted-foreground">
                              {formatUptime(uptime)}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>

            {/* Queues Section */}
            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Layers className="h-4 w-4 text-muted-foreground" />
                Queues
                <Badge variant="outline" className="text-muted-foreground">
                  {queues.length}
                </Badge>
              </div>
              {queues.length === 0 ? (
                <div className="flex items-center gap-2 p-4 bg-muted/30 rounded-lg">
                  <span className="text-sm text-muted-foreground">No queue data available.</span>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Queue</TableHead>
                        <TableHead className="text-right">Messages</TableHead>
                        <TableHead className="text-right">Consumers</TableHead>
                        <TableHead>State</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {queues.map((queue, idx) => (
                        <TableRow key={`${queue.name ?? 'queue'}-${idx}`}>
                          <TableCell className="font-medium">{queue.name ?? '—'}</TableCell>
                          <TableCell className="text-right tabular-nums">
                            {queue.messages !== undefined ? (
                              <span className={queue.messages > 0 ? 'text-amber-500 font-medium' : ''}>
                                {queue.messages}
                              </span>
                            ) : '—'}
                          </TableCell>
                          <TableCell className="text-right tabular-nums">{queue.consumers ?? '—'}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{queue.state ?? 'unknown'}</Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>

            {/* Broker Section */}
            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Database className="h-4 w-4 text-muted-foreground" />
                Broker
              </div>
              <div className="p-4 bg-muted/30 rounded-lg">
                <BrokerCard broker={broker} />
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
