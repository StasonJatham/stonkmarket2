import { useCallback, useEffect, useState } from 'react';
import {
  getCeleryBroker,
  getCeleryQueues,
  getCeleryWorkers,
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
import { AlertCircle, Database, RefreshCw, Server, Layers } from 'lucide-react';

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

function formatLoadAvg(loadavg: CeleryWorkerInfo['loadavg']): string {
  if (!loadavg) return '—';
  if (Array.isArray(loadavg)) return loadavg.join(', ');
  return String(loadavg);
}

export function CeleryMonitorPanel() {
  const [workers, setWorkers] = useState<CeleryWorkersResponse>({});
  const [queues, setQueues] = useState<CeleryQueueInfo[]>([]);
  const [broker, setBroker] = useState<CeleryBrokerInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    const [workersResult, queuesResult, brokerResult] = await Promise.allSettled([
      getCeleryWorkers(),
      getCeleryQueues(),
      getCeleryBroker(),
    ]);

    if (workersResult.status === 'fulfilled') {
      setWorkers(workersResult.value);
    } else {
      setError('Failed to load Celery workers');
    }

    if (queuesResult.status === 'fulfilled') {
      setQueues(queuesResult.value);
    } else {
      setError('Failed to load Celery queues');
    }

    if (brokerResult.status === 'fulfilled') {
      setBroker(brokerResult.value);
    } else {
      setError('Failed to load Celery broker info');
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 15000);
    return () => clearInterval(interval);
  }, [loadStats]);

  const workerEntries = Object.entries(workers);
  const brokerPayload = broker ? JSON.stringify(broker, null, 2) : '';

  return (
    <Card className="mt-6">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Celery Monitor
            </CardTitle>
            <CardDescription>
              Worker status, queue depth, and broker connectivity (via Flower)
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={loadStats} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <div className="flex items-center gap-2 bg-danger/10 text-danger p-3 rounded-lg">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {isLoading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : (
          <>
            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Server className="h-4 w-4 text-muted-foreground" />
                Workers
                <Badge variant="outline" className="text-muted-foreground">
                  {workerEntries.length}
                </Badge>
              </div>
              {workerEntries.length === 0 ? (
                <p className="text-sm text-muted-foreground">No workers reported by Flower.</p>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Worker</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Active</TableHead>
                        <TableHead>Processed</TableHead>
                        <TableHead>Concurrency</TableHead>
                        <TableHead>Load</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {workerEntries.map(([name, info]) => {
                        const active = getActiveCount(info);
                        const processed = getProcessedCount(info);
                        const concurrency = getConcurrency(info);
                        return (
                          <TableRow key={name}>
                            <TableCell className="font-medium">{name}</TableCell>
                            <TableCell>
                              <Badge variant="outline">{info.status || 'unknown'}</Badge>
                            </TableCell>
                            <TableCell>{active ?? '—'}</TableCell>
                            <TableCell>{processed ?? '—'}</TableCell>
                            <TableCell>{concurrency ?? '—'}</TableCell>
                            <TableCell>{formatLoadAvg(info.loadavg)}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Layers className="h-4 w-4 text-muted-foreground" />
                Queues
              </div>
              {queues.length === 0 ? (
                <p className="text-sm text-muted-foreground">No queue data reported.</p>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Queue</TableHead>
                        <TableHead>Messages</TableHead>
                        <TableHead>Consumers</TableHead>
                        <TableHead>State</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {queues.map((queue, idx) => (
                        <TableRow key={`${queue.name ?? 'queue'}-${idx}`}>
                          <TableCell className="font-medium">{queue.name ?? '—'}</TableCell>
                          <TableCell>{queue.messages ?? '—'}</TableCell>
                          <TableCell>{queue.consumers ?? '—'}</TableCell>
                          <TableCell>{queue.state ?? '—'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center gap-2 mb-3 text-sm font-medium">
                <Database className="h-4 w-4 text-muted-foreground" />
                Broker
              </div>
              {brokerPayload ? (
                <pre className="bg-muted/30 rounded-lg p-3 text-xs overflow-auto max-h-48">{brokerPayload}</pre>
              ) : (
                <p className="text-sm text-muted-foreground">No broker data reported.</p>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
