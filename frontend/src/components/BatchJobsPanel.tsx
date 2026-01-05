import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getBatchJobs,
  cancelBatchJob,
  deleteBatchJob,
  type BatchJob,
  type BatchJobStatus,
} from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Clock,
  XCircle,
  Pause,
  Zap,
  Package,
  MoreHorizontal,
  Trash2,
  StopCircle,
} from 'lucide-react';
import { DataTableControls } from '@/components/ui/data-table-controls';

function getStatusBadge(status: BatchJobStatus) {
  switch (status) {
    case 'pending':
      return (
        <Badge variant="outline" className="text-muted-foreground">
          <Clock className="h-3 w-3 mr-1" />
          Pending
        </Badge>
      );
    case 'validating':
      return (
        <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          Validating
        </Badge>
      );
    case 'in_progress':
      return (
        <Badge className="bg-chart-4/20 text-chart-4 border-chart-4/30">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          Processing
        </Badge>
      );
    case 'finalizing':
      return (
        <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30">
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
          Finalizing
        </Badge>
      );
    case 'completed':
      return (
        <Badge className="bg-success/20 text-success border-success/30">
          <CheckCircle2 className="h-3 w-3 mr-1" />
          Completed
        </Badge>
      );
    case 'failed':
      return (
        <Badge className="bg-danger/20 text-danger border-danger/30">
          <XCircle className="h-3 w-3 mr-1" />
          Failed
        </Badge>
      );
    case 'expired':
      return (
        <Badge variant="outline" className="text-warning">
          <AlertCircle className="h-3 w-3 mr-1" />
          Expired
        </Badge>
      );
    case 'cancelled':
      return (
        <Badge variant="outline" className="text-muted-foreground">
          <Pause className="h-3 w-3 mr-1" />
          Cancelled
        </Badge>
      );
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function getJobTypeBadge(jobType: string) {
  switch (jobType.toLowerCase()) {
    case 'rating':
      return (
        <Badge variant="outline" className="bg-chart-1/10 text-chart-1 border-chart-1/30">
          <Zap className="h-3 w-3 mr-1" />
          Rating
        </Badge>
      );
    case 'bio':
      return (
        <Badge variant="outline" className="bg-chart-2/10 text-chart-2 border-chart-2/30">
          <Package className="h-3 w-3 mr-1" />
          Bio
        </Badge>
      );
    case 'summary':
      return (
        <Badge variant="outline" className="bg-chart-3/10 text-chart-3 border-chart-3/30">
          <Package className="h-3 w-3 mr-1" />
          Summary
        </Badge>
      );
    default:
      return <Badge variant="outline">{jobType}</Badge>;
  }
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '—';
  const date = new Date(dateStr);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatCost(cost: number | null): string {
  if (cost === null || cost === undefined) return '—';
  return `$${cost.toFixed(4)}`;
}

export function BatchJobsPanel() {
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [activeCount, setActiveCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalJobs, setTotalJobs] = useState(0);
  const activeCountRef = useRef(0);

  async function loadJobs() {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getBatchJobs(pageSize, (currentPage - 1) * pageSize, true);
      setJobs(response.jobs);
      setActiveCount(response.active_count);
      activeCountRef.current = response.active_count;
      setTotalJobs(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load batch jobs');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCancel(batchId: string) {
    setActionLoading(batchId);
    setError(null);
    try {
      await cancelBatchJob(batchId);
      await loadJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel batch job');
    } finally {
      setActionLoading(null);
    }
  }

  async function handleDelete(jobId: number, batchId: string) {
    setActionLoading(batchId);
    setError(null);
    try {
      await deleteBatchJob(jobId);
      await loadJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete batch job');
    } finally {
      setActionLoading(null);
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: loadJobs defined in component scope
  useEffect(() => {
    loadJobs();
    
    // Poll for updates if there are active jobs
    const interval = setInterval(() => {
      if (activeCountRef.current > 0) {
        loadJobs();
      }
    }, 10000); // Every 10 seconds
    
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage, pageSize]);

  useEffect(() => {
    setCurrentPage(1);
  }, [pageSize]);

  const totalPages = Math.ceil(totalJobs / pageSize);

  useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(totalJobs / pageSize));
    if (currentPage > maxPage) {
      setCurrentPage(1);
    }
  }, [currentPage, pageSize, totalJobs]);

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              AI Batch Jobs
              {activeCount > 0 && (
                <Badge className="bg-chart-4/20 text-chart-4 border-chart-4/30 ml-2">
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                  {activeCount} active
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              OpenAI Batch API jobs for cheap bulk AI processing (50% cheaper than real-time)
            </CardDescription>
          </div>
          <Button variant="outline" onClick={loadJobs} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2 bg-danger/10 text-danger p-3 rounded-lg mb-4"
            >
              <AlertCircle className="h-4 w-4" />
              <span className="flex-1 text-sm">{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <DataTableControls
          currentPage={currentPage}
          totalPages={totalPages}
          totalItems={totalJobs}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={setPageSize}
          itemName="batch jobs"
        />

        {/* Table */}
        {isLoading && jobs.length === 0 ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Package className="h-12 w-12 mx-auto mb-4 opacity-30" />
            <p>No batch jobs found</p>
            <p className="text-sm mt-1">Batch jobs are created automatically by scheduled tasks</p>
          </div>
        ) : (
          <>
            <div className="space-y-4 md:hidden">
              {jobs.map((job) => {
                const progressPct = job.total_requests > 0
                  ? Math.round((job.completed_requests / job.total_requests) * 100)
                  : 0;
                const isActive = ['pending', 'validating', 'in_progress', 'finalizing'].includes(job.status);

                return (
                  <motion.div
                    key={job.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="rounded-lg border bg-card p-4 space-y-3"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="space-y-2">
                        {getJobTypeBadge(job.job_type)}
                        {getStatusBadge(job.status)}
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            disabled={actionLoading === job.batch_id}
                          >
                            {actionLoading === job.batch_id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <MoreHorizontal className="h-4 w-4" />
                            )}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          {isActive && (
                            <DropdownMenuItem 
                              onClick={() => handleCancel(job.batch_id)}
                              className="text-destructive focus:text-destructive"
                            >
                              <StopCircle className="h-4 w-4 mr-2" />
                              Cancel Job
                            </DropdownMenuItem>
                          )}
                          {!isActive && (
                            <DropdownMenuItem 
                              onClick={() => handleDelete(job.id, job.batch_id)}
                              className="text-destructive focus:text-destructive"
                            >
                              <Trash2 className="h-4 w-4 mr-2" />
                              Delete Job
                            </DropdownMenuItem>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="space-y-2">
                      <Progress 
                        value={progressPct} 
                        className={`h-2 ${isActive ? 'animate-pulse' : ''}`}
                      />
                      <div className="text-xs text-muted-foreground">
                        {job.completed_requests}/{job.total_requests}
                        {job.failed_requests > 0 && (
                          <span className="text-danger ml-1">
                            ({job.failed_requests} failed)
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <div>
                        <p className="uppercase tracking-wide text-[10px]">Cost</p>
                        <p className="text-sm font-mono text-foreground">
                          {job.actual_cost_usd !== null 
                            ? formatCost(job.actual_cost_usd)
                            : job.estimated_cost_usd !== null 
                              ? `~${formatCost(job.estimated_cost_usd)}`
                              : '—'
                          }
                        </p>
                      </div>
                      <div>
                        <p className="uppercase tracking-wide text-[10px]">Created</p>
                        <p className="text-sm text-foreground">{formatDate(job.created_at)}</p>
                      </div>
                      <div>
                        <p className="uppercase tracking-wide text-[10px]">Completed</p>
                        <p className="text-sm text-foreground">{formatDate(job.completed_at)}</p>
                      </div>
                      <div>
                        <p className="uppercase tracking-wide text-[10px]">Requests</p>
                        <p className="text-sm text-foreground">{job.total_requests}</p>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            <div className="hidden md:block overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Progress</TableHead>
                    <TableHead>Cost</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Completed</TableHead>
                    <TableHead className="w-[50px]">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <AnimatePresence mode="popLayout">
                    {jobs.map((job) => {
                      const progressPct = job.total_requests > 0
                        ? Math.round((job.completed_requests / job.total_requests) * 100)
                        : 0;
                      const isActive = ['pending', 'validating', 'in_progress', 'finalizing'].includes(job.status);
                      
                      return (
                        <motion.tr
                          key={job.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          className="border-b"
                        >
                          <TableCell>
                            {getJobTypeBadge(job.job_type)}
                          </TableCell>
                          <TableCell>
                            {getStatusBadge(job.status)}
                          </TableCell>
                          <TableCell className="min-w-[150px]">
                            <div className="space-y-1">
                              <Progress 
                                value={progressPct} 
                                className={`h-2 ${isActive ? 'animate-pulse' : ''}`}
                              />
                              <div className="text-xs text-muted-foreground">
                                {job.completed_requests}/{job.total_requests}
                                {job.failed_requests > 0 && (
                                  <span className="text-danger ml-1">
                                    ({job.failed_requests} failed)
                                  </span>
                                )}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {job.actual_cost_usd !== null 
                              ? formatCost(job.actual_cost_usd)
                              : job.estimated_cost_usd !== null 
                                ? <span className="text-muted-foreground">~{formatCost(job.estimated_cost_usd)}</span>
                                : '—'
                            }
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(job.created_at)}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(job.completed_at)}
                          </TableCell>
                          <TableCell>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button 
                                  variant="ghost" 
                                  size="sm"
                                  disabled={actionLoading === job.batch_id}
                                >
                                  {actionLoading === job.batch_id ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <MoreHorizontal className="h-4 w-4" />
                                  )}
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                {isActive && (
                                  <DropdownMenuItem 
                                    onClick={() => handleCancel(job.batch_id)}
                                    className="text-destructive focus:text-destructive"
                                  >
                                    <StopCircle className="h-4 w-4 mr-2" />
                                    Cancel Job
                                  </DropdownMenuItem>
                                )}
                                {!isActive && (
                                  <DropdownMenuItem 
                                    onClick={() => handleDelete(job.id, job.batch_id)}
                                    className="text-destructive focus:text-destructive"
                                  >
                                    <Trash2 className="h-4 w-4 mr-2" />
                                    Delete Job
                                  </DropdownMenuItem>
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </motion.tr>
                      );
                    })}
                  </AnimatePresence>
                </TableBody>
              </Table>
            </div>
          </>
        )}
        
        {/* Info */}
        <div className="mt-4 text-xs text-muted-foreground bg-muted/30 rounded-lg p-3">
          <strong>Note:</strong> Batch jobs use OpenAI's Batch API which is 50% cheaper than real-time requests.
          Jobs are processed within 24 hours. Active jobs auto-refresh every 10 seconds.
        </div>
      </CardContent>
    </Card>
  );
}
