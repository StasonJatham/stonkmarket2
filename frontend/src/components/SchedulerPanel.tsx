import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  getCronJobs,
  getCronLogs,
  updateCronJob,
  runCronJobNow,
  getBatchJobs,
  cancelBatchJob,
  deleteBatchJob,
  type CronJob,
  type CronLogEntry,
  type BatchJob,
} from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Play,
  RefreshCw,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  Search,
  Settings,
  Calendar,
  Filter,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  X,
  MoreHorizontal,
  Workflow,
  CircleDot,
  Bot,
  Timer,
} from 'lucide-react';
import { CronBuilder, validateCron } from '@/components/CronBuilder';
import { CeleryMonitorPanel } from '@/components/CeleryMonitorPanel';
import { cn } from '@/lib/utils';

// =============================================================================
// UNIFIED LOG ENTRY TYPE
// =============================================================================
// Combines cron job logs and batch job logs into a unified format

interface UnifiedLogEntry {
  id: string;
  name: string;
  type: 'cron' | 'batch';
  status: string;
  message: string | null;
  created_at: string;
  duration_ms?: number | null;
  // Batch-specific fields
  batch_id?: string;
  job_type?: string;
  total_requests?: number;
  completed_requests?: number;
  failed_requests?: number;
  estimated_cost_usd?: number | null;
  actual_cost_usd?: number | null;
  completed_at?: string | null;
}

// =============================================================================
// EXPANDABLE LOG ROW COMPONENT
// =============================================================================

interface LogRowProps {
  log: UnifiedLogEntry;
  formatDate: (date: string) => string;
  getStatusBadge: (status: string) => React.ReactNode;
  onCancelBatch?: (batchId: string) => Promise<void>;
  onDeleteBatch?: (jobId: number) => Promise<void>;
}

function LogRow({ log, formatDate, getStatusBadge, onCancelBatch, onDeleteBatch }: LogRowProps) {
  const [expanded, setExpanded] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const formatDuration = (ms?: number | null): string => {
    if (!ms) return '—';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    const mins = Math.floor(ms / 60000);
    const secs = Math.floor((ms % 60000) / 1000);
    return `${mins}m ${secs}s`;
  };

  const handleCancel = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!log.batch_id || !onCancelBatch) return;
    setActionLoading(true);
    try {
      await onCancelBatch(log.batch_id);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!log.id || !onDeleteBatch) return;
    const jobId = parseInt(log.id.replace('batch-', ''));
    if (isNaN(jobId)) return;
    setActionLoading(true);
    try {
      await onDeleteBatch(jobId);
    } finally {
      setActionLoading(false);
    }
  };

  const isActive = log.type === 'batch' && 
    ['pending', 'validating', 'in_progress', 'finalizing'].includes(log.status);
  const canDelete = log.type === 'batch' && 
    ['completed', 'failed', 'cancelled', 'expired'].includes(log.status);

  return (
    <>
      <TableRow 
        className={cn("cursor-pointer hover:bg-muted/50", expanded && "bg-muted/30")}
        onClick={() => setExpanded(!expanded)}
      >
        <TableCell className="font-medium">
          <div className="flex items-center gap-2">
            <ChevronDown className={cn(
              "h-4 w-4 text-muted-foreground transition-transform",
              expanded && "rotate-180"
            )} />
            <span>{log.name}</span>
            {log.type === 'batch' && (
              <Badge variant="outline" className="text-xs">
                <Bot className="h-3 w-3 mr-1" />
                AI
              </Badge>
            )}
          </div>
        </TableCell>
        <TableCell className="text-muted-foreground whitespace-nowrap">
          {formatDate(log.created_at)}
        </TableCell>
        <TableCell>{getStatusBadge(log.status)}</TableCell>
        <TableCell className="text-right tabular-nums text-sm text-muted-foreground">
          {log.type === 'batch' && log.total_requests ? (
            <span>{log.completed_requests}/{log.total_requests}</span>
          ) : (
            formatDuration(log.duration_ms)
          )}
        </TableCell>
        <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
          {log.message || (log.type === 'batch' ? log.job_type : '—')}
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="bg-muted/20 hover:bg-muted/20">
          <TableCell colSpan={5} className="p-4">
            <div className="space-y-3">
              {/* Full message */}
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">Full Message</div>
                <div className="text-sm bg-background/50 p-3 rounded-md font-mono whitespace-pre-wrap break-all max-h-48 overflow-auto">
                  {log.message || 'No message'}
                </div>
              </div>
              
              {/* Stats row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-xs text-muted-foreground">Type</div>
                  <div className="font-medium capitalize">{log.type === 'batch' ? 'AI Batch' : 'Cron Job'}</div>
                </div>
                {log.duration_ms && (
                  <div>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <Timer className="h-3 w-3" /> Duration
                    </div>
                    <div className="font-medium">{formatDuration(log.duration_ms)}</div>
                  </div>
                )}
                {log.type === 'batch' && (
                  <>
                    <div>
                      <div className="text-xs text-muted-foreground">Requests</div>
                      <div className="font-medium">
                        {log.completed_requests}/{log.total_requests}
                        {log.failed_requests ? ` (${log.failed_requests} failed)` : ''}
                      </div>
                    </div>
                    {log.actual_cost_usd && (
                      <div>
                        <div className="text-xs text-muted-foreground">Cost</div>
                        <div className="font-medium">${log.actual_cost_usd.toFixed(4)}</div>
                      </div>
                    )}
                    {log.batch_id && (
                      <div className="col-span-2">
                        <div className="text-xs text-muted-foreground">Batch ID</div>
                        <div className="font-mono text-xs">{log.batch_id}</div>
                      </div>
                    )}
                  </>
                )}
              </div>
              
              {/* Batch job actions */}
              {log.type === 'batch' && (isActive || canDelete) && (
                <div className="flex gap-2 pt-2 border-t">
                  {isActive && onCancelBatch && (
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={handleCancel}
                      disabled={actionLoading}
                    >
                      {actionLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <XCircle className="h-4 w-4 mr-2" />
                      )}
                      Cancel Job
                    </Button>
                  )}
                  {canDelete && onDeleteBatch && (
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="text-destructive hover:text-destructive"
                      onClick={handleDelete}
                      disabled={actionLoading}
                    >
                      {actionLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <X className="h-4 w-4 mr-2" />
                      )}
                      Delete
                    </Button>
                  )}
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

// =============================================================================
// PIPELINE DISPLAY COMPONENT
// =============================================================================

interface PipelineDisplayProps {
  orchestrator: CronJob;
  steps: CronJob[];
  runningJob: string | null;
  onRunJob: (name: string) => void;
  onEditSchedule: (job: CronJob) => void;
}

function PipelineDisplay({
  orchestrator,
  steps,
  runningJob,
  onRunJob,
  onEditSchedule,
}: PipelineDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Sort steps by pipeline_step
  const sortedSteps = (() =>
    [...steps].sort((a, b) => (a.pipeline_step || 0) - (b.pipeline_step || 0))
  )();

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }

  function getStepStatusIcon(job: CronJob) {
    if (runningJob === job.name) {
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    }
    if (job.last_status === 'success' || job.last_status === 'ok') {
      return <CheckCircle className="h-4 w-4 text-success" />;
    }
    if (job.last_status === 'error') {
      return <XCircle className="h-4 w-4 text-danger" />;
    }
    return <CircleDot className="h-4 w-4 text-muted-foreground" />;
  }

  return (
    <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Workflow className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle className="text-lg flex items-center gap-2">
                  Market Close Pipeline
                  <Badge variant="outline" className="text-xs font-normal">
                    {steps.length} steps
                  </Badge>
                </CardTitle>
                <CardDescription className="text-sm">
                  Runs daily after market close • Each step waits for the previous to complete
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-xs">
                {orchestrator.cron}
              </Badge>
              {orchestrator.next_run && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="secondary" className="text-xs">
                      <Clock className="h-3 w-3 mr-1" />
                      {formatDate(orchestrator.next_run)}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>Next pipeline run</TooltipContent>
                </Tooltip>
              )}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onClick={() => onRunJob(orchestrator.name)}
                    disabled={runningJob === orchestrator.name}
                  >
                    {runningJob === orchestrator.name ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Run full pipeline
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onEditSchedule(orchestrator)}>
                    <Settings className="h-4 w-4 mr-2" />
                    Edit schedule
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <ChevronDown
                    className={cn(
                      'h-4 w-4 transition-transform',
                      isExpanded ? '' : '-rotate-90'
                    )}
                  />
                </Button>
              </CollapsibleTrigger>
            </div>
          </div>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="pt-4">
            {/* Pipeline visualization */}
            <div className="relative">
              {/* Connection line */}
              <div className="absolute left-[19px] top-8 bottom-8 w-0.5 bg-border" />

              {/* Steps */}
              <div className="space-y-3">
                {sortedSteps.map((step, index) => (
                  <motion.div
                    key={step.name}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="relative flex items-center gap-4 pl-10"
                  >
                    {/* Step indicator */}
                    <div className="absolute left-0 flex items-center justify-center w-10">
                      <div
                        className={cn(
                          'w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium border-2 bg-background',
                          step.last_status === 'error'
                            ? 'border-danger text-danger'
                            : step.last_status === 'success' || step.last_status === 'ok'
                            ? 'border-success text-success'
                            : 'border-muted-foreground/30 text-muted-foreground'
                        )}
                      >
                        {step.pipeline_step}
                      </div>
                    </div>

                    {/* Step content */}
                    <div className="flex-1 flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors">
                      <div className="flex items-center gap-3">
                        {getStepStatusIcon(step)}
                        <div>
                          <div className="font-medium text-sm">{step.name}</div>
                          {step.description && (
                            <div className="text-xs text-muted-foreground line-clamp-1 max-w-md">
                              {step.description}
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {step.last_duration_ms && (
                          <span className="text-xs text-muted-foreground">
                            {(step.last_duration_ms / 1000).toFixed(1)}s
                          </span>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-7 w-7">
                              <MoreHorizontal className="h-3.5 w-3.5" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              onClick={() => onRunJob(step.name)}
                              disabled={runningJob === step.name}
                            >
                              {runningJob === step.name ? (
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              ) : (
                                <Play className="h-4 w-4 mr-2" />
                              )}
                              Run this step only
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Pipeline summary */}
            {orchestrator.last_run && (
              <div className="mt-4 pt-4 border-t flex items-center justify-between text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <span>Last run:</span>
                  <span className="font-medium">{formatDate(orchestrator.last_run)}</span>
                </div>
                {orchestrator.last_status && (
                  <Badge
                    variant={
                      orchestrator.last_status === 'success' || orchestrator.last_status === 'ok'
                        ? 'default'
                        : 'destructive'
                    }
                    className={cn(
                      orchestrator.last_status === 'success' || orchestrator.last_status === 'ok'
                        ? 'bg-success/20 text-success hover:bg-success/30'
                        : ''
                    )}
                  >
                    {orchestrator.last_status === 'success' || orchestrator.last_status === 'ok' ? (
                      <CheckCircle className="h-3 w-3 mr-1" />
                    ) : (
                      <XCircle className="h-3 w-3 mr-1" />
                    )}
                    {orchestrator.last_status}
                  </Badge>
                )}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

// =============================================================================
// MAIN SCHEDULER PANEL
// =============================================================================

export function SchedulerPanel() {
  const [cronJobs, setCronJobs] = useState<CronJob[]>([]);
  const [cronLogs, setCronLogs] = useState<CronLogEntry[]>([]);
  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([]);
  const [totalLogs, setTotalLogs] = useState(0);
  const [isLoadingJobs, setIsLoadingJobs] = useState(true);
  const [isLoadingLogs, setIsLoadingLogs] = useState(true);
  const [runningJob, setRunningJob] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Log filters
  const [logSearch, setLogSearch] = useState('');
  const [logStatus, setLogStatus] = useState<string | undefined>();
  const [logPage, setLogPage] = useState(0);
  const logsPerPage = 20;
  const [jobSearch, setJobSearch] = useState('');
  const [logType, setLogType] = useState<'all' | 'cron' | 'batch'>('all');

  // Cron editor
  const [editingJob, setEditingJob] = useState<CronJob | null>(null);
  const [newCronValue, setNewCronValue] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Separate jobs into pipeline and standalone
  const { pipelineOrchestrator, pipelineSteps, standaloneJobs } = (() => {
    const orchestrator = cronJobs.find((j) => j.name === 'market_close_pipeline');
    const steps = cronJobs.filter((j) => j.pipeline === 'market_close_pipeline');
    const standalone = cronJobs.filter(
      (j) => j.is_scheduled !== false && j.name !== 'market_close_pipeline'
    );
    return {
      pipelineOrchestrator: orchestrator,
      pipelineSteps: steps,
      standaloneJobs: standalone,
    };
  })();

  // Unified logs: combine cron logs and batch jobs into single list
  const unifiedLogs = ((): UnifiedLogEntry[] => {
    const logs: UnifiedLogEntry[] = [];

    // Add cron logs
    if (logType === 'all' || logType === 'cron') {
      for (const log of cronLogs) {
        logs.push({
          id: `cron-${log.name}-${log.created_at}`,
          name: log.name,
          type: 'cron',
          status: log.status,
          message: log.message,
          created_at: log.created_at,
          duration_ms: log.duration_ms,
        });
      }
    }

    // Add batch jobs as log entries
    if (logType === 'all' || logType === 'batch') {
      for (const job of batchJobs) {
        // Map batch status to log status
        let status: string = job.status;
        if (job.status === 'completed') status = 'success';
        if (job.status === 'failed' || job.status === 'cancelled') status = 'error';

        // Apply search filter
        if (logSearch) {
          const search = logSearch.toLowerCase();
          if (!job.job_type.toLowerCase().includes(search) &&
              !job.batch_id.toLowerCase().includes(search)) {
            continue;
          }
        }

        // Apply status filter
        if (logStatus) {
          if (logStatus === 'success' && job.status !== 'completed') continue;
          if (logStatus === 'error' && !['failed', 'cancelled'].includes(job.status)) continue;
        }

        logs.push({
          id: `batch-${job.id}`,
          name: job.job_type,
          type: 'batch',
          status,
          message: job.status === 'failed' ? 'Batch job failed' : null,
          created_at: job.created_at || '',
          batch_id: job.batch_id,
          job_type: job.job_type,
          total_requests: job.total_requests,
          completed_requests: job.completed_requests,
          failed_requests: job.failed_requests,
          estimated_cost_usd: job.estimated_cost_usd,
          actual_cost_usd: job.actual_cost_usd,
          completed_at: job.completed_at,
        });
      }
    }

    // Sort by created_at descending
    logs.sort((a, b) => {
      const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return dateB - dateA;
    });

    return logs;
  })();

  async function loadCronJobs() {
    setIsLoadingJobs(true);
    try {
      const jobs = await getCronJobs();
      setCronJobs(jobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cron jobs');
    } finally {
      setIsLoadingJobs(false);
    }
  }

  async function loadCronLogs() {
    setIsLoadingLogs(true);
    try {
      const [logsResponse, batchResponse] = await Promise.all([
        getCronLogs(
          logsPerPage,
          logPage * logsPerPage,
          logSearch || undefined,
          logStatus
        ),
        getBatchJobs(50, 0, true), // Get recent batch jobs
      ]);
      setCronLogs(logsResponse.logs);
      setTotalLogs(logsResponse.total);
      setBatchJobs(batchResponse.jobs);
    } catch (err) {
      console.error('Failed to load logs:', err);
    } finally {
      setIsLoadingLogs(false);
    }
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: loadCronJobs defined in component scope
  useEffect(() => {
    loadCronJobs();
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: loadCronLogs defined in component scope
  useEffect(() => {
    loadCronLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logPage, logSearch, logStatus]);

  async function handleRunNow(name: string) {
    setRunningJob(name);
    try {
      await runCronJobNow(name);
      await loadCronLogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run job');
    } finally {
      setRunningJob(null);
    }
  }

  async function handleCancelBatch(batchId: string) {
    try {
      await cancelBatchJob(batchId);
      await loadCronLogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel batch job');
    }
  }

  async function handleDeleteBatch(jobId: number) {
    try {
      await deleteBatchJob(jobId);
      await loadCronLogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete batch job');
    }
  }

  async function handleSaveCron() {
    if (!editingJob || !newCronValue) return;

    setIsSaving(true);
    try {
      const updated = await updateCronJob(editingJob.name, newCronValue);
      setCronJobs((jobs) => jobs.map((j) => (j.name === editingJob.name ? updated : j)));
      setEditingJob(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update schedule');
    } finally {
      setIsSaving(false);
    }
  }

  function openCronEditor(job: CronJob) {
    setEditingJob(job);
    setNewCronValue(job.cron);
  }

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }

  function getStatusBadge(status: string) {
    switch (status) {
      case 'success':
      case 'ok':
        return (
          <Badge variant="default" className="bg-success/20 text-success hover:bg-success/30">
            <CheckCircle className="h-3 w-3 mr-1" />
            Success
          </Badge>
        );
      case 'error':
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Error
          </Badge>
        );
      default:
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            {status}
          </Badge>
        );
    }
  }

  const totalPages = Math.ceil(totalLogs / logsPerPage);

  // Filter standalone jobs by search
  const filteredStandaloneJobs = standaloneJobs.filter((job) => {
    const query = jobSearch.trim().toLowerCase();
    if (!query) return true;
    return (
      job.name.toLowerCase().includes(query) ||
      job.cron.toLowerCase().includes(query) ||
      (job.description || '').toLowerCase().includes(query)
    );
  });

  // Sort by next run
  const sortedStandaloneJobs = [...filteredStandaloneJobs].sort((a, b) => {
    const timeA = a.next_run ? new Date(a.next_run).getTime() : Infinity;
    const timeB = b.next_run ? new Date(b.next_run).getTime() : Infinity;
    return timeA - timeB;
  });

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid gap-6">
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-2 bg-danger/10 text-danger p-4 rounded-xl border border-danger/20"
          >
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setError(null)}>
              <X className="h-4 w-4" />
            </Button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Pipeline Display */}
      {isLoadingJobs ? (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-4 w-96 mt-2" />
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      ) : pipelineOrchestrator && pipelineSteps.length > 0 ? (
        <PipelineDisplay
          orchestrator={pipelineOrchestrator}
          steps={pipelineSteps}
          runningJob={runningJob}
          onRunJob={handleRunNow}
          onEditSchedule={openCronEditor}
        />
      ) : null}

      {/* Standalone Jobs */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5 text-muted-foreground" />
                Standalone Tasks
              </CardTitle>
              <CardDescription>
                Independent jobs with their own schedules
              </CardDescription>
            </div>
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search jobs..."
                value={jobSearch}
                onChange={(event) => setJobSearch(event.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoadingJobs ? (
            <div className="space-y-2">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : filteredStandaloneJobs.length === 0 ? (
            <div className="text-center py-10 text-muted-foreground">
              <Clock className="h-10 w-10 mx-auto mb-3 opacity-20" />
              <p>
                {standaloneJobs.length === 0
                  ? 'No standalone jobs configured'
                  : 'No jobs match your search'}
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Cron</TableHead>
                  <TableHead>Next Run</TableHead>
                  <TableHead className="w-[60px]" />
                </TableRow>
              </TableHeader>
              <TableBody>
                <AnimatePresence mode="popLayout">
                  {sortedStandaloneJobs.map((job) => {
                    const nextRuns =
                      job.next_runs && job.next_runs.length > 0
                        ? job.next_runs
                        : job.next_run
                        ? [job.next_run]
                        : [];
                    return (
                      <motion.tr
                        key={job.name}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="border-b"
                      >
                        <TableCell className="font-medium">
                          {job.description ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="cursor-help underline decoration-dotted">
                                  {job.name}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                {job.description}
                              </TooltipContent>
                            </Tooltip>
                          ) : (
                            job.name
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="font-mono text-xs">
                            {job.cron}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {job.next_run ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="cursor-help">{formatDate(job.next_run)}</span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <div className="space-y-1">
                                  <p className="text-xs font-medium">Next runs</p>
                                  {nextRuns.map((run, index) => (
                                    <p key={`${job.name}-next-${index}`} className="text-xs">
                                      {formatDate(run)}
                                    </p>
                                  ))}
                                </div>
                              </TooltipContent>
                            </Tooltip>
                          ) : (
                            '—'
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem
                                onClick={() => handleRunNow(job.name)}
                                disabled={runningJob === job.name}
                              >
                                {runningJob === job.name ? (
                                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                ) : (
                                  <Play className="h-4 w-4 mr-2" />
                                )}
                                Run now
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => openCronEditor(job)}>
                                <Settings className="h-4 w-4 mr-2" />
                                Edit schedule
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </motion.tr>
                    );
                  })}
                </AnimatePresence>
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Unified Job Logs - Cron jobs + AI Batch jobs */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-muted-foreground" />
              Execution Logs
            </CardTitle>
            <Tabs value={logType} onValueChange={(v) => setLogType(v as 'all' | 'cron' | 'batch')}>
              <TabsList className="h-8">
                <TabsTrigger value="all" className="text-xs px-3 h-6">All</TabsTrigger>
                <TabsTrigger value="cron" className="text-xs px-3 h-6">
                  <Clock className="h-3 w-3 mr-1" />
                  Cron
                </TabsTrigger>
                <TabsTrigger value="batch" className="text-xs px-3 h-6">
                  <Bot className="h-3 w-3 mr-1" />
                  AI Batch
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search logs..."
                value={logSearch}
                onChange={(e) => {
                  setLogSearch(e.target.value);
                  setLogPage(0);
                }}
                className="pl-9"
              />
              {logSearch && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7"
                  onClick={() => {
                    setLogSearch('');
                    setLogPage(0);
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <Filter className="h-4 w-4 mr-2" />
                  {logStatus ? `Status: ${logStatus}` : 'All Status'}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() => {
                    setLogStatus(undefined);
                    setLogPage(0);
                  }}
                >
                  All Status
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => {
                    setLogStatus('success');
                    setLogPage(0);
                  }}
                >
                  Success
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => {
                    setLogStatus('error');
                    setLogPage(0);
                  }}
                >
                  Error
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <Button variant="outline" size="sm" onClick={loadCronLogs}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoadingLogs ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : unifiedLogs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Calendar className="h-12 w-12 mb-4 opacity-20" />
              <p>No logs found</p>
              {logSearch && <p className="text-sm mt-1">Try adjusting your search</p>}
            </div>
          ) : (
            <ScrollArea className="h-[400px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Job</TableHead>
                    <TableHead>Time</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Duration</TableHead>
                    <TableHead>Message</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {unifiedLogs.slice(0, logsPerPage).map((log) => (
                    <LogRow
                      key={log.id}
                      log={log}
                      formatDate={formatDate}
                      getStatusBadge={getStatusBadge}
                      onCancelBatch={handleCancelBatch}
                      onDeleteBatch={handleDeleteBatch}
                    />
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}

          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-4 border-t mt-4">
              <p className="text-sm text-muted-foreground">
                Showing {logPage * logsPerPage + 1} -{' '}
                {Math.min((logPage + 1) * logsPerPage, totalLogs)} of {totalLogs}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setLogPage((p) => p - 1)}
                  disabled={logPage === 0}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-muted-foreground px-2">
                  Page {logPage + 1} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setLogPage((p) => p + 1)}
                  disabled={logPage >= totalPages - 1}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <CeleryMonitorPanel />

      {/* Cron Editor Dialog */}
      <Dialog
        open={!!editingJob}
        onOpenChange={(open) => {
          if (!open) {
            setEditingJob(null);
          }
        }}
      >
        <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Schedule</DialogTitle>
            <DialogDescription>
              Configure the cron schedule for {editingJob?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <CronBuilder value={newCronValue} onChange={setNewCronValue} />
          </div>
          <DialogFooter>
            <Button
              onClick={handleSaveCron}
              disabled={isSaving || !newCronValue || !validateCron(newCronValue).valid}
            >
              {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
