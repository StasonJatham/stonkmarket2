import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { 
  getCronJobs, 
  getCronLogs, 
  updateCronJob,
  runCronJobNow,
  refreshData 
} from '@/services/api';
import type { CronJob, CronLogEntry } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
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
  X
} from 'lucide-react';

// Cron presets
const cronPresets = [
  { label: 'Every minute', value: '* * * * *' },
  { label: 'Every 5 minutes', value: '*/5 * * * *' },
  { label: 'Every 15 minutes', value: '*/15 * * * *' },
  { label: 'Every hour', value: '0 * * * *' },
  { label: 'Every 6 hours', value: '0 */6 * * *' },
  { label: 'Daily at midnight', value: '0 0 * * *' },
  { label: 'Daily at 9 AM', value: '0 9 * * *' },
  { label: 'Daily at 6 PM', value: '0 18 * * *' },
  { label: 'Weekdays at 9 AM', value: '0 9 * * 1-5' },
  { label: 'Weekly on Sunday', value: '0 0 * * 0' },
];

export function AdminPage() {
  const { user } = useAuth();
  const [cronJobs, setCronJobs] = useState<CronJob[]>([]);
  const [cronLogs, setCronLogs] = useState<CronLogEntry[]>([]);
  const [totalLogs, setTotalLogs] = useState(0);
  const [isLoadingJobs, setIsLoadingJobs] = useState(true);
  const [isLoadingLogs, setIsLoadingLogs] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [runningJob, setRunningJob] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Log filters
  const [logSearch, setLogSearch] = useState('');
  const [logStatus, setLogStatus] = useState<string | undefined>();
  const [logPage, setLogPage] = useState(0);
  const logsPerPage = 20;

  // Cron editor
  const [editingJob, setEditingJob] = useState<CronJob | null>(null);
  const [newCronValue, setNewCronValue] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    loadCronJobs();
  }, []);

  useEffect(() => {
    loadCronLogs();
  }, [logSearch, logStatus, logPage]);

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
      const response = await getCronLogs(
        logsPerPage, 
        logPage * logsPerPage,
        logSearch || undefined,
        logStatus
      );
      setCronLogs(response.logs);
      setTotalLogs(response.total);
    } catch (err) {
      console.error('Failed to load logs:', err);
    } finally {
      setIsLoadingLogs(false);
    }
  }

  async function handleRunNow(name: string) {
    setRunningJob(name);
    try {
      await runCronJobNow(name);
      // Refresh logs after running
      await loadCronLogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run job');
    } finally {
      setRunningJob(null);
    }
  }

  async function handleRefreshData() {
    setIsRefreshing(true);
    try {
      await refreshData();
      await loadCronJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsRefreshing(false);
    }
  }

  async function handleSaveCron() {
    if (!editingJob || !newCronValue) return;
    
    setIsSaving(true);
    try {
      const updated = await updateCronJob(editingJob.name, newCronValue);
      setCronJobs(jobs => jobs.map(j => j.name === editingJob.name ? updated : j));
      setEditingJob(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update schedule');
    } finally {
      setIsSaving(false);
    }
  }

  const openCronEditor = useCallback((job: CronJob) => {
    setEditingJob(job);
    setNewCronValue(job.cron);
  }, []);

  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleString();
  }

  function getStatusBadge(status: string) {
    switch (status) {
      case 'success':
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Manage scheduled jobs and view execution history
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-muted-foreground">
            {user?.username}
          </Badge>
          <Button onClick={handleRefreshData} disabled={isRefreshing} variant="outline">
            {isRefreshing ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh Data
          </Button>
        </div>
      </motion.div>

      {/* Error Alert */}
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
            <Button 
              variant="ghost" 
              size="icon"
              className="h-6 w-6"
              onClick={() => setError(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tabs */}
      <Tabs defaultValue="jobs" className="space-y-6">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="jobs">
            <Settings className="h-4 w-4 mr-2" />
            Scheduled Jobs
          </TabsTrigger>
          <TabsTrigger value="logs">
            <Calendar className="h-4 w-4 mr-2" />
            Execution Logs
          </TabsTrigger>
        </TabsList>

        {/* Jobs Tab */}
        <TabsContent value="jobs">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid gap-4"
          >
            {isLoadingJobs ? (
              <>
                {[...Array(3)].map((_, i) => (
                  <Skeleton key={i} className="h-32 rounded-xl" />
                ))}
              </>
            ) : cronJobs.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Clock className="h-12 w-12 mb-4 opacity-20" />
                  <p>No scheduled jobs configured</p>
                </CardContent>
              </Card>
            ) : (
              cronJobs.map((job, index) => (
                <motion.div
                  key={job.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="overflow-hidden">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <CardTitle className="text-xl flex items-center gap-2">
                            <Clock className="h-5 w-5 text-muted-foreground" />
                            {job.name}
                          </CardTitle>
                          {job.description && (
                            <CardDescription>{job.description}</CardDescription>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button 
                                variant="outline" 
                                size="sm"
                                onClick={() => openCronEditor(job)}
                              >
                                <Settings className="h-4 w-4 mr-1" />
                                Edit Schedule
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-md">
                              <DialogHeader>
                                <DialogTitle>Edit Schedule</DialogTitle>
                                <DialogDescription>
                                  Configure the cron schedule for {editingJob?.name}
                                </DialogDescription>
                              </DialogHeader>
                              <div className="space-y-4 py-4">
                                <div className="space-y-2">
                                  <Label>Cron Expression</Label>
                                  <Input
                                    value={newCronValue}
                                    onChange={(e) => setNewCronValue(e.target.value)}
                                    placeholder="* * * * *"
                                    className="font-mono"
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    Format: minute hour day month weekday
                                  </p>
                                </div>
                                
                                <Separator />
                                
                                <div className="space-y-2">
                                  <Label>Quick Presets</Label>
                                  <div className="grid grid-cols-2 gap-2">
                                    {cronPresets.map((preset) => (
                                      <Button
                                        key={preset.value}
                                        variant={newCronValue === preset.value ? 'default' : 'outline'}
                                        size="sm"
                                        className="justify-start text-xs"
                                        onClick={() => setNewCronValue(preset.value)}
                                      >
                                        {preset.label}
                                      </Button>
                                    ))}
                                  </div>
                                </div>
                              </div>
                              <DialogFooter>
                                <Button 
                                  onClick={handleSaveCron}
                                  disabled={isSaving || !newCronValue}
                                >
                                  {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                                  Save Changes
                                </Button>
                              </DialogFooter>
                            </DialogContent>
                          </Dialog>
                          
                          <Button
                            size="sm"
                            onClick={() => handleRunNow(job.name)}
                            disabled={runningJob === job.name}
                          >
                            {runningJob === job.name ? (
                              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                            ) : (
                              <Play className="h-4 w-4 mr-1" />
                            )}
                            Run Now
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
                        <Badge variant="outline" className="font-mono text-sm">
                          {job.cron}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))
            )}
          </motion.div>
        </TabsContent>

        {/* Logs Tab */}
        <TabsContent value="logs">
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col sm:flex-row gap-4">
                {/* Search */}
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

                {/* Status Filter */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">
                      <Filter className="h-4 w-4 mr-2" />
                      {logStatus ? `Status: ${logStatus}` : 'All Status'}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => { setLogStatus(undefined); setLogPage(0); }}>
                      All Status
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => { setLogStatus('success'); setLogPage(0); }}>
                      Success
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => { setLogStatus('error'); setLogPage(0); }}>
                      Error
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>

                {/* Refresh */}
                <Button variant="outline" onClick={loadCronLogs}>
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
              ) : cronLogs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Calendar className="h-12 w-12 mb-4 opacity-20" />
                  <p>No logs found</p>
                  {logSearch && (
                    <p className="text-sm mt-1">Try adjusting your search</p>
                  )}
                </div>
              ) : (
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Job</TableHead>
                        <TableHead>Time</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Message</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {cronLogs.map((log, i) => (
                        <TableRow key={`${log.name}-${log.created_at}-${i}`}>
                          <TableCell className="font-medium">{log.name}</TableCell>
                          <TableCell className="text-muted-foreground whitespace-nowrap">
                            {formatDate(log.created_at)}
                          </TableCell>
                          <TableCell>
                            {getStatusBadge(log.status)}
                          </TableCell>
                          <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
                            {log.message || '—'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              )}

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-4 border-t mt-4">
                  <p className="text-sm text-muted-foreground">
                    Showing {logPage * logsPerPage + 1} - {Math.min((logPage + 1) * logsPerPage, totalLogs)} of {totalLogs}
                  </p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setLogPage(p => p - 1)}
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
                      onClick={() => setLogPage(p => p + 1)}
                      disabled={logPage >= totalPages - 1}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
