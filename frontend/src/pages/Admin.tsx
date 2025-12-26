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
import { Separator } from '@/components/ui/separator';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
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
  X,
  Menu,
  MoreHorizontal,
  TrendingUp,
  User,
  Cog,
  Lightbulb,
  Sparkles
} from 'lucide-react';
import { CronBuilder, validateCron } from '@/components/CronBuilder';
import { SymbolManager } from '@/components/SymbolManager';
import { UserSettings } from '@/components/UserSettings';
import { MFASetup } from '@/components/MFASetup';
import { ApiKeyManager } from '@/components/ApiKeyManager';
import { UserApiKeyManager } from '@/components/UserApiKeyManager';
import { SystemSettings } from '@/components/SystemSettings';
import { SuggestionManager } from '@/components/SuggestionManager';
import { AIManager } from '@/components/AIManager';
import { CeleryMonitorPanel } from '@/components/CeleryMonitorPanel';
import { useSEO } from '@/lib/seo';

export function AdminPage() {
  // SEO - noindex for admin page
  useSEO({
    title: 'Admin Dashboard',
    description: 'StonkMarket administration panel.',
    noindex: true, // Don't index admin pages
  });

  const { user } = useAuth();
  const [activeSection, setActiveSection] = useState(() => {
    if (typeof window === 'undefined') return 'symbols';
    return localStorage.getItem('admin-section') || localStorage.getItem('admin-tab') || 'symbols';
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
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
  const [jobSearch, setJobSearch] = useState('');

  // Cron editor
  const [editingJob, setEditingJob] = useState<CronJob | null>(null);
  const [newCronValue, setNewCronValue] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const loadCronJobs = useCallback(async () => {
    setIsLoadingJobs(true);
    try {
      const jobs = await getCronJobs();
      setCronJobs(jobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cron jobs');
    } finally {
      setIsLoadingJobs(false);
    }
  }, []);

  const loadCronLogs = useCallback(async () => {
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
  }, [logPage, logSearch, logStatus]);

  useEffect(() => {
    loadCronJobs();
  }, [loadCronJobs]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    localStorage.setItem('admin-section', activeSection);
  }, [activeSection]);

  useEffect(() => {
    loadCronLogs();
  }, [loadCronLogs]);

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
  const filteredCronJobs = cronJobs.filter((job) => {
    const query = jobSearch.trim().toLowerCase();
    if (!query) return true;
    return (
      job.name.toLowerCase().includes(query) ||
      job.cron.toLowerCase().includes(query) ||
      (job.description || '').toLowerCase().includes(query)
    );
  });
  const sortedCronJobs = [...filteredCronJobs].sort((a, b) => {
    const timeA = a.next_run ? new Date(a.next_run).getTime() : Infinity;
    const timeB = b.next_run ? new Date(b.next_run).getTime() : Infinity;
    return timeA - timeB;
  });

  const sections = [
    {
      id: 'symbols',
      label: 'Symbols',
      icon: TrendingUp,
      content: <SymbolManager onError={(msg) => setError(msg)} />,
    },
    {
      id: 'suggestions',
      label: 'Suggestions',
      icon: Lightbulb,
      content: <SuggestionManager />,
    },
    {
      id: 'ai',
      label: 'AI Content',
      icon: Sparkles,
      content: <AIManager />,
    },
    {
      id: 'system',
      label: 'System',
      icon: Cog,
      content: (
        <div className="max-w-2xl space-y-6">
          <SystemSettings />
          <ApiKeyManager />
        </div>
      ),
    },
    {
      id: 'account',
      label: 'Account',
      icon: User,
      content: (
        <div className="max-w-2xl space-y-6">
          <UserSettings />
          <MFASetup />
          <UserApiKeyManager 
            onError={(msg) => setError(msg)}
            onSuccess={() => { /* Could add toast */ }}
          />
        </div>
      ),
    },
    {
      id: 'scheduler',
      label: 'Scheduler',
      icon: Clock,
      content: (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid gap-6"
        >
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5 text-muted-foreground" />
                    Scheduled Tasks
                  </CardTitle>
                  <CardDescription>
                    Manage recurring jobs and run them on demand
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
              ) : filteredCronJobs.length === 0 ? (
                <div className="text-center py-10 text-muted-foreground">
                  <Clock className="h-10 w-10 mx-auto mb-3 opacity-20" />
                  <p>
                    {cronJobs.length === 0 ? 'No scheduled jobs configured' : 'No jobs match your search'}
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
                      {sortedCronJobs.map((job) => {
                        const nextRuns = job.next_runs && job.next_runs.length > 0
                          ? job.next_runs
                          : job.next_run ? [job.next_run] : [];
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
                                    <span className="cursor-help">
                                      {formatDate(job.next_run)}
                                    </span>
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

          {/* Logs Section (inline in Scheduler) */}
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-muted-foreground" />
                Job Logs
              </CardTitle>
              <div className="flex flex-col sm:flex-row gap-4 mt-4">
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
                    <Button variant="outline" size="sm">
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
              ) : cronLogs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Calendar className="h-12 w-12 mb-4 opacity-20" />
                  <p>No logs found</p>
                  {logSearch && (
                    <p className="text-sm mt-1">Try adjusting your search</p>
                  )}
                </div>
              ) : (
                <ScrollArea className="h-[300px]">
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
          
          <CeleryMonitorPanel />
        </motion.div>
      ),
    },
  ];

  const activeItem = sections.find((section) => section.id === activeSection) || sections[0];

  const buildNavButton = (section: (typeof sections)[number], collapsed: boolean) => {
    const Icon = section.icon;
    const isActive = activeSection === section.id;
    const button = (
      <Button
        key={section.id}
        variant={isActive ? 'secondary' : 'ghost'}
        size={collapsed ? 'icon' : 'sm'}
        className={collapsed ? 'w-10 justify-center' : 'w-full justify-start'}
        onClick={() => {
          setActiveSection(section.id);
          setMobileNavOpen(false);
        }}
      >
        <Icon className="h-4 w-4" />
        {!collapsed && <span className="ml-2">{section.label}</span>}
      </Button>
    );

    if (!collapsed) {
      return button;
    }

    return (
      <Tooltip key={section.id}>
        <TooltipTrigger asChild>
          {button}
        </TooltipTrigger>
        <TooltipContent>{section.label}</TooltipContent>
      </Tooltip>
    );
  };

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
            Manage your account, symbols, and scheduled jobs
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
            <SheetTrigger asChild>
              <Button variant="outline" size="icon" className="lg:hidden">
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="p-0">
              <SheetHeader className="px-4 pt-4">
                <SheetTitle>Settings</SheetTitle>
              </SheetHeader>
              <div className="flex flex-col gap-2 px-2 pb-4">
                {sections.map((section) => buildNavButton(section, false))}
              </div>
            </SheetContent>
          </Sheet>
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

      <div className="flex gap-6">
        <aside
          className={`hidden lg:flex flex-col gap-3 rounded-xl border bg-card p-3 ${sidebarCollapsed ? 'w-16' : 'w-60'}`}
        >
          <div className="flex items-center justify-between px-1">
            {!sidebarCollapsed && (
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Sections
              </span>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              className="h-7 w-7"
            >
              {sidebarCollapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          </div>
          <Separator />
          <nav className={`flex flex-col gap-2 ${sidebarCollapsed ? 'items-center' : ''}`}>
            {sections.map((section) => buildNavButton(section, sidebarCollapsed))}
          </nav>
        </aside>
        <div className="flex-1 min-w-0">
          {activeItem.content}
        </div>
      </div>

      <Dialog open={!!editingJob} onOpenChange={(open) => {
        if (!open) {
          setEditingJob(null);
        }
      }}>
        <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Schedule</DialogTitle>
            <DialogDescription>
              Configure the cron schedule for {editingJob?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <CronBuilder
              value={newCronValue}
              onChange={setNewCronValue}
            />
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
    </div>
  );
}
