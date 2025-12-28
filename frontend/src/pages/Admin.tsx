import { useState, useEffect, type ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { LucideIcon } from 'lucide-react';
import {
  Menu,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Lightbulb,
  Sparkles,
  User,
  Clock,
  BarChart2,
  Trash2,
  KeyRound,
  Info,
  AlertCircle,
  X,
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { refreshData } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { SymbolManager } from '@/components/SymbolManager';
import { UserSettings } from '@/components/UserSettings';
import { MFASetup } from '@/components/MFASetup';
import { UserApiKeyManager } from '@/components/UserApiKeyManager';
import { SystemSettings } from '@/components/SystemSettings';
import { SuggestionManager } from '@/components/SuggestionManager';
import { AIManager } from '@/components/AIManager';
import { AIPersonasPanel } from '@/components/AIPersonasPanel';
import { SchedulerPanel } from '@/components/SchedulerPanel';
import { useSEO } from '@/lib/seo';

type AdminSection = {
  id: string;
  label: string;
  icon: LucideIcon;
  group: string;
  description?: string;
  content: ReactNode;
};

export function AdminPage() {
  useSEO({
    title: 'Admin Dashboard',
    description: 'StonkMarket administration panel.',
    noindex: true,
  });

  const { user } = useAuth();
  const [activeSection, setActiveSection] = useState(() => {
    if (typeof window === 'undefined') return 'symbols';
    return localStorage.getItem('admin-section') || localStorage.getItem('admin-tab') || 'symbols';
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    localStorage.setItem('admin-section', activeSection);
  }, [activeSection]);

  async function handleRefreshData() {
    setIsRefreshing(true);
    try {
      await refreshData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsRefreshing(false);
    }
  }

  const sections: AdminSection[] = [
    {
      id: 'symbols',
      label: 'Symbols',
      icon: TrendingUp,
      group: 'Data',
      description: 'Manage tracked tickers and ingest settings.',
      content: <SymbolManager onError={(msg) => setError(msg)} />,
    },
    {
      id: 'suggestions',
      label: 'Suggestions',
      icon: Lightbulb,
      group: 'Data',
      description: 'Review, approve, and fix incoming stock ideas.',
      content: <SuggestionManager />,
    },
    {
      id: 'ai',
      label: 'AI Content',
      icon: Sparkles,
      group: 'AI',
      description: 'Monitor personas, AI bios, ratings, and batch jobs.',
      content: (
        <div className="space-y-6">
          <AIPersonasPanel />
          <AIManager />
        </div>
      ),
    },
    {
      id: 'settings-ai',
      label: 'AI Settings',
      icon: Sparkles,
      group: 'Settings',
      description: 'Configure AI models, enrichment, and batch throughput.',
      content: <SystemSettings defaultTab="ai" showTabs={false} />,
    },
    {
      id: 'settings-trading',
      label: 'Trading Settings',
      icon: BarChart2,
      group: 'Settings',
      description: 'Tune thresholds, costs, risk rules, and validation.',
      content: <SystemSettings defaultTab="trading" showTabs={false} />,
    },
    {
      id: 'settings-benchmarks',
      label: 'Benchmarks',
      icon: BarChart2,
      group: 'Settings',
      description: 'Maintain benchmark and sector ETF references.',
      content: <SystemSettings defaultTab="benchmarks" showTabs={false} />,
    },
    {
      id: 'settings-maintenance',
      label: 'Maintenance',
      icon: Trash2,
      group: 'Settings',
      description: 'Control cleanup and retention policies.',
      content: <SystemSettings defaultTab="maintenance" showTabs={false} />,
    },
    {
      id: 'settings-integrations',
      label: 'Integrations',
      icon: KeyRound,
      group: 'Settings',
      description: 'Manage API keys and external providers.',
      content: <SystemSettings defaultTab="integrations" showTabs={false} />,
    },
    {
      id: 'settings-about',
      label: 'About',
      icon: Info,
      group: 'Settings',
      description: 'Application status and runtime metadata.',
      content: <SystemSettings defaultTab="about" showTabs={false} />,
    },
    {
      id: 'account',
      label: 'Account',
      icon: User,
      group: 'Account',
      description: 'Profile, MFA, and personal API keys.',
      content: (
        <div className="max-w-2xl space-y-6">
          <UserSettings />
          <MFASetup />
          <UserApiKeyManager
            onError={(msg) => setError(msg)}
            onSuccess={() => { /* optional toast */ }}
          />
        </div>
      ),
    },
    {
      id: 'scheduler',
      label: 'Scheduler',
      icon: Clock,
      group: 'Operations',
      description: 'Cron jobs, Celery workers, and queue health.',
      content: <SchedulerPanel />,
    },
  ];

  const activeItem = sections.find((section) => section.id === activeSection) || sections[0];

  const groupedSections = sections.reduce<{ label: string; items: AdminSection[] }[]>((groups, section) => {
    const existing = groups.find((group) => group.label === section.group);
    if (existing) {
      existing.items.push(section);
    } else {
      groups.push({ label: section.group, items: [section] });
    }
    return groups;
  }, []);

  const buildNavButton = (section: AdminSection, collapsed: boolean) => {
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
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            {activeItem.label}
          </h1>
          <p className="text-muted-foreground">
            {activeItem.description || 'Manage your account, symbols, and scheduled jobs'}
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
              <div className="flex flex-col gap-4 px-4 pb-6">
                {groupedSections.map((group, index) => (
                  <div key={group.label} className="space-y-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      {group.label}
                    </span>
                    <div className="flex flex-col gap-2">
                      {group.items.map((section) => buildNavButton(section, false))}
                    </div>
                    {index < groupedSections.length - 1 && <Separator />}
                  </div>
                ))}
              </div>
            </SheetContent>
          </Sheet>
          <Badge variant="outline" className="text-muted-foreground">
            {user?.username}
          </Badge>
          <Button onClick={handleRefreshData} disabled={isRefreshing} variant="outline">
            {isRefreshing ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4 mr-2" />
            )}
            Refresh Data
          </Button>
        </div>
      </motion.div>

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
          className={`hidden lg:flex flex-col gap-3 rounded-xl border bg-card p-3 ${sidebarCollapsed ? 'w-16' : 'w-64'}`}
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
          <nav className={`flex flex-col gap-3 ${sidebarCollapsed ? 'items-center' : ''}`}>
            {groupedSections.map((group, index) => (
              <div key={group.label} className="space-y-2 w-full">
                {!sidebarCollapsed && (
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {group.label}
                  </span>
                )}
                <div className={`flex flex-col gap-2 ${sidebarCollapsed ? 'items-center' : ''}`}>
                  {group.items.map((section) => buildNavButton(section, sidebarCollapsed))}
                </div>
                {index < groupedSections.length - 1 && !sidebarCollapsed && <Separator />}
              </div>
            ))}
          </nav>
        </aside>
        <div className="flex-1 min-w-0">
          {activeItem.content}
        </div>
      </div>
    </div>
  );
}
