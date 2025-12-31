import { useMemo } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  CircleAlert,
  Lightbulb,
  Shield,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// =============================================================================
// Types matching backend AIPortfolioAnalysis schema
// =============================================================================

interface AIInsight {
  type: 'positive' | 'warning' | 'neutral';
  text: string;
}

interface AIActionItem {
  priority: 1 | 2 | 3;
  action: string;
}

interface AIRiskAlert {
  severity: 'high' | 'medium' | 'low';
  alert: string;
}

interface AIPortfolioAnalysis {
  health: 'strong' | 'good' | 'fair' | 'weak';
  headline: string;
  insights: AIInsight[];
  actions: AIActionItem[];
  risks: AIRiskAlert[];
}

interface AIPortfolioAnalysisProps {
  summary: string | null | undefined;
  className?: string;
}

// =============================================================================
// Health Badge Component
// =============================================================================

function HealthBadge({ health }: { health: AIPortfolioAnalysis['health'] }) {
  const config = {
    strong: {
      icon: ShieldCheck,
      label: 'Strong',
      variant: 'default' as const,
      className: 'bg-success/10 text-success border-success/20 hover:bg-success/20',
    },
    good: {
      icon: Shield,
      label: 'Good',
      variant: 'secondary' as const,
      className: 'bg-primary/10 text-primary border-primary/20 hover:bg-primary/20',
    },
    fair: {
      icon: ShieldQuestion,
      label: 'Fair',
      variant: 'secondary' as const,
      className: 'bg-warning/10 text-warning border-warning/20 hover:bg-warning/20',
    },
    weak: {
      icon: ShieldAlert,
      label: 'Needs Attention',
      variant: 'destructive' as const,
      className: 'bg-destructive/10 text-destructive border-destructive/20 hover:bg-destructive/20',
    },
  };

  const { icon: Icon, label, className } = config[health];

  return (
    <Badge variant="outline" className={cn('gap-1 px-2 py-1', className)}>
      <Icon className="h-3.5 w-3.5" />
      {label}
    </Badge>
  );
}

// =============================================================================
// Insight Icon
// =============================================================================

function InsightIcon({ type }: { type: AIInsight['type'] }) {
  if (type === 'positive') {
    return <CheckCircle2 className="h-4 w-4 text-success flex-shrink-0" />;
  }
  if (type === 'warning') {
    return <AlertTriangle className="h-4 w-4 text-warning flex-shrink-0" />;
  }
  return <Lightbulb className="h-4 w-4 text-muted-foreground flex-shrink-0" />;
}

// =============================================================================
// Priority Badge
// =============================================================================

function PriorityBadge({ priority }: { priority: AIActionItem['priority'] }) {
  const config = {
    1: { label: 'High', className: 'bg-destructive/10 text-destructive border-destructive/20' },
    2: { label: 'Med', className: 'bg-warning/10 text-warning border-warning/20' },
    3: { label: 'Low', className: 'bg-muted text-muted-foreground border-border' },
  };

  const { label, className } = config[priority];

  return (
    <Badge variant="outline" className={cn('text-xs px-1.5 py-0', className)}>
      {label}
    </Badge>
  );
}

// =============================================================================
// Severity Icon
// =============================================================================

function SeverityIcon({ severity }: { severity: AIRiskAlert['severity'] }) {
  const className = {
    high: 'text-destructive',
    medium: 'text-warning',
    low: 'text-muted-foreground',
  }[severity];

  return <CircleAlert className={cn('h-4 w-4 flex-shrink-0', className)} />;
}

// =============================================================================
// Main Component
// =============================================================================

export function AIPortfolioAnalysis({ summary, className }: AIPortfolioAnalysisProps) {
  // Parse the JSON summary
  const analysis = useMemo<AIPortfolioAnalysis | null>(() => {
    if (!summary) return null;

    try {
      const parsed = JSON.parse(summary);
      // Basic validation
      if (
        parsed.health &&
        parsed.headline &&
        Array.isArray(parsed.insights)
      ) {
        return parsed as AIPortfolioAnalysis;
      }
    } catch {
      // Not valid JSON - might be legacy markdown format
    }
    return null;
  }, [summary]);

  // Pending state
  if (!summary) {
    return (
      <div className={cn('flex items-center gap-3 rounded-lg border border-dashed p-4 text-muted-foreground', className)}>
        <Sparkles className="h-5 w-5 animate-pulse" />
        <div className="text-sm">
          <p className="font-medium">Analysis pending</p>
          <p className="text-xs">AI analysis will be generated automatically. Check back soon!</p>
        </div>
      </div>
    );
  }

  // Invalid/old format - prompt regeneration
  if (!analysis) {
    return (
      <div className={cn('flex items-center gap-3 rounded-lg border border-dashed p-4 text-muted-foreground', className)}>
        <Sparkles className="h-5 w-5" />
        <div className="text-sm">
          <p className="font-medium">Analysis needs refresh</p>
          <p className="text-xs">AI analysis will be regenerated on the next scheduled run.</p>
        </div>
      </div>
    );
  }

  // Structured format
  return (
    <div className={cn('space-y-4', className)}>
      {/* Header with health badge and headline */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <HealthBadge health={analysis.health} />
          </div>
          <p className="text-sm font-medium text-foreground leading-relaxed">
            {analysis.headline}
          </p>
        </div>
      </div>

      {/* Insights - compact grid */}
      {analysis.insights.length > 0 && (
        <div className="grid gap-2 sm:grid-cols-2">
          {analysis.insights.map((insight, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 rounded-md bg-muted/50 p-2"
            >
              <InsightIcon type={insight.type} />
              <p className="text-xs text-muted-foreground leading-relaxed">
                {insight.text}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Actions and Risks in 2-column layout on larger screens */}
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Actions */}
        {analysis.actions.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
              <TrendingUp className="h-3.5 w-3.5" />
              Actions
            </div>
            <div className="space-y-1.5">
              {analysis.actions.map((action, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-2 text-xs"
                >
                  <PriorityBadge priority={action.priority} />
                  <span className="text-foreground leading-relaxed">{action.action}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Risks */}
        {analysis.risks.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
              <AlertTriangle className="h-3.5 w-3.5" />
              Risk Alerts
            </div>
            <div className="space-y-1.5">
              {analysis.risks.map((risk, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-2 text-xs"
                >
                  <SeverityIcon severity={risk.severity} />
                  <span className="text-muted-foreground leading-relaxed">{risk.alert}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No actions needed message */}
        {analysis.actions.length === 0 && analysis.risks.length === 0 && (
          <div className="sm:col-span-2 flex items-center gap-2 text-xs text-success">
            <CheckCircle2 className="h-4 w-4" />
            <span>No immediate actions needed. Portfolio looks healthy!</span>
          </div>
        )}
      </div>
    </div>
  );
}
