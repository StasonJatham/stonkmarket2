import { 
  TrendingDown, 
  BarChart3, 
  Activity, 
  Target, 
  Percent, 
  CalendarDays, 
  ArrowDown, 
  CircleCheck, 
  Building2, 
  Scale, 
  TriangleAlert,
  ChartNoAxesGantt,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

// Dip type badge - shows whether dip is stock-specific, mixed, or market-wide
export function getDipTypeBadge(dipType: string, colorblindMode: boolean) {
  const config: Record<string, { color: string; colorblindColor: string; label: string; desc: string }> = {
    STOCK_SPECIFIC: { 
      color: 'bg-success/20 text-success border-success/30', 
      colorblindColor: 'bg-blue-500/20 text-blue-500 border-blue-500/30',
      label: 'Stock',
      desc: 'This dip is specific to this stock, not market-wide' 
    },
    MIXED: { 
      color: 'bg-chart-2/20 text-chart-2 border-chart-2/30', 
      colorblindColor: 'bg-purple-500/20 text-purple-500 border-purple-500/30',
      label: 'Mixed',
      desc: 'Part stock-specific, part market-driven dip' 
    },
    MARKET_DIP: { 
      color: 'bg-chart-4/20 text-chart-4 border-chart-4/30', 
      colorblindColor: 'bg-orange-500/20 text-orange-500 border-orange-500/30',
      label: 'Market',
      desc: 'This dip is primarily driven by overall market decline' 
    },
  };
  const item = config[dipType] || { color: '', colorblindColor: '', label: dipType, desc: '' };
  return (
    <Badge 
      variant="outline" 
      className={`${colorblindMode ? item.colorblindColor : item.color} font-medium`} 
      title={item.desc}
    >
      {item.label}
    </Badge>
  );
}

interface DipFinderMethodologyProps {
  colorblindMode: boolean;
}

export function DipFinderMethodology({ colorblindMode }: DipFinderMethodologyProps) {
  return (
    <div className="space-y-8">
      <Separator />
      
      {/* Introduction */}
      <div className="bg-gradient-to-br from-primary/5 via-primary/3 to-transparent rounded-2xl p-5 border border-primary/10">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-primary/10 shrink-0">
            <Target className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold mb-2">How DipFinder Works</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              DipFinder identifies potential buying opportunities by finding stocks that have dropped significantly 
              from their recent highs. But not all dips are created equal—we score each opportunity based on 
              how deep the dip is, how healthy the company is, and how stable the stock typically behaves.
            </p>
          </div>
        </div>
      </div>

      {/* Opportunity Score Section */}
      <div className="space-y-4">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <ChartNoAxesGantt className="h-5 w-5 text-muted-foreground" />
          The Opportunity Score
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Each stock gets a score from 0-100 based on three weighted factors. Higher scores indicate 
          potentially better buying opportunities—deeper dips in quality, stable companies.
        </p>
        
        <div className="grid gap-3 md:grid-cols-3">
          {/* Dip Score Card */}
          <div className="bg-muted/30 rounded-xl p-5 border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <div className={`p-2 rounded-lg ${colorblindMode ? 'bg-orange-500/15' : 'bg-danger/15'}`}>
                <TrendingDown className={`h-5 w-5 ${colorblindMode ? 'text-orange-500' : 'text-danger'}`} />
              </div>
              <div>
                <span className="font-semibold">Dip Score</span>
                <Badge variant="outline" className="ml-2 text-xs">45%</Badge>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              The largest weight goes to dip magnitude. We measure how much the stock has fallen, 
              how rare this dip is historically, and whether the decline exceeds the stock's typical patterns.
              A minimum 10% drop is required to qualify.
            </p>
          </div>

          {/* Quality Score Card */}
          <div className="bg-muted/30 rounded-xl p-5 border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <div className={`p-2 rounded-lg ${colorblindMode ? 'bg-blue-500/15' : 'bg-success/15'}`}>
                <CircleCheck className={`h-5 w-5 ${colorblindMode ? 'text-blue-500' : 'text-success'}`} />
              </div>
              <div>
                <span className="font-semibold">Quality Score</span>
                <Badge variant="outline" className="ml-2 text-xs">30%</Badge>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Is this a healthy company? We analyze profitability (gross margins, ROE), 
              balance sheet strength (debt ratios), cash flow generation, 
              and growth trajectory to filter out fundamentally weak businesses.
            </p>
          </div>

          {/* Stability Score Card */}
          <div className="bg-muted/30 rounded-xl p-5 border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <div className={`p-2 rounded-lg ${colorblindMode ? 'bg-purple-500/15' : 'bg-chart-2/15'}`}>
                <Activity className={`h-5 w-5 ${colorblindMode ? 'text-purple-500' : 'text-chart-2'}`} />
              </div>
              <div>
                <span className="font-semibold">Stability Score</span>
                <Badge variant="outline" className="ml-2 text-xs">25%</Badge>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Stable stocks with lower volatility get higher scores. We look at daily price swings, 
              beta (correlation to market), maximum historical drawdowns, 
              and typical dip patterns to identify stocks likely to recover smoothly.
            </p>
          </div>
        </div>
      </div>

      {/* What Makes a Good Dip */}
      <div className="space-y-4">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <ArrowDown className="h-5 w-5 text-muted-foreground" />
          What Makes a Good Dip?
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          The dip score itself is calculated from four components, each measuring a different 
          aspect of the price decline:
        </p>
        
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="bg-card rounded-xl p-4 border">
            <div className="flex items-center gap-2 mb-2">
              <Percent className="h-5 w-5 text-muted-foreground" />
              <span className="font-semibold text-sm">Magnitude</span>
            </div>
            <p className="text-sm text-muted-foreground">
              How far has the stock fallen from its peak? Bigger drops score higher, up to 40 points 
              for drops approaching 50% or more.
            </p>
          </div>

          <div className="bg-card rounded-xl p-4 border">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="h-5 w-5 text-muted-foreground" />
              <span className="font-semibold text-sm">Rarity</span>
            </div>
            <p className="text-sm text-muted-foreground">
              How unusual is this dip compared to history? We use percentile ranking to award up to 
              25 points for truly exceptional declines.
            </p>
          </div>

          <div className="bg-card rounded-xl p-4 border">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-5 w-5 text-muted-foreground" />
              <span className="font-semibold text-sm">Excess</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Is this dip bigger than the stock's typical pattern? Dips exceeding normal behavior 
              by more than 2x earn up to 20 bonus points.
            </p>
          </div>

          <div className="bg-card rounded-xl p-4 border">
            <div className="flex items-center gap-2 mb-2">
              <CalendarDays className="h-5 w-5 text-muted-foreground" />
              <span className="font-semibold text-sm">Persistence</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Has the dip been confirmed over time? Sustained dips of 3+ days show conviction and 
              can add up to 10 points to the score.
            </p>
          </div>
        </div>
      </div>

      {/* Understanding Dip Types */}
      <div className="space-y-4">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <Scale className="h-5 w-5 text-muted-foreground" />
          Understanding Dip Types
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Not all dips have the same cause. We classify each dip based on how it compares to the 
          overall market movement:
        </p>
        
        <div className="grid gap-3 md:grid-cols-3">
          <div className="bg-card rounded-xl p-4 border flex flex-col gap-3">
            <div className="flex items-center gap-2">
              {getDipTypeBadge('STOCK_SPECIFIC', colorblindMode)}
              <span className="font-semibold text-sm">Stock-Specific</span>
            </div>
            <p className="text-sm text-muted-foreground flex-1">
              The stock is falling much more than the market—at least 2× the market decline. 
              This could signal company-specific news or overreaction to earnings.
            </p>
          </div>

          <div className="bg-card rounded-xl p-4 border flex flex-col gap-3">
            <div className="flex items-center gap-2">
              {getDipTypeBadge('MIXED', colorblindMode)}
              <span className="font-semibold text-sm">Mixed</span>
            </div>
            <p className="text-sm text-muted-foreground flex-1">
              The stock is declining somewhat faster than the market (1.25-2×). 
              Part of the move is market-wide, but there's additional stock-specific pressure.
            </p>
          </div>

          <div className="bg-card rounded-xl p-4 border flex flex-col gap-3">
            <div className="flex items-center gap-2">
              {getDipTypeBadge('MARKET_DIP', colorblindMode)}
              <span className="font-semibold text-sm">Market Dip</span>
            </div>
            <p className="text-sm text-muted-foreground flex-1">
              The stock is moving roughly with the market. The decline is primarily driven by 
              broad market conditions rather than anything specific to this company.
            </p>
          </div>
        </div>
      </div>

      {/* Quality & Stability Deep Dive */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="bg-muted/20 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <Building2 className={`h-5 w-5 ${colorblindMode ? 'text-blue-500' : 'text-success'}`} />
            <span className="font-semibold">Quality Metrics</span>
          </div>
          <div className="space-y-2 text-sm text-muted-foreground">
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Profitability:</strong> Gross margin, operating margin, ROE</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Balance Sheet:</strong> Debt-to-equity, current ratio</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Cash Flow:</strong> Free cash flow generation</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Growth:</strong> Revenue and earnings momentum</span>
            </div>
          </div>
        </div>

        <div className="bg-muted/20 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <Activity className={`h-5 w-5 ${colorblindMode ? 'text-purple-500' : 'text-chart-2'}`} />
            <span className="font-semibold">Stability Metrics</span>
          </div>
          <div className="space-y-2 text-sm text-muted-foreground">
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Volatility:</strong> Daily price fluctuations (lower is better)</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Beta:</strong> Correlation to market moves (closer to 1 is better)</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Max Drawdown:</strong> Worst historical decline</span>
            </div>
            <div className="flex items-start gap-2">
              <CircleCheck className="h-4 w-4 shrink-0 mt-0.5 text-muted-foreground/70" />
              <span><strong>Typical Dip:</strong> Median dip pattern over time</span>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-chart-4/5 border border-chart-4/20 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <TriangleAlert className="h-5 w-5 text-chart-4 shrink-0 mt-0.5" />
          <div className="text-sm text-muted-foreground">
            <p className="font-medium text-foreground mb-1">Not Financial Advice</p>
            <p>
              DipFinder is an educational tool to help identify potential opportunities. 
              Scores are based on historical patterns and fundamental data, not predictions of future performance. 
              Always do your own research before making investment decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
