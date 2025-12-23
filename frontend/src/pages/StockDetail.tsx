import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  getStockInfo, 
  getStockChart, 
  getDipCard,
  type StockInfo, 
  type ChartDataPoint,
  type DipCard 
} from '@/services/api';
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { StockChart } from '@/components/StockChart';
import { 
  TrendingDown, 
  ArrowLeft,
  Building2,
  DollarSign,
  BarChart3,
  AlertTriangle,
  Calendar,
  ExternalLink
} from 'lucide-react';

// Generate stock-specific structured data
function generateStockJsonLd(symbol: string, info: StockInfo | null, dipCard: DipCard | null) {
  const baseSchema = {
    '@context': 'https://schema.org',
    '@type': 'FinancialProduct',
    'name': info?.name || symbol,
    'description': `Stock analysis and dip tracking for ${info?.name || symbol} (${symbol}). View recovery potential, price charts, and AI-powered insights.`,
    'category': 'Stock',
    'provider': {
      '@type': 'Organization',
      'name': 'StonkMarket',
      'url': 'https://stonkmarket.de',
    },
  };

  if (dipCard) {
    return {
      ...baseSchema,
      'additionalProperty': [
        {
          '@type': 'PropertyValue',
          'name': 'Dip Percentage',
          'value': `${(dipCard.dip_pct * 100).toFixed(1)}%`,
        },
        {
          '@type': 'PropertyValue',
          'name': 'Days Below High',
          'value': dipCard.days_below?.toString() || 'N/A',
        },
      ],
    };
  }

  return baseSchema;
}

export function StockDetailPage() {
  const { symbol } = useParams<{ symbol: string }>();
  const upperSymbol = symbol?.toUpperCase() || '';
  
  const [info, setInfo] = useState<StockInfo | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [dipCard, setDipCard] = useState<DipCard | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // SEO for stock detail page
  useSEO({
    title: info ? `${info.name} (${upperSymbol}) - Stock Analysis` : `${upperSymbol} Stock Analysis`,
    description: info 
      ? `Track ${info.name} (${upperSymbol}) stock dips and recovery potential. Current sector: ${info.sector || 'N/A'}. View price charts and AI analysis.`
      : `Stock analysis and dip tracking for ${upperSymbol}. View recovery potential and price charts.`,
    keywords: `${upperSymbol}, ${info?.name || ''}, stock analysis, dip tracking, recovery potential, ${info?.sector || 'stocks'}`,
    canonical: `/stock/${upperSymbol.toLowerCase()}`,
    jsonLd: [
      generateBreadcrumbJsonLd([
        { name: 'Home', url: '/' },
        { name: 'Stocks', url: '/' },
        { name: upperSymbol, url: `/stock/${upperSymbol.toLowerCase()}` },
      ]),
      generateStockJsonLd(upperSymbol, info, dipCard),
    ],
  });

  const loadData = useCallback(async () => {
    if (!upperSymbol) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Load all data in parallel
      const [infoData, chartResult, dipCardResult] = await Promise.allSettled([
        getStockInfo(upperSymbol),
        getStockChart(upperSymbol, 365),
        getDipCard(upperSymbol),
      ]);

      if (infoData.status === 'fulfilled') {
        setInfo(infoData.value);
      }
      
      if (chartResult.status === 'fulfilled') {
        setChartData(chartResult.value);
      }
      
      if (dipCardResult.status === 'fulfilled') {
        setDipCard(dipCardResult.value);
      }

      // If none succeeded, show error
      if (infoData.status === 'rejected' && chartResult.status === 'rejected') {
        setError('Stock not found or data unavailable');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stock data');
    } finally {
      setIsLoading(false);
    }
  }, [upperSymbol]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
        <Skeleton className="h-[400px]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <AlertTriangle className="h-16 w-16 text-muted-foreground" />
        <h1 className="text-2xl font-bold">Stock Not Found</h1>
        <p className="text-muted-foreground">{error}</p>
        <Button asChild>
          <Link to="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
      </div>
    );
  }

  const dipPct = dipCard?.dip_pct ? (dipCard.dip_pct * 100).toFixed(1) : null;
  const minDipPct = dipCard?.min_dip_pct ? (dipCard.min_dip_pct * 100).toFixed(1) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Back button */}
      <Button variant="ghost" size="sm" asChild>
        <Link to="/">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Link>
      </Button>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            {info?.name || upperSymbol}
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline">{upperSymbol}</Badge>
            {info?.sector && (
              <Badge variant="secondary">
                <Building2 className="mr-1 h-3 w-3" />
                {info.sector}
              </Badge>
            )}
          </div>
        </div>
        
        {dipCard?.ai_rating && (
          <div className="text-right">
            <div className="text-sm text-muted-foreground">AI Rating</div>
            <Badge variant={
              dipCard.ai_rating === 'strong_buy' || dipCard.ai_rating === 'buy' ? 'default' :
              dipCard.ai_rating === 'hold' ? 'secondary' : 'destructive'
            } className="text-lg px-3 py-1">
              {dipCard.ai_rating.replace('_', ' ').toUpperCase()}
            </Badge>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-3">
        {dipPct && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <TrendingDown className="h-4 w-4" />
                Current Dip
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-destructive">
                -{dipPct}%
              </div>
              <p className="text-xs text-muted-foreground">
                From reference high
              </p>
            </CardContent>
          </Card>
        )}

        {minDipPct && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <TrendingDown className="h-4 w-4" />
                Max Dip
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-chart-4">
                -{minDipPct}%
              </div>
              <p className="text-xs text-muted-foreground">
                Lowest point reached
              </p>
            </CardContent>
          </Card>
        )}

        {info?.market_cap && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Market Cap
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${(info.market_cap / 1e9).toFixed(1)}B
              </div>
              <p className="text-xs text-muted-foreground">
                USD
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Price Chart */}
      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Price History (1 Year)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <StockChart symbol={upperSymbol} data={chartData} />
          </CardContent>
        </Card>
      )}

      {/* AI Analysis */}
      {dipCard?.ai_reasoning && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              AI Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {dipCard.ai_rating && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Rating:</span>
                <Badge variant={
                  dipCard.ai_rating === 'buy' ? 'default' :
                  dipCard.ai_rating === 'hold' ? 'secondary' : 'destructive'
                }>
                  {dipCard.ai_rating.toUpperCase()}
                </Badge>
              </div>
            )}
            <p className="text-sm text-muted-foreground leading-relaxed">
              {dipCard.ai_reasoning}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Disclaimer */}
      <Card className="bg-muted/50 border-dashed">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
            <div className="text-sm text-muted-foreground">
              <strong className="text-foreground">Disclaimer:</strong> This is not financial advice. 
              All information is for educational purposes only. Past performance does not guarantee 
              future results. Always do your own research and consult a financial advisor.
            </div>
          </div>
        </CardContent>
      </Card>

      {/* External Links */}
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" asChild>
          <a 
            href={`https://finance.yahoo.com/quote/${upperSymbol}`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Yahoo Finance
            <ExternalLink className="ml-2 h-3 w-3" />
          </a>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <a 
            href={`https://www.google.com/finance/quote/${upperSymbol}:NASDAQ`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Google Finance
            <ExternalLink className="ml-2 h-3 w-3" />
          </a>
        </Button>
      </div>
    </motion.div>
  );
}
