import { Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { useSEO } from '@/lib/seo';
import {
  TrendingDown,
  PieChart,
  Bell,
  Heart,
  ChevronRight,
  Plus,
  Zap,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  TrendingUp,
} from 'lucide-react';
import { getRanking, getDipCards, type DipCard, type DipStock, type RankingResponse } from '@/services/api';
import { useNotificationRules, useNotificationSummary } from '@/features/notifications/api/queries';
import { queryKeys } from '@/lib/query';

// Components
function PortfolioWidget() {
  // Portfolio data not yet implemented - show placeholder
  const hasPortfolio = false;
  const holdingsCount = 0;
  const totalValue = 0;
  const dayChangePercent = 0;

  if (!hasPortfolio || holdingsCount === 0) {
    return (
      <Card className="col-span-1 md:col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PieChart className="h-5 w-5" />
            Portfolio
          </CardTitle>
          <CardDescription>Track your investments</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8 text-center">
          <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mb-4">
            <Plus className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-muted-foreground mb-4">No portfolio yet</p>
          <Button asChild>
            <Link to="/portfolio">
              Create Portfolio
              <ChevronRight className="h-4 w-4 ml-1" />
            </Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const isPositive = dayChangePercent >= 0;

  return (
    <Card className="col-span-1 md:col-span-2">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <PieChart className="h-5 w-5" />
            Portfolio
          </CardTitle>
          <CardDescription>{holdingsCount} holdings</CardDescription>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/portfolio">
            View <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        <div className="flex items-end justify-between">
          <div>
            <div className="text-3xl font-bold">
              ${totalValue.toLocaleString()}
            </div>
            <div className={`flex items-center gap-1 text-sm ${isPositive ? 'text-success' : 'text-destructive'}`}>
              {isPositive ? (
                <ArrowUpRight className="h-4 w-4" />
              ) : (
                <ArrowDownRight className="h-4 w-4" />
              )}
              {isPositive ? '+' : ''}{dayChangePercent.toFixed(2)}% today
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function RecentSignalsWidget() {
  const { data: ranking, isLoading } = useQuery<RankingResponse>({
    queryKey: queryKeys.stocks.ranking(false),
    queryFn: () => getRanking(false, false),
    staleTime: 1000 * 60 * 2,
  });

  const signals: DipStock[] = ranking?.ranking?.slice(0, 5) || [];

  return (
    <Card className="col-span-1 md:col-span-2">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Recent Signals
          </CardTitle>
          <CardDescription>Latest dip opportunities</CardDescription>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/dashboard">
            View All <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : signals.length === 0 ? (
          <p className="text-center text-muted-foreground py-4">No recent signals</p>
        ) : (
          <div className="space-y-2">
            {signals.map(signal => (
              <Link
                key={signal.symbol}
                to={`/stock/${signal.symbol}`}
                className="flex items-center justify-between p-2 rounded-lg hover:bg-muted transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <TrendingDown className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <div className="font-medium">{signal.symbol}</div>
                    <div className="text-xs text-muted-foreground truncate max-w-[150px]">{signal.name}</div>
                  </div>
                </div>
                <Badge variant="outline" className="text-primary">
                  {signal.depth?.toFixed(1)}% dip
                </Badge>
              </Link>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function SwipeMatchesWidget() {
  const { data, isLoading } = useQuery({
    queryKey: queryKeys.dips.cards(10),
    queryFn: () => getDipCards(false, false),
    staleTime: 1000 * 60 * 5,
  });

  const matches: DipCard[] = data?.cards?.slice(0, 10) || [];

  return (
    <Card className="col-span-1 md:col-span-2 lg:col-span-4">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Heart className="h-5 w-5" />
            DipSwipe Opportunities
          </CardTitle>
          <CardDescription>Stocks currently in a dip</CardDescription>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/swipe">
            Swipe <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex gap-4">
            {[1, 2, 3, 4].map(i => (
              <Skeleton key={i} className="h-24 w-24 rounded-lg shrink-0" />
            ))}
          </div>
        ) : matches.length === 0 ? (
          <p className="text-center text-muted-foreground py-4">No dips found</p>
        ) : (
          <ScrollArea className="w-full whitespace-nowrap">
            <div className="flex gap-4">
              {matches.map(match => (
                <Link
                  key={match.symbol}
                  to={`/stock/${match.symbol}`}
                  className="shrink-0 w-28 p-3 rounded-lg border hover:border-primary transition-colors text-center"
                >
                  <Avatar className="h-10 w-10 mx-auto mb-2">
                    <AvatarImage src={`/api/logo/${match.symbol}/light`} alt={match.symbol} />
                    <AvatarFallback className="text-xs">{match.symbol.slice(0, 2)}</AvatarFallback>
                  </Avatar>
                  <div className="font-medium text-sm truncate">{match.symbol}</div>
                  <div className="text-xs text-destructive">
                    {match.dip_pct.toFixed(1)}% dip
                  </div>
                </Link>
              ))}
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}

function NotificationsWidget() {
  const { data: rules = [], isLoading: rulesLoading } = useNotificationRules();
  const { data: summary, isLoading: summaryLoading } = useNotificationSummary();

  const isLoading = rulesLoading || summaryLoading;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/notifications">
            Manage <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Active Rules</span>
              <Badge variant="secondary">{rules.length}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Delivered Today</span>
              <span className="text-sm font-medium">{summary?.notifications_today ?? 0}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function QuickActionsWidget() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Quick Actions
        </CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-2">
        <Button variant="outline" size="sm" asChild className="h-auto py-3 flex-col gap-1">
          <Link to="/swipe">
            <Heart className="h-4 w-4" />
            <span className="text-xs">DipSwipe</span>
          </Link>
        </Button>
        <Button variant="outline" size="sm" asChild className="h-auto py-3 flex-col gap-1">
          <Link to="/portfolio">
            <Plus className="h-4 w-4" />
            <span className="text-xs">Add Holding</span>
          </Link>
        </Button>
        <Button variant="outline" size="sm" asChild className="h-auto py-3 flex-col gap-1">
          <Link to="/notifications">
            <Bell className="h-4 w-4" />
            <span className="text-xs">Create Alert</span>
          </Link>
        </Button>
        <Button variant="outline" size="sm" asChild className="h-auto py-3 flex-col gap-1">
          <Link to="/dashboard">
            <TrendingUp className="h-4 w-4" />
            <span className="text-xs">Explore</span>
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}

export function UserHomePage() {
  const { user } = useAuth();

  useSEO({
    title: 'Home - StonkMarket',
    description: 'Your personalized stock analysis dashboard',
    noindex: true,
  });

  return (
    <div className="container max-w-6xl py-6 space-y-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">
            Welcome back{user?.username ? `, ${user.username}` : ''}
          </h1>
          <p className="text-muted-foreground">
            Here's what's happening with your investments
          </p>
        </div>
      </div>

      {/* Bento Grid */}
      <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        {/* Portfolio Summary - 2 cols */}
        <PortfolioWidget />
        
        {/* Recent Signals - 2 cols */}
        <RecentSignalsWidget />
        
        {/* DipSwipe Matches - Full width */}
        <SwipeMatchesWidget />
        
        {/* Notifications - 1 col */}
        <NotificationsWidget />
        
        {/* Quick Actions - 1 col */}
        <QuickActionsWidget />
      </div>
    </div>
  );
}

export default UserHomePage;
