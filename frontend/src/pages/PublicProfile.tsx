import { useParams, Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Lock, Eye, Settings, Calendar, TrendingUp, ThumbsUp, ThumbsDown, Share2 } from 'lucide-react';
import { useSEO } from '@/lib/seo';

type ProfileState = 'loading' | 'owner' | 'public' | 'private';

export function PublicProfilePage() {
  const { username } = useParams<{ username: string }>();
  const { user: currentUser } = useAuth();

  useSEO({
    title: `@${username} - StonkMarket`,
    description: `View ${username}'s profile on StonkMarket`,
  });

  // Determine profile state
  const isOwner = currentUser?.username === username;
  
  // For now, all non-owner profiles are private (no public profile API yet)
  // This will be updated when we add the API endpoint
  const profileState: ProfileState = isOwner ? 'owner' : 'private';

  const initials = username?.slice(0, 2).toUpperCase() || 'U';

  // Owner viewing their own public profile
  if (profileState === 'owner') {
    return (
      <div className="min-h-[calc(100vh-4rem)]">
        {/* Owner Banner */}
        <div className="bg-primary/10 border-b border-primary/20 px-4 py-3">
          <div className="container max-w-4xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <Eye className="h-4 w-4 text-primary" />
              <span className="text-muted-foreground">
                You're viewing your public profile as others see it
              </span>
            </div>
            <Button variant="outline" size="sm" asChild>
              <Link to="/settings/profile">
                <Settings className="h-4 w-4 mr-2" />
                Edit Profile
              </Link>
            </Button>
          </div>
        </div>

        <div className="container max-w-4xl py-8 space-y-6">
          {/* Profile Header */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
                <Avatar className="h-24 w-24">
                  <AvatarImage src={undefined} alt={username} />
                  <AvatarFallback className="text-2xl">{initials}</AvatarFallback>
                </Avatar>
                
                <div className="flex-1 text-center sm:text-left">
                  <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-2">
                    <h1 className="text-2xl font-bold">@{username}</h1>
                    {currentUser?.is_admin && (
                      <Badge variant="outline" className="w-fit mx-auto sm:mx-0">Admin</Badge>
                    )}
                  </div>
                  <div className="flex items-center justify-center sm:justify-start gap-2 text-sm text-muted-foreground">
                    <Calendar className="h-4 w-4" />
                    <span>Member since {new Date().getFullYear()}</span>
                  </div>
                </div>

                <Button variant="outline" size="sm" className="gap-2">
                  <Share2 className="h-4 w-4" />
                  Share Profile
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Stats Overview */}
          <div className="grid gap-4 sm:grid-cols-3">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <TrendingUp className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold">0</div>
                    <div className="text-sm text-muted-foreground">Watchlist Items</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-success/10 flex items-center justify-center">
                    <ThumbsUp className="h-5 w-5 text-success" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold">0</div>
                    <div className="text-sm text-muted-foreground">Dips Liked</div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-destructive/10 flex items-center justify-center">
                    <ThumbsDown className="h-5 w-5 text-destructive" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold">0</div>
                    <div className="text-sm text-muted-foreground">Dips Passed</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Public Watchlist */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Public Watchlist
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <p>Your public watchlist is empty.</p>
                <p className="text-sm mt-1">
                  Add stocks to your watchlist and enable sharing in{' '}
                  <Link to="/settings/privacy" className="text-primary hover:underline">
                    Privacy Settings
                  </Link>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  // Private profile (fallback for non-owner)
  return (
    <div className="container max-w-2xl py-12">
      <Card className="text-center">
        <CardHeader className="pb-2">
          <div className="flex flex-col items-center gap-4">
            <Avatar className="h-24 w-24">
              <AvatarFallback className="text-2xl bg-muted">
                <Lock className="h-8 w-8 text-muted-foreground" />
              </AvatarFallback>
            </Avatar>
            <div className="space-y-2">
              <h1 className="text-2xl font-bold">@{username}</h1>
              <Badge variant="secondary" className="gap-1">
                <Lock className="h-3 w-3" />
                Private
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            This profile is private. The user has chosen not to share their profile publicly.
          </p>
          
          {currentUser && (
            <Button variant="outline" asChild>
              <Link to="/dashboard">
                <TrendingUp className="h-4 w-4 mr-2" />
                Explore Stocks
              </Link>
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default PublicProfilePage;
