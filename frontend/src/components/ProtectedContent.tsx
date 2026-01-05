import { useState, useEffect } from 'react';
import { Shield, User, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

/**
 * Bot protection wrapper for sensitive content
 * 
 * Multiple layers of protection:
 * 1. Content not rendered in initial HTML (prevents static scraping)
 * 2. Requires user interaction (click) to reveal
 * 3. Simple challenge that's easy for humans but adds friction for bots
 * 4. Delayed render to ensure JavaScript execution
 * 5. Session storage to remember verification within session
 */

const STORAGE_KEY = 'human_verified';
const VERIFICATION_EXPIRY = 30 * 60 * 1000; // 30 minutes

interface ProtectedContentProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
}

function isVerified(): boolean {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (!stored) return false;
    
    const { timestamp } = JSON.parse(stored);
    const now = Date.now();
    
    // Check if verification is still valid
    if (now - timestamp < VERIFICATION_EXPIRY) {
      return true;
    }
    
    // Expired, clean up
    sessionStorage.removeItem(STORAGE_KEY);
    return false;
  } catch {
    return false;
  }
}

function setVerified(): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      timestamp: Date.now(),
    }));
  } catch {
    // sessionStorage not available
  }
}

export function ProtectedContent({ 
  children, 
  title = 'Contact Information',
  description = 'This information is protected from automated scrapers.'
}: ProtectedContentProps) {
  const [verified, setVerifiedState] = useState(false);
  const [showChallenge, setShowChallenge] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Check if already verified on mount
  useEffect(() => {
    // Delay mount to ensure JS execution (blocks simple SSR scrapers)
    const timer = setTimeout(() => {
      setMounted(true);
      if (isVerified()) {
        setVerifiedState(true);
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);

  function handleVerify() {
    setVerified();
    setVerifiedState(true);
    setShowChallenge(false);
  }

  function handleShowChallenge() {
    setShowChallenge(true);
  }

  // Don't render anything until mounted (prevents SSR/static scraping)
  if (!mounted) {
    return (
      <div className="animate-pulse space-y-2">
        <div className="h-4 bg-muted rounded w-3/4"></div>
        <div className="h-4 bg-muted rounded w-1/2"></div>
        <div className="h-4 bg-muted rounded w-2/3"></div>
      </div>
    );
  }

  // Already verified - show content
  if (verified) {
    return (
      <div className="relative">
        <div className="absolute -left-3 top-0 bottom-0 w-0.5 bg-green-500/30"></div>
        {children}
      </div>
    );
  }

  // Show verification UI
  return (
    <div className="space-y-4 p-4 rounded-lg bg-muted/30 border border-border">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Shield className="h-5 w-5" />
        <span className="font-medium">{title}</span>
      </div>
      
      <p className="text-sm text-muted-foreground">
        {description}
      </p>

      {!showChallenge ? (
        <Button 
          variant="outline" 
          size="sm"
          onClick={handleShowChallenge}
          className="gap-2"
        >
          <User className="h-4 w-4" />
          I'm a human, show me
        </Button>
      ) : (
        <div className="space-y-3 p-3 rounded-md bg-background border">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-warning mt-0.5" />
            <p className="text-sm">
              Please confirm you're not a bot by clicking the button below.
              This helps protect personal data from automated scraping.
            </p>
          </div>
          <Button 
            size="sm"
            onClick={handleVerify}
            className="gap-2"
          >
            <CheckCircle2 className="h-4 w-4" />
            Yes, I'm a human
          </Button>
        </div>
      )}
    </div>
  );
}
