import { useEffect } from 'react';

/**
 * Umami Analytics Integration
 * Privacy-focused, GDPR-compliant analytics
 * 
 * Environment variables:
 * - VITE_UMAMI_WEBSITE_ID: Your Umami website ID
 * - VITE_UMAMI_SRC: URL to your Umami tracking script
 */

declare global {
  interface Window {
    umami?: {
      track: (event: string, data?: Record<string, unknown>) => void;
    };
  }
}

export function useUmami() {
  useEffect(() => {
    const websiteId = import.meta.env.VITE_UMAMI_WEBSITE_ID;
    const src = import.meta.env.VITE_UMAMI_SRC;

    // Only load if configured
    if (!websiteId || !src) {
      return;
    }

    // Check if already loaded
    if (document.querySelector(`script[data-website-id="${websiteId}"]`)) {
      return;
    }

    // Create and inject Umami script
    const script = document.createElement('script');
    script.defer = true;
    script.src = src;
    script.setAttribute('data-website-id', websiteId);
    
    // Umami privacy settings
    script.setAttribute('data-auto-track', 'true');
    script.setAttribute('data-do-not-track', 'true'); // Respect DNT header
    script.setAttribute('data-cache', 'true');

    document.head.appendChild(script);

    return () => {
      // Cleanup on unmount (for hot reload)
      script.remove();
    };
  }, []);
}

// Track custom events
export function trackEvent(event: string, data?: Record<string, unknown>) {
  if (typeof window !== 'undefined' && window.umami) {
    window.umami.track(event, data);
  }
}

// Component to include in App
export function UmamiAnalytics() {
  useUmami();
  return null;
}
