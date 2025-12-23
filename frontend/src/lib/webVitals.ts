/**
 * Web Vitals measurement and reporting module
 * 
 * Captures Core Web Vitals metrics:
 * - LCP (Largest Contentful Paint): Loading performance
 * - INP (Interaction to Next Paint): Interactivity
 * - CLS (Cumulative Layout Shift): Visual stability
 * - FCP (First Contentful Paint): Initial rendering
 * - TTFB (Time to First Byte): Server response time
 */

// Metric types
export interface WebVitalsMetric {
  name: 'LCP' | 'INP' | 'CLS' | 'FCP' | 'TTFB';
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  id: string;
  navigationType: string;
}

// Thresholds for each metric (in milliseconds or unitless for CLS)
const THRESHOLDS = {
  LCP: { good: 2500, poor: 4000 },      // ms
  INP: { good: 200, poor: 500 },         // ms
  CLS: { good: 0.1, poor: 0.25 },        // unitless
  FCP: { good: 1800, poor: 3000 },       // ms
  TTFB: { good: 800, poor: 1800 },       // ms
};

// Rate metric performance
function getRating(name: keyof typeof THRESHOLDS, value: number): 'good' | 'needs-improvement' | 'poor' {
  const { good, poor } = THRESHOLDS[name];
  if (value <= good) return 'good';
  if (value >= poor) return 'poor';
  return 'needs-improvement';
}

// Collected metrics storage
const collectedMetrics: WebVitalsMetric[] = [];

// Callback type
type MetricCallback = (metric: WebVitalsMetric) => void;
const callbacks: MetricCallback[] = [];

/**
 * Subscribe to Web Vitals metrics
 */
export function onWebVitals(callback: MetricCallback): void {
  callbacks.push(callback);
  // Send any already collected metrics
  collectedMetrics.forEach(callback);
}

/**
 * Report a metric to all subscribers
 */
function reportMetric(metric: WebVitalsMetric): void {
  collectedMetrics.push(metric);
  callbacks.forEach(cb => cb(metric));
  
  // Log in development
  if (import.meta.env.DEV) {
    const color = metric.rating === 'good' ? 'green' : metric.rating === 'poor' ? 'red' : 'orange';
    console.log(
      `%c[Web Vitals] ${metric.name}: ${metric.value.toFixed(2)} (${metric.rating})`,
      `color: ${color}; font-weight: bold;`
    );
  }
}

/**
 * Initialize Web Vitals collection using PerformanceObserver
 */
export function initWebVitals(): void {
  if (typeof window === 'undefined' || !('PerformanceObserver' in window)) {
    console.warn('[Web Vitals] PerformanceObserver not supported');
    return;
  }

  // LCP - Largest Contentful Paint
  try {
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number };
      if (lastEntry) {
        reportMetric({
          name: 'LCP',
          value: lastEntry.startTime,
          rating: getRating('LCP', lastEntry.startTime),
          delta: lastEntry.startTime,
          id: crypto.randomUUID?.() || String(Date.now()),
          navigationType: 'navigate',
        });
      }
    });
    lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
  } catch (e) {
    // LCP not supported
  }

  // FCP - First Contentful Paint
  try {
    const fcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const fcpEntry = entries.find(e => e.name === 'first-contentful-paint');
      if (fcpEntry) {
        reportMetric({
          name: 'FCP',
          value: fcpEntry.startTime,
          rating: getRating('FCP', fcpEntry.startTime),
          delta: fcpEntry.startTime,
          id: crypto.randomUUID?.() || String(Date.now()),
          navigationType: 'navigate',
        });
      }
    });
    fcpObserver.observe({ type: 'paint', buffered: true });
  } catch (e) {
    // FCP not supported
  }

  // CLS - Cumulative Layout Shift
  try {
    let clsValue = 0;
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const layoutShift = entry as PerformanceEntry & { hadRecentInput: boolean; value: number };
        if (!layoutShift.hadRecentInput) {
          clsValue += layoutShift.value;
        }
      }
    });
    clsObserver.observe({ type: 'layout-shift', buffered: true });

    // Report CLS on page hide
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        reportMetric({
          name: 'CLS',
          value: clsValue,
          rating: getRating('CLS', clsValue),
          delta: clsValue,
          id: crypto.randomUUID?.() || String(Date.now()),
          navigationType: 'navigate',
        });
      }
    });
  } catch (e) {
    // CLS not supported
  }

  // TTFB - Time to First Byte
  try {
    const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (navEntry) {
      const ttfb = navEntry.responseStart - navEntry.requestStart;
      reportMetric({
        name: 'TTFB',
        value: ttfb,
        rating: getRating('TTFB', ttfb),
        delta: ttfb,
        id: crypto.randomUUID?.() || String(Date.now()),
        navigationType: 'navigate',
      });
    }
  } catch (e) {
    // TTFB not supported
  }

  // INP - Interaction to Next Paint (approximation using event timing)
  try {
    let maxINP = 0;
    const inpObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const eventTiming = entry as PerformanceEntry & { processingStart: number; startTime: number; duration: number };
        const inp = eventTiming.processingStart - eventTiming.startTime + eventTiming.duration;
        if (inp > maxINP) {
          maxINP = inp;
        }
      }
    });
    inpObserver.observe({ type: 'event', buffered: true });

    // Report INP on page hide
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden' && maxINP > 0) {
        reportMetric({
          name: 'INP',
          value: maxINP,
          rating: getRating('INP', maxINP),
          delta: maxINP,
          id: crypto.randomUUID?.() || String(Date.now()),
          navigationType: 'navigate',
        });
      }
    });
  } catch (e) {
    // INP not supported
  }
}

/**
 * Send collected metrics to the backend for aggregation
 */
export async function sendWebVitals(endpoint = '/api/metrics/vitals'): Promise<void> {
  if (collectedMetrics.length === 0) return;

  try {
    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        metrics: collectedMetrics,
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
      }),
      keepalive: true, // Ensure request completes even if page unloads
    });
  } catch (e) {
    console.warn('[Web Vitals] Failed to send metrics:', e);
  }
}

/**
 * Get summary of collected metrics
 */
export function getWebVitalsSummary(): Record<string, { value: number; rating: string } | undefined> {
  const summary: Record<string, { value: number; rating: string }> = {};
  
  for (const metric of collectedMetrics) {
    // Keep the latest value for each metric
    summary[metric.name] = { value: metric.value, rating: metric.rating };
  }
  
  return summary;
}

/**
 * Check if all Core Web Vitals pass thresholds
 */
export function passesWebVitals(): boolean {
  const required = ['LCP', 'INP', 'CLS'];
  const summary = getWebVitalsSummary();
  
  return required.every(name => {
    const metric = summary[name];
    return metric && metric.rating !== 'poor';
  });
}
