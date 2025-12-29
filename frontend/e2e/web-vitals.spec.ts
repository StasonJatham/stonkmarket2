import { test, expect, type Page, type Request } from '@playwright/test';

const API_URL = process.env.API_URL || 'http://localhost:8000';

const THRESHOLDS = {
  LCP_MS: Number(process.env.PERF_LCP_MS || 4000),
  CLS: Number(process.env.PERF_CLS || 0.25),
  FCP_MS: Number(process.env.PERF_FCP_MS || 3000),
  PAGE_LOAD_MS: Number(process.env.PERF_PAGE_LOAD_MS || 5000),
  PAGE_LOAD_TIMEOUT_MS: Number(process.env.PERF_PAGE_LOAD_TIMEOUT_MS || 15000),
  NETWORK_IDLE_TIMEOUT_MS: Number(process.env.PERF_NETWORK_IDLE_TIMEOUT_MS || 15000),
  SETTLE_MS: Number(process.env.PERF_SETTLE_MS || 1500),
  API_SLOW_MS: Number(process.env.PERF_API_SLOW_MS || 2000),
};

type WebVitals = {
  lcp: number;
  cls: number;
  fcp: number;
};

type ApiTiming = {
  url: string;
  status: number;
  durationMs: number;
};

const API_PATH_FRAGMENT = '/api/';

function startApiTiming(page: Page): ApiTiming[] {
  const timings: ApiTiming[] = [];
  const startTimes = new Map<Request, number>();

  page.on('request', (request) => {
    const url = request.url();
    if (!url.includes(API_PATH_FRAGMENT)) {
      return;
    }
    if (request.method() !== 'GET') {
      return;
    }
    startTimes.set(request, Date.now());
  });

  page.on('response', (response) => {
    const request = response.request();
    const startTime = startTimes.get(request);
    if (startTime === undefined) {
      return;
    }
    timings.push({
      url: response.url(),
      status: response.status(),
      durationMs: Date.now() - startTime,
    });
    startTimes.delete(request);
  });

  return timings;
}

async function collectMetrics(page: Page, path: string) {
  await page.addInitScript(() => {
    const metrics = { lcp: 0, cls: 0, fcp: 0 };
    const g = globalThis as { __webVitals?: typeof metrics };
    g.__webVitals = metrics;

    const supported =
      typeof PerformanceObserver === 'undefined'
        ? []
        : PerformanceObserver.supportedEntryTypes || [];

    if (supported.includes('largest-contentful-paint')) {
      new PerformanceObserver((list) => {
        const entries = list.getEntries();
        if (entries.length > 0) {
          const lastEntry = entries[entries.length - 1];
          metrics.lcp = lastEntry.startTime;
        }
      }).observe({ type: 'largest-contentful-paint', buffered: true });
    }

    if (supported.includes('layout-shift')) {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const shift = entry as { hadRecentInput?: boolean; value?: number };
          if (!shift.hadRecentInput) {
            metrics.cls += shift.value || 0;
          }
        }
      }).observe({ type: 'layout-shift', buffered: true });
    }

    if (supported.includes('paint')) {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-contentful-paint') {
            metrics.fcp = entry.startTime;
          }
        }
      }).observe({ type: 'paint', buffered: true });
    }
  });

  const apiTimings = startApiTiming(page);
  const startTime = Date.now();

  await page.goto(path, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('load', { timeout: THRESHOLDS.PAGE_LOAD_TIMEOUT_MS });

  const loadTime = Date.now() - startTime;

  await page
    .waitForLoadState('networkidle', {
      timeout: THRESHOLDS.NETWORK_IDLE_TIMEOUT_MS,
    })
    .catch(() => null);

  await page.waitForTimeout(THRESHOLDS.SETTLE_MS);

  const vitals = (await page.evaluate(() => {
    const g = globalThis as { __webVitals?: WebVitals };
    return g.__webVitals || { lcp: 0, cls: 0, fcp: 0 };
  })) as WebVitals;

  return { loadTime, vitals, apiTimings };
}

function assertApiTimings(timings: ApiTiming[], contextLabel: string) {
  const slow = timings.filter((entry) => entry.durationMs > THRESHOLDS.API_SLOW_MS);

  const slowSummary = slow
    .slice(0, 5)
    .map((entry) => `${entry.status} ${entry.durationMs}ms ${entry.url}`)
    .join(' | ');

  expect(
    slow,
    `${contextLabel} slow API responses: ${slowSummary || 'none'}`
  ).toEqual([]);
}

test.describe('Web Vitals & API performance', () => {
  test('dashboard meets core vitals and API budgets', async ({ page }) => {
    const { loadTime, vitals, apiTimings } = await collectMetrics(page, '/');

    expect(loadTime).toBeLessThan(THRESHOLDS.PAGE_LOAD_MS);
    if (vitals.lcp > 0) {
      expect(vitals.lcp).toBeLessThan(THRESHOLDS.LCP_MS);
    }
    expect(vitals.cls).toBeLessThan(THRESHOLDS.CLS);

    if (vitals.fcp > 0) {
      expect(vitals.fcp).toBeLessThan(THRESHOLDS.FCP_MS);
    }

    assertApiTimings(apiTimings, 'Dashboard');

    const imageStats = await page.evaluate(() => {
      const images = Array.from(document.querySelectorAll('img'));
      const sample = images.slice(0, 50);
      let missingDimensions = 0;
      let lazyCount = 0;

      for (const img of sample) {
        const style = getComputedStyle(img);
        const hasWidth = img.hasAttribute('width') || style.width !== 'auto';
        const hasHeight = img.hasAttribute('height') || style.height !== 'auto';

        if (!hasWidth || !hasHeight) {
          missingDimensions += 1;
        }

        if (img.loading === 'lazy') {
          lazyCount += 1;
        }
      }

      return {
        total: images.length,
        sampled: sample.length,
        missingDimensions,
        lazyCount,
      };
    });

    expect(imageStats.missingDimensions).toBeLessThan(5);

    if (imageStats.total > 5) {
      expect(imageStats.lazyCount).toBeGreaterThan(0);
    }
  });

  test('swipe page meets core vitals and API budgets', async ({ page }) => {
    const { loadTime, vitals, apiTimings } = await collectMetrics(page, '/swipe');

    expect(loadTime).toBeLessThan(THRESHOLDS.PAGE_LOAD_MS);
    if (vitals.lcp > 0) {
      expect(vitals.lcp).toBeLessThan(THRESHOLDS.LCP_MS);
    }
    expect(vitals.cls).toBeLessThan(THRESHOLDS.CLS);

    if (vitals.fcp > 0) {
      expect(vitals.fcp).toBeLessThan(THRESHOLDS.FCP_MS);
    }

    assertApiTimings(apiTimings, 'Swipe');
  });
});

test.describe('API response times', () => {
  test('key API endpoints respond quickly', async ({ request }) => {
    const endpoints = [
      {
        name: 'ranking',
        url: `${API_URL}/api/dips/ranking`,
        maxMs: 2000,
      },
      {
        name: 'swipe cards',
        url: `${API_URL}/api/swipe/cards?limit=5`,
        maxMs: 2000,
      },
      {
        name: 'dipfinder latest',
        url: `${API_URL}/api/dipfinder/latest?limit=5`,
        maxMs: 3000,
      },
    ];

    for (const endpoint of endpoints) {
      const startTime = Date.now();
      const response = await request.get(endpoint.url);
      const durationMs = Date.now() - startTime;

      expect(
        response.ok(),
        `${endpoint.name} returned ${response.status()}`
      ).toBe(true);
      expect(
        durationMs,
        `${endpoint.name} took ${durationMs}ms`
      ).toBeLessThan(endpoint.maxMs);
    }
  });
});
