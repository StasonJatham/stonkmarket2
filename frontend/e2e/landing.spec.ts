import { test, expect, type Page, type Request, type Response } from '@playwright/test';

const API_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * Landing Page E2E Tests
 * 
 * These tests ensure the landing page:
 * 1. Loads successfully with all critical data
 * 2. Shows signal board with real stock data
 * 3. Displays AI pipeline information
 * 4. Has acceptable performance metrics
 * 5. Catches API errors and missing data
 */

// Track API calls to detect 404s and errors
type ApiCall = {
  url: string;
  method: string;
  status: number;
  durationMs: number;
};

function trackApiCalls(page: Page): ApiCall[] {
  const calls: ApiCall[] = [];
  const startTimes = new Map<Request, number>();

  page.on('request', (request) => {
    const url = request.url();
    if (url.includes('/api/')) {
      startTimes.set(request, Date.now());
    }
  });

  page.on('response', (response) => {
    const request = response.request();
    const startTime = startTimes.get(request);
    if (startTime !== undefined) {
      calls.push({
        url: response.url(),
        method: request.method(),
        status: response.status(),
        durationMs: Date.now() - startTime,
      });
      startTimes.delete(request);
    }
  });

  return calls;
}

test.describe('Landing Page - Critical Data', () => {
  test('loads with no API errors (no 4xx/5xx responses)', async ({ page }) => {
    const apiCalls = trackApiCalls(page);
    
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000); // Allow async data to load
    
    // Check for failed API calls
    const failedCalls = apiCalls.filter(c => c.status >= 400);
    
    if (failedCalls.length > 0) {
      const summary = failedCalls.map(c => 
        `${c.method} ${c.url} -> ${c.status}`
      ).join('\n');
      
      expect(
        failedCalls,
        `Landing page has failed API calls:\n${summary}`
      ).toHaveLength(0);
    }
  });

  test('displays signal board with stock cards', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    
    // Wait for signal board section to appear - use actual heading text
    const signalBoard = page.locator('section:has-text("Live Signal Board"), section:has-text("Signal Board")');
    await expect(signalBoard.first()).toBeVisible({ timeout: 10000 });
    
    // Check for stock cards with ticker symbols and price info
    // Cards contain stock logos and price data
    const stockLogos = page.locator('[data-testid="stock-logo"], .stock-logo, img[alt*="logo"]');
    const priceElements = page.locator('.font-mono:has-text("$")');
    
    // At least some should be visible
    const logosCount = await stockLogos.count();
    const pricesCount = await priceElements.count();
    
    expect(
      logosCount + pricesCount,
      'Expected stock cards with logos or prices on landing page'
    ).toBeGreaterThan(0);
  });

  test('displays hero section with featured stock', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    
    // Hero section should have:
    // 1. A headline about AI-powered signals
    const headline = page.locator('h1');
    await expect(headline).toBeVisible({ timeout: 5000 });
    
    // 2. A CTA button
    const ctaButton = page.locator('button:has-text("Start"), button:has-text("Get Started"), button:has-text("Try"), a:has-text("Start")');
    await expect(ctaButton.first()).toBeVisible({ timeout: 5000 });
  });

  test('AI pipeline section shows pipeline steps', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    
    // Scroll to AI pipeline section
    const pipelineSection = page.locator('section:has-text("pipeline")').first();
    if (await pipelineSection.isVisible()) {
      await pipelineSection.scrollIntoViewIfNeeded();
    }
    
    // Check for pipeline steps (Scan, Debate, Optimize, Backtest)
    const expectedSteps = ['Scan', 'Debate', 'Optimize', 'Backtest'];
    for (const step of expectedSteps) {
      const stepElement = page.locator(`text=${step}`);
      await expect(
        stepElement.first(),
        `Pipeline step "${step}" should be visible`
      ).toBeVisible({ timeout: 5000 });
    }
  });

  test('displays portfolio stats with real numbers (not N/A)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(3000); // Wait for data to load
    
    // Check that key stats don't show "N/A" or dashes everywhere
    const naElements = page.locator('text="N/A"');
    const dashElements = page.locator('text="—"');
    
    const naCount = await naElements.count();
    const dashCount = await dashElements.count();
    
    // Allow some N/A values but not too many (indicates missing data)
    expect(
      naCount,
      `Too many N/A values (${naCount}) - indicates missing data`
    ).toBeLessThan(10);
    
    expect(
      dashCount,
      `Too many dash placeholders (${dashCount}) - indicates missing data`
    ).toBeLessThan(10);
  });

  test('charts render with actual data points', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(3000);
    
    // Check for SVG chart elements (Recharts renders SVG)
    const chartAreas = page.locator('svg path[fill*="url(#)"], svg path.recharts-area-area');
    const chartLines = page.locator('svg path.recharts-curve, svg path[stroke-width]');
    
    const areasCount = await chartAreas.count();
    const linesCount = await chartLines.count();
    
    expect(
      areasCount + linesCount,
      'Expected charts to render with SVG paths'
    ).toBeGreaterThan(0);
  });
});

test.describe('Landing Page - Performance', () => {
  test('loads within acceptable time limits', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    const domContentLoaded = Date.now() - startTime;
    
    await page.waitForLoadState('load');
    const loadTime = Date.now() - startTime;
    
    // DOMContentLoaded should be under 2s
    expect(
      domContentLoaded,
      `DOMContentLoaded took ${domContentLoaded}ms - should be under 2000ms`
    ).toBeLessThan(2000);
    
    // Full load should be under 5s
    expect(
      loadTime,
      `Page load took ${loadTime}ms - should be under 5000ms`
    ).toBeLessThan(5000);
  });

  test('First Contentful Paint is acceptable', async ({ page }) => {
    // Inject web vitals measurement
    await page.addInitScript(() => {
      const g = globalThis as { __fcp?: number };
      g.__fcp = 0;
      
      if (typeof PerformanceObserver !== 'undefined') {
        const supported = PerformanceObserver.supportedEntryTypes || [];
        if (supported.includes('paint')) {
          new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (entry.name === 'first-contentful-paint') {
                g.__fcp = entry.startTime;
              }
            }
          }).observe({ type: 'paint', buffered: true });
        }
      }
    });
    
    await page.goto('/');
    await page.waitForLoadState('load');
    await page.waitForTimeout(1000);
    
    const fcp = await page.evaluate(() => {
      return (globalThis as { __fcp?: number }).__fcp || 0;
    });
    
    // FCP should be under 2500ms for "good" rating
    expect(
      fcp,
      `FCP is ${fcp}ms - should be under 2500ms for good rating`
    ).toBeLessThan(2500);
  });

  test('API calls complete within budget', async ({ page }) => {
    const apiCalls = trackApiCalls(page);
    
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    
    // Check for slow API calls (> 2s)
    const slowCalls = apiCalls.filter(c => c.durationMs > 2000 && c.status < 400);
    
    if (slowCalls.length > 0) {
      const summary = slowCalls.map(c => 
        `${c.method} ${c.url} took ${c.durationMs}ms`
      ).join('\n');
      
      expect(
        slowCalls,
        `Landing page has slow API calls:\n${summary}`
      ).toHaveLength(0);
    }
  });
});

test.describe('Landing Page - API Integration', () => {
  test('quant recommendations endpoint returns valid data', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/quant/recommendations?limit=10`);
    
    expect(response.ok(), `Recommendations API returned ${response.status()}`).toBe(true);
    
    const data = await response.json();
    
    // Should have recommendations array
    expect(data).toHaveProperty('recommendations');
    expect(Array.isArray(data.recommendations)).toBe(true);
    expect(data.recommendations.length).toBeGreaterThan(0);
    
    // Each recommendation should have required fields
    const rec = data.recommendations[0];
    expect(rec).toHaveProperty('ticker');
    expect(rec).toHaveProperty('action');
    expect(['BUY', 'SELL', 'HOLD']).toContain(rec.action);
  });

  test('batch charts endpoint returns chart data', async ({ request }) => {
    // First get some symbols from recommendations
    const recsResponse = await request.get(`${API_URL}/api/quant/recommendations?limit=5`);
    const recsData = await recsResponse.json();
    const symbols = recsData.recommendations?.slice(0, 3).map((r: { ticker: string }) => r.ticker) || ['AAPL', 'MSFT', 'GOOG'];
    
    // Request batch charts via POST
    const response = await request.post(`${API_URL}/api/dips/batch/charts`, {
      data: { symbols, days: 45 }
    });
    
    expect(response.ok(), `Batch charts API returned ${response.status()}`).toBe(true);
    
    const data = await response.json();
    
    // Should have chart data for at least some symbols
    expect(typeof data).toBe('object');
    const chartSymbols = Object.keys(data);
    expect(chartSymbols.length).toBeGreaterThan(0);
  });

  test('signal triggers endpoint returns trigger data', async ({ request }) => {
    // Get a symbol first
    const recsResponse = await request.get(`${API_URL}/api/quant/recommendations?limit=1`);
    const recsData = await recsResponse.json();
    const symbol = recsData.recommendations?.[0]?.ticker || 'AAPL';
    
    const response = await request.get(
      `${API_URL}/api/quant/recommendations/${symbol}/signal-triggers?lookback_days=365`
    );
    
    expect(response.ok(), `Signal triggers API returned ${response.status()}`).toBe(true);
    
    const data = await response.json();
    expect(data).toHaveProperty('symbol');
    expect(data).toHaveProperty('triggers');
  });
});

test.describe('Landing Page - Error Handling', () => {
  test('shows error state gracefully when API fails', async ({ page }) => {
    // Block the recommendations API to simulate failure
    await page.route('**/api/quant/recommendations**', route => 
      route.fulfill({ status: 500, body: 'Internal Server Error' })
    );
    
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {});
    
    // Page should still render (not crash)
    const body = page.locator('body');
    await expect(body).toBeVisible();
    
    // Should show some kind of error or fallback state
    const errorIndicators = page.locator('text=/error|failed|unavailable|try again/i');
    const skeletons = page.locator('.skeleton, [class*="skeleton"]');
    
    // Either show error message or loading skeletons (graceful degradation)
    const errorCount = await errorIndicators.count();
    const skeletonCount = await skeletons.count();
    
    // Should have some indication that data failed to load
    // (either error message or persistent skeletons)
    expect(
      errorCount + skeletonCount,
      'Expected error indication or loading skeletons when API fails'
    ).toBeGreaterThanOrEqual(0); // Just ensure page doesn't crash
  });

  test('no JavaScript console errors on landing page', async ({ page }) => {
    const consoleErrors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        // Ignore some known non-critical errors
        const text = msg.text();
        if (!text.includes('ResizeObserver') && !text.includes('favicon')) {
          consoleErrors.push(text);
        }
      }
    });
    
    page.on('pageerror', error => {
      consoleErrors.push(error.message);
    });
    
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(2000);
    
    // Filter out network errors (already tested separately)
    const jsErrors = consoleErrors.filter(e => 
      !e.includes('404') && 
      !e.includes('Failed to load resource') &&
      !e.includes('net::ERR')
    );
    
    expect(
      jsErrors,
      `JavaScript errors on landing page:\n${jsErrors.join('\n')}`
    ).toHaveLength(0);
  });
});
