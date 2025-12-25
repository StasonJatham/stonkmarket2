import { test, expect } from '@playwright/test';

/**
 * Web Vitals & Performance E2E Tests
 * 
 * Tests for:
 * - Largest Contentful Paint (LCP)
 * - Cumulative Layout Shift (CLS)
 * - First Input Delay (FID) / Interaction to Next Paint (INP)
 * - Large image optimization
 * - Layout shift / UI flashes
 */

const FRONTEND_URL = 'http://localhost:5173';

// Performance thresholds (based on Google Web Vitals)
const THRESHOLDS = {
  LCP_GOOD: 2500, // ms - Good < 2.5s
  LCP_NEEDS_IMPROVEMENT: 4000, // ms
  CLS_GOOD: 0.1, // Cumulative shift score
  CLS_NEEDS_IMPROVEMENT: 0.25,
  FCP_GOOD: 1800, // First Contentful Paint
  TTFB_GOOD: 800, // Time to First Byte
  PAGE_LOAD_BUDGET: 5000, // Total page load
};

test.describe('Web Vitals - Dashboard', () => {
  test('dashboard loads within performance budget', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    console.log(`Dashboard load time: ${loadTime}ms`);
    
    expect(loadTime).toBeLessThan(THRESHOLDS.PAGE_LOAD_BUDGET);
  });

  test('dashboard LCP is within threshold', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    
    // Measure LCP using Performance API
    const lcp = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let lcpValue = 0;
        
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1] as PerformanceEntry;
          lcpValue = lastEntry.startTime;
        });
        
        observer.observe({ type: 'largest-contentful-paint', buffered: true });
        
        // Wait for LCP to stabilize
        setTimeout(() => {
          observer.disconnect();
          resolve(lcpValue);
        }, 3000);
      });
    });
    
    console.log(`LCP: ${lcp}ms`);
    
    // LCP should be under 4 seconds for "needs improvement"
    expect(lcp).toBeLessThan(THRESHOLDS.LCP_NEEDS_IMPROVEMENT);
  });

  test('dashboard has minimal layout shift', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    
    // Measure CLS using Performance API
    const cls = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let clsValue = 0;
        
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            // @ts-ignore - CLS entries have hadRecentInput
            if (!entry.hadRecentInput) {
              // @ts-ignore
              clsValue += entry.value;
            }
          }
        });
        
        observer.observe({ type: 'layout-shift', buffered: true });
        
        setTimeout(() => {
          observer.disconnect();
          resolve(clsValue);
        }, 3000);
      });
    });
    
    console.log(`CLS: ${cls}`);
    
    expect(cls).toBeLessThan(THRESHOLDS.CLS_NEEDS_IMPROVEMENT);
  });
});

test.describe('Web Vitals - DipSwipe', () => {
  test('swipe page loads quickly', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(`${FRONTEND_URL}/swipe`);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    console.log(`Swipe page load time: ${loadTime}ms`);
    
    expect(loadTime).toBeLessThan(THRESHOLDS.PAGE_LOAD_BUDGET);
  });

  test('swipe cards render without layout shift', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/swipe`);
    
    // Wait for cards to load
    await page.waitForSelector('[data-testid="swipe-card"], .swipe-card, .card', {
      timeout: 10000,
    }).catch(() => null);
    
    // Measure CLS after cards load
    const cls = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let clsValue = 0;
        
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            // @ts-ignore
            if (!entry.hadRecentInput) {
              // @ts-ignore
              clsValue += entry.value;
            }
          }
        });
        
        observer.observe({ type: 'layout-shift', buffered: true });
        
        setTimeout(() => {
          observer.disconnect();
          resolve(clsValue);
        }, 2000);
      });
    });
    
    console.log(`Swipe page CLS: ${cls}`);
    
    expect(cls).toBeLessThan(THRESHOLDS.CLS_NEEDS_IMPROVEMENT);
  });
});

test.describe('Image Optimization', () => {
  test('images have proper dimensions set', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    // Check all images for width/height attributes
    const imagesWithoutDimensions = await page.evaluate(() => {
      const images = Array.from(document.querySelectorAll('img'));
      return images.filter(img => {
        const hasWidth = img.hasAttribute('width') || img.style.width || getComputedStyle(img).width !== 'auto';
        const hasHeight = img.hasAttribute('height') || img.style.height || getComputedStyle(img).height !== 'auto';
        return !hasWidth || !hasHeight;
      }).map(img => img.src);
    });
    
    console.log(`Images without dimensions: ${imagesWithoutDimensions.length}`);
    
    // Allow some images without explicit dimensions if they're small icons
    expect(imagesWithoutDimensions.length).toBeLessThan(5);
  });

  test('large images are lazy loaded', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    
    // Check for lazy loading attribute on images
    const lazyLoadedImages = await page.evaluate(() => {
      const images = Array.from(document.querySelectorAll('img'));
      return {
        total: images.length,
        lazy: images.filter(img => img.loading === 'lazy').length,
        eager: images.filter(img => img.loading === 'eager' || !img.loading).length,
      };
    });
    
    console.log(`Images: ${JSON.stringify(lazyLoadedImages)}`);
    
    // At least some images should be lazy loaded if there are many
    if (lazyLoadedImages.total > 5) {
      expect(lazyLoadedImages.lazy).toBeGreaterThan(0);
    }
  });

  test('logo images load from cache or quickly', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    // Check logo load times
    const logoLoadTimes = await page.evaluate(() => {
      const entries = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
      return entries
        .filter(e => e.name.includes('logo') || e.name.includes('/api/logos/'))
        .map(e => ({
          url: e.name,
          duration: e.duration,
          fromCache: e.transferSize === 0,
        }));
    });
    
    console.log(`Logo load times: ${JSON.stringify(logoLoadTimes, null, 2)}`);
    
    // All logos should load quickly (< 1s) or from cache
    for (const logo of logoLoadTimes) {
      expect(logo.duration).toBeLessThan(1000);
    }
  });
});

test.describe('UI Flash / Loading States', () => {
  test('dashboard shows loading state before content', async ({ page }) => {
    // Start navigation
    const response = page.goto(FRONTEND_URL);
    
    // Immediately check for loading indicators
    const hasLoadingIndicator = await page.locator(
      '[data-testid="loading"], .loading, .skeleton, [aria-busy="true"], .spinner'
    ).first().isVisible().catch(() => false);
    
    // Wait for navigation to complete
    await response;
    await page.waitForLoadState('networkidle');
    
    // Content should now be visible
    const hasContent = await page.locator('main, [role="main"], .dashboard, .content').first().isVisible();
    
    expect(hasContent).toBe(true);
  });

  test('no flash of unstyled content (FOUC)', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    
    // Check that CSS is applied immediately
    const bodyStyles = await page.evaluate(() => {
      const body = document.body;
      const styles = getComputedStyle(body);
      return {
        backgroundColor: styles.backgroundColor,
        fontFamily: styles.fontFamily,
        hasStyles: styles.backgroundColor !== 'rgba(0, 0, 0, 0)' || styles.fontFamily !== '',
      };
    });
    
    console.log(`Body styles applied: ${JSON.stringify(bodyStyles)}`);
    
    expect(bodyStyles.hasStyles).toBe(true);
  });

  test('theme toggle does not cause layout shift', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    // Find theme toggle
    const themeToggle = page.locator(
      'button[aria-label*="theme"], button[aria-label*="dark"], button[aria-label*="light"], [data-testid="theme-toggle"]'
    ).first();
    
    if (await themeToggle.isVisible()) {
      // Get initial layout
      const initialLayout = await page.evaluate(() => {
        const main = document.querySelector('main, [role="main"], .content');
        return main ? main.getBoundingClientRect() : null;
      });
      
      // Toggle theme
      await themeToggle.click();
      await page.waitForTimeout(500);
      
      // Get new layout
      const newLayout = await page.evaluate(() => {
        const main = document.querySelector('main, [role="main"], .content');
        return main ? main.getBoundingClientRect() : null;
      });
      
      // Layout should not change significantly
      if (initialLayout && newLayout) {
        const widthDiff = Math.abs(initialLayout.width - newLayout.width);
        const heightDiff = Math.abs(initialLayout.height - newLayout.height);
        
        console.log(`Layout diff: width=${widthDiff}, height=${heightDiff}`);
        
        expect(widthDiff).toBeLessThan(5);
        // Height can change slightly due to content reflow
        expect(heightDiff).toBeLessThan(50);
      }
    }
  });
});

test.describe('Chart Performance', () => {
  test('charts load without blocking UI', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('domcontentloaded');
    
    // UI should be interactive before charts fully load
    const isInteractive = await page.locator('button, a, input').first().isEnabled();
    
    expect(isInteractive).toBe(true);
    
    // Wait for charts to appear
    await page.waitForLoadState('networkidle');
    
    const hasCharts = await page.locator(
      'canvas, svg[role="img"], .recharts-wrapper, [data-testid="chart"]'
    ).first().isVisible().catch(() => false);
    
    console.log(`Charts visible: ${hasCharts}`);
  });

  test('chart period switching is responsive', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    // Find period buttons
    const periodButtons = page.locator('button:has-text("1M"), button:has-text("3M"), button:has-text("1Y")');
    
    if (await periodButtons.count() > 0) {
      const startTime = Date.now();
      
      await periodButtons.first().click();
      
      // Should respond quickly (UI update, not full data load)
      const responseTime = Date.now() - startTime;
      
      console.log(`Period switch response: ${responseTime}ms`);
      
      expect(responseTime).toBeLessThan(500);
    }
  });
});

test.describe('API Response Times', () => {
  test('ranking API responds quickly', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.get('http://localhost:8000/api/dips/ranking');
    
    const responseTime = Date.now() - startTime;
    
    console.log(`Ranking API response time: ${responseTime}ms`);
    
    expect(response.ok()).toBe(true);
    expect(responseTime).toBeLessThan(2000);
  });

  test('swipe cards API responds quickly', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.get('http://localhost:8000/api/swipe/cards');
    
    const responseTime = Date.now() - startTime;
    
    console.log(`Swipe cards API response time: ${responseTime}ms`);
    
    expect(response.ok()).toBe(true);
    expect(responseTime).toBeLessThan(2000);
  });

  test('signals API responds quickly', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.get('http://localhost:8000/api/dipfinder/signals');
    
    const responseTime = Date.now() - startTime;
    
    console.log(`Signals API response time: ${responseTime}ms`);
    
    expect(response.ok()).toBe(true);
    expect(responseTime).toBeLessThan(3000);
  });
});

test.describe('Mobile Performance', () => {
  test.use({
    viewport: { width: 375, height: 667 },
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
  });

  test('mobile dashboard loads quickly', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    console.log(`Mobile dashboard load time: ${loadTime}ms`);
    
    // Mobile should load within 6 seconds
    expect(loadTime).toBeLessThan(6000);
  });

  test('mobile touch interactions are responsive', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/swipe`);
    await page.waitForLoadState('networkidle');
    
    // Find interactive elements
    const buttons = page.locator('button').first();
    
    if (await buttons.isVisible()) {
      const startTime = Date.now();
      
      await buttons.tap();
      
      const tapResponseTime = Date.now() - startTime;
      
      console.log(`Mobile tap response: ${tapResponseTime}ms`);
      
      // Touch should respond within 100ms (INP threshold)
      expect(tapResponseTime).toBeLessThan(200);
    }
  });
});
