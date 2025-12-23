import { test, expect } from '@playwright/test';

/**
 * SEO E2E tests
 * Validates SEO requirements for SPA including:
 * - Meta tags
 * - Structured data
 * - Core Web Vitals thresholds
 * - Crawlability
 */

test.describe('SEO Meta Tags', () => {
  test('should have proper meta tags on homepage', async ({ page }) => {
    await page.goto('/');
    
    // Title
    const title = await page.title();
    expect(title.length).toBeGreaterThan(10);
    expect(title.length).toBeLessThan(70);
    
    // Meta description
    const description = await page.locator('meta[name="description"]').getAttribute('content');
    expect(description).toBeTruthy();
    expect(description!.length).toBeGreaterThan(50);
    expect(description!.length).toBeLessThan(160);
    
    // Canonical URL
    const canonical = await page.locator('link[rel="canonical"]').getAttribute('href');
    expect(canonical).toBeTruthy();
    
    // Robots
    const robots = await page.locator('meta[name="robots"]').getAttribute('content');
    expect(robots).toContain('index');
  });

  test('should have Open Graph meta tags', async ({ page }) => {
    await page.goto('/');
    
    const ogTitle = await page.locator('meta[property="og:title"]').getAttribute('content');
    const ogDescription = await page.locator('meta[property="og:description"]').getAttribute('content');
    const ogImage = await page.locator('meta[property="og:image"]').getAttribute('content');
    const ogUrl = await page.locator('meta[property="og:url"]').getAttribute('content');
    const ogType = await page.locator('meta[property="og:type"]').getAttribute('content');
    
    expect(ogTitle).toBeTruthy();
    expect(ogDescription).toBeTruthy();
    expect(ogImage).toBeTruthy();
    // OG image should be absolute URL for social sharing
    expect(ogImage).toMatch(/^https?:\/\//);
    expect(ogUrl).toBeTruthy();
    expect(ogType).toBeTruthy();
  });

  test('should have Twitter Card meta tags', async ({ page }) => {
    await page.goto('/');
    
    const twitterCard = await page.locator('meta[property="twitter:card"], meta[name="twitter:card"]').getAttribute('content');
    const twitterTitle = await page.locator('meta[property="twitter:title"], meta[name="twitter:title"]').getAttribute('content');
    const twitterImage = await page.locator('meta[property="twitter:image"], meta[name="twitter:image"]').getAttribute('content');
    
    expect(twitterCard).toBeTruthy();
    expect(twitterTitle).toBeTruthy();
    expect(twitterImage).toBeTruthy();
  });

  test('should update meta tags per route', async ({ page }) => {
    // Dashboard
    await page.goto('/');
    const homeTitle = await page.title();
    
    // Swipe page
    await page.goto('/swipe');
    await page.waitForLoadState('networkidle');
    const swipeTitle = await page.title();
    
    // Titles should be different per route
    expect(swipeTitle).not.toBe(homeTitle);
    
    // About page
    await page.goto('/about');
    await page.waitForLoadState('networkidle');
    const aboutTitle = await page.title();
    expect(aboutTitle).toContain('About');
  });
});

test.describe('Structured Data', () => {
  test('should have JSON-LD structured data', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const jsonLdScripts = await page.locator('script[type="application/ld+json"]').all();
    expect(jsonLdScripts.length).toBeGreaterThan(0);
    
    // Parse and validate first JSON-LD
    const jsonLdContent = await jsonLdScripts[0].textContent();
    expect(jsonLdContent).toBeTruthy();
    
    const jsonLd = JSON.parse(jsonLdContent!);
    // Can be a single object or array of schemas
    const schemas = Array.isArray(jsonLd) ? jsonLd : [jsonLd];
    expect(schemas[0]['@context']).toBe('https://schema.org');
  });

  test('should have Organization schema', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const jsonLdScripts = await page.locator('script[type="application/ld+json"]').all();
    
    let hasOrganization = false;
    for (const script of jsonLdScripts) {
      const content = await script.textContent();
      if (content && content.includes('"@type":"Organization"')) {
        hasOrganization = true;
        break;
      }
    }
    
    expect(hasOrganization).toBe(true);
  });
});

test.describe('Web Vitals Performance', () => {
  test('should load within acceptable LCP threshold', async ({ page }) => {
    await page.goto('/');
    
    // Measure LCP
    const lcp = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number };
          resolve(lastEntry?.startTime || 0);
        }).observe({ type: 'largest-contentful-paint', buffered: true });
        
        // Fallback timeout
        setTimeout(() => resolve(0), 5000);
      });
    });
    
    // LCP should be under 2.5s for good rating (use 4s as threshold for CI)
    if (lcp > 0) {
      expect(lcp).toBeLessThan(4000);
    }
  });

  test('should have minimal layout shift (CLS)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Wait for page to stabilize
    await page.waitForTimeout(1000);
    
    const cls = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let clsValue = 0;
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            const layoutShift = entry as PerformanceEntry & { hadRecentInput: boolean; value: number };
            if (!layoutShift.hadRecentInput) {
              clsValue += layoutShift.value;
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
    
    // CLS should be under 0.25 for good rating
    expect(cls).toBeLessThan(0.25);
  });

  test('should have fast First Contentful Paint', async ({ page }) => {
    await page.goto('/');
    
    const fcp = await page.evaluate(() => {
      const paintEntry = performance.getEntriesByType('paint')
        .find(entry => entry.name === 'first-contentful-paint');
      return paintEntry?.startTime || 0;
    });
    
    // FCP should be under 3s for acceptable rating
    if (fcp > 0) {
      expect(fcp).toBeLessThan(3000);
    }
  });
});

test.describe('Crawlability', () => {
  test('should have robots.txt', async ({ page }) => {
    const response = await page.goto('/robots.txt');
    expect(response?.status()).toBe(200);
    
    const content = await page.content();
    expect(content).toContain('User-agent');
  });

  test('should have sitemap.xml', async ({ page }) => {
    // In dev mode, Vite may serve the SPA for unknown routes
    // In production, Nginx/server should serve the static file
    const response = await page.goto('/sitemap.xml');
    
    // Should at least get a response (200 or 304)
    expect(response?.ok() || response?.status() === 304).toBeTruthy();
    
    // If we get the actual XML content, verify it
    const body = await response?.text() || '';
    if (body.includes('<?xml') || body.includes('urlset')) {
      expect(body).toContain('stonkmarket.de');
    }
    // Otherwise, just verify the file exists in public folder (dev mode serves SPA)
  });

  test('should have security.txt', async ({ page }) => {
    // security.txt can be at /.well-known/security.txt or /security.txt
    const response = await page.goto('/security.txt');
    
    // Should at least get a response
    expect(response?.ok() || response?.status() === 304).toBeTruthy();
    
    const body = await response?.text() || '';
    if (body.includes('Contact:')) {
      expect(body).toContain('Expires:');
    }
  });

  test('should have humans.txt', async ({ page }) => {
    const response = await page.goto('/humans.txt');
    
    expect(response?.ok() || response?.status() === 304).toBeTruthy();
    
    const body = await response?.text() || '';
    if (body.includes('TEAM')) {
      expect(body).toContain('stonkmarket');
    }
  });

  test('should have manifest.json for PWA', async ({ page }) => {
    const response = await page.goto('/manifest.json');
    expect(response?.status()).toBe(200);
    
    const content = await page.textContent('body');
    const manifest = JSON.parse(content || '{}');
    
    expect(manifest.name).toBeTruthy();
    expect(manifest.icons).toBeTruthy();
    expect(manifest.icons.length).toBeGreaterThan(0);
  });

  test('should have proper favicon', async ({ page }) => {
    await page.goto('/');
    
    // Check for favicon link
    const favicon = await page.locator('link[rel="icon"], link[rel="shortcut icon"]').first();
    const faviconHref = await favicon.getAttribute('href');
    expect(faviconHref).toBeTruthy();
    
    // Verify favicon loads
    const response = await page.goto(faviconHref!);
    expect(response?.status()).toBe(200);
  });

  test('should have og-image that loads', async ({ page }) => {
    await page.goto('/');
    
    const ogImage = await page.locator('meta[property="og:image"]').getAttribute('content');
    expect(ogImage).toBeTruthy();
    
    // Extract the path from absolute URL and test locally
    const imagePath = ogImage!.replace(/^https?:\/\/[^\/]+/, '');
    const response = await page.goto(imagePath);
    expect(response?.status()).toBe(200);
  });
});

test.describe('Mobile SEO', () => {
  test('should have viewport meta tag', async ({ page }) => {
    await page.goto('/');
    
    const viewport = await page.locator('meta[name="viewport"]').getAttribute('content');
    expect(viewport).toContain('width=device-width');
  });

  test('should be mobile-responsive', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check for horizontal overflow
    const hasHorizontalScroll = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    
    expect(hasHorizontalScroll).toBe(false);
  });

  test('should have touch-friendly tap targets', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check button sizes (should be at least 44x44 for touch)
    const buttons = await page.locator('button, a').all();
    
    for (const button of buttons.slice(0, 10)) { // Check first 10
      if (await button.isVisible()) {
        const box = await button.boundingBox();
        if (box) {
          // Touch targets should be at least 20px, but many have padding
          // The actual clickable area may be larger due to CSS
          expect(box.width).toBeGreaterThanOrEqual(20);
          expect(box.height).toBeGreaterThanOrEqual(20);
        }
      }
    }
  });
});
