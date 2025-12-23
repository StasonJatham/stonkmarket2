import { test, expect } from '@playwright/test';

/**
 * Dashboard page E2E tests
 * Tests the main dashboard functionality including:
 * - Page load and basic rendering
 * - Stock cards display
 * - Navigation and interactions
 */

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the dashboard page', async ({ page }) => {
    // Check that the page loads without errors
    await expect(page).toHaveTitle(/stonkmarket|dip/i);
  });

  test('should display header with navigation', async ({ page }) => {
    // Check for main navigation elements
    const header = page.locator('header');
    await expect(header).toBeVisible();
  });

  test('should display stock cards or ranking table', async ({ page }) => {
    // Wait for data to load
    await page.waitForLoadState('networkidle');
    
    // Should have either stock cards or a ranking table
    const hasStockCards = await page.locator('[data-testid="stock-card"]').count() > 0;
    const hasRankingTable = await page.locator('table').count() > 0;
    const hasStockList = await page.locator('[class*="card"], [class*="stock"]').count() > 0;
    
    expect(hasStockCards || hasRankingTable || hasStockList).toBeTruthy();
  });

  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    
    // Page should still be usable
    const body = page.locator('body');
    await expect(body).toBeVisible();
    
    // No horizontal overflow
    const bodyWidth = await body.evaluate(el => el.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 10); // Small tolerance
  });

  test('should toggle dark/light theme', async ({ page }) => {
    // Look for theme toggle button
    const themeToggle = page.locator('[data-testid="theme-toggle"], button:has-text("theme"), button[aria-label*="theme"]').first();
    
    if (await themeToggle.isVisible()) {
      // Get initial theme
      const initialIsDark = await page.evaluate(() => 
        document.documentElement.classList.contains('dark')
      );
      
      // Click toggle
      await themeToggle.click();
      
      // Theme should change
      const newIsDark = await page.evaluate(() => 
        document.documentElement.classList.contains('dark')
      );
      
      expect(newIsDark).not.toBe(initialIsDark);
    }
  });
});

test.describe('Navigation', () => {
  test('should navigate to different sections', async ({ page }) => {
    await page.goto('/');
    
    // Look for navigation links
    const navLinks = page.locator('nav a, header a');
    const linkCount = await navLinks.count();
    
    if (linkCount > 0) {
      // Click first link and verify navigation
      const firstLink = navLinks.first();
      const href = await firstLink.getAttribute('href');
      
      if (href && !href.startsWith('http')) {
        await firstLink.click();
        await page.waitForLoadState('networkidle');
        // Page should load without errors
        await expect(page.locator('body')).toBeVisible();
      }
    }
  });
});

test.describe('Stock Interactions', () => {
  test('should open stock details when clicking a card', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Find a clickable stock element
    const stockElement = page.locator('[data-testid="stock-card"], [class*="stock"], table tbody tr').first();
    
    if (await stockElement.isVisible()) {
      await stockElement.click();
      
      // Wait for any detail panel/modal to appear
      await page.waitForTimeout(500);
      
      // Check for detail content (chart, info, etc.)
      const hasDetails = 
        await page.locator('[class*="chart"], [class*="detail"], [class*="panel"], [role="dialog"]').count() > 0;
      
      // Either details appeared or we navigated to detail page
      const urlChanged = !page.url().endsWith('/');
      expect(hasDetails || urlChanged).toBeTruthy();
    }
  });
});
