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
    // Wait for content to load by looking for specific elements instead of networkidle
    // TanStack Query keeps connections open, so networkidle never resolves
    // Use data-slot="card" which is on all shadcn Card components
    await page.waitForSelector('[data-slot="card"], table', { timeout: 15000 });
    
    // Should have either stock cards or a ranking table
    const hasStockCards = await page.locator('[data-slot="card"]').count() > 0;
    const hasRankingTable = await page.locator('table').count() > 0;
    
    expect(hasStockCards || hasRankingTable).toBeTruthy();
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
      
      // Wait for theme transition to complete
      await page.waitForTimeout(300);
      
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
    // Wait for cards to load using data-slot="card" which is on all shadcn cards
    await page.waitForSelector('[data-slot="card"]', { timeout: 15000 });
    
    // Find a clickable stock element - use data-slot="card" which contains stock info
    const stockElement = page.locator('[data-slot="card"]').first();
    
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

  test('should switch chart periods on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    // Wait for cards to appear using data-slot="card"
    await page.waitForSelector('[data-slot="card"]', { timeout: 15000 });
    
    // Wait for animations to complete
    await page.waitForTimeout(1000);
    
    // Find a stock card - use data-slot="card" 
    const stockCards = page.locator('[data-slot="card"]');
    const stockCount = await stockCards.count();
    
    if (stockCount > 0) {
      // Get the first visible stock card
      const firstCard = stockCards.first();
      
      // Wait for the card to be in a stable state (animations complete)
      await page.waitForTimeout(500);
      
      // Use force click to bypass any overlay issues
      await firstCard.click({ force: true, timeout: 5000 });
      
      // Wait for sheet to open
      await page.waitForTimeout(1000);
      
      // Look for chart period buttons (1M, 3M, 6M, 1Y, etc.)
      const periodButtons = page.locator('button:has-text("1M"), button:has-text("3M"), button:has-text("1Y")');
      const periodCount = await periodButtons.count();
      
      console.log(`Found ${periodCount} period buttons`);
      
      if (periodCount > 0) {
        // Try clicking 1M button
        const oneMonthBtn = page.locator('button:has-text("1M")').first();
        if (await oneMonthBtn.isVisible()) {
          console.log('Clicking 1M button...');
          await oneMonthBtn.click();
          await page.waitForTimeout(1000);
          
          // Check for any errors in console
          const consoleErrors: string[] = [];
          page.on('console', msg => {
            if (msg.type() === 'error') {
              consoleErrors.push(msg.text());
            }
          });
          
          // Verify chart is still visible or loading
          const chartArea = page.locator('[class*="chart"], canvas, svg');
          const chartVisible = await chartArea.count() > 0;
          console.log(`Chart area visible: ${chartVisible}`);
          
          // Check for any loading or error states
          const hasError = await page.locator('[class*="error"], [class*="danger"]').count() > 0;
          const isLoading = await page.locator('[class*="loading"], [class*="spinner"], [class*="animate-spin"]').count() > 0;
          
          console.log(`Has error: ${hasError}, Is loading: ${isLoading}`);
          console.log(`Console errors: ${consoleErrors.join(', ')}`);
          
          // The chart should be visible (not in permanent error state)
          expect(chartVisible || isLoading).toBeTruthy();
        }
        
        // Try clicking 1Y button
        const oneYearBtn = page.locator('button:has-text("1Y")').first();
        if (await oneYearBtn.isVisible()) {
          console.log('Clicking 1Y button...');
          await oneYearBtn.click();
          await page.waitForTimeout(1000);
          
          const chartArea = page.locator('[class*="chart"], canvas, svg');
          const chartVisible = await chartArea.count() > 0;
          console.log(`Chart area visible after 1Y: ${chartVisible}`);
          
          expect(chartVisible).toBeTruthy();
        }
      }
    }
  });
});

test.describe('Suggest Stock Feature', () => {
  test('should open suggest stock dialog', async ({ page }) => {
    await page.goto('/');
    // Wait for page content to load instead of networkidle
    await page.waitForSelector('button:has-text("Suggest"), nav, header', { timeout: 15000 });
    
    // Find and click the suggest button
    const suggestButton = page.locator('button:has-text("Suggest")').first();
    
    if (await suggestButton.isVisible()) {
      await suggestButton.click();
      
      // Dialog should open
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();
      
      // Should have search input (placeholder is "e.g., AAPL or Apple")
      const searchInput = dialog.locator('input[placeholder*="AAPL"], input[placeholder*="Apple"], input[type="text"]').first();
      await expect(searchInput).toBeVisible();
    }
  });
  
  test('should search for stocks in suggest dialog', async ({ page }) => {
    await page.goto('/');
    // Wait for page content to load instead of networkidle
    await page.waitForSelector('button:has-text("Suggest"), nav, header', { timeout: 15000 });
    
    // Open suggest dialog
    const suggestButton = page.locator('button:has-text("Suggest")').first();
    
    if (await suggestButton.isVisible()) {
      await suggestButton.click();
      
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();
      
      // Type in search
      const searchInput = dialog.locator('input');
      await searchInput.fill('Tesla');
      
      // Wait for search results
      await page.waitForTimeout(500);
      
      // Should show search results
      const results = dialog.locator('button:has-text("TSLA"), button:has-text("Tesla")');
      const resultCount = await results.count();
      console.log(`Found ${resultCount} Tesla results`);
      
      // Click on a result
      if (resultCount > 0) {
        await results.first().click();
        await page.waitForTimeout(500);
        
        // Should show stock preview with submit button
        const submitBtn = dialog.locator('button:has-text("Submit")');
        await expect(submitBtn).toBeVisible();
      }
    }
  });
});
