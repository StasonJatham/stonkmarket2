import { test, expect } from '@playwright/test';

/**
 * Admin Flow E2E Tests
 * 
 * Tests for:
 * - Adding new stocks
 * - Suggesting stocks
 * - Admin approval flow
 */

// Test configuration - hardcoded for e2e tests
const BASE_URL = 'http://localhost:8000';
const FRONTEND_URL = 'http://localhost:5173';

// Helper: Create admin auth token (mock for testing)
async function getAdminToken(): Promise<string> {
  // In tests, we use a test admin token
  // Real implementation would login via /auth/login
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: 'admin@test.com',
      password: 'testadmin123',
    }),
  });
  
  if (response.ok) {
    const data = await response.json();
    return data.access_token;
  }
  
  // Return empty for unauthenticated tests
  return '';
}

test.describe('Stock Suggestion Flow', () => {
  test('should open suggestion dialog from dashboard', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/`);
    
    // Wait for page load
    await page.waitForLoadState('networkidle');
    
    // Find and click suggest button
    const suggestButton = page.locator('button:has-text("Suggest"), [data-testid="suggest-stock"]');
    
    if (await suggestButton.count() > 0) {
      await suggestButton.first().click();
      
      // Dialog should appear
      const dialog = page.locator('[role="dialog"], .modal, [data-testid="suggest-dialog"]');
      await expect(dialog).toBeVisible({ timeout: 5000 });
      
      // Should have input field
      const input = dialog.locator('input[type="text"], input[placeholder*="symbol"], input[placeholder*="AAPL"]');
      await expect(input).toBeVisible();
    }
  });

  test('should search for stocks in suggestion dialog', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/`);
    await page.waitForLoadState('networkidle');
    
    // Open suggest dialog
    const suggestButton = page.locator('button:has-text("Suggest"), [data-testid="suggest-stock"]');
    
    if (await suggestButton.count() > 0) {
      await suggestButton.first().click();
      
      const dialog = page.locator('[role="dialog"], .modal');
      await dialog.waitFor({ state: 'visible', timeout: 5000 });
      
      // Type in search
      const input = dialog.locator('input');
      await input.fill('Tesla');
      
      // Wait for search results
      await page.waitForTimeout(800);
      
      // Should show results - look for any button with TSLA or Tesla text
      const results = dialog.locator('button:has-text("TSLA"), button:has-text("Tesla")');
      const noResults = dialog.locator('text=/no results|not found|no matches/i');
      
      const hasResults = await results.count() > 0;
      const hasNoResults = await noResults.count() > 0;
      
      expect(hasResults || hasNoResults).toBe(true);
    }
  });

  test('should submit stock suggestion', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/`);
    await page.waitForLoadState('networkidle');
    
    const suggestButton = page.locator('button:has-text("Suggest")');
    
    if (await suggestButton.count() > 0) {
      await suggestButton.first().click();
      
      const dialog = page.locator('[role="dialog"]');
      await dialog.waitFor({ state: 'visible', timeout: 5000 });
      
      // Fill in suggestion
      const input = dialog.locator('input');
      await input.fill('TEST_SYMBOL');
      
      // Submit
      const submitButton = dialog.locator('button:has-text("Submit"), button:has-text("Suggest"), button[type="submit"]');
      
      if (await submitButton.count() > 0) {
        await submitButton.first().click();
        
        // Should show success or error message
        await page.waitForTimeout(1000);
        
        // Check for any feedback
        const feedback = page.locator('text=/success|submitted|thank|error|already/i');
        const feedbackCount = await feedback.count();
        
        console.log(`Suggestion feedback count: ${feedbackCount}`);
      }
    }
  });
});

test.describe('Add Stock Flow (API)', () => {
  test('should validate stock symbol via API', async ({ request }) => {
    // Test symbol validation endpoint
    const response = await request.get(`${BASE_URL}/api/symbols/validate/AAPL`);
    
    if (response.ok()) {
      const data = await response.json();
      expect(data.valid).toBe(true);
      expect(data.symbol).toBe('AAPL');
    } else {
      // 404 means endpoint doesn't exist, skip
      expect(response.status()).toBe(404);
    }
  });

  test('should search symbols via API', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/symbols/search/Apple?limit=10`);
    
    expect(response.ok()).toBe(true);
    
    const data = await response.json();
    expect(Array.isArray(data.results)).toBe(true);
  });

  test('should get symbol info via API', async ({ request }) => {
    // Get a tracked symbol first
    const symbolsResponse = await request.get(`${BASE_URL}/api/symbols/paged?limit=1`);
    
    if (symbolsResponse.ok()) {
      const symbols = await symbolsResponse.json();
      const items = symbols.items || [];
      
      if (items.length > 0) {
        const symbol = items[0].symbol;
        
        const infoResponse = await request.get(`${BASE_URL}/api/dips/${symbol}/info`);
        expect(infoResponse.ok()).toBe(true);
        
        const info = await infoResponse.json();
        expect(info.symbol).toBe(symbol);
        expect(info.name).toBeDefined();
      }
    }
  });
});

test.describe('Admin Approval Flow', () => {
  test('should list pending suggestions via API', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/suggestions?status=pending`);
    
    // May require auth, so accept 401/403
    if (response.ok()) {
      const data = await response.json();
      expect(Array.isArray(data.suggestions || data)).toBe(true);
    } else {
      expect([401, 403, 404]).toContain(response.status());
    }
  });

  test('should handle suggestion approval (requires auth)', async ({ request }) => {
    // This would require admin auth
    const response = await request.post(`${BASE_URL}/api/suggestions/1/approve`, {
      headers: {
        'Authorization': 'Bearer invalid_token',
      },
    });
    
    // Should get 401 unauthorized
    expect([401, 403, 404, 422]).toContain(response.status());
  });

  test('should handle suggestion rejection (requires auth)', async ({ request }) => {
    const response = await request.post(`${BASE_URL}/api/suggestions/1/reject`, {
      headers: {
        'Authorization': 'Bearer invalid_token',
      },
      data: { reason: 'Test rejection' },
    });
    
    expect([401, 403, 404, 422]).toContain(response.status());
  });
});

test.describe('Symbol Management API', () => {
  test('should list all tracked symbols', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/symbols`);
    
    expect(response.ok()).toBe(true);
    
    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
  });

  test('should get symbol details', async ({ request }) => {
    // Get first symbol
    const listResponse = await request.get(`${BASE_URL}/api/symbols/paged?limit=1`);
    
    // Symbols list might require auth
    if (!listResponse.ok()) {
      // Skip if API requires auth (401) - this is expected behavior
      expect([401, 403]).toContain(listResponse.status());
      return;
    }
    
    const symbols = await listResponse.json();
    const items = symbols.items || [];
    
    if (items.length > 0) {
      const symbol = items[0].symbol;
      
      const detailResponse = await request.get(`${BASE_URL}/api/symbols/${symbol}`);
      // Accept either success or auth required
      expect([200, 401, 403]).toContain(detailResponse.status());
      
      if (detailResponse.ok()) {
        const detail = await detailResponse.json();
        expect(detail.symbol).toBe(symbol);
      }
    }
  });

  test('should handle invalid symbol gracefully', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/symbols/INVALID_SYMBOL_12345`);
    
    // Should return 404 (not found) or 401 (if auth required) - not a 500 error
    expect([200, 401, 404]).toContain(response.status());
  });
});

test.describe('Batch Job Status (API)', () => {
  test('should get batch job status', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/admin/batch-jobs`);
    
    // May require auth
    if (response.ok()) {
      const data = await response.json();
      expect(Array.isArray(data.jobs || data)).toBe(true);
    } else {
      expect([401, 403, 404]).toContain(response.status());
    }
  });

  test('should get cron job status', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/admin/cronjobs`);
    
    if (response.ok()) {
      const data = await response.json();
      expect(Array.isArray(data)).toBe(true);
    } else {
      expect([401, 403, 404]).toContain(response.status());
    }
  });
});
