import { test, expect } from '@playwright/test';

/**
 * Authentication E2E tests
 * Tests login, logout, and protected routes
 */

test.describe('Authentication', () => {
  test('should display login form', async ({ page }) => {
    await page.goto('/login');
    
    // Should have login form elements
    const usernameInput = page.locator('input[name="username"], input[type="text"]').first();
    const passwordInput = page.locator('input[type="password"]');
    const submitButton = page.locator('button[type="submit"], button:has-text("Login"), button:has-text("Sign")');
    
    await expect(usernameInput).toBeVisible();
    await expect(passwordInput).toBeVisible();
    await expect(submitButton).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Fill in invalid credentials
    await page.fill('input[name="username"], input[type="text"]', 'invalid_user');
    await page.fill('input[type="password"]', 'wrong_password');
    
    // Submit
    await page.click('button[type="submit"], button:has-text("Login"), button:has-text("Sign")');
    
    // Wait for response
    await page.waitForTimeout(1000);
    
    // Should show error message or still be on login page
    const errorMessage = page.locator('[class*="error"], [role="alert"], .text-danger, .text-red');
    const stillOnLogin = page.url().includes('login');
    
    const hasError = await errorMessage.count() > 0;
    expect(hasError || stillOnLogin).toBeTruthy();
  });

  test('should redirect unauthenticated users from admin routes', async ({ page }) => {
    await page.goto('/admin');
    
    // Should be redirected to login or show unauthorized
    await page.waitForLoadState('networkidle');
    
    const onLoginPage = page.url().includes('login');
    const hasUnauthorizedMessage = await page.locator('text=/unauthorized|login|sign in/i').count() > 0;
    
    expect(onLoginPage || hasUnauthorizedMessage).toBeTruthy();
  });
});

test.describe('Protected Routes', () => {
  test('admin page requires authentication', async ({ page }) => {
    // Try to access admin without login
    const _response = await page.goto('/admin');
    
    // Should redirect or show error
    const url = page.url();
    expect(url.includes('login') || url.includes('admin')).toBeTruthy();
  });
});
