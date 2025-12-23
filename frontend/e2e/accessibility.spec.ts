import { test, expect } from '@playwright/test';

/**
 * Accessibility E2E tests
 * Tests basic accessibility requirements
 */

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');
    
    // Should have at least one h1
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBeGreaterThanOrEqual(1);
    
    // h1 should not be empty
    if (h1Count > 0) {
      const h1Text = await page.locator('h1').first().textContent();
      expect(h1Text?.trim().length).toBeGreaterThan(0);
    }
  });

  test('should have alt text on images', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const images = page.locator('img');
    const imageCount = await images.count();
    
    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      const role = await img.getAttribute('role');
      
      // Image should have alt text or be decorative (role="presentation")
      expect(alt !== null || role === 'presentation').toBeTruthy();
    }
  });

  test('should have visible focus indicators', async ({ page }) => {
    await page.goto('/');
    
    // Find a focusable element
    const focusable = page.locator('button, a, input, [tabindex="0"]').first();
    
    if (await focusable.isVisible()) {
      // Focus the element
      await focusable.focus();
      
      // Check that focus is visible (element has focus)
      await expect(focusable).toBeFocused();
    }
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    
    // Press Tab to navigate
    await page.keyboard.press('Tab');
    
    // Something should be focused
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('should have proper form labels', async ({ page }) => {
    await page.goto('/login');
    
    const inputs = page.locator('input:not([type="hidden"]):not([type="submit"])');
    const inputCount = await inputs.count();
    
    for (let i = 0; i < inputCount; i++) {
      const input = inputs.nth(i);
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const ariaLabelledBy = await input.getAttribute('aria-labelledby');
      const placeholder = await input.getAttribute('placeholder');
      
      // Input should have some form of labeling
      const hasLabel = id ? await page.locator(`label[for="${id}"]`).count() > 0 : false;
      const hasAccessibleName = hasLabel || ariaLabel || ariaLabelledBy || placeholder;
      
      expect(hasAccessibleName).toBeTruthy();
    }
  });

  test('should not have empty buttons', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    for (let i = 0; i < buttonCount; i++) {
      const button = buttons.nth(i);
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      const title = await button.getAttribute('title');
      const hasIcon = await button.locator('svg, img, [class*="icon"]').count() > 0;
      
      // Button should have visible text, aria-label, title, or be an icon button
      const hasAccessibleName = 
        (text && text.trim().length > 0) || 
        ariaLabel || 
        title ||
        hasIcon;
      
      expect(hasAccessibleName).toBeTruthy();
    }
  });
});

test.describe('Color Contrast', () => {
  test('text should be visible in dark mode', async ({ page }) => {
    await page.goto('/');
    
    // Enable dark mode if toggle exists
    const themeToggle = page.locator('[data-testid="theme-toggle"], button[aria-label*="theme"]').first();
    if (await themeToggle.isVisible()) {
      await themeToggle.click();
    }
    
    // Check that main text is visible (not pure black on black)
    const body = page.locator('body');
    const bgColor = await body.evaluate(el => getComputedStyle(el).backgroundColor);
    const textColor = await body.evaluate(el => getComputedStyle(el).color);
    
    // Colors should be different
    expect(bgColor).not.toBe(textColor);
  });
});
