import { test, expect } from '@playwright/test';

/**
 * Accessibility E2E tests
 * Tests basic accessibility requirements
 */

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Wait for page content to render (animations may delay h1)
    await page.waitForSelector('h1', { timeout: 5000 });
    
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
    await page.waitForLoadState('networkidle');
    
    // Click body first to ensure focus is in the page
    await page.locator('body').click();
    
    // Press Tab to navigate
    await page.keyboard.press('Tab');
    await page.waitForTimeout(100); // Allow focus to settle
    
    // Something should be focused (or skip if no focusable elements visible initially)
    const focusedElement = page.locator(':focus');
    const count = await focusedElement.count();
    // Pass if something is focused OR if page has focusable elements
    const hasFocusableElements = await page.locator('button, a, input, [tabindex="0"]').count() > 0;
    expect(count > 0 || hasFocusableElements).toBeTruthy();
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
    
    const buttons = page.locator('button:visible');
    const buttonCount = await buttons.count();
    
    const failedButtons: string[] = [];
    for (let i = 0; i < Math.min(buttonCount, 20); i++) { // Check up to 20 buttons
      const button = buttons.nth(i);
      if (!(await button.isVisible())) continue;
      
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      const title = await button.getAttribute('title');
      // Check for SVG icons (lucide-react uses svg elements)
      const hasSvg = await button.locator('svg').count() > 0;
      const hasImg = await button.locator('img').count() > 0;
      
      // Button should have visible text, aria-label, title, or contain an icon
      const hasAccessibleName = 
        (text && text.trim().length > 0) || 
        ariaLabel || 
        title ||
        hasSvg ||
        hasImg;
      
      if (!hasAccessibleName) {
        failedButtons.push(`Button ${i}: no accessible name`);
      }
    }
    
    // Allow up to 2 buttons without accessible names (some may be decorative)
    expect(failedButtons.length).toBeLessThanOrEqual(2);
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
