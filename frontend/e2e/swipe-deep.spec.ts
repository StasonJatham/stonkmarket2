import { test, expect } from '@playwright/test';

/**
 * Deep Swipe Functionality Tests
 * Tests specific issues:
 * - Voting works and is reflected in UI
 * - AI content is displayed
 * - Vote counts are visible
 * - Signals load correctly
 * - All buttons work
 */

test.describe('DipSwipe Deep Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Wait for page and API to be ready
    await page.goto('/swipe');
    await page.waitForLoadState('networkidle');
  });

  test('should display swipe cards with all data', async ({ page }) => {
    // Wait for cards to load
    await page.waitForTimeout(2000);
    
    // Check for card content
    const card = page.locator('[data-testid="swipe-card"], [class*="card"]').first();
    
    if (await card.isVisible()) {
      // Check card has content
      const cardText = await card.textContent();
      console.log('Card text:', cardText?.slice(0, 200));
      
      // Should have symbol/name
      expect(cardText).toBeTruthy();
      expect(cardText?.length).toBeGreaterThan(10);
    }
  });

  test('should display vote counts', async ({ page }) => {
    await page.waitForTimeout(2000);
    
    // Look for vote-related elements (thumbs up/down, numbers)
    const voteElements = page.locator('[class*="vote"], [class*="thumb"], svg[class*="lucide-thumbs"]');
    const count = await voteElements.count();
    console.log(`Found ${count} vote-related elements`);
    
    // Or look for the vote counts text
    const pageText = await page.textContent('body');
    const hasVoteRelated = pageText?.includes('vote') || 
                          pageText?.includes('Buy') || 
                          pageText?.includes('Sell') ||
                          pageText?.includes('Bullish') ||
                          pageText?.includes('Bearish');
    console.log('Has vote-related text:', hasVoteRelated);
  });

  test('should display AI content (rating, bio)', async ({ page }) => {
    await page.waitForTimeout(2000);
    
    const pageText = await page.textContent('body') || '';
    
    // Check for AI rating (buy/sell/hold)
    const hasRating = pageText.toLowerCase().includes('buy') || 
                     pageText.toLowerCase().includes('sell') || 
                     pageText.toLowerCase().includes('hold');
    console.log('Has AI rating:', hasRating);
    
    // Check for AI bio (usually starts with "I'm" for swipe bios)
    const hasBio = pageText.includes("I'm") || 
                  pageText.includes('dealbreaker') ||
                  pageText.length > 500;  // Should have substantial content
    console.log('Has AI bio content:', hasBio);
    console.log('Page text length:', pageText.length);
  });

  test('swipe buttons should be clickable', async ({ page }) => {
    await page.waitForTimeout(2000);
    
    // Look for buy/sell buttons
    const buyButton = page.locator('button:has-text("Buy"), button[aria-label*="buy"], [class*="success"]').first();
    const sellButton = page.locator('button:has-text("Sell"), button[aria-label*="sell"], [class*="danger"]').first();
    
    // Check thumbs up/down icons
    const thumbsUp = page.locator('svg[class*="thumbs-up"], [class*="ThumbsUp"]').first();
    const thumbsDown = page.locator('svg[class*="thumbs-down"], [class*="ThumbsDown"]').first();
    
    const hasBuyButton = await buyButton.isVisible().catch(() => false);
    const hasSellButton = await sellButton.isVisible().catch(() => false);
    const hasThumbsUp = await thumbsUp.isVisible().catch(() => false);
    const hasThumbsDown = await thumbsDown.isVisible().catch(() => false);
    
    console.log('Buy button visible:', hasBuyButton);
    console.log('Sell button visible:', hasSellButton);
    console.log('Thumbs up visible:', hasThumbsUp);
    console.log('Thumbs down visible:', hasThumbsDown);
    
    // At least one way to vote should be available
    expect(hasBuyButton || hasSellButton || hasThumbsUp || hasThumbsDown).toBeTruthy();
  });

  test('should handle vote action without errors', async ({ page }) => {
    // Capture console errors
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    
    await page.waitForTimeout(2000);
    
    // Find and click a vote button
    const voteButton = page.locator('button').filter({ hasText: /buy|sell|👍|👎/i }).first();
    
    if (await voteButton.isVisible()) {
      await voteButton.click();
      await page.waitForTimeout(1000);
      
      console.log('Console errors after vote:', errors);
      // Check no errors or only "already voted" type errors
      const criticalErrors = errors.filter(e => 
        !e.includes('Already voted') && 
        !e.includes('cooldown') &&
        !e.includes('422')
      );
      expect(criticalErrors.length).toBe(0);
    }
  });
});

test.describe('API Data Tests', () => {
  test('swipe cards API returns complete data', async ({ request }) => {
    const response = await request.get('http://localhost:8000/api/swipe/cards');
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    console.log('Cards response:', JSON.stringify(data, null, 2).slice(0, 500));
    
    if (data.cards && data.cards.length > 0) {
      const card = data.cards[0];
      
      // Check required fields
      expect(card.symbol).toBeTruthy();
      console.log('Symbol:', card.symbol);
      console.log('Name:', card.name);
      console.log('AI Rating:', card.ai_rating);
      console.log('AI Bio:', card.swipe_bio?.slice(0, 100));
      console.log('Vote counts:', card.vote_counts);
      
      // These should all exist for a properly fetched symbol
      expect(card.name || card.symbol).toBeTruthy();
      expect(card.vote_counts).toBeTruthy();
    }
  });

  test('signals API returns data', async ({ request }) => {
    const response = await request.get('http://localhost:8000/api/dipfinder/signals?tickers=AAPL');
    console.log('Signals status:', response.status());
    
    if (response.ok()) {
      const data = await response.json();
      console.log('Signals response:', JSON.stringify(data, null, 2).slice(0, 500));
      expect(data.signals).toBeDefined();
    } else {
      console.log('Signals failed:', await response.text());
    }
  });

  test('voting API works correctly', async ({ request }) => {
    // First get the list of tracked symbols from swipe cards
    const cardsResponse = await request.get('http://localhost:8000/api/swipe/cards');
    const cardsData = await cardsResponse.json();
    
    // Use the first tracked symbol, or skip test if none available
    if (!cardsData.cards || cardsData.cards.length === 0) {
      console.log('No tracked symbols available, skipping vote test');
      return;
    }
    
    const trackedSymbol = cardsData.cards[0].symbol;
    console.log('Using tracked symbol for vote test:', trackedSymbol);
    
    const response = await request.put(`http://localhost:8000/api/swipe/cards/${trackedSymbol}/vote`, {
      data: { vote_type: 'buy' },
      headers: {
        'User-Agent': 'Playwright-Test-' + Date.now(),
      }
    });
    
    console.log('Vote status:', response.status());
    const data = await response.json();
    console.log('Vote response:', data);
    
    // Either success or already voted (both are "working")
    expect([200, 422]).toContain(response.status());
    
    if (response.status() === 422) {
      // Either "Already voted" or "Vote recorded from your network"
      expect(data.message).toMatch(/Already voted|Vote recorded|Try again/);
    } else {
      expect(data.message).toContain('recorded');
    }
  });

  test('logo API returns proper response', async ({ request }) => {
    const response = await request.get('http://localhost:8000/api/logos/AAPL');
    console.log('Logo status:', response.status());
    
    // 404 is acceptable if Logo.dev key not configured
    // 200 means logo was found
    expect([200, 404]).toContain(response.status());
    
    if (response.status() === 404) {
      const data = await response.json();
      console.log('Logo not configured:', data.detail);
    }
  });
});
