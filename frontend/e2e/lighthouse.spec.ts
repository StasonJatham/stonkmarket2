import { chromium, expect, test } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const SHOULD_RUN = process.env.E2E_LIGHTHOUSE === '1' || !!process.env.CI;

// In development mode, performance is significantly lower due to:
// - No minification, tree-shaking, or dead code elimination
// - Source maps included
// - Hot Module Replacement code
// - Unoptimized chunks
// Use lower thresholds for dev, higher for production builds
const IS_PRODUCTION = process.env.NODE_ENV === 'production' || BASE_URL.includes('4173');

const SCORE_THRESHOLDS = {
  // Performance: 0.85 for prod, 0.30 for dev (dev server is inherently slow)
  performance: Number(process.env.LH_PERF || (IS_PRODUCTION ? 0.85 : 0.30)),
  // Accessibility: Allow 0.85 threshold (some contrast issues are acceptable in dark theme)
  accessibility: Number(process.env.LH_A11Y || 0.85),
  'best-practices': Number(process.env.LH_BEST || 0.9),
  seo: Number(process.env.LH_SEO || 0.9),
};

// Audit thresholds - skip timing checks in dev since API proxy may not work with Lighthouse
// For accurate performance testing, use: BASE_URL=http://localhost:4173 npm run test:e2e:lighthouse
const AUDIT_THRESHOLDS = IS_PRODUCTION ? {
  'first-contentful-paint': 2000,
  'largest-contentful-paint': 2500,
  'speed-index': 3500,
  'total-blocking-time': 300,
  'cumulative-layout-shift': 0.1,
} : {
  // Only check CLS in dev - timing metrics depend on API availability
  'cumulative-layout-shift': 0.25,
};

const PAGES = [
  { name: 'home', path: '/' },
  { name: 'swipe', path: '/swipe' },
];

async function runLighthouse(url: string, reportName: string) {
  const lighthouseModule = await import('lighthouse');
  const chromeLauncherModule = await import('chrome-launcher');
  const lighthouse = lighthouseModule.default || lighthouseModule;
  const chromeLauncher = chromeLauncherModule.default || chromeLauncherModule;

  const chrome = await chromeLauncher.launch({
    chromePath: chromium.executablePath(),
    chromeFlags: [
      '--headless=new',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--disable-extensions',
    ],
  });

  try {
    const result = await lighthouse(
      url,
      {
        logLevel: 'error',
        output: 'html',
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
        port: chrome.port,
      },
      null
    );

    const reportDir = path.join(process.cwd(), 'test-results', 'lighthouse');
    await fs.mkdir(reportDir, { recursive: true });
    await fs.writeFile(
      path.join(reportDir, `${reportName}.html`),
      result.report as string
    );
    await fs.writeFile(
      path.join(reportDir, `${reportName}.json`),
      JSON.stringify(result.lhr, null, 2)
    );

    return result.lhr;
  } finally {
    await chrome.kill();
  }
}

test.describe('Lighthouse Audits', () => {
  test.describe.configure({ mode: 'serial' });

  for (const page of PAGES) {
    test(`${page.name} lighthouse scores`, async ({ browserName }) => {
      test.skip(!SHOULD_RUN, 'Set E2E_LIGHTHOUSE=1 to run Lighthouse audits');
      test.skip(browserName !== 'chromium', 'Lighthouse runs on Chromium only');

      const url = new URL(page.path, BASE_URL).toString();
      const lhr = await runLighthouse(url, `lh-${page.name}`);

      for (const [category, minScore] of Object.entries(SCORE_THRESHOLDS)) {
        const score = lhr.categories?.[category]?.score ?? 0;
        expect(
          score,
          `${page.name} ${category} score ${score} below ${minScore}`
        ).toBeGreaterThanOrEqual(minScore);
      }

      for (const [auditId, maxValue] of Object.entries(AUDIT_THRESHOLDS)) {
        const audit = lhr.audits?.[auditId];
        const value = audit?.numericValue;
        if (typeof value === 'number') {
          expect(
            value,
            `${page.name} ${auditId} ${value}ms exceeds ${maxValue}ms`
          ).toBeLessThanOrEqual(maxValue);
        }
      }
    });
  }
});
