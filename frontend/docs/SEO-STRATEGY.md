# SEO Strategy for StonkMarket (Financial SPA)

## Overview

This document outlines the SEO strategy for StonkMarket, a React SPA for tracking stock dips. As a YMYL (Your Money Your Life) site, we must prioritize E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness).

---

## 1. Current Implementation Status

### ✅ Implemented

| Feature | Status | Files |
|---------|--------|-------|
| Per-route meta tags | ✅ | `src/lib/seo.tsx` - custom `useSEO()` hook |
| Structured Data (JSON-LD) | ✅ | `src/lib/structuredData.tsx` - Organization, WebSite, SoftwareApplication |
| Breadcrumb Schema | ✅ | Added to About, Privacy, Imprint, Contact, DipSwipe pages |
| FAQ Schema | ✅ | About page with 5 FAQs |
| Sitemap.xml | ✅ | `public/sitemap.xml` - 8 routes with lastmod |
| robots.txt | ✅ | Allows crawling, blocks /admin and /api |
| security.txt | ✅ | RFC 9116 compliant |
| humans.txt | ✅ | Team and tech stack info |
| OG/Twitter Cards | ✅ | Absolute URLs, proper dimensions |
| Canonical URLs | ✅ | Auto-generated per route |
| Critical CSS | ✅ | Inlined in `<head>` |
| Code Splitting | ✅ | All pages lazy-loaded |
| Vendor Chunking | ✅ | react, recharts, framer-motion, radix separated |
| Web Vitals | ✅ | Measured and sent to analytics |

### ⚠️ Recommendations (Not Yet Implemented)

| Feature | Priority | Notes |
|---------|----------|-------|
| **Prerendering/SSG** | HIGH | Use @prerenderer/rollup-plugin or SSR for key pages |
| **Dynamic Stock Pages** | MEDIUM | Create `/stock/:symbol` routes for individual stocks |
| **Hreflang** | LOW | Add if multi-language support is added |
| **News/Blog Section** | MEDIUM | Would help with topical authority |

---

## 2. Technical SEO Architecture

### URL Structure

```
/                    → Homepage (Dashboard)
/swipe               → DipSwipe feature
/signals             → Advanced signals (auth required)
/suggest             → Community suggestions
/about               → Methodology & FAQ
/contact             → Contact form
/privacy             → Privacy policy (GDPR)
/imprint             → Legal imprint
/login               → Admin login (noindex)
/admin               → Admin panel (noindex)
```

### Meta Tag Management

All pages use the `useSEO()` hook:

```tsx
import { useSEO, generateBreadcrumbJsonLd } from '@/lib/seo';

export function MyPage() {
  useSEO({
    title: 'Page Title',
    description: 'Meta description for this page',
    keywords: 'relevant, keywords',
    canonical: '/my-page',
    noindex: false, // set true for private pages
    jsonLd: generateBreadcrumbJsonLd([
      { name: 'Home', url: '/' },
      { name: 'My Page', url: '/my-page' },
    ]),
  });
  // ...
}
```

### Structured Data (JSON-LD)

Base schemas (injected once in App root):
- `Organization` - Site operator info
- `WebSite` - Site info with search action
- `SoftwareApplication` - Describes the web app

Page-specific schemas:
- `BreadcrumbList` - Navigation path
- `FAQPage` - Q&A content (About page)

---

## 3. E-E-A-T Compliance (Financial/YMYL)

### Required Trust Signals

| Signal | Location | Status |
|--------|----------|--------|
| Risk Disclaimer | About page, prominent | ✅ |
| Data Sources | About page | ✅ |
| Methodology | About page | ✅ |
| Privacy Policy | /privacy | ✅ |
| Legal Imprint | /imprint | ✅ |
| Contact Info | /contact | ✅ |
| "Not Financial Advice" | Multiple pages | ✅ |

### Content Guidelines

1. **Never present as investment advice**
2. **Always cite data sources** (Yahoo Finance)
3. **Show clear disclaimers** on any financial metrics
4. **Avoid guaranteed returns language**
5. **Include risk warnings** near any suggestions

---

## 4. Performance & Core Web Vitals

### Current Bundle Analysis

| Chunk | Size (gzip) | Notes |
|-------|-------------|-------|
| charts | 111 KB | Recharts - largest chunk |
| radix | 38 KB | UI components |
| animation | 38 KB | Framer Motion |
| index | 87 KB | Main app code |
| react-vendor | 17 KB | React + Router |

### Optimization Strategies

1. **Code Splitting** ✅ - All pages lazy-loaded
2. **Vendor Chunks** ✅ - Separated for caching
3. **Critical CSS** ✅ - Inlined in head
4. **Preconnect** ✅ - Analytics endpoint
5. **Image Optimization** ✅ - WebP with fallback

### Web Vitals Targets

| Metric | Target | Current |
|--------|--------|---------|
| LCP | < 2.5s | ✅ Tested |
| CLS | < 0.1 | ✅ Tested |
| INP | < 200ms | ✅ Tested |
| FCP | < 1.8s | ✅ Tested |

---

## 5. Future Improvements

### HIGH Priority

1. **Add Prerendering**
   - Use `@prerenderer/rollup-plugin` for build-time HTML generation
   - Or deploy to Netlify/Vercel with prerender feature enabled
   - Key pages: `/`, `/about`, `/swipe`, `/suggest`

2. **Dynamic Stock Pages**
   - Create `/stock/:symbol` routes (e.g., `/stock/AAPL`)
   - Generate stock-specific JSON-LD (`FinancialQuote` schema)
   - Add to sitemap dynamically

### MEDIUM Priority

3. **Educational Content**
   - Add `/learn` section with investing basics
   - Create topical authority for "stock dip" related queries
   - Use Article schema for educational content

4. **Internal Linking**
   - Link from stock cards to methodology
   - Cross-link related stocks
   - Add breadcrumbs to all pages

### NICE TO HAVE

5. **Multi-language Support**
   - German version (site is .de domain)
   - Add hreflang tags
   - Separate sitemaps per language

6. **AMP Version**
   - For news/article content only
   - Not recommended for app functionality

---

## 6. Deployment Checklist

### Before Launch

- [ ] Verify all pages have `useSEO()` calls
- [ ] Test OG images render correctly on social media
- [ ] Validate structured data with Google Rich Results Test
- [ ] Check Core Web Vitals with PageSpeed Insights
- [ ] Submit sitemap to Google Search Console
- [ ] Verify robots.txt doesn't block important pages

### Monitoring

- [ ] Set up Google Search Console
- [ ] Monitor Core Web Vitals in GSC
- [ ] Track rankings for target keywords
- [ ] Watch for crawl errors
- [ ] Review structured data errors

---

## 7. Target Keywords

### Primary (Branded)
- stonkmarket
- stonk market de

### Secondary (Feature-based)
- stock dip tracker
- buy the dip stocks
- stock recovery potential
- S&P 500 dips

### Long-tail (Informational)
- what is a stock dip
- when to buy stock dips
- stock dip analysis tool

---

*Last updated: 2025-12-23*
