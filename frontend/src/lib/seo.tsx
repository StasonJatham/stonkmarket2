/**
 * SEO utilities for React 19 SPA
 * 
 * Provides per-route meta tag management without external dependencies.
 * Uses React 19's document.title and meta tag manipulation.
 */

import { useEffect } from 'react';

export interface SEOProps {
  title?: string;
  description?: string;
  keywords?: string;
  canonical?: string;
  ogType?: 'website' | 'article';
  ogImage?: string;
  noindex?: boolean;
  // Structured data
  jsonLd?: Record<string, unknown> | Record<string, unknown>[];
}

// Base site info
const SITE_NAME = 'StonkMarket';
const SITE_URL = 'https://stonkmarket.de';
const DEFAULT_DESCRIPTION = 'Track market dips and identify stocks with high recovery potential. Real-time analysis of S&P 500 and MSCI World Index benchmarks.';
const DEFAULT_IMAGE = '/og-image.png';

/**
 * Format page title with site name
 */
function formatTitle(pageTitle?: string): string {
  if (!pageTitle) return `${SITE_NAME} - Discover Stock Dips with Recovery Potential`;
  return `${pageTitle} | ${SITE_NAME}`;
}

/**
 * Update or create a meta tag
 */
function setMetaTag(name: string, content: string, isProperty = false): void {
  const attrName = isProperty ? 'property' : 'name';
  let meta = document.querySelector(`meta[${attrName}="${name}"]`) as HTMLMetaElement | null;
  
  if (!meta) {
    meta = document.createElement('meta');
    meta.setAttribute(attrName, name);
    document.head.appendChild(meta);
  }
  
  meta.content = content;
}

/**
 * Set or update canonical link
 */
function setCanonicalLink(url: string): void {
  let link = document.querySelector('link[rel="canonical"]') as HTMLLinkElement | null;
  
  if (!link) {
    link = document.createElement('link');
    link.rel = 'canonical';
    document.head.appendChild(link);
  }
  
  link.href = url;
}

/**
 * Validate JSON-LD data has required @context property
 */
function isValidJsonLd(item: unknown): item is Record<string, unknown> {
  return (
    typeof item === 'object' &&
    item !== null &&
    '@context' in item &&
    typeof (item as Record<string, unknown>)['@context'] === 'string'
  );
}

/**
 * Inject JSON-LD structured data
 */
function setJsonLd(data: Record<string, unknown> | Record<string, unknown>[]): void {
  // Remove existing JSON-LD scripts (except the base organization one)
  document.querySelectorAll('script[type="application/ld+json"][data-seo="page"]').forEach(el => el.remove());
  
  // Validate and filter the data
  let validData: Record<string, unknown> | Record<string, unknown>[];
  
  if (Array.isArray(data)) {
    // Filter out any invalid items from the array
    const validItems = data.filter(isValidJsonLd);
    if (validItems.length === 0) {
      console.warn('SEO: No valid JSON-LD items to inject');
      return;
    }
    validData = validItems;
  } else {
    // Single object - validate it
    if (!isValidJsonLd(data)) {
      console.warn('SEO: Invalid JSON-LD data, missing @context', data);
      return;
    }
    validData = data;
  }
  
  const script = document.createElement('script');
  script.type = 'application/ld+json';
  script.setAttribute('data-seo', 'page');
  script.textContent = JSON.stringify(validData);
  document.head.appendChild(script);
}

/**
 * React hook for managing SEO meta tags
 * Call this in each page component to set page-specific meta
 */
export function useSEO(props: SEOProps): void {
  const {
    title,
    description = DEFAULT_DESCRIPTION,
    keywords,
    canonical,
    ogType = 'website',
    ogImage = DEFAULT_IMAGE,
    noindex = false,
    jsonLd,
  } = props;

  useEffect(() => {
    // Set document title
    document.title = formatTitle(title);
    
    // Basic meta tags
    setMetaTag('description', description);
    setMetaTag('title', formatTitle(title));
    if (keywords) setMetaTag('keywords', keywords);
    
    // Robots
    setMetaTag('robots', noindex ? 'noindex, nofollow' : 'index, follow');
    
    // Open Graph
    setMetaTag('og:type', ogType, true);
    setMetaTag('og:title', formatTitle(title), true);
    setMetaTag('og:description', description, true);
    setMetaTag('og:image', ogImage.startsWith('/') ? `${SITE_URL}${ogImage}` : ogImage, true);
    setMetaTag('og:site_name', SITE_NAME, true);
    
    // Twitter Card
    setMetaTag('twitter:title', formatTitle(title), true);
    setMetaTag('twitter:description', description, true);
    setMetaTag('twitter:image', ogImage.startsWith('/') ? `${SITE_URL}${ogImage}` : ogImage, true);
    
    // Canonical URL
    const canonicalUrl = canonical 
      ? (canonical.startsWith('/') ? `${SITE_URL}${canonical}` : canonical)
      : `${SITE_URL}${window.location.pathname}`;
    setCanonicalLink(canonicalUrl);
    setMetaTag('og:url', canonicalUrl, true);
    setMetaTag('twitter:url', canonicalUrl, true);
    
    // JSON-LD structured data
    if (jsonLd) {
      setJsonLd(jsonLd);
    }
    
    // Cleanup: reset to defaults when component unmounts
    return () => {
      // Don't reset on unmount - next page will set its own
    };
  }, [title, description, keywords, canonical, ogType, ogImage, noindex, jsonLd]);
}

/**
 * Generate breadcrumb JSON-LD
 */
export function generateBreadcrumbJsonLd(items: { name: string; url: string }[]): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    'itemListElement': items.map((item, index) => ({
      '@type': 'ListItem',
      'position': index + 1,
      'name': item.name,
      'item': item.url.startsWith('/') ? `${SITE_URL}${item.url}` : item.url,
    })),
  };
}

/**
 * Generate FAQ JSON-LD
 */
export function generateFAQJsonLd(faqs: { question: string; answer: string }[]): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    'mainEntity': faqs.map(faq => ({
      '@type': 'Question',
      'name': faq.question,
      'acceptedAnswer': {
        '@type': 'Answer',
        'text': faq.answer,
      },
    })),
  };
}

/**
 * Stock/Financial instrument JSON-LD
 */
export function generateStockJsonLd(stock: {
  symbol: string;
  name: string;
  description?: string;
  exchange?: string;
  sector?: string;
}): Record<string, unknown> {
  return {
    '@context': 'https://schema.org',
    '@type': 'FinancialProduct',
    'name': `${stock.symbol} - ${stock.name}`,
    'description': stock.description || `Stock analysis and dip tracking for ${stock.name} (${stock.symbol})`,
    'category': stock.sector || 'Equity',
    'provider': {
      '@type': 'Organization',
      'name': stock.exchange || 'NASDAQ/NYSE',
    },
  };
}

/**
 * SEO constants for reuse
 */
export const SEO_DEFAULTS = {
  SITE_NAME,
  SITE_URL,
  DEFAULT_DESCRIPTION,
  DEFAULT_IMAGE,
};
