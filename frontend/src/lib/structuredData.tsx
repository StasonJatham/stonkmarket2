/**
 * Base Structured Data (JSON-LD) for the entire site
 * 
 * Injects Organization, WebSite, and base schema that applies site-wide.
 * This component should be rendered once in the app root.
 */

/* eslint-disable react-refresh/only-export-components -- This module exports schemas alongside components */
import { useEffect } from 'react';
import { SEO_DEFAULTS } from './seo';

const { SITE_NAME, SITE_URL } = SEO_DEFAULTS;

/**
 * Base organization schema - who operates the site
 */
const organizationSchema = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  '@id': `${SITE_URL}/#organization`,
  'name': SITE_NAME,
  'url': SITE_URL,
  'logo': {
    '@type': 'ImageObject',
    'url': `${SITE_URL}/favicon.svg`,
    'width': 512,
    'height': 512,
  },
  'description': 'Financial analytics platform for tracking market dips and identifying stocks with recovery potential.',
  'sameAs': [
    // Add social media profiles here when available
  ],
  'contactPoint': {
    '@type': 'ContactPoint',
    'contactType': 'customer support',
    'url': `${SITE_URL}/contact`,
  },
};

/**
 * WebSite schema - search action and site info
 */
const websiteSchema = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  '@id': `${SITE_URL}/#website`,
  'name': SITE_NAME,
  'url': SITE_URL,
  'description': 'Track market dips and identify stocks with high recovery potential.',
  'publisher': {
    '@id': `${SITE_URL}/#organization`,
  },
  // Enable sitelinks search box in Google
  'potentialAction': {
    '@type': 'SearchAction',
    'target': {
      '@type': 'EntryPoint',
      'urlTemplate': `${SITE_URL}/?search={search_term_string}`,
    },
    'query-input': 'required name=search_term_string',
  },
};

/**
 * SoftwareApplication schema - describes the web app
 */
const softwareAppSchema = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  'name': SITE_NAME,
  'applicationCategory': 'FinanceApplication',
  'operatingSystem': 'Web Browser',
  'description': 'Financial analytics web application for tracking market dips and identifying stocks with recovery potential. Features AI-powered analysis, benchmark comparison, and community voting.',
  'offers': {
    '@type': 'Offer',
    'price': '0',
    'priceCurrency': 'USD',
    'description': 'Free to use',
  },
  'featureList': [
    'Real-time stock dip tracking',
    'AI-powered stock analysis',
    'S&P 500 and MSCI World benchmark comparison',
    'Community sentiment voting',
    'Portfolio performance aggregation',
  ],
  'screenshot': `${SITE_URL}/og-image.png`,
  'author': {
    '@id': `${SITE_URL}/#organization`,
  },
};

/**
 * Inject base JSON-LD schemas into document head
 */
export function BaseStructuredData(): null {
  useEffect(() => {
    // Check if already injected
    if (document.querySelector('script[data-seo="base"]')) {
      return;
    }

    const schemas = [organizationSchema, websiteSchema, softwareAppSchema];
    
    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.setAttribute('data-seo', 'base');
    script.textContent = JSON.stringify(schemas);
    document.head.appendChild(script);

    return () => {
      // Keep base schema on unmount - it's site-wide
    };
  }, []);

  return null;
}

/**
 * Financial disclaimer schema (for E-E-A-T)
 */
export const financialDisclaimerSchema = {
  '@context': 'https://schema.org',
  '@type': 'WebPage',
  'speakable': {
    '@type': 'SpeakableSpecification',
    'cssSelector': ['.risk-disclaimer', '.disclaimer-text'],
  },
  'specialty': 'Financial Analysis',
  'about': {
    '@type': 'Thing',
    'name': 'Stock Market Analysis',
    'description': 'Analysis of stock market dips and recovery potential',
  },
  'disclaimer': 'This is not financial advice. All information provided is for educational purposes only. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.',
};
