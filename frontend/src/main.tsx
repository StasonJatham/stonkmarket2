import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { initWebVitals, sendWebVitals } from './lib/webVitals'
import { apiCache } from './lib/cache'

// Expose cache utilities for debugging
declare global {
  interface Window {
    stonkmarket: {
      clearCache: () => void;
      cacheStats: () => { size: number; keys: string[]; memorySize: number; storageSize: number };
      invalidateChart: (symbol?: string) => void;
    };
  }
}

window.stonkmarket = {
  clearCache: () => {
    apiCache.clear();
    console.log('✅ All cache cleared (memory + localStorage)');
  },
  cacheStats: () => apiCache.stats(),
  invalidateChart: (symbol?: string) => {
    if (symbol) {
      apiCache.invalidate(new RegExp(`^chart:${symbol}:`));
      console.log(`✅ Chart cache cleared for ${symbol}`);
    } else {
      apiCache.invalidate(/^chart:/);
      console.log('✅ All chart caches cleared');
    }
  },
};

// Initialize Web Vitals measurement
initWebVitals()

// Send metrics when page is about to unload
window.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    sendWebVitals()
  }
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Dispatch event for prerenderer to know when app is ready
// This is used by vite-plugin-prerender to capture the rendered HTML
document.dispatchEvent(new Event('render-event'))
