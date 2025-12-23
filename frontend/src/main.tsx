import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { initWebVitals, sendWebVitals } from './lib/webVitals'

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
