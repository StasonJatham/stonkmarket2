import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Watch for file changes in Docker
    watch: {
      usePolling: true,
    },
    // HMR configuration for Docker
    hmr: {
      host: 'localhost',
      port: 5173,
    },
    proxy: {
      '/api': {
        // In Docker, use container name; locally use localhost
        target: process.env.DOCKER_ENV ? 'http://api:8000' : 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Optimize chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks for better caching
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'charts': ['recharts'],
          'animation': ['framer-motion'],
          'radix': [
            '@radix-ui/react-dialog',
            '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-label',
            '@radix-ui/react-scroll-area',
            '@radix-ui/react-separator',
            '@radix-ui/react-slider',
            '@radix-ui/react-slot',
            '@radix-ui/react-switch',
            '@radix-ui/react-tabs',
            '@radix-ui/react-tooltip',
          ],
        },
      },
    },
    // Enable minification with esbuild (faster than terser)
    minify: 'esbuild',
    // Generate source maps for production debugging
    sourcemap: false,
    // Increase chunk size warning limit (we're chunking properly now)
    chunkSizeWarningLimit: 500,
    // CSS code splitting
    cssCodeSplit: true,
    // Asset inlining threshold
    assetsInlineLimit: 4096,
  },
  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'recharts', 'framer-motion'],
  },
})
