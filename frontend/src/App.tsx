import { lazy, Suspense, memo } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Feature flags from environment
const ENABLE_LEGAL_PAGES = import.meta.env.VITE_ENABLE_LEGAL_PAGES === 'true';
import { AuthProvider } from '@/context/AuthContext';
import { ThemeProvider } from '@/context/ThemeContext';
import { DipProvider } from '@/context/DipContext';
import { Layout } from '@/components/layout/Layout';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { UmamiAnalytics } from '@/lib/analytics';
import { BaseStructuredData } from '@/lib/structuredData';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard').then(m => ({ default: m.Dashboard })));
const LoginPage = lazy(() => import('@/pages/Login').then(m => ({ default: m.LoginPage })));
const AdminPage = lazy(() => import('@/pages/Admin').then(m => ({ default: m.AdminPage })));
const DipSwipePage = lazy(() => import('@/pages/DipSwipe').then(m => ({ default: m.DipSwipePage })));
const SuggestionsPage = lazy(() => import('@/pages/Suggestions').then(m => ({ default: m.SuggestionsPage })));
const PrivacyPage = lazy(() => import('@/pages/Privacy').then(m => ({ default: m.PrivacyPage })));
const ImprintPage = lazy(() => import('@/pages/Imprint').then(m => ({ default: m.ImprintPage })));
const ContactPage = lazy(() => import('@/pages/Contact').then(m => ({ default: m.ContactPage })));
const AboutPage = lazy(() => import('@/pages/About').then(m => ({ default: m.AboutPage })));
const StockDetailPage = lazy(() => import('@/pages/StockDetail').then(m => ({ default: m.StockDetailPage })));
const LearnPage = lazy(() => import('@/pages/Learn').then(m => ({ default: m.LearnPage })));
const PortfolioPage = lazy(() => import('@/pages/Portfolio').then(m => ({ default: m.PortfolioPage })));

// Loading fallback component - minimal to avoid layout shift
const PageLoader = memo(function PageLoader() {
  return (
    <div className="min-h-[60vh]" />
  );
});

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <DipProvider>
          <BrowserRouter>
            <UmamiAnalytics />
            <BaseStructuredData />
          <ErrorBoundary>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/swipe" element={<DipSwipePage />} />
                <Route path="/suggest" element={<SuggestionsPage />} />
                <Route path="/about" element={<AboutPage />} />
                <Route path="/learn" element={<LearnPage />} />
                <Route path="/stock/:symbol" element={<StockDetailPage />} />
                {ENABLE_LEGAL_PAGES ? (
                  <>
                    <Route path="/privacy" element={<PrivacyPage />} />
                    <Route path="/imprint" element={<ImprintPage />} />
                  </>
                ) : (
                  <>
                    <Route path="/privacy" element={<Navigate to="/" replace />} />
                    <Route path="/imprint" element={<Navigate to="/" replace />} />
                  </>
                )}
                <Route path="/contact" element={<ContactPage />} />
                <Route
                  path="/admin"
                  element={
                    <ProtectedRoute>
                      <AdminPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/portfolio"
                  element={
                    <ProtectedRoute>
                      <PortfolioPage />
                    </ProtectedRoute>
                  }
                />
                </Route>
              </Routes>
            </Suspense>
          </ErrorBoundary>
          </BrowserRouter>
        </DipProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
