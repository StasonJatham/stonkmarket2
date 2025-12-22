import { lazy, Suspense, memo } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { ThemeProvider } from '@/context/ThemeContext';
import { Layout } from '@/components/layout/Layout';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { UmamiAnalytics } from '@/lib/analytics';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard').then(m => ({ default: m.Dashboard })));
const LoginPage = lazy(() => import('@/pages/Login').then(m => ({ default: m.LoginPage })));
const AdminPage = lazy(() => import('@/pages/Admin').then(m => ({ default: m.AdminPage })));
const DipSwipePage = lazy(() => import('@/pages/DipSwipe').then(m => ({ default: m.DipSwipePage })));
const DipFinderPage = lazy(() => import('@/pages/DipFinder').then(m => ({ default: m.DipFinderPage })));
const SuggestionsPage = lazy(() => import('@/pages/Suggestions').then(m => ({ default: m.SuggestionsPage })));
const PrivacyPage = lazy(() => import('@/pages/Privacy').then(m => ({ default: m.PrivacyPage })));
const ImprintPage = lazy(() => import('@/pages/Imprint').then(m => ({ default: m.ImprintPage })));
const ContactPage = lazy(() => import('@/pages/Contact').then(m => ({ default: m.ContactPage })));

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
        <BrowserRouter>
          <UmamiAnalytics />
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/swipe" element={<DipSwipePage />} />
                <Route path="/suggest" element={<SuggestionsPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="/imprint" element={<ImprintPage />} />
                <Route path="/contact" element={<ContactPage />} />
                <Route
                  path="/signals"
                  element={
                    <ProtectedRoute>
                      <DipFinderPage />
                    </ProtectedRoute>
                  }
                />
                <Route
                  path="/admin"
                  element={
                    <ProtectedRoute>
                      <AdminPage />
                    </ProtectedRoute>
                  }
                />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
