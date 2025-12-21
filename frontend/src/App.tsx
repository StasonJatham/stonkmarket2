import { lazy, Suspense, memo } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { ThemeProvider } from '@/context/ThemeContext';
import { Layout } from '@/components/layout/Layout';
import { ProtectedRoute } from '@/components/ProtectedRoute';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard').then(m => ({ default: m.Dashboard })));
const LoginPage = lazy(() => import('@/pages/Login').then(m => ({ default: m.LoginPage })));
const AdminPage = lazy(() => import('@/pages/Admin').then(m => ({ default: m.AdminPage })));
const DipSwipePage = lazy(() => import('@/pages/DipSwipe').then(m => ({ default: m.DipSwipePage })));
const DipFinderPage = lazy(() => import('@/pages/DipFinder').then(m => ({ default: m.DipFinderPage })));

// Loading fallback component
const PageLoader = memo(function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 border-3 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-muted-foreground">Loading...</p>
      </div>
    </div>
  );
});

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/swipe" element={<DipSwipePage />} />
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
