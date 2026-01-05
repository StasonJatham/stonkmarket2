import { lazy, Suspense, memo } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryProvider } from '@/lib/query';
import { AuthProvider } from '@/context/AuthContext';
import { ThemeProvider } from '@/context/ThemeContext';
import { Layout } from '@/components/layout/Layout';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { UmamiAnalytics } from '@/lib/analytics';
import { BaseStructuredData } from '@/lib/structuredData';

// Feature flags from environment
const ENABLE_LEGAL_PAGES = import.meta.env.VITE_ENABLE_LEGAL_PAGES === 'true';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard').then(m => ({ default: m.Dashboard })));
const Landing = lazy(() => import('@/pages/Landing').then(m => ({ default: m.Landing })));
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
const WatchlistPage = lazy(() => import('@/pages/Watchlist').then(m => ({ default: m.WatchlistPage })));
const NotificationsPage = lazy(() => import('@/pages/Notifications').then(m => ({ default: m.NotificationsPage })));
const SettingsLayout = lazy(() => import('@/pages/Settings').then(m => ({ default: m.SettingsLayout })));
const PublicProfilePage = lazy(() => import('@/pages/PublicProfile').then(m => ({ default: m.PublicProfilePage })));
const UserHomePage = lazy(() => import('@/pages/UserHome').then(m => ({ default: m.UserHomePage })));

// Settings sub-pages
const ProfileSettings = lazy(() => import('@/pages/settings/ProfileSettings').then(m => ({ default: m.ProfileSettings })));
const SecuritySettings = lazy(() => import('@/pages/settings/SecuritySettings').then(m => ({ default: m.SecuritySettings })));
const AppearanceSettings = lazy(() => import('@/pages/settings/AppearanceSettings').then(m => ({ default: m.AppearanceSettings })));
const NotificationsSettings = lazy(() => import('@/pages/settings/NotificationsSettings').then(m => ({ default: m.NotificationsSettings })));
const ApiKeySettings = lazy(() => import('@/pages/settings/ApiKeySettings').then(m => ({ default: m.ApiKeySettings })));
const ConnectionSettings = lazy(() => import('@/pages/settings/ConnectionSettings').then(m => ({ default: m.ConnectionSettings })));
const PrivacySettings = lazy(() => import('@/pages/settings/PrivacySettings').then(m => ({ default: m.PrivacySettings })));

// Loading fallback component - minimal to avoid layout shift
const PageLoader = memo(function PageLoader() {
  return (
    <div className="min-h-[60vh]" />
  );
});

function App() {
  return (
    <QueryProvider>
      <ThemeProvider>
        <AuthProvider>
          <BrowserRouter>
            <UmamiAnalytics />
            <BaseStructuredData />
            <ErrorBoundary>
              <Suspense fallback={<PageLoader />}>
                <Routes>
                  <Route path="/login" element={<LoginPage />} />
                  <Route element={<Layout />}>
                    <Route path="/" element={<Landing />} />
                    <Route path="/dashboard" element={<Dashboard />} />
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
                      path="/home"
                      element={
                        <ProtectedRoute>
                          <UserHomePage />
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
                    <Route
                      path="/portfolio"
                      element={
                        <ProtectedRoute>
                          <PortfolioPage />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/watchlist"
                      element={
                        <ProtectedRoute>
                          <WatchlistPage />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/notifications"
                      element={
                        <ProtectedRoute>
                          <NotificationsPage />
                        </ProtectedRoute>
                      }
                    />
                    <Route
                      path="/settings"
                      element={
                        <ProtectedRoute>
                          <SettingsLayout />
                        </ProtectedRoute>
                      }
                    >
                      <Route path="profile" element={<ProfileSettings />} />
                      <Route path="security" element={<SecuritySettings />} />
                      <Route path="appearance" element={<AppearanceSettings />} />
                      <Route path="notifications" element={<NotificationsSettings />} />
                      <Route path="api-keys" element={<ApiKeySettings />} />
                      <Route path="connections" element={<ConnectionSettings />} />
                      <Route path="privacy" element={<PrivacySettings />} />
                    </Route>
                    <Route path="/u/:username" element={<PublicProfilePage />} />
                  </Route>
                </Routes>
              </Suspense>
            </ErrorBoundary>
          </BrowserRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryProvider>
  );
}

export default App;
