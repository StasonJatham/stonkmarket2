'use client';

import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import { ThemeToggle } from '@/components/ThemeToggle';
import { SuggestStockDialog } from '@/components/SuggestStockDialog';
import { TrendingUp, Settings, LogOut, Heart, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';

const navLinks = [
  { href: '/', label: 'Dashboard' },
  { href: '/swipe', label: 'DipSwipe', icon: Heart },
  { href: '/learn', label: 'Learn' },
  { href: '/signals', label: 'Signals', icon: BarChart3 },
];

const adminLinks = [
  { href: '/admin', label: 'Settings', icon: Settings },
];

export function Header() {
  const { user, logout } = useAuth();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const toggleMenu = () => setMobileMenuOpen(!mobileMenuOpen);
  const closeMenu = () => setMobileMenuOpen(false);

  return (
    <>
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2" onClick={closeMenu}>
            <TrendingUp className="h-6 w-6" />
            <span className="font-semibold text-lg tracking-tight">StonkMarket</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={`text-sm font-medium transition-colors hover:text-foreground flex items-center gap-1.5 ${
                    isActive(link.href) ? 'text-foreground' : 'text-muted-foreground'
                  }`}
                >
                  {Icon && <Icon className="h-4 w-4" />}
                  {link.label}
                </Link>
              );
            })}
            {user && adminLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                className={`text-sm font-medium transition-colors hover:text-foreground flex items-center gap-1.5 ${
                  isActive(link.href) ? 'text-foreground' : 'text-muted-foreground'
                }`}
              >
                <link.icon className="h-4 w-4" />
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Desktop Actions */}
          <div className="hidden md:flex items-center gap-3">
            <SuggestStockDialog />
            <ThemeToggle />
            {user ? (
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">{user.username}</span>
                <Button variant="ghost" size="sm" onClick={logout} className="gap-2">
                  <LogOut className="h-4 w-4" />
                  <span className="sr-only sm:not-sr-only">Logout</span>
                </Button>
              </div>
            ) : (
              <Link to="/login">
                <Button variant="default" size="sm">Sign in</Button>
              </Link>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            <button
              onClick={toggleMenu}
              className="relative h-10 w-10 flex items-center justify-center rounded-md hover:bg-muted transition-colors"
              aria-label="Toggle menu"
            >
              <motion.div
                animate={mobileMenuOpen ? 'open' : 'closed'}
                className="relative w-5 h-4"
              >
                <motion.span
                  variants={{
                    closed: { rotate: 0, y: 0 },
                    open: { rotate: 45, y: 6 },
                  }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-0 left-0 w-5 h-0.5 bg-foreground rounded-full origin-center"
                />
                <motion.span
                  variants={{
                    closed: { opacity: 1, x: 0 },
                    open: { opacity: 0, x: -10 },
                  }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-1.5 left-0 w-5 h-0.5 bg-foreground rounded-full"
                />
                <motion.span
                  variants={{
                    closed: { rotate: 0, y: 0 },
                    open: { rotate: -45, y: -6 },
                  }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-3 left-0 w-5 h-0.5 bg-foreground rounded-full origin-center"
                />
              </motion.div>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Full-Page Navigation */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-background md:hidden"
          >
            <motion.nav
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              className="flex flex-col items-center justify-center h-full gap-8"
            >
              {navLinks.map((link, i) => (
                <motion.div
                  key={link.href}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + i * 0.05 }}
                >
                  <Link
                    to={link.href}
                    onClick={closeMenu}
                    className={`text-3xl font-medium transition-colors ${
                      isActive(link.href) ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {link.label}
                  </Link>
                </motion.div>
              ))}
              
              {user && adminLinks.map((link, i) => (
                <motion.div
                  key={link.href}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.15 + i * 0.05 }}
                >
                  <Link
                    to={link.href}
                    onClick={closeMenu}
                    className={`text-3xl font-medium transition-colors flex items-center gap-3 ${
                      isActive(link.href) ? 'text-foreground' : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <link.icon className="h-6 w-6" />
                    {link.label}
                  </Link>
                </motion.div>
              ))}

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
                className="pt-8 border-t border-border w-48 space-y-3"
              >
                <SuggestStockDialog />
                {user ? (
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => { logout(); closeMenu(); }}
                    className="w-full gap-2"
                  >
                    <LogOut className="h-4 w-4" />
                    Sign out
                  </Button>
                ) : (
                  <Link to="/login" onClick={closeMenu} className="block">
                    <Button variant="default" size="lg" className="w-full">
                      Sign in
                    </Button>
                  </Link>
                )}
              </motion.div>
            </motion.nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
