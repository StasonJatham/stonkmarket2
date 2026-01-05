import { createContext, useContext, useState, type ReactNode } from 'react';
import { 
  login as authLogin, 
  logout as authLogout,
  logoutAll as authLogoutAll,
  getCurrentUser,
} from '@/services/auth';
import type { User } from '@/services/auth';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isLoading: boolean;
  login: (username: string, password: string, mfaCode?: string) => Promise<void>;
  logout: () => Promise<void>;
  logoutAll: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  // Initialize state from localStorage synchronously to avoid flash
  const [user, setUser] = useState<User | null>(() => getCurrentUser());
  const [isLoading, setIsLoading] = useState(false);

  async function login(username: string, password: string, mfaCode?: string) {
    setIsLoading(true);
    try {
      await authLogin(username, password, mfaCode);
      const currentUser = getCurrentUser();
      setUser(currentUser);
    } finally {
      setIsLoading(false);
    }
  }

  async function logout() {
    setIsLoading(true);
    try {
      await authLogout();
    } finally {
      setUser(null);
      setIsLoading(false);
    }
  }

  async function logoutAll() {
    setIsLoading(true);
    try {
      await authLogoutAll();
    } finally {
      setUser(null);
      setIsLoading(false);
    }
  }

  const value = {
    user,
    isAuthenticated: !!user,
    isAdmin: user?.is_admin ?? false,
    isLoading,
    login,
    logout,
    logoutAll,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
