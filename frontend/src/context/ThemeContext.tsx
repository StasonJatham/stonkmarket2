import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface CustomColors {
  up: string;
  down: string;
}

const DEFAULT_COLORS: CustomColors = {
  up: '#22c55e', // Green
  down: '#ef4444', // Red
};

const COLORBLIND_COLORS: CustomColors = {
  up: '#3b82f6', // Blue
  down: '#f97316', // Orange
};

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resolvedTheme: 'light' | 'dark';
  colorblindMode: boolean;
  setColorblindMode: (enabled: boolean) => void;
  customColors: CustomColors;
  setCustomColors: (colors: CustomColors) => void;
  resetColors: () => void;
  /** Get the currently active colors (respects colorblind mode) */
  getActiveColors: () => CustomColors;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window !== 'undefined') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return 'light';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('theme') as Theme) || 'system';
    }
    return 'system';
  });

  const [colorblindMode, setColorblindModeState] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('colorblind-mode') === 'true';
    }
    return false;
  });

  const [customColors, setCustomColorsState] = useState<CustomColors>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('custom-colors');
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          return DEFAULT_COLORS;
        }
      }
    }
    return DEFAULT_COLORS;
  });

  const resolvedTheme = theme === 'system' ? getSystemTheme() : theme;

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(resolvedTheme);
    localStorage.setItem('theme', theme);
  }, [theme, resolvedTheme]);

  useEffect(() => {
    const root = document.documentElement;
    if (colorblindMode) {
      root.classList.add('colorblind');
    } else {
      root.classList.remove('colorblind');
    }
    localStorage.setItem('colorblind-mode', String(colorblindMode));
  }, [colorblindMode]);

  // Apply custom/colorblind colors to CSS variables
  // This ensures text-success and text-danger use the right colors
  useEffect(() => {
    const root = document.documentElement;
    const activeColors = colorblindMode ? COLORBLIND_COLORS : customColors;
    
    // Check if we're using non-default colors
    const isUsingCustomColors = 
      colorblindMode || 
      customColors.up !== DEFAULT_COLORS.up || 
      customColors.down !== DEFAULT_COLORS.down;
    
    if (isUsingCustomColors) {
      // Override the CSS variables directly so text-success/text-danger work
      root.style.setProperty('--success', activeColors.up);
      root.style.setProperty('--danger', activeColors.down);
      root.style.setProperty('--color-success', activeColors.up);
      root.style.setProperty('--color-danger', activeColors.down);
    } else {
      // Remove inline styles to use CSS defaults
      root.style.removeProperty('--success');
      root.style.removeProperty('--danger');
      root.style.removeProperty('--color-success');
      root.style.removeProperty('--color-danger');
    }
    
    // Also set the custom color variables for components that use them directly
    root.style.setProperty('--color-success-custom', activeColors.up);
    root.style.setProperty('--color-danger-custom', activeColors.down);
  }, [colorblindMode, customColors]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'system') {
        const root = document.documentElement;
        root.classList.remove('light', 'dark');
        root.classList.add(getSystemTheme());
      }
    };
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  function setTheme(newTheme: Theme) {
    setThemeState(newTheme);
  }

  function setColorblindMode(enabled: boolean) {
    setColorblindModeState(enabled);
  }

  function setCustomColors(colors: CustomColors) {
    setCustomColorsState(colors);
    localStorage.setItem('custom-colors', JSON.stringify(colors));
    // Apply immediately if not in colorblind mode
    const root = document.documentElement;
    root.style.setProperty('--color-success-custom', colors.up);
    root.style.setProperty('--color-danger-custom', colors.down);
  }

  function resetColors() {
    setCustomColors(DEFAULT_COLORS);
  }

  function getActiveColors() {
    return colorblindMode ? COLORBLIND_COLORS : customColors;
  }

  const value = {
    theme,
    setTheme,
    resolvedTheme,
    colorblindMode,
    setColorblindMode,
    customColors,
    setCustomColors,
    resetColors,
    getActiveColors,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
