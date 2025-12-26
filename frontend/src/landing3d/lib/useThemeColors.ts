import { useMemo } from 'react';
import { Color } from 'three';

/**
 * Theme-aware color palette for the 3D landing page.
 * Supports colorblind mode and custom accent colors from ThemeContext.
 */
export interface ThemeColors {
  // Base colors (black/white adapted for dark theme)
  background: string;
  foreground: string;
  
  // Accent colors (from theme - up/down or colorblind friendly)
  positive: string;
  negative: string;
  
  // Neutral accents
  primary: string;
  secondary: string;
  muted: string;
  
  // Pre-converted Three.js colors
  positiveColor: Color;
  negativeColor: Color;
  primaryColor: Color;
  secondaryColor: Color;
}

// Default color palette - works for both light and dark themes
const DEFAULT_COLORS = {
  // Standard green/red
  positive: '#22c55e',
  negative: '#ef4444',
};

const COLORBLIND_COLORS = {
  // Blue/orange for colorblind safety
  positive: '#3b82f6',
  negative: '#f97316',
};

export function getThemeColors(
  colorblindMode: boolean = false,
  customColors?: { up: string; down: string },
  isDarkMode: boolean = true
): ThemeColors {
  // Determine which accent colors to use
  const positive = colorblindMode ? COLORBLIND_COLORS.positive : (customColors?.up || DEFAULT_COLORS.positive);
  const negative = colorblindMode ? COLORBLIND_COLORS.negative : (customColors?.down || DEFAULT_COLORS.negative);
  
  return {
    // Base - always dark for the 3D scene
    background: isDarkMode ? '#030508' : '#f8fafc',
    foreground: isDarkMode ? '#e5edff' : '#0f172a',
    
    // Theme accents
    positive,
    negative,
    
    // Neutral accents - monochrome with slight blue tint
    primary: '#5588ff',
    secondary: '#8899bb',
    muted: '#4a5568',
    
    // Pre-converted for Three.js
    positiveColor: new Color(positive),
    negativeColor: new Color(negative),
    primaryColor: new Color('#5588ff'),
    secondaryColor: new Color('#8899bb'),
  };
}

/**
 * Hook to get theme colors for use in Three.js components.
 * Uses CSS custom properties set by ThemeContext.
 */
export function useThemeColors(): ThemeColors {
  return useMemo(() => {
    // Read from CSS custom properties if available
    if (typeof document !== 'undefined') {
      const root = document.documentElement;
      const style = getComputedStyle(root);
      
      const colorblindMode = root.classList.contains('colorblind');
      const isDarkMode = root.classList.contains('dark');
      
      // Try to read custom colors from CSS variables
      const customUp = style.getPropertyValue('--color-success-custom')?.trim();
      const customDown = style.getPropertyValue('--color-danger-custom')?.trim();
      
      const customColors = (customUp && customDown) 
        ? { up: customUp, down: customDown }
        : undefined;
      
      return getThemeColors(colorblindMode, customColors, isDarkMode);
    }
    
    return getThemeColors(false, undefined, true);
  }, []);
}

/**
 * Get a neutral color between positive and negative for use in gradients
 */
export function getNeutralColor(colors: ThemeColors): Color {
  const neutral = new Color();
  neutral.lerpColors(colors.positiveColor, colors.negativeColor, 0.5);
  return neutral;
}

/**
 * Get color based on a value (-1 to 1), interpolating between negative and positive
 */
export function getValueColor(value: number, colors: ThemeColors): Color {
  const result = new Color();
  if (value >= 0) {
    result.lerpColors(colors.secondaryColor, colors.positiveColor, Math.min(value, 1));
  } else {
    result.lerpColors(colors.secondaryColor, colors.negativeColor, Math.min(-value, 1));
  }
  return result;
}
