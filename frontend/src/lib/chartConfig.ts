/**
 * Central chart animation configuration.
 * 
 * All charts should use these settings for consistent,
 * smooth line morphing animations.
 * 
 * Key principles:
 * - Use linear easing for natural data transitions
 * - Keep duration moderate (500-800ms) for responsiveness
 * - Always keep animation active for smooth morphing
 */

export const CHART_ANIMATION = {
  /** Enable smooth morphing animations */
  isAnimationActive: true,
  
  /** Duration in milliseconds - moderate for responsiveness */
  animationDuration: 600,
  
  /** 
   * Easing function - 'ease-out' gives a natural deceleration
   * Options: 'ease', 'ease-in', 'ease-out', 'ease-in-out', 'linear'
   */
  animationEasing: 'ease-out' as const,
  
  /** 
   * Initial delay before animation starts (ms)
   * Keep at 0 for immediate response
   */
  animationBegin: 0,
} as const;

/**
 * Animation props to spread on Line/Area components.
 * Usage: <Line {...CHART_LINE_ANIMATION} dataKey="value" />
 */
export const CHART_LINE_ANIMATION = {
  isAnimationActive: CHART_ANIMATION.isAnimationActive,
  animationDuration: CHART_ANIMATION.animationDuration,
  animationEasing: CHART_ANIMATION.animationEasing,
  animationBegin: CHART_ANIMATION.animationBegin,
} as const;

/**
 * For mini charts (stock cards), use faster animations
 */
export const CHART_MINI_ANIMATION = {
  isAnimationActive: true,
  animationDuration: 400,
  animationEasing: 'ease-out' as const,
  animationBegin: 0,
} as const;

/**
 * Generate a stable animation ID based on data length.
 * This helps Recharts understand when to morph vs redraw.
 * 
 * @param dataLength - Length of the data array
 * @param prefix - Optional prefix for uniqueness
 */
export function getAnimationId(dataLength: number, prefix = 'chart'): string {
  return `${prefix}-${dataLength}`;
}
