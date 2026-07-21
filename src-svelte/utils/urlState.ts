/**
 * URL State Management Utilities
 *
 * Type-safe utilities for syncing application state with URL parameters.
 * Supports tab navigation persistence across page reloads.
 *
 * Usage:
 * ```typescript
 * import { getTabFromURL, setTabInURL, TabType } from '@/utils/urlState';
 *
 * // On component mount
 * const initialTab = getTabFromURL() || 'runtime';
 *
 * // When tab changes
 * setTabInURL('training');
 * ```
 */

import { logger } from '../services/logging/logger';

/**
 * Valid tab types for the unified page
 */
export type TabType = 'runtime' | 'collect' | 'training';

/**
 * Get the current tab from URL query parameters
 *
 * @returns The validated tab type from URL, or null if not present/invalid
 *
 * @example
 * // URL: ?tab=training
 * getTabFromURL() // returns 'training'
 *
 * // URL: ?tab=invalid
 * getTabFromURL() // returns null
 *
 * // URL: (no tab parameter)
 * getTabFromURL() // returns null
 */
export function getTabFromURL(): TabType | null {
  // Check for browser environment (SSR safety)
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const urlParams = new URLSearchParams(window.location.search);
    const tabParam = urlParams.get('tab');

    // Validate tab parameter
    if (tabParam && isValidTabType(tabParam)) {
      return tabParam;
    }

    return null;
  } catch (error) {
    // If parsing fails, return null (fallback to default)
    logger.error('storage', '[urlState] Failed to parse tab from URL:', error);
    return null;
  }
}

/**
 * Set the tab parameter in the URL without reloading the page
 *
 * Uses window.history.replaceState to update the URL without polluting
 * browser history. This allows users to bookmark or share the current tab.
 *
 * @param tab - The tab type to set in the URL
 *
 * @example
 * setTabInURL('training');
 * // URL becomes: ?tab=training (or ?tab=training&log=debug if other params exist)
 */
export function setTabInURL(tab: TabType): void {
  // Check for browser environment (SSR safety)
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', tab);

    // Use replaceState to avoid polluting browser history
    window.history.replaceState({}, '', url.toString());
  } catch (error) {
    // If updating fails, log error but don't crash
    logger.error('storage', '[urlState] Failed to set tab in URL:', error);
  }
}

/**
 * Type guard to check if a string is a valid TabType
 *
 * @param value - The string to validate
 * @returns True if the value is a valid TabType
 */
function isValidTabType(value: string): value is TabType {
  return ['runtime', 'collect', 'training'].includes(value);
}
