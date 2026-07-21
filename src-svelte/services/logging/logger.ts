/**
 * URL Parameter-Based Logging Service
 *
 * Provides controlled logging with category-based filtering and URL parameter configuration.
 *
 * URL Parameter Format:
 * - ?log=training,detection - Enable specific categories (INFO level)
 * - ?log=all - Enable all categories (INFO level)
 * - ?log=debug - Enable DEBUG level for all categories
 * - ?log=training:debug,detection:info - Category-specific levels
 * - ?log=none or no parameter - Only WARN and ERROR (production-safe default)
 *
 * Usage:
 * ```typescript
 * import { logger } from '@/services/logging/logger';
 *
 * logger.debug('detection', '[Classifier] Input features:', features);
 * logger.info('training', '[Training] Starting cross-validation');
 * logger.warn('storage', '[Storage] Low quota remaining');
 * logger.error('worker', '[Worker] Failed to initialize:', error);
 * ```
 */

import { ALL_LOG_CATEGORIES, LogCategory, LogConfig, LogLevel } from './types';

// tauri-plugin-log numeric LogLevel (Trace=1..Error=5). Only warn/error are
// forwarded to the native file log, so only those two are needed here.
const NATIVE_LOG_WARN = 4;
const NATIVE_LOG_ERROR = 5;

/** True only inside the packaged Tauri webview, where the IPC bridge exists. */
function tauriRuntimePresent(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== 'undefined'
  );
}

/** Flatten console-style args into a single log line without ever throwing. */
function formatLogArgs(args: unknown[]): string {
  return args
    .map((arg) => {
      if (typeof arg === 'string') return arg;
      if (arg instanceof Error) return arg.stack ?? `${arg.name}: ${arg.message}`;
      try {
        return JSON.stringify(arg);
      } catch {
        return String(arg);
      }
    })
    .join(' ');
}

/**
 * Best-effort mirror of a warn/error line into the Rust file log via
 * tauri-plugin-log's invoke command. Additive to the console output, guarded on
 * the Tauri runtime, and swallowing every failure so logging can never throw or
 * reject into caller code (and so non-Tauri/vitest environments are untouched:
 * the module is only imported when the runtime is present).
 */
function forwardToNativeLog(level: number, category: LogCategory, args: unknown[]): void {
  if (!tauriRuntimePresent()) return;
  try {
    const message = `[${category}] ${formatLogArgs(args)}`;
    void import('@tauri-apps/api/core')
      .then(({ invoke }) => invoke('plugin:log|log', { level, message }))
      .catch(() => {
        /* best-effort: never surface a logging-transport failure */
      });
  } catch {
    /* best-effort: guard against synchronous import/runtime failures */
  }
}

class Logger {
  private config: LogConfig;

  constructor() {
    // Initialize with default config (production-safe: only WARN and ERROR)
    this.config = {
      enabled: false,
      minLevel: LogLevel.WARN,
      categories: new Map(),
    };

    // Parse URL parameters if in browser environment
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const logParam = urlParams.get('log');
      if (logParam) {
        this.setFromURLParam(logParam);
      }
    }
  }

  /**
   * Configure logger from URL parameter value
   *
   * This method can be called at runtime to reconfigure the logger
   * without requiring a page reload.
   *
   * @param logParam - The value of the 'log' URL parameter
   *
   * @example
   * logger.setFromURLParam('debug'); // Enable debug logging
   * logger.setFromURLParam('training,detection'); // Enable specific categories
   * logger.setFromURLParam('none'); // Disable all logging (errors only)
   */
  public setFromURLParam(logParam: string): void {
    try {
      if (!logParam || logParam === 'none') {
        // Production mode: only WARN and ERROR
        this.config.enabled = false;
        this.config.minLevel = LogLevel.WARN;
        this.config.categories.clear();
        return;
      }

      if (logParam === 'all') {
        // Enable all categories at INFO level
        this.config.enabled = true;
        this.config.minLevel = LogLevel.INFO;
        this.enableAllCategories(LogLevel.INFO);
        return;
      }

      if (logParam === 'debug') {
        // Enable all categories at DEBUG level
        this.config.enabled = true;
        this.config.minLevel = LogLevel.DEBUG;
        this.enableAllCategories(LogLevel.DEBUG);
        return;
      }

      // Parse comma-separated category:level pairs
      this.config.enabled = true;
      this.config.minLevel = LogLevel.INFO; // Default for enabled logging
      this.config.categories.clear();

      const parts = logParam.split(',');
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        if (trimmed.includes(':')) {
          // Format: category:level
          const [categoryStr, levelStr] = trimmed.split(':');
          const category = categoryStr.trim() as LogCategory;
          const level = this.parseLevelString(levelStr.trim());

          if (this.isValidCategory(category) && level !== null) {
            this.config.categories.set(category, level);
          }
        } else {
          // Format: category (default to INFO level)
          const category = trimmed as LogCategory;
          if (this.isValidCategory(category)) {
            this.config.categories.set(category, LogLevel.INFO);
          }
        }
      }

      // If categories were specified, update min level to lowest enabled
      if (this.config.categories.size > 0) {
        const levels = Array.from(this.config.categories.values());
        this.config.minLevel = Math.min(...levels);
      }
    } catch (error) {
      // If parsing fails, fall back to production mode
      console.error('[Logger] Failed to parse log parameter:', error);
      this.config.enabled = false;
      this.config.minLevel = LogLevel.WARN;
      this.config.categories.clear();
    }
  }

  /**
   * Enable all categories at the specified level
   */
  private enableAllCategories(level: LogLevel): void {
    for (const category of ALL_LOG_CATEGORIES) {
      this.config.categories.set(category, level);
    }
  }

  /**
   * Parse level string to LogLevel enum
   */
  private parseLevelString(levelStr: string): LogLevel | null {
    const normalized = levelStr.toLowerCase();
    switch (normalized) {
      case 'debug':
        return LogLevel.DEBUG;
      case 'info':
        return LogLevel.INFO;
      case 'warn':
      case 'warning':
        return LogLevel.WARN;
      case 'error':
        return LogLevel.ERROR;
      default:
        return null;
    }
  }

  /**
   * Check if a string is a valid category
   */
  private isValidCategory(category: string): category is LogCategory {
    return ALL_LOG_CATEGORIES.includes(category as LogCategory);
  }

  /**
   * Check if logging is enabled for a specific category and level
   * @public - Exposed for conditional expensive logging operations
   */
  public isEnabled(category: LogCategory, level: LogLevel): boolean {
    // Always allow WARN and ERROR regardless of config (production-safe)
    if (level >= LogLevel.WARN) {
      return true;
    }

    // If logging is disabled, only WARN/ERROR pass through
    if (!this.config.enabled) {
      return false;
    }

    // Check if category is explicitly enabled
    const categoryLevel = this.config.categories.get(category);
    if (categoryLevel !== undefined) {
      return level >= categoryLevel;
    }

    // Category not explicitly enabled
    return false;
  }

  /**
   * Check if debug logging is enabled for a category
   * Use this to guard expensive debug logging operations
   */
  public isDebugEnabled(category: LogCategory): boolean {
    return this.isEnabled(category, LogLevel.DEBUG);
  }

  /**
   * Log a DEBUG message
   */
  debug(category: LogCategory, ...args: unknown[]): void {
    if (this.isEnabled(category, LogLevel.DEBUG)) {
      console.log(...args);
    }
  }

  /**
   * Log an INFO message
   */
  info(category: LogCategory, ...args: unknown[]): void {
    if (this.isEnabled(category, LogLevel.INFO)) {
      console.log(...args);
    }
  }

  /**
   * Log a WARN message
   */
  warn(category: LogCategory, ...args: unknown[]): void {
    if (this.isEnabled(category, LogLevel.WARN)) {
      console.warn(...args);
      forwardToNativeLog(NATIVE_LOG_WARN, category, args);
    }
  }

  /**
   * Log an ERROR message
   */
  error(category: LogCategory, ...args: unknown[]): void {
    if (this.isEnabled(category, LogLevel.ERROR)) {
      console.error(...args);
      forwardToNativeLog(NATIVE_LOG_ERROR, category, args);
    }
  }

  /**
   * Get current configuration (for debugging)
   */
  getConfig(): LogConfig {
    return {
      enabled: this.config.enabled,
      minLevel: this.config.minLevel,
      categories: new Map(this.config.categories),
    };
  }

  /**
   * Set configuration programmatically (for Web Workers)
   */
  setConfig(config: Partial<LogConfig>): void {
    if (config.enabled !== undefined) {
      this.config.enabled = config.enabled;
    }
    if (config.minLevel !== undefined) {
      this.config.minLevel = config.minLevel;
    }
    if (config.categories !== undefined) {
      this.config.categories = new Map(config.categories);
    }
  }
}

// Export singleton instance
export const logger = new Logger();
