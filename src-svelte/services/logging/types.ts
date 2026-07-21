/**
 * Logging Service Types
 *
 * Type definitions for the URL parameter-based logging system.
 */

/**
 * Log levels (in order of verbosity)
 */
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

/**
 * All available log categories (single source of truth)
 */
export const ALL_LOG_CATEGORIES = ['detection', 'training', 'worker', 'storage', 'debug', 'preprocessing'] as const;

/**
 * Log categories for different subsystems
 */
export type LogCategory = typeof ALL_LOG_CATEGORIES[number];

/**
 * Configuration for a specific category
 */
export interface CategoryConfig {
  category: LogCategory;
  level: LogLevel;
}

/**
 * Logging configuration
 */
export interface LogConfig {
  enabled: boolean;
  minLevel: LogLevel;
  categories: Map<LogCategory, LogLevel>;
}
