import { vi } from 'vitest';
/**
 * Logger Service Tests
 */

// Unmock the logger for these tests (we want to test the real logger)
vi.unmock('../logger');

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn().mockResolvedValue(undefined) }));

import { invoke } from '@tauri-apps/api/core';
import { logger } from '../logger';
import { ALL_LOG_CATEGORIES, LogLevel, type LogCategory } from '../types';

const invokeMock = invoke as unknown as ReturnType<typeof vi.fn>;

// Mock window.location
const mockLocation = (search: string) => {
  delete (global as any).window;
  (global as any).window = {
    location: {
      search,
    },
  };
};

// Non-Tauri window: no __TAURI_INTERNALS__, so forwarding must never fire.
// Tauri window: presence of the invoke bridge enables best-effort forwarding.
const mockTauriWindow = (search = '') => {
  delete (global as any).window;
  (global as any).window = {
    location: { search },
    __TAURI_INTERNALS__: { invoke: () => Promise.resolve() },
  };
};

describe('Logger Service', () => {
  let consoleLogSpy: any;
  let consoleWarnSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    // Spy on console methods
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined as any);
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined as any);
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined as any);
  });

  afterEach(() => {
    // Restore console methods
    consoleLogSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  describe('Native log forwarding (Tauri runtime)', () => {
    beforeEach(() => {
      mockTauriWindow('');
      logger.setConfig({ enabled: false, minLevel: LogLevel.WARN, categories: new Map() });
      invokeMock.mockClear();
      invokeMock.mockResolvedValue(undefined);
    });

    it('forwards ERROR to the native file log at level 5', async () => {
      logger.error('worker', 'boom', new Error('bad'));
      await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledTimes(1));
      const [command, payload] = invokeMock.mock.calls[0] as [string, { level: number; message: string }];
      expect(command).toBe('plugin:log|log');
      expect(payload.level).toBe(5);
      expect(payload.message).toContain('[worker]');
      expect(payload.message).toContain('boom');
    });

    it('forwards WARN at level 4', async () => {
      logger.warn('storage', 'careful');
      await vi.waitFor(() => expect(invokeMock).toHaveBeenCalledTimes(1));
      expect((invokeMock.mock.calls[0][1] as { level: number }).level).toBe(4);
    });

    it('does not forward INFO or DEBUG', async () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.DEBUG,
        categories: new Map([['detection', LogLevel.DEBUG]]),
      });
      logger.info('detection', 'info');
      logger.debug('detection', 'debug');
      await new Promise((resolve) => setTimeout(resolve, 15));
      expect(invokeMock).not.toHaveBeenCalled();
    });

    it('never throws when the native transport rejects', async () => {
      invokeMock.mockRejectedValue(new Error('no permission'));
      expect(() => logger.error('worker', 'still fine')).not.toThrow();
      await new Promise((resolve) => setTimeout(resolve, 15));
    });
  });

  describe('Default Behavior (Production Mode)', () => {
    beforeEach(() => {
      mockLocation('');
      logger.setConfig({ enabled: false, minLevel: LogLevel.WARN, categories: new Map() });
    });

    it('should not log DEBUG messages by default', () => {
      logger.debug('detection', 'debug message');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should not log INFO messages by default', () => {
      logger.info('detection', 'info message');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should log WARN messages by default', () => {
      logger.warn('detection', 'warn message');
      expect(consoleWarnSpy).toHaveBeenCalledWith('warn message');
    });

    it('should log ERROR messages by default', () => {
      logger.error('detection', 'error message');
      expect(consoleErrorSpy).toHaveBeenCalledWith('error message');
    });
  });

  describe('URL Parameter: ?log=none', () => {
    beforeEach(() => {
      mockLocation('?log=none');
      logger.setFromURLParam('none');
    });

    it('should only log WARN and ERROR', () => {
      logger.debug('detection', 'debug');
      logger.info('detection', 'info');
      logger.warn('detection', 'warn');
      logger.error('detection', 'error');

      expect(consoleLogSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe('URL Parameter: ?log=all', () => {
    beforeEach(() => {
      mockLocation('?log=all');
      logger.setFromURLParam('all');
    });

    it('should log INFO and above for all categories', () => {
      for (const category of ALL_LOG_CATEGORIES) {
        logger.info(category, `${category} info`);
      }
      logger.warn('worker', 'worker warn');
      logger.error('storage', 'storage error');

      expect(consoleLogSpy).toHaveBeenCalledTimes(ALL_LOG_CATEGORIES.length);
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
    });

    it('should not log DEBUG for any category', () => {
      logger.debug('detection', 'debug');
      logger.debug('training', 'debug');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });
  });

  describe('URL Parameter: ?log=debug', () => {
    beforeEach(() => {
      mockLocation('?log=debug');
      logger.setFromURLParam('debug');
    });

    it('should log DEBUG and above for all categories', () => {
      for (const category of ALL_LOG_CATEGORIES) {
        logger.debug(category, `${category} debug`);
      }
      logger.warn('worker', 'warn');
      logger.error('storage', 'error');

      expect(consoleLogSpy).toHaveBeenCalledTimes(ALL_LOG_CATEGORIES.length);
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe('URL Parameter: ?log=training,detection', () => {
    beforeEach(() => {
      mockLocation('?log=training,detection');
      logger.setFromURLParam('training,detection');
    });

    it('should log INFO for specified categories', () => {
      logger.info('detection', 'detection info');
      logger.info('training', 'training info');
      expect(consoleLogSpy).toHaveBeenCalledTimes(2);
    });

    it('should not log INFO for non-specified categories', () => {
      logger.info('worker', 'worker info');
      logger.info('storage', 'storage info');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should not log DEBUG for any category', () => {
      logger.debug('detection', 'debug');
      logger.debug('training', 'debug');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should always log WARN and ERROR for all categories', () => {
      logger.warn('worker', 'worker warn');
      logger.error('storage', 'storage error');
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe('URL Parameter: ?log=training:debug,detection:info', () => {
    beforeEach(() => {
      mockLocation('?log=training:debug,detection:info');
      logger.setFromURLParam('training:debug,detection:info');
    });

    it('should log DEBUG for training category', () => {
      logger.debug('training', 'training debug');
      expect(consoleLogSpy).toHaveBeenCalledWith('training debug');
    });

    it('should not log DEBUG for detection category', () => {
      logger.debug('detection', 'detection debug');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should log INFO for detection category', () => {
      logger.info('detection', 'detection info');
      expect(consoleLogSpy).toHaveBeenCalledWith('detection info');
    });

    it('should log INFO for training category', () => {
      logger.info('training', 'training info');
      expect(consoleLogSpy).toHaveBeenCalledWith('training info');
    });
  });

  describe('Multiple Arguments', () => {
    beforeEach(() => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });
    });

    it('should pass all arguments to console.log', () => {
      const obj = { foo: 'bar' };
      const arr = [1, 2, 3];
      logger.info('detection', 'Message:', obj, arr, 42);
      expect(consoleLogSpy).toHaveBeenCalledWith('Message:', obj, arr, 42);
    });

    it('should pass all arguments to logger.warn', () => {
      logger.warn('detection', 'Warning:', 'multiple', 'args');
      expect(consoleWarnSpy).toHaveBeenCalledWith('Warning:', 'multiple', 'args');
    });

    it('should pass all arguments to logger.error', () => {
      const error = new Error('test');
      logger.error('detection', 'Error:', error);
      expect(consoleErrorSpy).toHaveBeenCalledWith('Error:', error);
    });
  });

  describe('Edge Cases', () => {
    it('should handle invalid category gracefully', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });

      // TypeScript would prevent this, but test runtime behavior
      (logger.info as any)('invalid_category', 'test');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should handle empty log calls', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });

      logger.info('detection');
      expect(consoleLogSpy).toHaveBeenCalledWith();
    });

    it('should handle null and undefined arguments', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });

      logger.info('detection', null, undefined);
      expect(consoleLogSpy).toHaveBeenCalledWith(null, undefined);
    });
  });

  describe('getConfig', () => {
    it('should return current configuration', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.DEBUG,
        categories: new Map([
          ['detection', LogLevel.DEBUG],
          ['training', LogLevel.INFO],
        ]),
      });

      const config = logger.getConfig();
      expect(config.enabled).toBe(true);
      expect(config.minLevel).toBe(LogLevel.DEBUG);
      expect(config.categories.get('detection')).toBe(LogLevel.DEBUG);
      expect(config.categories.get('training')).toBe(LogLevel.INFO);
    });

    it('should return a copy of categories map', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });

      const config1 = logger.getConfig();
      const config2 = logger.getConfig();

      // Should be different Map instances
      expect(config1.categories).not.toBe(config2.categories);
      // But with same content
      expect(config1.categories.get('detection')).toBe(config2.categories.get('detection'));
    });
  });

  describe('setConfig', () => {
    it('should update enabled flag', () => {
      logger.setConfig({ enabled: true });
      expect(logger.getConfig().enabled).toBe(true);

      logger.setConfig({ enabled: false });
      expect(logger.getConfig().enabled).toBe(false);
    });

    it('should update minLevel', () => {
      logger.setConfig({ minLevel: LogLevel.DEBUG });
      expect(logger.getConfig().minLevel).toBe(LogLevel.DEBUG);

      logger.setConfig({ minLevel: LogLevel.ERROR });
      expect(logger.getConfig().minLevel).toBe(LogLevel.ERROR);
    });

    it('should update categories map', () => {
      const categories = new Map([
        ['detection', LogLevel.DEBUG],
        ['training', LogLevel.INFO],
      ]) as Map<LogCategory, LogLevel>;
      logger.setConfig({ categories });
      expect(logger.getConfig().categories.get('detection')).toBe(LogLevel.DEBUG);
      expect(logger.getConfig().categories.get('training')).toBe(LogLevel.INFO);
    });

    it('should allow partial updates', () => {
      logger.setConfig({
        enabled: true,
        minLevel: LogLevel.INFO,
        categories: new Map([['detection', LogLevel.INFO]]),
      });

      // Update only enabled flag
      logger.setConfig({ enabled: false });
      const config = logger.getConfig();
      expect(config.enabled).toBe(false);
      expect(config.minLevel).toBe(LogLevel.INFO); // unchanged
      expect(config.categories.get('detection')).toBe(LogLevel.INFO); // unchanged
    });
  });
});
