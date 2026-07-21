import { flushSync } from 'svelte';
import { vi } from 'vitest';
/**
 * Unit tests for useAutoCapture hook
 *
 * Tests interval-based auto-capture timing with dynamic adjustment.
 */

import { useAutoCapture, type AutoCaptureConfig } from '../useAutoCapture';

type HookHarness = {
  config: AutoCaptureConfig;
  rerender: (changes: Partial<AutoCaptureConfig>) => void;
  unmount: () => void;
};

const mountedHooks: Array<() => void> = [];

function mountHook(initialConfig: AutoCaptureConfig): HookHarness {
  const config = $state<AutoCaptureConfig>({ ...initialConfig });
  let dispose: (() => void) | undefined;
  let disposed = false;

  dispose = $effect.root(() => {
    useAutoCapture(config);
  });
  flushSync();

  const unmount = (): void => {
    if (disposed) {
      return;
    }

    disposed = true;
    dispose?.();
    dispose = undefined;
    flushSync();
  };

  mountedHooks.push(unmount);

  return {
    config,
    rerender: changes => {
      Object.assign(config, changes);
      flushSync();
    },
    unmount,
  };
}

async function advanceTimersByTime(milliseconds: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(milliseconds);
  flushSync();
}

describe('useAutoCapture', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });

  afterEach(() => {
    while (mountedHooks.length > 0) {
      mountedHooks.pop()?.();
    }
    vi.useRealTimers();
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  describe('basic functionality', () => {
    it('should not run when disabled', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);

      mountHook({
        enabled: false,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(5000);

      expect(mockOnCapture).not.toHaveBeenCalled();
    });

    it('should not run when mode is posture-change', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);

      mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'posture-change',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(5000);

      expect(mockOnCapture).not.toHaveBeenCalled();
    });

    it('should call onCapture callback at specified interval', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);

      mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });
  });

  describe('dynamic timing', () => {
    it('should adjust next timeout based on processing duration', async () => {
      const processingTimeMs = 200;
      const intervalMs = 1000;
      const mockOnCapture = vi.fn<() => Promise<void>>().mockImplementation(async () => {
        await vi.advanceTimersByTimeAsync(processingTimeMs);
      });

      mountHook({
        enabled: true,
        intervalSeconds: intervalMs / 1000,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(intervalMs);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      await advanceTimersByTime(800);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });

    it('should use Math.max(0, ...) when processing exceeds interval', async () => {
      const processingTimeMs = 1500;
      const intervalMs = 1000;
      const mockOnCapture = vi.fn<() => Promise<void>>().mockImplementation(async () => {
        return new Promise<void>(resolve => {
          setTimeout(resolve, processingTimeMs);
        });
      });

      mountHook({
        enabled: true,
        intervalSeconds: intervalMs / 1000,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(intervalMs + processingTimeMs);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      await advanceTimersByTime(100);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });
  });

  describe('error handling', () => {
    it('should continue scheduling on callback errors', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>()
        .mockRejectedValueOnce(new Error('First capture failed'))
        .mockResolvedValueOnce(undefined)
        .mockResolvedValueOnce(undefined);

      mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(3);
    });

    it('should use full interval on error (no processing time)', async () => {
      const intervalMs = 1000;
      let callCount = 0;
      const mockOnCapture = vi.fn<() => Promise<void>>().mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.reject(new Error('Capture failed'));
        }
        return Promise.resolve(undefined);
      });

      mountHook({
        enabled: true,
        intervalSeconds: intervalMs / 1000,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(intervalMs);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      await advanceTimersByTime(intervalMs);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });
  });

  describe('effect lifecycle', () => {
    it('should cleanup timeout on unmount', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const harness = mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.unmount();
      await advanceTimersByTime(2000);

      expect(mockOnCapture).toHaveBeenCalledTimes(1);
    });

    it('does not recreate a timer when an in-flight capture resolves after unmount', async () => {
      let resolveCapture!: () => void;
      const mockOnCapture = vi.fn<() => Promise<void>>().mockReturnValue(
        new Promise<void>((resolve) => { resolveCapture = resolve; }),
      );
      const harness = mountHook({ enabled: true, intervalSeconds: 1, mode: 'interval', onCapture: mockOnCapture });
      await advanceTimersByTime(1000);
      harness.unmount();
      resolveCapture();
      await Promise.resolve();
      await advanceTimersByTime(5000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    });

    it('does not recreate a timer when an in-flight capture rejects after disable', async () => {
      let rejectCapture!: (error: Error) => void;
      const mockOnCapture = vi.fn<() => Promise<void>>().mockReturnValue(
        new Promise<void>((_resolve, reject) => { rejectCapture = reject; }),
      );
      const harness = mountHook({ enabled: true, intervalSeconds: 1, mode: 'interval', onCapture: mockOnCapture });
      await advanceTimersByTime(1000);
      harness.rerender({ enabled: false });
      rejectCapture(new Error('late failure'));
      await Promise.resolve();
      await advanceTimersByTime(5000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
    });

    it('should restart when enabled changes', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const harness = mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.rerender({ enabled: false });
      await advanceTimersByTime(2000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.rerender({ enabled: true });
      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });

    it('should restart when interval changes', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const harness = mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.rerender({ intervalSeconds: 2 });
      await advanceTimersByTime(2000);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });

    it('should restart when mode changes', async () => {
      const mockOnCapture = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const harness = mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.rerender({ mode: 'posture-change' });
      await advanceTimersByTime(2000);
      expect(mockOnCapture).toHaveBeenCalledTimes(1);

      harness.rerender({ mode: 'interval' });
      await advanceTimersByTime(1000);
      expect(mockOnCapture).toHaveBeenCalledTimes(2);
    });

    it('should NOT restart when only callback changes (uses ref)', async () => {
      const mockOnCapture1 = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const mockOnCapture2 = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
      const harness = mountHook({
        enabled: true,
        intervalSeconds: 1,
        mode: 'interval',
        onCapture: mockOnCapture1,
      });

      await advanceTimersByTime(1000);
      expect(mockOnCapture1).toHaveBeenCalledTimes(1);
      expect(mockOnCapture2).not.toHaveBeenCalled();

      harness.rerender({ onCapture: mockOnCapture2 });
      await advanceTimersByTime(1000);

      expect(mockOnCapture1).toHaveBeenCalledTimes(1);
      expect(mockOnCapture2).toHaveBeenCalledTimes(1);
    });
  });
});
