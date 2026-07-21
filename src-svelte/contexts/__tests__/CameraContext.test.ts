import { beforeEach, describe, expect, it, vi } from 'vitest';

const svelteContext = vi.hoisted(() => ({
  getContext: vi.fn(),
  setContext: vi.fn(),
}));

vi.mock('svelte', () => svelteContext);

import {
  CameraProvider,
  type CameraContextValue,
  useCameraContext,
} from '../CameraContext';

describe('CameraContext', () => {
  const createMockContextValue = (): CameraContextValue => ({
    inferenceResult: null,
    fps: 30,
  });

  let currentContext: CameraContextValue | null;

  beforeEach(() => {
    currentContext = null;
    svelteContext.getContext.mockImplementation(() => currentContext);
    svelteContext.setContext.mockImplementation(
      (_key: symbol, value: CameraContextValue) => {
        currentContext = value;
      },
    );
  });

  describe('CameraProvider', () => {
    it('should provide updated context value when value changes', () => {
      const contextValue1 = createMockContextValue();
      CameraProvider(contextValue1);

      expect(useCameraContext().fps).toBe(30);

      const contextValue2 = createMockContextValue();
      contextValue2.fps = 60;
      CameraProvider(contextValue2);

      expect(useCameraContext().fps).toBe(60);
      expect(svelteContext.setContext).toHaveBeenCalledTimes(2);
    });
  });

  describe('useCameraContext', () => {
    it('should throw error when used outside provider', () => {
      expect(() => useCameraContext()).toThrow(
        'useCameraContext must be used within CameraProvider',
      );
    });
  });
});
