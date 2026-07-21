import { flushSync } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { useActionHistory } from '../useActionHistory';
import type { CaptureAction } from '@/services/dataset/types';
import { FrameLabel } from '@/services/dataset/types';

type ActionHistoryHook = ReturnType<typeof useActionHistory>;

type HookHarness = {
  result: ActionHistoryHook;
  unmount: () => void;
};

const mountedHooks: Array<() => void> = [];

function mountHook(): HookHarness {
  let result!: ActionHistoryHook;
  let disposed = false;
  const dispose = $effect.root(() => {
    result = useActionHistory();
  });

  flushSync();

  const unmount = (): void => {
    if (disposed) {
      return;
    }

    disposed = true;
    dispose();
    flushSync();
  };

  mountedHooks.push(unmount);

  return { result, unmount };
}

function createMockAction(
  overrides: Partial<CaptureAction> = {},
): CaptureAction {
  return {
    frameId: `frame-${Date.now()}`,
    timestamp: Date.now(),
    label: FrameLabel.GOOD,
    thumbnailUrl: 'data:image/webp;base64,mock',
    captureSource: 'manual',
    ...overrides,
  };
}

afterEach(() => {
  while (mountedHooks.length > 0) {
    mountedHooks.pop()?.();
  }
});

describe('useActionHistory', () => {
  describe('initialization', () => {
    it('should initialize with empty history', () => {
      const { result } = mountHook();

      expect(result.canUndo).toBe(false);
      expect(result.lastAction).toBeNull();
    });
  });

  describe('push', () => {
    it('should add action to history', () => {
      const { result } = mountHook();
      const action = createMockAction();

      flushSync(() => {
        result.push(action);
      });

      expect(result.canUndo).toBe(true);
      expect(result.lastAction).toEqual(action);
    });

    it('should update lastAction when pushing multiple actions', () => {
      const { result } = mountHook();
      const action1 = createMockAction({ frameId: 'frame-1' });
      const action2 = createMockAction({ frameId: 'frame-2' });

      flushSync(() => {
        result.push(action1);
        result.push(action2);
      });

      expect(result.lastAction).toEqual(action2);
      expect(result.canUndo).toBe(true);
    });

    it('should maintain max history size of 5', () => {
      const { result } = mountHook();
      const actions = Array.from({ length: 7 }, (_, i) =>
        createMockAction({ frameId: `frame-${i}` }),
      );

      flushSync(() => {
        actions.forEach(action => result.push(action));
      });

      expect(result.lastAction).toEqual(actions[6]);

      const undoneActions: (CaptureAction | null)[] = [];
      flushSync(() => {
        let action = result.undo();
        while (action !== null) {
          undoneActions.push(action);
          action = result.undo();
        }
      });

      expect(undoneActions.length).toBe(5);
    });
  });

  describe('undo', () => {
    it('should pop and return last action', () => {
      const { result } = mountHook();
      const action = createMockAction();

      flushSync(() => {
        result.push(action);
      });

      let poppedAction: CaptureAction | null = null;
      flushSync(() => {
        poppedAction = result.undo();
      });

      expect(poppedAction).toEqual(action);
      expect(result.canUndo).toBe(false);
      expect(result.lastAction).toBeNull();
    });

    it('should return null when history is empty', () => {
      const { result } = mountHook();

      let poppedAction: CaptureAction | null = null;
      flushSync(() => {
        poppedAction = result.undo();
      });

      expect(poppedAction).toBeNull();
      expect(result.canUndo).toBe(false);
    });

    it('should update lastAction to previous action after undo', () => {
      const { result } = mountHook();
      const action1 = createMockAction({ frameId: 'frame-1' });
      const action2 = createMockAction({ frameId: 'frame-2' });

      flushSync(() => {
        result.push(action1);
        result.push(action2);
      });

      expect(result.lastAction).toEqual(action2);

      flushSync(() => {
        result.undo();
      });

      expect(result.lastAction).toEqual(action1);
      expect(result.canUndo).toBe(true);
    });

    it('should handle multiple undos correctly', () => {
      const { result } = mountHook();
      const actions = Array.from({ length: 3 }, (_, i) =>
        createMockAction({ frameId: `frame-${i}` }),
      );

      flushSync(() => {
        actions.forEach(action => result.push(action));
      });

      const undoneActions: (CaptureAction | null)[] = [];
      flushSync(() => {
        undoneActions.push(result.undo());
        undoneActions.push(result.undo());
        undoneActions.push(result.undo());
      });

      expect(undoneActions[0]).toEqual(actions[2]);
      expect(undoneActions[1]).toEqual(actions[1]);
      expect(undoneActions[2]).toEqual(actions[0]);
      expect(result.canUndo).toBe(false);
    });
  });

  describe('clear', () => {
    it('should remove all actions from history', () => {
      const { result } = mountHook();

      flushSync(() => {
        result.push(createMockAction({ frameId: 'frame-1' }));
        result.push(createMockAction({ frameId: 'frame-2' }));
        result.push(createMockAction({ frameId: 'frame-3' }));
      });

      expect(result.canUndo).toBe(true);

      flushSync(() => {
        result.clear();
      });

      expect(result.canUndo).toBe(false);
      expect(result.lastAction).toBeNull();

      let poppedAction: CaptureAction | null = null;
      flushSync(() => {
        poppedAction = result.undo();
      });

      expect(poppedAction).toBeNull();
    });
  });

  describe('integration scenarios', () => {
    it('should handle push-undo-push sequence', () => {
      const { result } = mountHook();
      const action1 = createMockAction({ frameId: 'frame-1' });
      const action2 = createMockAction({ frameId: 'frame-2' });

      flushSync(() => {
        result.push(action1);
      });

      expect(result.lastAction).toEqual(action1);

      flushSync(() => {
        result.undo();
      });

      expect(result.canUndo).toBe(false);

      flushSync(() => {
        result.push(action2);
      });

      expect(result.lastAction).toEqual(action2);
      expect(result.canUndo).toBe(true);
    });
  });
});
