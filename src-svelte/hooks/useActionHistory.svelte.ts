/**
 * Svelte state helper for managing undo action history.
 *
 * Provides a ref-based circular buffer for tracking capture actions
 * that can be undone. Maximum 5 actions are retained, oldest are
 * automatically removed when capacity is exceeded.
 */

import type { CaptureAction } from '../services/dataset/types';

const MAX_HISTORY_SIZE = 5;

/**
 * Return type for useActionHistory.
 */
export interface UseActionHistoryReturn {
  push: (action: CaptureAction) => void;
  undo: () => CaptureAction | null;
  canUndo: boolean;
  lastAction: CaptureAction | null;
  clear: () => void;
}

/**
 * Manages the in-memory history of capture actions for undo functionality.
 */
export function useActionHistory(): UseActionHistoryReturn {
  const historyRef: { current: CaptureAction[] } = { current: [] };
  let canUndo = $state(false);
  let lastAction = $state.raw<CaptureAction | null>(null);

  const push = (action: CaptureAction): void => {
    historyRef.current.push(action);

    if (historyRef.current.length > MAX_HISTORY_SIZE) {
      historyRef.current.shift();
    }

    canUndo = true;
    lastAction = action;
  };

  const undo = (): CaptureAction | null => {
    const action = historyRef.current.pop();

    if (!action) {
      return null;
    }

    const hasMore = historyRef.current.length > 0;
    canUndo = hasMore;
    lastAction = hasMore
      ? historyRef.current[historyRef.current.length - 1]
      : null;

    return action;
  };

  const clear = (): void => {
    historyRef.current = [];
    canUndo = false;
    lastAction = null;
  };

  return {
    push,
    undo,
    get canUndo() {
      return canUndo;
    },
    get lastAction() {
      return lastAction;
    },
    clear,
  };
}
