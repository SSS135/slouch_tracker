import { onMount } from 'svelte';
import { nativeClient, type NativeClient } from '../lib/native/client';

interface GlobalShortcutHandlers {
  onCaptureGood: () => void;
  onCaptureBad: () => void;
  onCaptureAway: () => void;
}
export interface GlobalShortcutState {
  readonly registered: boolean;
  readonly error: Error | null;
}

let audioContext: AudioContext | null = null;

function playSound(): void {
  try {
    audioContext ??= new AudioContext();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.frequency.value = 880;
    oscillator.type = 'sine';
    gain.gain.setValueAtTime(0.3, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
  } catch {
    // Shortcut capture must continue when confirmation audio is unavailable.
  }
}

export function useGlobalShortcuts(
  handlers: GlobalShortcutHandlers,
  client: NativeClient = nativeClient,
): GlobalShortcutState {
  let currentHandlers = handlers;
  let registered = $state(false);
  let error = $state<Error | null>(null);

  $effect(() => { currentHandlers = handlers; });
  onMount(() => {
    let disposed = false;
    let setupFailed = false;
    let unlisten: (() => void) | undefined;
    let cleanupCalled = false;
    const cleanup = (): void => {
      if (cleanupCalled) return;
      cleanupCalled = true;
      unlisten?.();
      unlisten = undefined;
    };

    const listener = client.onShortcutCapture((label) => {
      if (disposed) return;
      let action: (() => void) | undefined;
      if (label === 'good') action = currentHandlers.onCaptureGood;
      else if (label === 'bad') action = currentHandlers.onCaptureBad;
      else if (label === 'away') action = currentHandlers.onCaptureAway;
      if (!action) return;
      playSound();
      action();
    });

    void listener.then((listenerCleanup) => {
      if (disposed || setupFailed) listenerCleanup();
      else unlisten = listenerCleanup;
    }).catch((cause: unknown) => {
      setupFailed = true;
      if (!disposed) error = cause instanceof Error ? cause : new Error(String(cause));
    });

    void client.getShortcutStatus().then(async (status) => {
      await listener;
      if (!disposed && !setupFailed) {
        registered = status.registered;
        error = null;
      }
    }).catch((cause: unknown) => {
      setupFailed = true;
      void listener.then(() => cleanup()).catch(() => undefined);
      if (!disposed) {
        registered = false;
        error = cause instanceof Error ? cause : new Error(String(cause));
      }
    });

    return () => {
      disposed = true;
      registered = false;
      cleanup();
      void listener.then((listenerCleanup) => {
        if (!cleanupCalled) listenerCleanup();
      }).catch(() => undefined);
    };
  });

  return {
    get registered() { return registered; },
    get error() { return error; },
  };
}
