import { getContext, setContext } from 'svelte';
import type {
  AppStatus,
  ClassifierMetadata_Serialize,
  FeatureMetadata_Serialize,
  ShortcutStatus,
} from '@generated/bindings';
import { nativeClient, type NativeClient } from '../native/client';

export interface NativeAppState {
  readonly status: AppStatus | null;
  readonly classifiers: ClassifierMetadata_Serialize[];
  readonly features: FeatureMetadata_Serialize[];
  readonly shortcutStatus: ShortcutStatus | null;
  readonly loading: boolean;
  readonly error: Error | null;
  initialize(): Promise<void>;
  reconcile(status: AppStatus): void;
  refresh(): Promise<void>;
}

const NATIVE_APP_CONTEXT = Symbol('native-app-context');

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

const initializationPromises = new WeakMap<object, Promise<void>>();

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/** Single native-inference initializer shared by providers and camera adapters. */
export async function initializeNativeInference(client: NativeClient = nativeClient): Promise<void> {
  const key = client as object;
  const existing = initializationPromises.get(key);
  if (existing) return existing;

  const promise = (async () => {
    const status = await client.appStatus();
    if (status.inferenceReady) return;
    try {
      await client.initializeInference();
    } catch (cause) {
      const message = asError(cause).message.toLowerCase();
      if (!message.includes('busy') && !message.includes('already initializing')) throw cause;
      for (let attempt = 0; attempt < 600; attempt += 1) {
        await delay(25);
        if ((await client.appStatus()).inferenceReady) return;
      }
      throw cause;
    }
  })();

  initializationPromises.set(key, promise);
  try {
    await promise;
  } finally {
    if (initializationPromises.get(key) === promise) initializationPromises.delete(key);
  }
}

export function createNativeAppState(client: NativeClient = nativeClient): NativeAppState {
  let status = $state<AppStatus | null>(null);
  let classifiers = $state<ClassifierMetadata_Serialize[]>([]);
  let features = $state<FeatureMetadata_Serialize[]>([]);
  let shortcutStatus = $state<ShortcutStatus | null>(null);
  let loading = $state(false);
  let error = $state<Error | null>(null);

  async function loadMetadata(): Promise<void> {
    const [nextStatus, nextClassifiers, nextFeatures, nextShortcutStatus] = await Promise.all([
      client.appStatus(),
      client.getClassifierRegistry(),
      client.getFeatureRegistry(),
      client.getShortcutStatus(),
    ]);
    status = nextStatus;
    classifiers = nextClassifiers;
    features = nextFeatures;
    shortcutStatus = nextShortcutStatus;
  }

  async function run(operation: () => Promise<void>): Promise<void> {
    loading = true;
    error = null;
    try {
      await operation();
    } catch (cause) {
      error = asError(cause);
      throw cause;
    } finally {
      loading = false;
    }
  }

  return {
    get status() {
      return status;
    },
    get classifiers() {
      return classifiers;
    },
    get features() {
      return features;
    },
    get shortcutStatus() {
      return shortcutStatus;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    async initialize() {
      await run(async () => {
        await initializeNativeInference(client);
        await loadMetadata();
      });
    },
    reconcile(nextStatus) {
      status = nextStatus;
      error = null;
      loading = false;
    },
    async refresh() {
      await run(loadMetadata);
    },
  };
}

export function provideNativeAppState(client: NativeClient = nativeClient): NativeAppState {
  const state = createNativeAppState(client);
  setContext(NATIVE_APP_CONTEXT, state);
  return state;
}

export function useNativeAppState(): NativeAppState {
  const state = getContext<NativeAppState | undefined>(NATIVE_APP_CONTEXT);
  if (!state) {
    throw new Error('useNativeAppState must be used beneath AppProviders.');
  }
  return state;
}
