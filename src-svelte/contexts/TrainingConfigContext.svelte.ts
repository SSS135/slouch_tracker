import { getContext, setContext } from 'svelte';
import type {
  ClassifierConfig,
  ClassifierMetadata_Serialize,
  DimensionalityReductionConfig,
  FeatureId,
  FeatureMetadata_Serialize,
  NormalizationMode,
  TrainingSettings_Deserialize,
  TrainingSettings_Serialize,
} from '@generated/bindings';
import { nativeClient } from '../lib/native/client';
import { logger } from '../services/logging/logger';

export interface TrainingConfig {
  classifierConfig: ClassifierConfig;
  dimReductionConfig: DimensionalityReductionConfig;
  postureFeatureTypes: FeatureId[];
  presenceFeatureTypes: FeatureId[];
  normalizationMode: NormalizationMode;
  cvFolds: number;
}
export interface TrainingConfigContextValue {
  readonly config: TrainingConfig;
  readonly classifiers: ClassifierMetadata_Serialize[];
  readonly features: FeatureMetadata_Serialize[];
  readonly ready: boolean;
  readonly loading: boolean;
  readonly error: string | null;
  updateClassifierConfig(config: ClassifierConfig): void;
  updateDimReductionConfig(config: DimensionalityReductionConfig): void;
  updatePostureFeatureTypes(featureTypes: FeatureId[]): void;
  updatePresenceFeatureTypes(featureTypes: FeatureId[]): void;
  updateNormalizationMode(mode: NormalizationMode): void;
  updateCvFolds(folds: number): void;
  flushToStorage(): Promise<void>;
  reconcile(settings: TrainingSettings_Serialize | null): void;
  reload(): Promise<void>;
}

export const DEFAULT_CONFIG: TrainingConfig = {
  classifierConfig: { classifierId: 'mlp', params: {} },
  dimReductionConfig: { method: 'none', components: 64 },
  postureFeatureTypes: ['engineered_features'],
  presenceFeatureTypes: ['rtmdet_engineered', 'keypoint_scores'],
  normalizationMode: 'z_score',
  cvFolds: 5,
};

const CONTEXT = Symbol('training-config-context');
const cloneDefault = (): TrainingConfig => ({
  ...DEFAULT_CONFIG,
  classifierConfig: { ...DEFAULT_CONFIG.classifierConfig, params: { ...DEFAULT_CONFIG.classifierConfig.params } },
  dimReductionConfig: { ...DEFAULT_CONFIG.dimReductionConfig },
  postureFeatureTypes: [...DEFAULT_CONFIG.postureFeatureTypes],
  presenceFeatureTypes: [...DEFAULT_CONFIG.presenceFeatureTypes],
});

function defaultParams(metadata: ClassifierMetadata_Serialize | undefined): ClassifierConfig['params'] {
  return Object.fromEntries(Object.entries(metadata?.params ?? {}).map(([name, definition]) => [name, definition.default]));
}

function serialize(config: TrainingConfig): TrainingSettings_Deserialize {
  return { ...config, featureTypes: null, lastUpdated: Date.now() };
}

export function createTrainingConfigContext(): TrainingConfigContextValue {
  let config = $state<TrainingConfig>(cloneDefault());
  let classifiers = $state<ClassifierMetadata_Serialize[]>([]);
  let features = $state<FeatureMetadata_Serialize[]>([]);
  let ready = $state(false);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let debounce: ReturnType<typeof setTimeout> | null = null;
  let userRevision = $state(0);
  let classifierRegistryLoaded = false;
  let featureRegistryLoaded = false;
  let loadGeneration = 0;

  const registriesLoaded = (): boolean => classifierRegistryLoaded && featureRegistryLoaded;

  const applySettings = (saved: TrainingSettings_Serialize | null): void => {
    const savedClassifier = classifiers.find((entry) => entry.id === saved?.classifierConfig.classifierId);
    const defaultClassifier = classifiers.find((entry) => entry.id === DEFAULT_CONFIG.classifierConfig.classifierId);
    const classifier = savedClassifier ?? defaultClassifier;

    if (saved && !savedClassifier) {
      logger.warn(
        'training',
        `Invalid classifier "${saved.classifierConfig.classifierId}" in saved settings, resetting to default`,
      );
    }

    config = saved ? {
      classifierConfig: savedClassifier
        ? { classifierId: savedClassifier.id, params: saved.classifierConfig.params }
        : classifier
          ? { classifierId: classifier.id, params: defaultParams(classifier) }
          : cloneDefault().classifierConfig,
      dimReductionConfig: saved.dimReductionConfig,
      postureFeatureTypes: saved.postureFeatureTypes,
      presenceFeatureTypes: saved.presenceFeatureTypes,
      normalizationMode: saved.normalizationMode ?? DEFAULT_CONFIG.normalizationMode,
      cvFolds: saved.cvFolds,
    } : {
      ...cloneDefault(),
      classifierConfig: classifier
        ? { classifierId: classifier.id, params: defaultParams(classifier) }
        : cloneDefault().classifierConfig,
    };
  };

  const load = async (): Promise<void> => {
    const generation = ++loadGeneration;
    loading = true;
    ready = false;
    error = null;
    if (debounce) clearTimeout(debounce);
    debounce = null;
    try {
      const [saved, registry, featureRegistry] = await Promise.all([
        nativeClient.getTrainingSettings(),
        nativeClient.getClassifierRegistry(),
        nativeClient.getFeatureRegistry(),
      ]);
      if (generation !== loadGeneration) return;
      classifiers = registry;
      features = featureRegistry;
      classifierRegistryLoaded = true;
      featureRegistryLoaded = true;
      applySettings(saved);
      userRevision = 0;
      error = null;
      ready = true;
    } catch (cause) {
      if (generation !== loadGeneration) return;
      error = cause instanceof Error ? cause.message : String(cause);
      logger.error('training', 'Failed to load native training settings:', cause);
      throw cause;
    } finally {
      if (generation === loadGeneration) loading = false;
    }
  };

  $effect(() => { void load().catch(() => undefined); });
  $effect(() => {
    config;
    userRevision;
    if (!ready || userRevision === 0) return;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => {
      debounce = null;
      void nativeClient.saveTrainingSettings(serialize(config)).catch((cause: unknown) => {
        logger.error('training', 'Failed to persist native training settings:', cause);
      });
    }, 300);
    return () => { if (debounce) clearTimeout(debounce); };
  });

  $effect(() => {
    const handleBeforeUnload = (): void => {
      if (debounce) clearTimeout(debounce);
      debounce = null;
      if (!ready || userRevision === 0) return;
      void nativeClient.saveTrainingSettings(serialize(config)).catch((cause: unknown) => {
        logger.error('training', 'Failed to flush native training settings on unload:', cause);
      });
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  });

  const value: TrainingConfigContextValue = {
    get config() { return config; },
    get classifiers() { return classifiers; },
    get features() { return features; },
    get ready() { return ready; },
    get loading() { return loading; },
    get error() { return error; },
    updateClassifierConfig: (classifierConfig) => { if (ready) { config = { ...config, classifierConfig }; userRevision += 1; } },
    updateDimReductionConfig: (dimReductionConfig) => { if (ready) { config = { ...config, dimReductionConfig }; userRevision += 1; } },
    updatePostureFeatureTypes: (postureFeatureTypes) => { if (ready) { config = { ...config, postureFeatureTypes }; userRevision += 1; } },
    updatePresenceFeatureTypes: (presenceFeatureTypes) => { if (ready) { config = { ...config, presenceFeatureTypes }; userRevision += 1; } },
    updateNormalizationMode: (normalizationMode) => { if (ready) { config = { ...config, normalizationMode }; userRevision += 1; } },
    updateCvFolds: (cvFolds) => { if (ready) { config = { ...config, cvFolds }; userRevision += 1; } },
    async flushToStorage() {
      if (!ready) throw new Error(error ? `Training settings are unavailable: ${error}` : 'Training settings are still loading.');
      if (debounce) clearTimeout(debounce);
      debounce = null;
      await nativeClient.saveTrainingSettings(serialize(config));
    },
    reconcile(settings) {
      loadGeneration += 1;
      if (debounce) clearTimeout(debounce);
      debounce = null;
      loading = false;
      applySettings(settings);
      userRevision = 0;
      if (registriesLoaded()) {
        error = null;
        ready = true;
      }
    },
    reload: load,
  };
  setContext(CONTEXT, value);
  return value;
}

export function TrainingConfigProvider(): TrainingConfigContextValue { return createTrainingConfigContext(); }
export function useTrainingConfig(): TrainingConfigContextValue {
  const context = getContext<TrainingConfigContextValue | undefined>(CONTEXT);
  if (!context) throw new Error('useTrainingConfig must be used within TrainingConfigProvider');
  return context;
}
