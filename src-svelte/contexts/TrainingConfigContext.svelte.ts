import { getContext, setContext } from 'svelte';
import type {
  ClassifierConfig,
  ClassifierMetadata_Serialize,
  DimensionalityReductionConfig,
  FeatureId,
  FeatureMetadata_Serialize,
  NormalizationMode,
  ParameterDefinition_Serialize,
  ParameterValue,
  TrainingSettings_Deserialize,
  TrainingSettings_Serialize,
} from '@generated/bindings';
import { nativeClient } from '../lib/native/client';
import { canonicalizeFeatureIds } from '../services/dataset/featureOrder';
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

// Default posture features: [nlf_backbone_max] — the max-pooled NLF-L backbone embedding
// (512 dims), pinned as the app default for the backbone-embedding pipeline; the user can
// change it in the Training tab. Classifier default stays the MLP logistic head
// (hiddenLayers 0) with class weighting + PCA + z-score from in-app benchmarking.
// classifierConfig.params holds only the deviations from the registry defaults;
// applySettings overlays them onto the full registry parameter set.
export const DEFAULT_CONFIG: TrainingConfig = {
  classifierConfig: { classifierId: 'mlp', params: { hiddenLayers: 0, useClassWeights: true } },
  dimReductionConfig: { method: 'pca', components: 30 },
  postureFeatureTypes: ['nlf_backbone_max'],
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

// A registry parameter is integer-valued when it is explicitly declared `integer`, or when it is a
// numeric range/number whose step is a whole number (e.g. maxIterations step 10, knn k / kmeans
// nClusters step 1). Such values must reach the native backend as integers: UI sliders can
// accumulate floating-point drift (99.999…) and the native settings/model schemas type these
// fields as unsigned integers, which reject non-integer JSON. Fractional-step params (learningRate,
// weightDecay, temperature, labelSmoothing, …) are genuine floats and are left untouched.
function isIntegerParam(definition: ParameterDefinition_Serialize | undefined): boolean {
  if (!definition) return false;
  const type: string = definition.type;
  if (type === 'integer') return true;
  if (type === 'range' || type === 'number') {
    const step = definition.step;
    return typeof step === 'number' && Number.isInteger(step) && step >= 1;
  }
  return false;
}

export function coerceParamValue(
  definition: ParameterDefinition_Serialize | undefined,
  value: ParameterValue,
): ParameterValue {
  return typeof value === 'number' && isIntegerParam(definition) ? Math.round(value) : value;
}

// Build a full parameter set from a classifier's registry defaults, coercing every integer-typed
// param so registry values (and any future fractional defaults) can never inject a float into the
// pipeline. This is the single source of truth for "params from registry" across every emission
// point (context defaults + classifier switch/reset in the selector UI).
export function defaultParams(metadata: ClassifierMetadata_Serialize | undefined): ClassifierConfig['params'] {
  return Object.fromEntries(
    Object.entries(metadata?.params ?? {}).map(([name, definition]) => [name, coerceParamValue(definition, definition.default)]),
  );
}

// Build the fresh/reset classifier config from a registry entry. The base is the full set of
// registry defaults (so every parameter is present for validation and the UI); for the app-wide
// default classifier we overlay DEFAULT_CONFIG's benchmarked parameter overrides on top.
function defaultClassifierConfig(classifier: ClassifierMetadata_Serialize | undefined): ClassifierConfig {
  if (!classifier) return cloneDefault().classifierConfig;
  const params = classifier.id === DEFAULT_CONFIG.classifierConfig.classifierId
    ? { ...defaultParams(classifier), ...DEFAULT_CONFIG.classifierConfig.params }
    : defaultParams(classifier);
  return { classifierId: classifier.id, params };
}

// Canonicalize both feature lists at the native boundary so the payload always satisfies the
// backend's unique-and-ascending contract, no matter how the config was assembled (click order,
// legacy stored settings, or an out-of-order reconcile).
function serialize(
  config: TrainingConfig,
  registryOrder: readonly FeatureId[],
): TrainingSettings_Deserialize {
  return {
    ...config,
    postureFeatureTypes: canonicalizeFeatureIds(config.postureFeatureTypes, registryOrder),
    presenceFeatureTypes: canonicalizeFeatureIds(config.presenceFeatureTypes, registryOrder),
    featureTypes: null,
    lastUpdated: Date.now(),
  };
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
  const registryOrder = (): FeatureId[] => features.map((entry) => entry.id);

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
        : defaultClassifierConfig(classifier),
      dimReductionConfig: saved.dimReductionConfig,
      postureFeatureTypes: canonicalizeFeatureIds(saved.postureFeatureTypes, registryOrder()),
      presenceFeatureTypes: canonicalizeFeatureIds(saved.presenceFeatureTypes, registryOrder()),
      normalizationMode: saved.normalizationMode ?? DEFAULT_CONFIG.normalizationMode,
      cvFolds: saved.cvFolds,
    } : {
      ...cloneDefault(),
      classifierConfig: defaultClassifierConfig(classifier),
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
      void nativeClient.saveTrainingSettings(serialize(config, registryOrder())).catch((cause: unknown) => {
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
      void nativeClient.saveTrainingSettings(serialize(config, registryOrder())).catch((cause: unknown) => {
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
    updatePostureFeatureTypes: (postureFeatureTypes) => { if (ready) { config = { ...config, postureFeatureTypes: canonicalizeFeatureIds(postureFeatureTypes, registryOrder()) }; userRevision += 1; } },
    updatePresenceFeatureTypes: (presenceFeatureTypes) => { if (ready) { config = { ...config, presenceFeatureTypes: canonicalizeFeatureIds(presenceFeatureTypes, registryOrder()) }; userRevision += 1; } },
    updateNormalizationMode: (normalizationMode) => { if (ready) { config = { ...config, normalizationMode }; userRevision += 1; } },
    updateCvFolds: (cvFolds) => { if (ready) { config = { ...config, cvFolds }; userRevision += 1; } },
    async flushToStorage() {
      if (!ready) throw new Error(error ? `Training settings are unavailable: ${error}` : 'Training settings are still loading.');
      if (debounce) clearTimeout(debounce);
      debounce = null;
      await nativeClient.saveTrainingSettings(serialize(config, registryOrder()));
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
