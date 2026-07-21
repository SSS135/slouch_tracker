import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import { KMeansLogisticClassifier, type KMeansLogisticParams } from '../../src/services/ml/kmeansLogisticClassifier';
import { kmeans } from '../../src/services/ml/kmeans';
import type { SerializedKMeansLogistic } from '../../src/services/ml/types';
import { encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/classifiers/kmeans-logistic-v1.json';
const generator = 'scripts/oracles/generate-kmeans-logistic.ts';
const mixedRows = [[-3, -2], [-2, -3], [-1, -1], [-0.5, -1.5], [0.5, 1.5], [1, 1], [2, 3], [3, 2]];
const mixedLabels = [0, 1, 0, 1, 0, 1, 0, 1];
const separatedRows = [[-3, -2], [-2.2, -3], [-1, -0.8], [0.35, 0.8], [0.8, 1.7], [1.2, 1], [2.4, 3.2], [3.3, 2.1]];
const separatedLabels = [0, 0, 0, 1, 1, 1, 1, 1];
const probes = [[-2, -2], [0, 0], [2, 2]];
const vectors = (rows: number[][]) => rows.map((row) => new Float32Array(row));
const routingWeights = (centroids: number[][], probe: number[], temperature: number) => {
  const logits = centroids.map((centroid) => -Math.sqrt(centroid.reduce((sum, value, index) => sum + (probe[index] - value) ** 2, 0)) / temperature);
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
};

function trainObserved(rows: number[][], labels: number[], params: KMeansLogisticParams): unknown {
  const classifier = new KMeansLogisticClassifier({ params });
  try {
    classifier.train(vectors(rows), new Int32Array(labels));
    return { state: classifier.toJSON(), probabilities: probes.map((probe) => classifier.predictProba(new Float32Array(probe))) };
  } finally {
    classifier.dispose();
  }
}

function loadObserved(state: SerializedKMeansLogistic, params: KMeansLogisticParams): unknown {
  const classifier = KMeansLogisticClassifier.fromJSON(state, params);
  try {
    return { state: classifier.toJSON(), probabilities: probes.map((probe) => classifier.predictProba(new Float32Array(probe))) };
  } finally {
    classifier.dispose();
  }
}

export async function generateKmeansLogistic(write: boolean): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
  const definitions: Array<{ id: string; rows: number[][]; labels: number[]; params: KMeansLogisticParams }> = [
    { id: 'manual-mixed-temp1', rows: mixedRows, labels: mixedLabels, params: { nClusters: 2, temperature: 1, maxIterations: 2, weightDecay: 1 } },
    { id: 'manual-separated-temp-half', rows: separatedRows, labels: separatedLabels, params: { nClusters: 2, temperature: 0.5, maxIterations: 2, weightDecay: 1 } },
    { id: 'manual-separated-temp-two', rows: separatedRows, labels: separatedLabels, params: { nClusters: 2, temperature: 2, maxIterations: 2, weightDecay: 1 } },
    { id: 'two-sample-auto-k1', rows: [[-1, -0.8], [1.3, 1]], labels: [0, 1], params: { nClusters: 0, temperature: 1, maxIterations: 2, weightDecay: 1 } },
    {
      id: 'auto-selection-normal-range',
      rows: Array.from({ length: 30 }, (_, index) => [index % 3 * 5 + (index % 5) * 0.05, Math.floor(index / 3) * 0.03]),
      labels: Array.from({ length: 30 }, (_, index) => index % 2),
      params: { nClusters: 0, temperature: 1, maxIterations: 1, weightDecay: 1 },
    },
  ];
  const cases = definitions.map(({ id, rows, labels, params }) => {
    const classifier = new KMeansLogisticClassifier({ params });
    classifier.train(vectors(rows), new Int32Array(labels));
    const state = classifier.toJSON();
    const selectedK = state.centroids.length;
    const clustering = kmeans(vectors(rows), selectedK);
    const probabilities = probes.map((probe) => classifier.predictProba(new Float32Array(probe)));
    const loaded = KMeansLogisticClassifier.fromJSON(state, params);
    const loadedProbabilities = probes.map((probe) => loaded.predictProba(new Float32Array(probe)));
    classifier.dispose(); loaded.dispose();
    return {
      id, params, rows, labels, probes, state,
      trainingTrace: { selectedK, assignments: clustering.assignments, centroids: clustering.centroids.map((centroid) => Array.from(centroid)), clusterModelMask: state.clusterModels.map(Boolean) },
      routingWeights: probes.map((probe) => routingWeights(state.centroids, probe, state.temperature)),
      probabilities, loadedProbabilities,
    };
  });
  const base = cases[0].state;
  const baseParams = definitions[0].params;
  const legacyParams: KMeansLogisticParams = { temperature: 1, maxIterations: 2, nClusters: 2 };
  const legacy = KMeansLogisticClassifier.fromJSON(base, legacyParams);
  const legacyCases = [{
    id: 'legacy-load-absent-weight-decay',
    state: base,
    params: legacyParams,
    expectedWeightDecayDefault: 1,
    probes,
    loadedProbabilities: probes.map((probe) => legacy.predictProba(new Float32Array(probe))),
  }];
  legacy.dispose();
  const invalidCases = [
    { id: 'empty', operation: 'train', rows: [], labels: [], params: baseParams, nativeError: 'EmptyDataset', typescript: observe(() => trainObserved([], [], baseParams)) },
    { id: 'mismatch', operation: 'train', rows: [[1, 2]], labels: [], params: baseParams, nativeError: 'LengthMismatch', typescript: observe(() => trainObserved([[1, 2]], [], baseParams)) },
    { id: 'ragged', operation: 'train', rows: [[1, 2], [3]], labels: [0, 1], params: baseParams, nativeError: 'RaggedFeatures', typescript: observe(() => trainObserved([[1, 2], [3]], [0, 1], baseParams)) },
    { id: 'nonfinite-feature', operation: 'train', rows: [['NaN', 2], [3, 4]], labels: [0, 1], params: baseParams, nativeError: 'NonFiniteFeature', typescript: observe(() => trainObserved([[Number.NaN, 2], [3, 4]], [0, 1], baseParams)) },
    { id: 'nonpositive-temperature', operation: 'construct', params: { ...baseParams, temperature: 0 }, nativeError: 'InvalidState', typescript: observe(() => new KMeansLogisticClassifier({ params: { ...baseParams, temperature: 0 } }).classifierId) },
    { id: 'malformed-mask', operation: 'load', params: baseParams, state: { ...base, clusterModels: base.clusterModels.slice(1) }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, clusterModels: base.clusterModels.slice(1) }, baseParams)) },
    { id: 'empty-centroids', operation: 'load', params: baseParams, state: { ...base, centroids: [], clusterModels: [] }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, centroids: [], clusterModels: [] }, baseParams)) },
    { id: 'nonfinite-centroid', operation: 'load', params: baseParams, state: encodeObserved({ ...base, centroids: [[Number.NaN, ...base.centroids[0].slice(1)], ...base.centroids.slice(1)] }), nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, centroids: [[Number.NaN, ...base.centroids[0].slice(1)], ...base.centroids.slice(1)] }, baseParams)) },
    { id: 'nonfinite-global-mlp', operation: 'load', params: baseParams, state: encodeObserved({ ...base, globalModel: { ...base.globalModel, layerWeights: [[Number.NaN, ...base.globalModel.layerWeights[0].slice(1)]] } }), nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, globalModel: { ...base.globalModel, layerWeights: [[Number.NaN, ...base.globalModel.layerWeights[0].slice(1)]] } }, baseParams)) },
  ];
  writeOrCheck(path, envelope('classifiers/kmeans-logistic-v1', generator, [
    'src/services/ml/kmeansLogisticClassifier.ts', 'src/services/ml/kmeans.ts', 'src/services/ml/mlpClassifier.ts',
    'src/services/ml/__tests__/kmeansLogisticClassifier.test.ts',
  ], `TensorFlow.js ${tf.version.tfjs} CPU + pure TypeScript K-Means`, cases, {
    legacyCases,
    invalidCases,
  }), write);
}

if (isMain(import.meta.url)) await generateKmeansLogistic(parseWriteFlag());
