import { KMeansPrototypeClassifier, type KMeansPrototypeParams } from '../../src/services/ml/kmeansPrototypeClassifier';
import { kmeans } from '../../src/services/ml/kmeans';
import type { SerializedKMeansPrototype } from '../../src/services/ml/types';
import { encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/classifiers/kmeans-prototype-v1.json';
const generator = 'scripts/oracles/generate-kmeans-prototype.ts';
const mixedRows = [[-3, -2], [-2, -3], [-1, -1], [-0.5, -1.5], [0.5, 1.5], [1, 1], [2, 3], [3, 2]];
const mixedLabels = [0, 1, 0, 1, 0, 1, 0, 1];
const separatedLabels = [0, 0, 0, 0, 1, 1, 1, 1];
const probes = [[-2, -2], [0, 0], [2, 2]];
const vectors = (rows: number[][]) => rows.map((row) => new Float32Array(row));
const routingWeights = (centroids: number[][], probe: number[], temperature: number) => {
  const logits = centroids.map((centroid) => -Math.sqrt(centroid.reduce((sum, value, index) => sum + (probe[index] - value) ** 2, 0)) / temperature);
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
};

function trainObserved(rows: number[][], labels: number[], params: KMeansPrototypeParams): unknown {
  const classifier = new KMeansPrototypeClassifier({ params });
  try {
    classifier.train(vectors(rows), new Int32Array(labels));
    return { state: classifier.toJSON(), probabilities: probes.map((probe) => classifier.predictProba(new Float32Array(probe))) };
  } finally {
    classifier.dispose();
  }
}

function loadObserved(state: SerializedKMeansPrototype, params: KMeansPrototypeParams): unknown {
  const classifier = KMeansPrototypeClassifier.fromJSON(state, params);
  try {
    return { state: classifier.toJSON(), probabilities: probes.map((probe) => classifier.predictProba(new Float32Array(probe))) };
  } finally {
    classifier.dispose();
  }
}

export function generateKmeansPrototype(write: boolean): void {
  const definitions: Array<{ id: string; rows: number[][]; labels: number[]; params: KMeansPrototypeParams }> = [
    { id: 'manual-mixed-temp1', rows: mixedRows, labels: mixedLabels, params: { nClusters: 2, temperature: 1 } },
    { id: 'manual-separated-temp-half', rows: mixedRows, labels: separatedLabels, params: { nClusters: 2, temperature: 0.5 } },
    { id: 'manual-separated-temp-two', rows: mixedRows, labels: separatedLabels, params: { nClusters: 2, temperature: 2 } },
    { id: 'auto-candidates', rows: mixedRows, labels: separatedLabels, params: { nClusters: 0, temperature: 1 } },
  ];
  const cases = definitions.map(({ id, rows, labels, params }) => {
    const classifier = new KMeansPrototypeClassifier({ params });
    classifier.train(vectors(rows), new Int32Array(labels));
    const state = classifier.toJSON();
    const selectedK = state.clusters.length;
    const clustering = kmeans(vectors(rows), selectedK);
    const probabilities = probes.map((probe) => classifier.predictProba(new Float32Array(probe)));
    const loaded = KMeansPrototypeClassifier.fromJSON(state, params);
    const loadedProbabilities = probes.map((probe) => loaded.predictProba(new Float32Array(probe)));
    classifier.dispose(); loaded.dispose();
    return {
      id, params, rows, labels, probes, state,
      trainingTrace: {
        selectedK,
        assignments: clustering.assignments,
        centroids: clustering.centroids.map((centroid) => Array.from(centroid)),
        prototypeMasks: state.clusters.map((cluster) => ({ good: cluster.prototypeGood !== null, bad: cluster.prototypeBad !== null })),
      },
      routingWeights: probes.map((probe) => routingWeights(state.clusters.map((cluster) => cluster.centroid), probe, state.temperature)),
      probabilities, loadedProbabilities,
    };
  });
  const base = cases[0].state;
  const baseParams = definitions[0].params;
  const invalidCases = [
    { id: 'missing-class', operation: 'train', rows: [[1], [2]], labels: [0, 0], params: baseParams, nativeError: 'MissingClass', typescript: observe(() => trainObserved([[1], [2]], [0, 0], baseParams)) },
    { id: 'empty', operation: 'train', rows: [], labels: [], params: baseParams, nativeError: 'EmptyDataset', typescript: observe(() => trainObserved([], [], baseParams)) },
    { id: 'mismatch', operation: 'train', rows: [[1]], labels: [], params: baseParams, nativeError: 'LengthMismatch', typescript: observe(() => trainObserved([[1]], [], baseParams)) },
    { id: 'ragged', operation: 'train', rows: [[1, 2], [3]], labels: [0, 1], params: baseParams, nativeError: 'RaggedFeatures', typescript: observe(() => trainObserved([[1, 2], [3]], [0, 1], baseParams)) },
    { id: 'nonfinite-feature', operation: 'train', rows: [['NaN'], [1]], labels: [0, 1], params: baseParams, nativeError: 'NonFiniteFeature', typescript: observe(() => trainObserved([[Number.NaN], [1]], [0, 1], baseParams)) },
    { id: 'nonpositive-temperature', operation: 'construct', params: { ...baseParams, temperature: 0 }, nativeError: 'InvalidState', typescript: observe(() => new KMeansPrototypeClassifier({ params: { ...baseParams, temperature: 0 } }).classifierId) },
    { id: 'empty-clusters', operation: 'load', params: baseParams, state: { ...base, clusters: [] }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, clusters: [] }, baseParams)) },
    { id: 'malformed-state', operation: 'load', params: baseParams, state: { ...base, globalPrototypeBad: [] }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, globalPrototypeBad: [] }, baseParams)) },
    { id: 'nonfinite-state', operation: 'load', params: baseParams, state: encodeObserved({ ...base, globalPrototypeGood: [Number.NaN, ...base.globalPrototypeGood.slice(1)] }), nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, globalPrototypeGood: [Number.NaN, ...base.globalPrototypeGood.slice(1)] }, baseParams)) },
    { id: 'malformed-cluster-prototype', operation: 'load', params: baseParams, state: { ...base, clusters: base.clusters.map((cluster, index) => index === 0 ? { ...cluster, prototypeGood: [1] } : cluster) }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, clusters: base.clusters.map((cluster, index) => index === 0 ? { ...cluster, prototypeGood: [1] } : cluster) }, baseParams)) },
  ];
  writeOrCheck(path, envelope('classifiers/kmeans-prototype-v1', generator, [
    'src/services/ml/kmeansPrototypeClassifier.ts', 'src/services/ml/kmeans.ts',
  ], 'pure TypeScript', cases, { invalidCases }), write);
}

if (isMain(import.meta.url)) generateKmeansPrototype(parseWriteFlag());
