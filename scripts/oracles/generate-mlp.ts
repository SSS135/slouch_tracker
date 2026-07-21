import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import { MLPClassifier, type MLPParams } from '../../src/services/ml/mlpClassifier';
import type { SerializedMLP } from '../../src/services/ml/types';
import { encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/classifiers/mlp-v1.json';
const generator = 'scripts/oracles/generate-mlp.ts';
const rows = [[-2, -0.7], [-1.4, -0.4], [-0.8, -1.1], [0.7, 1.2], [1.6, 0.4], [2.2, 1.7]];
const labels = [0, 0, 0, 1, 1, 1];
const probes = [[-1.25, -0.75], [0, 0], [1.25, 0.75]];

function trainObserved(dataRows: number[][], dataLabels: number[], params: MLPParams, dataProbes = probes): unknown {
  const classifier = new MLPClassifier({ params });
  try {
    classifier.train(dataRows.map((row) => new Float32Array(row)), new Int32Array(dataLabels));
    return {
      state: classifier.toJSON(),
      probabilities: dataProbes.map((probe) => classifier.predictProba(new Float32Array(probe))),
    };
  } finally {
    classifier.dispose();
  }
}

function loadObserved(state: SerializedMLP, params: MLPParams, probe = [1, 2]): unknown {
  const classifier = MLPClassifier.fromJSON(state, params);
  try {
    return { state: classifier.toJSON(), probability: classifier.predictProba(new Float32Array(probe)) };
  } finally {
    classifier.dispose();
  }
}

export async function generateMlp(write: boolean): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
  const common = { hiddenSize: 4, learningRate: 0.01, weightDecay: 0.01, labelSmoothing: 0 };
  const definitions: Array<{ id: string; params: MLPParams; rows?: number[][]; labels?: number[]; probes?: number[][] }> = [
    { id: 'zero-hidden-iter0', params: { ...common, hiddenLayers: 0, maxIterations: 0 } },
    { id: 'zero-hidden-iter1', params: { ...common, hiddenLayers: 0, maxIterations: 1 } },
    { id: 'zero-hidden-iter2', params: { ...common, hiddenLayers: 0, maxIterations: 2, labelSmoothing: 0.1 } },
    { id: 'zero-hidden-iter25', params: { ...common, hiddenLayers: 0, maxIterations: 25, labelSmoothing: 0.2 } },
    { id: 'one-hidden-iter2', params: { ...common, hiddenLayers: 1, maxIterations: 2, labelSmoothing: 0.1 } },
    { id: 'two-hidden-iter2', params: { ...common, hiddenLayers: 2, hiddenSize: 8, maxIterations: 2, labelSmoothing: 0.1 } },
    { id: 'weighted-imbalanced', params: { ...common, hiddenLayers: 0, maxIterations: 2, labelSmoothing: 0.1, useClassWeights: true }, rows: rows.slice(0, 4), labels: [0, 0, 0, 1] },
    {
      id: 'large-small-cancellation-iter2',
      params: { ...common, hiddenLayers: 0, maxIterations: 2, labelSmoothing: 0.1 },
      rows: [[16777216, 1, -16777216], [-16777216, -1, 16777216], [8388608, 0.00000011920928955078125, -8388608], [-8388608, -0.00000011920928955078125, 8388608]],
      labels: [0, 1, 0, 1],
      probes: [[16777216, 1, -16777216], [0, 0, 0], [-16777216, -1, 16777216]],
    },
  ];
  const cases = definitions.map(({ id, params, rows: customRows, labels: customLabels, probes: customProbes }) => {
    const dataRows = customRows ?? rows;
    const dataLabels = customLabels ?? labels;
    const dataProbes = customProbes ?? probes;
    const classifier = new MLPClassifier({ params });
    classifier.train(dataRows.map((row) => new Float32Array(row)), new Int32Array(dataLabels));
    const state = classifier.toJSON();
    const probabilities = dataProbes.map((probe) => classifier.predictProba(new Float32Array(probe)));
    const loaded = MLPClassifier.fromJSON(state, params);
    const loadedProbabilities = dataProbes.map((probe) => loaded.predictProba(new Float32Array(probe)));
    classifier.dispose();
    loaded.dispose();
    return { id, params, rows: dataRows, labels: dataLabels, probes: dataProbes, state, probabilities, loadedProbabilities };
  });

  const base = cases[0].state;
  const invalidCases = [
    { id: 'empty', operation: 'train', rows: [], labels: [], params: common, nativeError: 'EmptyDataset', typescript: observe(() => trainObserved([], [], common)) },
    { id: 'mismatch', operation: 'train', rows: [[1, 2]], labels: [], params: common, nativeError: 'LengthMismatch', typescript: observe(() => trainObserved([[1, 2]], [], common)) },
    { id: 'ragged', operation: 'train', rows: [[1, 2], [3]], labels: [0, 1], params: common, nativeError: 'RaggedFeatures', typescript: observe(() => trainObserved([[1, 2], [3]], [0, 1], common)) },
    { id: 'nan', operation: 'train', rows: [['NaN'], [1]], labels: [0, 1], params: common, nativeError: 'NonFiniteFeature', typescript: observe(() => trainObserved([[Number.NaN], [1]], [0, 1], common)) },
    { id: 'invalid-hidden-layers', operation: 'construct', params: { ...common, hiddenLayers: 3 }, nativeError: 'InvalidState', typescript: observe(() => new MLPClassifier({ params: { ...common, hiddenLayers: 3 } }).classifierId) },
    {
      id: 'malformed-state', operation: 'load', params: common,
      state: { ...base, layerWeights: [[]] }, nativeError: 'InvalidState',
      typescript: observe(() => loadObserved({ ...base, layerWeights: [[]] }, common)),
    },
    {
      id: 'nonfinite-state', operation: 'load', params: common,
      state: encodeObserved({ ...base, layerWeights: [[Number.NaN, ...base.layerWeights[0].slice(1)]] }), nativeError: 'InvalidState',
      typescript: observe(() => loadObserved({ ...base, layerWeights: [[Number.NaN, ...base.layerWeights[0].slice(1)]] }, common)),
    },
    {
      id: 'above-f32-max-state', operation: 'load', params: common,
      state: { ...base, layerWeights: [[3.5e38, ...base.layerWeights[0].slice(1)]] }, nativeError: 'InvalidState',
      typescript: observe(() => loadObserved({ ...base, layerWeights: [[3.5e38, ...base.layerWeights[0].slice(1)]] }, common)),
    },
    {
      id: 'nonfinite-prediction', operation: 'predict', params: common,
      state: base, probe: ['+Infinity', 0], nativeError: 'NonFiniteFeature',
      typescript: observe(() => loadObserved(base, common, [Number.POSITIVE_INFINITY, 0])),
    },
  ];

  writeOrCheck(path, envelope('classifiers/mlp-v1', generator, [
    'src/services/ml/mlpClassifier.ts', 'src/services/ml/adamw.ts', 'src/services/ml/config.ts',
    'src/services/ml/utils/classWeights.ts', 'src/services/ml/__tests__/mlpClassifier.test.ts',
  ], `TensorFlow.js ${tf.version.tfjs} CPU`, cases, { invalidCases }), write);
}

if (isMain(import.meta.url)) await generateMlp(parseWriteFlag());
