import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import { SVMClassifier, type SVMParams } from '../../src/services/ml/svmClassifier';
import type { SerializedSVM } from '../../src/services/ml/types';
import { bits, encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/classifiers/svm-v1.json';
const generator = 'scripts/oracles/generate-svm.ts';
const rows = [[-2, -0.7], [-1.4, -0.4], [-0.8, -1.1], [0.7, 1.2], [1.6, 0.4], [2.2, 1.7]];
const labels = [0, 0, 0, 1, 1, 1];
const probes = [[-1.25, -0.75], [0, 0], [1.25, 0.75]];

function trainObserved(dataRows: number[][], dataLabels: number[], params: SVMParams, dataProbes = probes): unknown {
  const classifier = new SVMClassifier({ params });
  try {
    classifier.train(dataRows.map((row) => new Float32Array(row)), new Int32Array(dataLabels));
    return { state: classifier.toJSON(), probabilities: dataProbes.map((probe) => classifier.predictProba(new Float32Array(probe))) };
  } finally {
    classifier.dispose();
  }
}

function loadObserved(state: SerializedSVM, params: SVMParams, probe = [1, 2]): unknown {
  const classifier = SVMClassifier.fromJSON(state, params);
  try {
    return { state: classifier.toJSON(), probability: classifier.predictProba(new Float32Array(probe)) };
  } finally {
    classifier.dispose();
  }
}

export async function generateSvm(write: boolean): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
  const definitions: Array<{ id: string; params: SVMParams; rows?: number[][]; labels?: number[]; probes?: number[][] }> = [];
  for (const maxIterations of [0, 1, 2, 25]) definitions.push({ id: `iter${maxIterations}`, params: { C: 1, maxIterations, useClassWeights: false } });
  definitions.push({ id: 'c-point-one', params: { C: 0.1, maxIterations: 2, useClassWeights: false } });
  definitions.push({ id: 'c-ten', params: { C: 10, maxIterations: 2, useClassWeights: false } });
  definitions.push({ id: 'weighted', params: { C: 1, maxIterations: 2, useClassWeights: true } });
  definitions.push({
    id: 'large-small-cancellation', params: { C: 1, maxIterations: 2, useClassWeights: false },
    rows: [[16777216, 1, -16777216], [-16777216, -1, 16777216], [8388608, 0.00000011920928955078125, -8388608], [-8388608, -0.00000011920928955078125, 8388608]],
    labels: [0, 1, 0, 1], probes: [[16777216, 1, -16777216], [0, 0, 0], [-16777216, -1, 16777216]],
  });
  const cases = definitions.map(({ id, params, rows: customRows, labels: customLabels, probes: customProbes }) => {
    const dataRows = customRows ?? rows;
    const dataLabels = customLabels ?? labels;
    const dataProbes = customProbes ?? probes;
    const classifier = new SVMClassifier({ params });
    classifier.train(dataRows.map((row) => new Float32Array(row)), new Int32Array(dataLabels));
    const state = classifier.toJSON();
    const probabilities = dataProbes.map((probe) => classifier.predictProba(new Float32Array(probe)));
    const decisionValues = dataProbes.map((probe) => tf.tidy(() => tf.tensor2d([probe]).matMul(tf.tensor2d(state.weights, [state.weights.length, 1])).add(tf.scalar(state.bias)).dataSync()[0]));
    const loaded = SVMClassifier.fromJSON(state, params);
    const loadedProbabilities = dataProbes.map((probe) => loaded.predictProba(new Float32Array(probe)));
    classifier.dispose();
    loaded.dispose();
    return { id, params, rows: dataRows, labels: dataLabels, probes: dataProbes, state, stateF32Bits: { weights: bits(state.weights), bias: bits([state.bias])[0] }, decisionValues, probabilities, loadedProbabilities };
  });
  const base = cases[0].state;
  const defaultParams = { C: 1, maxIterations: 2, useClassWeights: false };
  const invalidCases = [
    { id: 'empty', operation: 'train', rows: [], labels: [], params: defaultParams, nativeError: 'EmptyDataset', typescript: observe(() => trainObserved([], [], defaultParams)) },
    { id: 'mismatch', operation: 'train', rows: [[1]], labels: [], params: defaultParams, nativeError: 'LengthMismatch', typescript: observe(() => trainObserved([[1]], [], defaultParams)) },
    { id: 'ragged', operation: 'train', rows: [[1, 2], [3]], labels: [0, 1], params: defaultParams, nativeError: 'RaggedFeatures', typescript: observe(() => trainObserved([[1, 2], [3]], [0, 1], defaultParams)) },
    { id: 'nan', operation: 'train', rows: [['NaN'], [1]], labels: [0, 1], params: defaultParams, nativeError: 'NonFiniteFeature', typescript: observe(() => trainObserved([[Number.NaN], [1]], [0, 1], defaultParams)) },
    { id: 'invalid-c', operation: 'construct', params: { ...defaultParams, C: -1 }, nativeError: 'InvalidState', typescript: observe(() => new SVMClassifier({ params: { ...defaultParams, C: -1 } }).classifierId) },
    { id: 'malformed-state', operation: 'load', params: defaultParams, state: { ...base, weights: [] }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, weights: [] }, defaultParams, [])) },
    { id: 'nonfinite-state', operation: 'load', params: defaultParams, state: encodeObserved({ ...base, weights: [Number.NaN, ...base.weights.slice(1)] }), nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, weights: [Number.NaN, ...base.weights.slice(1)] }, defaultParams)) },
    { id: 'above-f32-max-state', operation: 'load', params: defaultParams, state: { ...base, weights: [3.5e38, ...base.weights.slice(1)] }, nativeError: 'InvalidState', typescript: observe(() => loadObserved({ ...base, weights: [3.5e38, ...base.weights.slice(1)] }, defaultParams)) },
    { id: 'nonfinite-prediction', operation: 'predict', params: defaultParams, state: base, probe: ['+Infinity', 0], nativeError: 'NonFiniteFeature', typescript: observe(() => loadObserved(base, defaultParams, [Number.POSITIVE_INFINITY, 0])) },
  ];
  writeOrCheck(path, envelope('classifiers/svm-v1', generator, [
    'src/services/ml/svmClassifier.ts', 'src/services/ml/sgd.ts', 'src/services/ml/config.ts', 'src/services/ml/utils/classWeights.ts',
  ], `TensorFlow.js ${tf.version.tfjs} CPU`, cases, { invalidCases }), write);
}

if (isMain(import.meta.url)) await generateSvm(parseWriteFlag());
