import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import { adamw } from '../../src/services/ml/adamw';
import { envelope, isMain, observeAsync, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/classifiers/adamw-v1.json';
const generator = 'scripts/oracles/generate-adamw.ts';
const settings = { learningRate: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-7 };

type WeightJson = { name: string; values: number[] };

async function optimizerWeights(optimizer: ReturnType<typeof adamw>): Promise<WeightJson[]> {
  const weights = await optimizer.getWeights();
  return weights.map((entry) => ({ name: entry.name, values: Array.from(entry.tensor.dataSync()) }));
}

export async function generateAdamw(write: boolean): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
  const run = async (id: string, name: string, initial: number[], gradient: number[], decay: number) => {
    const variable = tf.variable(tf.tensor1d(initial), true, name);
    const optimizer = adamw(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon, decay);
    const checkpoints: Record<string, number[]> = {};
    for (let step = 1; step <= 10; step++) {
      const grad = tf.tensor1d(gradient);
      optimizer.applyGradients([{ name, tensor: grad }]);
      grad.dispose();
      if ([1, 2, 5, 10].includes(step)) checkpoints[String(step)] = Array.from(variable.dataSync());
    }
    const weights = await optimizerWeights(optimizer);
    optimizer.dispose();
    variable.dispose();
    return { id, name, initial, gradient, decay, checkpoints, optimizerWeights: weights };
  };
  const cases = [
    await run('scalar-decay', 'oracle_weight_scalar', [1], [0.25], 0.1),
    await run('vector-decay', 'oracle_weight_vector', [1, -2, 3], [0.1, -0.2, 0.3], 0.01),
    await run('zero-gradient-decay', 'oracle_weight_zero', [1, -2, 3], [0, 0, 0], 0.1),
    await run('bias-excluded', 'oracle_bias', [1, -2, 3], [0, 0, 0], 0.1),
    await run('contains-bias-excluded', 'oracle_notabiasweight', [1], [0], 0.1),
  ];

  const firstName = 'oracle_order_first';
  const secondName = 'oracle_order_second_bias';
  const first = tf.variable(tf.tensor1d([1, -1]), true, firstName);
  const second = tf.variable(tf.tensor1d([2]), true, secondName);
  const ordered = adamw(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon, 0.1);
  const orderedCheckpoints: Array<Record<string, unknown>> = [];
  const gradientSteps: Array<[number[] | null, number[] | null]> = [
    [[0.25, -0.5], null],
    [null, [0.75]],
    [[-0.1, 0.2], [0.5]],
  ];
  for (let index = 0; index < gradientSteps.length; index++) {
    const [firstValues, secondValues] = gradientSteps[index];
    const firstGradient = firstValues ? tf.tensor1d(firstValues) : null;
    const secondGradient = secondValues ? tf.tensor1d(secondValues) : null;
    ordered.applyGradients({
      [firstName]: firstGradient as tf.Tensor,
      [secondName]: secondGradient as tf.Tensor,
    });
    firstGradient?.dispose();
    secondGradient?.dispose();
    orderedCheckpoints.push({
      step: index + 1,
      variables: [Array.from(first.dataSync()), Array.from(second.dataSync())],
      optimizerWeights: await optimizerWeights(ordered),
    });
  }
  const orderingCase = {
    id: 'null-gradient-source-order',
    decay: 0.1,
    names: [firstName, secondName],
    initial: [[1, -1], [2]],
    gradientSteps,
    checkpoints: orderedCheckpoints,
  };
  ordered.dispose();
  first.dispose();
  second.dispose();

  const continuationName = 'oracle_restore_weight';
  const continuationVariable = tf.variable(tf.tensor1d([1, -2, 3]), true, continuationName);
  const continuation = adamw(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon, 0.01);
  const continuationGradient = [0.1, -0.2, 0.3];
  for (let step = 0; step < 3; step++) {
    const gradient = tf.tensor1d(continuationGradient);
    continuation.applyGradients([{ name: continuationName, tensor: gradient }]);
    gradient.dispose();
  }
  const savedWeights = await optimizerWeights(continuation);
  const savedVariable = Array.from(continuationVariable.dataSync());
  const restoredVariable = tf.variable(tf.tensor1d(savedVariable), true, `${continuationName}_restored`);
  const restored = adamw(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon, 0.01);
  const restoreTensors = savedWeights.map((entry) => ({ name: entry.name, tensor: tf.tensor1d(entry.values) }));
  await restored.setWeights(restoreTensors);
  restoreTensors.forEach((entry) => entry.tensor.dispose());
  const continuationCheckpoints: Array<Record<string, unknown>> = [];
  for (let step = 4; step <= 7; step++) {
    const originalGradient = tf.tensor1d(continuationGradient);
    const restoredGradient = tf.tensor1d(continuationGradient);
    continuation.applyGradients([{ name: continuationName, tensor: originalGradient }]);
    restored.applyGradients([{ name: `${continuationName}_restored`, tensor: restoredGradient }]);
    originalGradient.dispose();
    restoredGradient.dispose();
    continuationCheckpoints.push({
      step,
      original: Array.from(continuationVariable.dataSync()),
      restored: Array.from(restoredVariable.dataSync()),
      restoredWeights: await optimizerWeights(restored),
    });
  }
  const continuationCase = {
    id: 'save-restore-continuation',
    decay: 0.01,
    name: continuationName,
    initial: [1, -2, 3],
    gradient: continuationGradient,
    saveAfterStep: 3,
    savedVariable,
    savedWeights,
    checkpoints: continuationCheckpoints,
  };
  continuation.dispose();
  restored.dispose();
  continuationVariable.dispose();
  restoredVariable.dispose();

  const invalidDefinitions: Array<{ id: string; weights: WeightJson[]; nativeError: string }> = [
    { id: 'missing-iteration', weights: [], nativeError: 'InvalidWeights' },
    { id: 'nonfinite-state', weights: [{ name: 'iter', values: [0] }, { name: 'w/m', values: [Number.NaN] }, { name: 'w/v', values: [0] }], nativeError: 'NonFiniteValue' },
    { id: 'dimension-mismatch', weights: [{ name: 'iter', values: [0] }, { name: 'w/m', values: [0, 0] }, { name: 'w/v', values: [0] }], nativeError: 'InvalidWeights' },
    { id: 'odd-moment-count', weights: [{ name: 'iter', values: [0] }, { name: 'w/m', values: [0] }], nativeError: 'InvalidWeights' },
    { id: 'negative-iteration', weights: [{ name: 'iter', values: [-1] }], nativeError: 'InvalidIteration' },
  ];
  const invalidCases = [];
  for (const definition of invalidDefinitions) {
    invalidCases.push({
      ...definition,
      weights: definition.weights.map((weight) => ({ ...weight, values: weight.values.map((value) => Number.isNaN(value) ? 'NaN' : value) })),
      typescript: await observeAsync(async () => {
        const optimizer = adamw(settings.learningRate, settings.beta1, settings.beta2, settings.epsilon, 0.01);
        const tensors = definition.weights.map((weight) => ({ name: weight.name, tensor: tf.tensor1d(weight.values) }));
        try {
          await optimizer.setWeights(tensors);
          return await optimizerWeights(optimizer);
        } finally {
          tensors.forEach((weight) => weight.tensor.dispose());
          optimizer.dispose();
        }
      }),
    });
  }

  writeOrCheck(path, envelope('classifiers/adamw-v1', generator, [
    'src/services/ml/adamw.ts',
    'src/services/ml/__tests__/adamw.test.ts',
  ], `TensorFlow.js ${tf.version.tfjs} CPU`, cases, {
    settings,
    adversarialCases: [orderingCase, continuationCase],
    invalidCases,
  }), write);
}

if (isMain(import.meta.url)) await generateAdamw(parseWriteFlag());
