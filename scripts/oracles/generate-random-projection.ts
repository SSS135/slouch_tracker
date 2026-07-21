import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import seedrandom from 'seedrandom';
import { RandomProjectionTransformer, type RandomProjectionState } from '../../src/services/ml/randomProjection';
import { bits, encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/math/random-projection-v1.json';
const generator = 'scripts/oracles/generate-random-projection.ts';
const samples = (rows: number[][]) => rows.map((row) => new Float32Array(row));
const f64Bits = (value: number) => {
  const bytes = Buffer.alloc(8);
  bytes.writeDoubleLE(value);
  return bytes.readBigUInt64LE().toString(16).padStart(16, '0');
};

export async function generateRandomProjection(write: boolean): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
  const definitions = [
    { id: 'two-by-three-seed42', nComponents: 2, seed: 42, rows: [[1, 2, 3], [-4, 5, -6]] },
    { id: 'dimension-increase', nComponents: 5, seed: 42, rows: [[1, 2, 3], [-4, 5, -6]] },
    { id: 'one-component-negative-zero', nComponents: 1, seed: -0, rows: [[0.25, -0.5, 1.5]] },
    { id: 'small-seed', nComponents: 2, seed: 1e-7, rows: [[1, 2, 3]] },
    { id: 'scientific-seed', nComponents: 2, seed: 1e21, rows: [[1, 2, 3]] },
    { id: 'cancellation-sensitive', nComponents: 2, seed: 42, rows: [[16777216, 1, -16777216, 0.00000011920928955078125]] },
  ];
  const cases = definitions.map(({ id, nComponents, seed, rows }) => {
    const input = samples(rows);
    const transformer = new RandomProjectionTransformer(nComponents, seed);
    const transformed = transformer.fitTransform(input).map((sample) => Array.from(sample));
    const state = transformer.toJSON();
    const loaded = RandomProjectionTransformer.fromJSON(state);
    const loadedTransform = input.map((sample) => Array.from(loaded.transform(sample)));
    const rng = seedrandom(seed.toString());
    const result = {
      id,
      nComponents,
      seed,
      seedString: seed.toString(),
      ...(() => { const values = Array.from({ length: 32 }, () => rng()); return { rngFirst32: values, rngFirst32Bits: values.map(f64Bits) }; })(),
      rows,
      state,
      matrixF32Bits: state.projectionMatrix.map(bits),
      transformed,
      transformedBits: transformed.map(bits),
      loadedTransform,
    };
    transformer.dispose();
    loaded.dispose();
    return result;
  });

  const refitDefinitions = [
    { id: 'refit-same-width', firstRows: [[1, 2, 3], [-4, 5, -6]], secondRows: [[7, 8, 9], [0.5, -0.25, 0.125]] },
    { id: 'refit-different-width', firstRows: [[1, 2, 3], [-4, 5, -6]], secondRows: [[7, 8, 9, 10], [0.5, -0.25, 0.125, -1]] },
  ];
  const adversarialCases = refitDefinitions.map(({ id, firstRows, secondRows }) => {
    const transformer = new RandomProjectionTransformer(2, 42);
    const firstTransformed = transformer.fitTransform(samples(firstRows)).map((sample) => Array.from(sample));
    const firstState = transformer.toJSON();
    const secondTransformed = transformer.fitTransform(samples(secondRows)).map((sample) => Array.from(sample));
    const secondState = transformer.toJSON();
    transformer.dispose();
    return {
      id,
      nComponents: 2,
      seed: 42,
      first: { rows: firstRows, state: firstState, matrixF32Bits: firstState.projectionMatrix.map(bits), transformed: firstTransformed },
      second: { rows: secondRows, state: secondState, matrixF32Bits: secondState.projectionMatrix.map(bits), transformed: secondTransformed },
    };
  });

  const base = cases[0].state;
  const invalidCases = [
    {
      id: 'zero-components', operation: 'construct', nComponents: 0, seed: 42,
      typescript: observe(() => new RandomProjectionTransformer(0, 42).getNComponents()), nativeError: 'InvalidComponents',
    },
    {
      id: 'empty', operation: 'fit', nComponents: 2, seed: 42, rows: [],
      typescript: observe(() => { const value = new RandomProjectionTransformer(2, 42); value.fit([]); return value.isFitted(); }), nativeError: 'EmptyDataset',
    },
    {
      id: 'empty-width', operation: 'fit', nComponents: 2, seed: 42, rows: [[]],
      typescript: observe(() => { const value = new RandomProjectionTransformer(2, 42); const output = value.fitTransform(samples([[]])); const result = { fitted: value.isFitted(), state: value.toJSON(), output }; value.dispose(); return result; }), nativeError: 'EmptyFeatureVector',
    },
    {
      id: 'ragged', operation: 'fitTransform', nComponents: 2, seed: 42, rows: [[1, 2], [3]],
      typescript: observe(() => { const value = new RandomProjectionTransformer(2, 42); return value.fitTransform(samples([[1, 2], [3]])); }), nativeError: 'RaggedDataset',
    },
    {
      id: 'dimension-mismatch', operation: 'transform', state: base, probe: [1, 2],
      typescript: observe(() => { const value = RandomProjectionTransformer.fromJSON(base); return value.transform(new Float32Array([1, 2])); }), nativeError: 'DimensionMismatch',
    },
    {
      id: 'nan', operation: 'transform', state: base, probe: ['NaN', 2, 3],
      typescript: observe(() => { const value = RandomProjectionTransformer.fromJSON(base); return value.transform(new Float32Array([Number.NaN, 2, 3])); }), nativeError: 'NonFiniteInput',
    },
    {
      id: 'unfitted-transform', operation: 'transform-unfitted', nComponents: 2, seed: 42,
      typescript: observe(() => new RandomProjectionTransformer(2, 42).transform(new Float32Array([1, 2]))), nativeError: 'NotFitted',
    },
    {
      id: 'unfitted-serialize', operation: 'serialize-unfitted', nComponents: 2, seed: 42,
      typescript: observe(() => new RandomProjectionTransformer(2, 42).toJSON()), nativeError: 'UnfittedSerialization',
    },
    {
      id: 'disposed-transform', operation: 'disposed-transform', state: base,
      typescript: observe(() => { const value = RandomProjectionTransformer.fromJSON(base); value.dispose(); const lifecycle = { isFitted: value.isFitted(), serialized: value.toJSON() }; try { return { ...lifecycle, transformed: value.transform(new Float32Array([1, 2, 3])) }; } catch (error) { return { ...lifecycle, transformError: error instanceof Error ? error.message : String(error) }; } }), nativeError: 'NotFitted',
    },
    ...([
      { id: 'corrupt-state-row-count', state: { ...base, projectionMatrix: base.projectionMatrix.slice(1) }, nativeError: 'MatrixRowCountMismatch' },
      { id: 'corrupt-state-row-width', state: { ...base, projectionMatrix: base.projectionMatrix.map((row, index) => index === 0 ? row.slice(1) : row) }, nativeError: 'MatrixRowLengthMismatch' },
      { id: 'corrupt-state-nonfinite', state: { ...base, projectionMatrix: base.projectionMatrix.map((row, index) => index === 0 ? [Number.NaN, ...row.slice(1)] : row) }, nativeError: 'NonFiniteMatrix' },
      { id: 'matrix-above-f32-max', state: { ...base, projectionMatrix: base.projectionMatrix.map((row, index) => index === 0 ? [3.5e38, ...row.slice(1)] : row) }, nativeError: 'MatrixValueOutOfRange' },
    ] as Array<{ id: string; state: RandomProjectionState; nativeError: string }>).map((entry) => ({
      ...entry,
      operation: 'load',
      state: encodeObserved(entry.state),
      typescript: observe(() => { const value = RandomProjectionTransformer.fromJSON(entry.state); const result = { fitted: value.isFitted(), transformed: value.transform(new Float32Array([1, 2, 3])) }; value.dispose(); return result; }),
    })),
  ];

  writeOrCheck(path, envelope('math/random-projection-v1', generator, [
    'src/services/ml/randomProjection.ts',
    'src/services/ml/config.ts',
    'src/services/ml/__tests__/randomProjection.test.ts',
  ], `TensorFlow.js ${tf.version.tfjs} CPU`, cases, { adversarialCases, invalidCases }), write);
}

if (isMain(import.meta.url)) await generateRandomProjection(parseWriteFlag());
