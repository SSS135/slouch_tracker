import { PCATransformer, type SerializedPCA } from '../../src/services/ml/pca';
import { encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/math/pca-v1.json';
const generator = 'scripts/oracles/generate-pca.ts';

function exerciseState(state: SerializedPCA, probe = [[1, 2, 3]]): unknown {
  const transformer = PCATransformer.fromJSON(state);
  const result = {
    outputDimension: transformer.getOutputDimension(),
    transformed: transformer.transform(probe),
    serialized: transformer.toJSON(),
  };
  transformer.dispose();
  return result;
}

export function generatePca(write: boolean): void {
  const definitions = [
    { id: 'tall-nondegenerate', components: 3, rows: [[1, 2, 3], [2, 1, 4], [3, 5, 2], [4, 2, 1], [5, 7, 6], [6, 4, 5]] },
    { id: 'wide-clamped', components: 9, rows: [[1, 2, 3, 4, 5, 6], [2, 4, 1, 5, 3, 7], [4, 1, 5, 2, 6, 3], [8, 3, 2, 7, 1, 4]] },
    { id: 'mean-offset', components: 2, rows: [[101, -48, 1003], [102, -44, 1001], [104, -49, 1005], [108, -43, 1002]] },
    { id: 'rank-deficient', components: 2, rows: [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]] },
    { id: 'near-equal-eigenvalues', components: 2, rows: [[1, 0.001, 0], [-1, -0.001, 0], [0, 1, 0.001], [0, -1, -0.001], [0.001, 0, 1], [-0.001, 0, -1]] },
    { id: 'repeated-eigenvalues', components: 2, rows: [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] },
    { id: 'square-one-component', components: 1, rows: [[3, 1, 2], [1, 4, 0], [2, 0, 5]] },
  ];
  const cases = definitions.map(({ id, rows, components }) => {
    const transformer = new PCATransformer();
    transformer.fit(rows, components);
    const state = transformer.toJSON();
    const transformed = transformer.transform(rows);
    const loaded = PCATransformer.fromJSON(state);
    const loadedTransform = loaded.transform(rows);
    transformer.dispose();
    loaded.dispose();
    return { id, requestedComponents: components, rows, state, transformed, loadedTransform };
  });

  const firstRows = [[1, 2, 3], [2, 1, 4], [4, 3, 2], [5, 7, 6]];
  const secondRows = [[10, -2], [12, -1], [15, 4], [18, 8], [20, 9]];
  const reused = new PCATransformer();
  reused.fit(firstRows, 2);
  const firstState = reused.toJSON();
  reused.fit(secondRows, 1);
  const secondState = reused.toJSON();
  const multipleFit = {
    id: 'multiple-fit-replaces-state',
    first: { rows: firstRows, requestedComponents: 2, state: firstState },
    second: {
      rows: secondRows,
      requestedComponents: 1,
      state: secondState,
      transformed: reused.transform(secondRows),
    },
  };
  reused.dispose();

  const base = cases[0].state;
  const corruptStates: Array<{ id: string; state: SerializedPCA; nativeError: string }> = [
    { id: 'corrupt-state-zero-components', state: { ...base, nComponents: 0 }, nativeError: 'InvalidState' },
    { id: 'corrupt-state-component-count', state: { ...base, nComponents: 2 }, nativeError: 'InvalidState' },
    { id: 'corrupt-state-width', state: { ...base, components: base.components.map((row, index) => index === 0 ? row.slice(1) : row) }, nativeError: 'InvalidState' },
    { id: 'corrupt-state-mean-width', state: { ...base, mean: base.mean.slice(1) }, nativeError: 'InvalidState' },
    { id: 'corrupt-state-explained-width', state: { ...base, explainedVariance: base.explainedVariance?.slice(1) }, nativeError: 'InvalidState' },
    { id: 'corrupt-state-component-nan', state: { ...base, components: base.components.map((row, index) => index === 0 ? [Number.NaN, ...row.slice(1)] : row) }, nativeError: 'NonFiniteState' },
    { id: 'corrupt-state-mean-infinity', state: { ...base, mean: [Number.POSITIVE_INFINITY, ...base.mean.slice(1)] }, nativeError: 'NonFiniteState' },
    { id: 'corrupt-state-explained-negative', state: { ...base, explainedVariance: [-1, ...(base.explainedVariance?.slice(1) ?? [])] }, nativeError: 'NonFiniteState' },
  ].map((entry) => ({
    ...entry,
    state: encodeObserved(entry.state) as SerializedPCA,
    typescript: observe(() => exerciseState(entry.state)),
  }));

  const invalidDefinitions = [
    { id: 'empty', input: [] as number[][], requestedComponents: 1, nativeError: 'EmptyData' },
    { id: 'empty-width', input: [[]], requestedComponents: 1, nativeError: 'EmptyData' },
    { id: 'zero-components', input: [[1], [2]], requestedComponents: 0, nativeError: 'InvalidComponentCount' },
    { id: 'negative-components', input: [[1], [2]], requestedComponents: -1, nativeError: 'InvalidComponentCount' },
    { id: 'ragged', input: [[1, 2], [3]], requestedComponents: 1, nativeError: 'RaggedData' },
    { id: 'nan', input: [[Number.NaN]], requestedComponents: 1, nativeError: 'NonFiniteValue' },
  ];
  const invalidCases = invalidDefinitions.map((entry) => ({
    ...entry,
    input: encodeObserved(entry.input),
    typescript: observe(() => {
      const transformer = new PCATransformer();
      transformer.fit(entry.input, entry.requestedComponents);
      const result = { state: transformer.toJSON(), transformed: transformer.transform(entry.input) };
      transformer.dispose();
      return result;
    }),
  })).concat(corruptStates as never[]);

  writeOrCheck(path, envelope('math/pca-v1', generator, [
    'src/services/ml/pca.ts',
    'src/services/ml/__tests__/pca.test.ts',
  ], 'ml-pca 4.1.1 center=true scale=false', cases, {
    adversarialCases: [multipleFit],
    invalidCases,
  }), write);
}

if (isMain(import.meta.url)) generatePca(parseWriteFlag());
