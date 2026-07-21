import seedrandom from 'seedrandom';
import { kmeans, kmeansWithTrace, selectBestKWithTrace } from '../../src/services/ml/kmeans';
import { encodeObserved, envelope, isMain, observe, parseWriteFlag, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/math/kmeans-v1.json';
const generator = 'scripts/oracles/generate-kmeans.ts';
const vectors = (rows: number[][]) => rows.map((row) => new Float32Array(row));
const f64Bits = (value: number) => {
  const bytes = Buffer.alloc(8);
  bytes.writeDoubleLE(value);
  return bytes.readBigUInt64LE().toString(16).padStart(16, '0');
};
const serialize = (result: ReturnType<typeof kmeans>) => ({
  k: result.k,
  centroids: result.centroids.map((centroid) => Array.from(centroid)),
  assignments: result.assignments,
  silhouetteScore: result.silhouetteScore,
});

export function generateKmeans(write: boolean): void {
  const datasets = [
    { id: 'empty-k3', rows: [] as number[][], k: 3, maxIter: 100, seed: 42 },
    { id: 'k-zero', rows: [[1, 2]], k: 0, maxIter: 100, seed: 42 },
    { id: 'one-sample-k3', rows: [[1.25, -2]], k: 3, maxIter: 100, seed: 42 },
    { id: 'two-clusters', rows: [[0, 0], [0.2, -0.1], [-0.1, 0.2], [8, 8], [8.2, 7.9], [7.8, 8.1]], k: 2, maxIter: 100, seed: 42 },
    { id: 'iteration-zero-checkpoint', rows: [[0, 0], [0.2, -0.1], [-0.1, 0.2], [8, 8], [8.2, 7.9], [7.8, 8.1]], k: 2, maxIter: 0, seed: 42 },
    { id: 'iteration-one-checkpoint', rows: [[0, 0], [0.2, -0.1], [-0.1, 0.2], [8, 8], [8.2, 7.9], [7.8, 8.1]], k: 2, maxIter: 1, seed: 42 },
    { id: 'identical-points', rows: [[3, 3], [3, 3], [3, 3], [3, 3]], k: 3, maxIter: 4, seed: 42 },
    { id: 'tie-distance-first-wins', rows: [[-1, 0], [1, 0], [0, 0], [9, 0]], k: 2, maxIter: 5, seed: -0 },
    { id: 'close-distance-f32-threshold', rows: [[0], [1], [1.0000001192092896], [1.000000238418579], [2]], k: 3, maxIter: 1, seed: 42 },
    { id: 'small-seed', rows: [[0], [1], [2], [9]], k: 2, maxIter: 5, seed: 1e-7 },
    { id: 'scientific-seed', rows: [[0], [1], [2], [9]], k: 2, maxIter: 5, seed: 1e21 },
  ];
  const cases = datasets.map((item) => {
    const rng = seedrandom(item.seed.toString());
    let result: ReturnType<typeof serialize> | undefined;
    let trace: ReturnType<typeof kmeansWithTrace>['trace'] | undefined;
    let typescriptError: string | undefined;
    try {
      const traced = kmeansWithTrace(vectors(item.rows), item.k, item.maxIter, item.seed);
      result = serialize(traced.result);
      trace = traced.trace;
    } catch (error) {
      typescriptError = error instanceof Error ? error.message : String(error);
    }
    return {
      ...item,
      seedString: item.seed.toString(),
      ...(() => { const values = Array.from({ length: 32 }, () => rng()); return { rngFirst32: values, rngFirst32Bits: values.map(f64Bits) }; })(),
      result,
      trace,
      typescriptError,
    };
  });
  const selectionRows = [[0, 0], [0.1, 0.2], [5, 5], [5.1, 4.9], [10, 0], [9.9, 0.2]];
  const selection = selectBestKWithTrace(vectors(selectionRows), [1, 2, 3, 4, 5], 42);
  cases.push({
    id: 'select-best-k',
    rows: selectionRows,
    kValues: [1, 2, 3, 4, 5],
    seed: 42,
    result: serialize(selection.result),
    runTrace: selection.runs,
  } as never);

  const invalidDefinitions = [
    { id: 'ragged', input: [[1, 2], [3]], nativeError: 'RaggedFeatures' },
    { id: 'empty-width', input: [[]], nativeError: 'EmptyFeatureVector' },
    { id: 'nan', input: [[Number.NaN]], nativeError: 'NonFiniteFeature' },
    { id: 'positive-infinity', input: [[Number.POSITIVE_INFINITY]], nativeError: 'NonFiniteFeature' },
    { id: 'negative-infinity', input: [[Number.NEGATIVE_INFINITY]], nativeError: 'NonFiniteFeature' },
  ];
  const invalidCases = invalidDefinitions.map((entry) => ({
    ...entry,
    input: encodeObserved(entry.input),
    k: 2,
    maxIter: 2,
    seed: 42,
    typescript: observe(() => serialize(kmeans(vectors(entry.input), 2, 2, 42))),
  }));

  writeOrCheck(path, envelope('math/kmeans-v1', generator, [
    'src/services/ml/kmeans.ts',
    'src/services/ml/config.ts',
    'src/services/ml/__tests__/kmeans.test.ts',
  ], 'pure TypeScript', cases, { invalidCases }), write);
}

if (isMain(import.meta.url)) generateKmeans(parseWriteFlag());
