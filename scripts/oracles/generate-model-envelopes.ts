import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs';
import { MLPClassifier } from '../../src/services/ml/mlpClassifier';
import { SVMClassifier } from '../../src/services/ml/svmClassifier';
import { KNNClassifier } from '../../src/services/ml/knnClassifier';
import { GaussianNBClassifier } from '../../src/services/ml/naiveBayesClassifier';
import { KMeansPrototypeClassifier } from '../../src/services/ml/kmeansPrototypeClassifier';
import { KMeansLogisticClassifier } from '../../src/services/ml/kmeansLogisticClassifier';
import { FeatureExtractor } from '../../src/services/ml/featureExtractor';
import { deserializeClassifier } from '../../src/services/ml/classifierFactory';
import { envelope, isMain, parseWriteFlag, sha256, writeOrCheck } from './common';

const generator = 'scripts/oracles/generate-model-envelopes.ts';
type RecordValue = { name: string; kind: number; dimensions?: number[]; bytes: Buffer };
const scalar = (name: string, kind: number, bytes: Buffer): RecordValue => ({ name, kind, bytes });
const utf8 = (name: string, value: string) => scalar(name, 7, Buffer.from(value));
const u32 = (name: string, value: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return scalar(name, 2, b); };
const u64 = (name: string, value: bigint) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(value); return scalar(name, 3, b); };
const i64 = (name: string, value: bigint) => { const b = Buffer.alloc(8); b.writeBigInt64LE(value); return scalar(name, 4, b); };
const f32Scalar = (name: string, value: number) => { const b = Buffer.alloc(4); b.writeFloatLE(value); return scalar(name, 5, b); };
const tensorF32 = (name: string, dimensions: number[], values: number[]): RecordValue => {
  const bytes = Buffer.alloc(values.length * 4); values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return { name, kind: 9, dimensions, bytes };
};
const tensorU32 = (name: string, dimensions: number[], values: number[]): RecordValue => {
  const bytes = Buffer.alloc(values.length * 4); values.forEach((value, index) => bytes.writeUInt32LE(value, index * 4));
  return { name, kind: 11, dimensions, bytes };
};
function encode(magic: string, stateVersion: number, records: RecordValue[], canonical = true): Buffer {
  if (canonical) records.sort((a, b) => Buffer.from(a.name).compare(Buffer.from(b.name)));
  const header = Buffer.alloc(12); header.write(magic, 0, 'ascii'); header.writeUInt16LE(1, 4); header.writeUInt16LE(stateVersion, 6); header.writeUInt32LE(records.length, 8);
  const chunks = [header];
  for (const record of records) {
    const name = Buffer.from(record.name); const rank = record.dimensions?.length ?? 0;
    const h = Buffer.alloc(2 + name.length + 2 + rank * 4 + 8);
    h.writeUInt16LE(name.length, 0); name.copy(h, 2); h[2 + name.length] = record.kind; h[3 + name.length] = rank;
    (record.dimensions ?? []).forEach((dimension, index) => h.writeUInt32LE(dimension, 4 + name.length + index * 4));
    h.writeBigUInt64LE(BigInt(record.bytes.length), 4 + name.length + rank * 4);
    chunks.push(h, record.bytes);
  }
  return Buffer.concat(chunks);
}
function svmRecords(role: 'presence' | 'posture', configHash: Buffer): RecordValue[] {
  return [
    utf8('classifier.id', 'svm'), u32('classifier.state_version', 1), u64('dataset.version', 7n),
    u32('feature.input_dimension', 5), utf8('feature.ids', 'posture_raw'), utf8('normalization.mode', 'none'),
    utf8('reduction.method', 'none'), u32('reduction.output_dimension', 5), utf8('role', role),
    i64('trained_at_ms', 1735689600000n), scalar('training_config.sha256', 8, configHash),
    tensorF32('svm.weights', [5], [0.25, -0.5, 0.75, -1, 1.25]), f32Scalar('svm.bias', 0.125), tensorF32('svm.class_weights', [2], [1, 1]),
  ];
}
function svmEnvelope(role: 'presence' | 'posture', configHash: Buffer): Buffer {
  return encode('SLMD', 1, svmRecords(role, configHash));
}
function nonsquareMlpEnvelope(configHash: Buffer): Buffer {
  const projection = Array.from({ length: 512 }, (_, index) => index % 257 === 0 ? 1 : 0);
  return encode('SLMD', 1, [
    utf8('classifier.id', 'mlp'), u32('classifier.state_version', 1), u64('dataset.version', 7n),
    u32('feature.input_dimension', 256), utf8('feature.ids', 'gau_features'), utf8('normalization.mode', 'none'),
    utf8('reduction.method', 'random_projection'), u32('reduction.output_dimension', 2),
    tensorF32('reduction.matrix', [2, 256], projection), utf8('reduction.rng', 'seedrandom'), utf8('reduction.seed', '42.00000000000000000'),
    utf8('role', 'posture'), i64('trained_at_ms', 1735689600000n), scalar('training_config.sha256', 8, configHash),
    tensorU32('mlp.layer_shapes', [2], [2, 3]), tensorF32('mlp.0.weights', [3, 2], [1, 4, 2, 5, 3, 6]),
    tensorF32('mlp.0.biases', [3], [0.1, 0.2, 0.3]), u32('mlp.hidden_layers', 0), u32('mlp.hidden_size', 64),
    tensorF32('mlp.class_weights', [2], [1, 1]),
  ]);
}

export async function generateModelEnvelopes(write: boolean): Promise<void> {
  await tf.setBackend('cpu'); await tf.ready();
  const rows = [[-2, -0.7], [-1.4, -0.4], [-0.8, -1.1], [0.7, 1.2], [1.6, 0.4], [2.2, 1.7]].map((row) => new Float32Array(row));
  const labels = new Int32Array([0, 0, 0, 1, 1, 1]);
  const classifiers = [
    new MLPClassifier({ params: { hiddenLayers: 0, maxIterations: 2 } }),
    new KNNClassifier({ params: { k: 3, kernel: 'cosine', gamma: 1 } }),
    new SVMClassifier({ params: { C: 1, maxIterations: 2 } }),
    new KMeansPrototypeClassifier({ params: { nClusters: 2, temperature: 1 } }),
    new GaussianNBClassifier({ params: { varianceSmoothing: 1e-6 } }),
    new KMeansLogisticClassifier({ params: { nClusters: 2, temperature: 1, maxIterations: 2, weightDecay: 1 } }),
  ];
  const featureExtractor = {
    featureTypes: ['posture_raw'], normalizationMode: 'none', dimReductionConfig: { method: 'none', components: 2 },
    concatenatedDimensions: 2, normalizationMean: null, normalizationStd: null, dimReductionTransformer: null,
  };
  const cases = classifiers.map((classifier) => {
    classifier.train(rows, labels); const state = classifier.toJSON();
    const modelJson = { featureExtractor, classifier: { classifierId: classifier.classifierId, state } };
    const classifierProbes = [[-1, -1], [0, 0], [1, 1]];
    const probabilities = classifierProbes.map((probe) => classifier.predictProba(new Float32Array(probe)));
    const loaded = deserializeClassifier(modelJson.classifier.classifierId, state, {});
    const loadedProbabilities = classifierProbes.map((probe) => loaded.predictProba(new Float32Array(probe)));
    classifier.dispose(); loaded.dispose();
    return { classifierId: modelJson.classifier.classifierId, modelJson, probes: classifierProbes, probabilities, loadedProbabilities };
  });
  const featureContainers = Array.from({ length: 10 }, (_, sample) => ({
    features: { gau_features: Float32Array.from({ length: 256 }, (_, feature) => Math.fround(sample * 0.1 + feature / 512)) },
    bbox: { x1: 0.1, y1: 0.1, x2: 0.9, y2: 0.9, score: 0.9, width: 0.8, height: 0.8 },
  }));
  const extractorDefinitions = [
    { id: 'none-none', normalizationMode: 'none' as const, dimReductionConfig: { method: 'none' as const, components: 256 } },
    { id: 'layer-none', normalizationMode: 'layer' as const, dimReductionConfig: { method: 'none' as const, components: 256 } },
    { id: 'zscore-none', normalizationMode: 'z_score' as const, dimReductionConfig: { method: 'none' as const, components: 256 } },
    { id: 'none-random-projection', normalizationMode: 'none' as const, dimReductionConfig: { method: 'random_projection' as const, components: 3 } },
    { id: 'none-pca', normalizationMode: 'none' as const, dimReductionConfig: { method: 'pca' as const, components: 3 } },
  ];
  const extractorVariants = extractorDefinitions.map((definition) => {
    const extractor = new FeatureExtractor({ featureTypes: ['gau_features'], normalizationMode: definition.normalizationMode, dimReductionConfig: definition.dimReductionConfig });
    extractor.fit(featureContainers, new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]));
    const state = extractor.toJSON();
    const transformed = Array.from(extractor.transform(featureContainers[3]));
    const loaded = FeatureExtractor.fromJSON(state);
    const loadedTransform = Array.from(loaded.transform(featureContainers[3]));
    extractor.dispose(); loaded.dispose();
    return { id: definition.id, state, transformed, loadedTransform };
  });
  const configContainer = encode('SLCF', 1, [utf8('pair.dataset_selection', 'all_labeled'), scalar('pair.include_reservoir', 1, Buffer.from([1]))]);
  const configHash = Buffer.from(sha256(configContainer), 'hex');
  const validSvm = svmEnvelope('presence', configHash);
  const corrupt = (id: string, bytes: Buffer, nativeOutcome = 'reject') => ({ id, bytesHex: bytes.toString('hex'), bytes: bytes.length, sha256: sha256(bytes), nativeOutcome });
  const badMagic = Buffer.from(validSvm); badMagic[0] = 0x58;
  const badVersion = Buffer.from(validSvm); badVersion.writeUInt16LE(2, 4);
  const nonfiniteRecords = svmRecords('presence', configHash).filter((record) => record.name !== 'svm.bias');
  nonfiniteRecords.push(f32Scalar('svm.bias', Number.POSITIVE_INFINITY));
  const invalidCases = [
    corrupt('bad-magic', badMagic),
    corrupt('bad-version', badVersion),
    corrupt('duplicate-record', encode('SLMD', 1, [...svmRecords('presence', configHash), utf8('role', 'presence')])),
    corrupt('unknown-record', encode('SLMD', 1, [...svmRecords('presence', configHash), utf8('unknown.record', 'x')])),
    corrupt('missing-record', encode('SLMD', 1, svmRecords('presence', configHash).filter((record) => record.name !== 'role'))),
    corrupt('noncanonical-order', encode('SLMD', 1, svmRecords('presence', configHash).reverse(), false)),
    corrupt('rank-shape-byte-mismatch', encode('SLMD', 1, svmRecords('presence', configHash).map((record) => record.name === 'svm.weights' ? tensorF32('svm.weights', [4], [0.25, -0.5, 0.75, -1, 1.25]) : record))),
    corrupt('trailing-bytes', Buffer.concat([validSvm, Buffer.from([0])])),
    corrupt('nonfinite', encode('SLMD', 1, nonfiniteRecords)),
    { id: 'checksum-mismatch', bytesHex: validSvm.toString('hex'), bytes: validSvm.length, sha256: '00'.repeat(32), nativeOutcome: 'sha256-mismatch' },
    { id: 'role-pair-mismatch', firstHex: validSvm.toString('hex'), secondHex: validSvm.toString('hex'), nativeOutcome: 'reject-same-role-pair' },
  ];
  const binaries = [
    { id: 'presence-svm', path: 'src-tauri/fixtures/models/presence-svm-v1.bin', bytes: validSvm },
    { id: 'posture-svm', path: 'src-tauri/fixtures/models/posture-svm-v1.bin', bytes: svmEnvelope('posture', configHash) },
    { id: 'training-config', path: 'src-tauri/fixtures/models/training-config-v1.bin', bytes: configContainer },
    { id: 'nonsquare-mlp', path: 'src-tauri/fixtures/models/nonsquare-mlp-v1.bin', bytes: nonsquareMlpEnvelope(configHash) },
  ];
  for (const binary of binaries) writeOrCheck(binary.path, binary.bytes, write);
  const index = binaries.map((binary) => ({ id: binary.id, path: binary.path, bytes: binary.bytes.length, sha256: sha256(binary.bytes) }));
  writeOrCheck('src-tauri/fixtures/models/model-envelope-v1-binaries.json', { schemaVersion: 1, generator: { path: generator, sha256: sha256(Buffer.from(await import('node:fs').then((fs) => fs.readFileSync(new URL(import.meta.url)))) ) }, binaries: index }, write);
  writeOrCheck('src-tauri/fixtures/models/model-envelope-v1.json', envelope('models/model-envelope-v1', generator, [
    'src/services/ml/model.ts', 'src/services/ml/types.ts', 'src/services/ml/featureExtractor.ts', 'src/services/ml/classifierFactory.ts',
    'src/services/validation/schemas.ts', 'src-tauri/model-format-v1.md',
  ], `TensorFlow.js ${tf.version.tfjs} CPU + independent TypeScript SLMD/SLCF encoder`, cases, {
    jsonContract: { modelToJsonKeys: ['featureExtractor', 'classifier'], callerAddedKeys: ['trainedAt', 'version'] },
    extractorVariants,
    binaryIndex: index, trainingConfigSha256: configHash.toString('hex'),
    nonsquareMlpTranspose: { sourceInOut: [1, 2, 3, 4, 5, 6], sourceShape: [2, 3], persistedOutIn: [1, 4, 2, 5, 3, 6], persistedShape: [3, 2] },
    invalidCases,
  }), write);
}

if (isMain(import.meta.url)) await generateModelEnvelopes(parseWriteFlag());
