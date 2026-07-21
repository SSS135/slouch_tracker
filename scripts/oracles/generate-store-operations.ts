import 'fake-indexeddb/auto';
import { envelope, isMain, parseWriteFlag, sha256, writeOrCheck } from './common';

const path = 'src-tauri/fixtures/store/store-operations-v1.json';
const generator = 'scripts/oracles/generate-store-operations.ts';
const dimensions: Record<string, number> = {
  rtmdet_extracted: 384,
  gau_features: 256,
  gau_features_max: 256,
  gau_features_std: 256,
  backbone_features: 768,
  backbone_features_max: 768,
  backbone_features_std: 768,
};
const vector = (length: number, offset: number) => Float32Array.from({ length }, (_, index) => Math.fround(offset + index / 1024));
const digest = (value: Float32Array) => sha256(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
async function deleteDatabase(name: string): Promise<void> {
  await new Promise<void>((resolve, reject) => { const request = indexedDB.deleteDatabase(name); request.onsuccess = () => resolve(); request.onerror = () => reject(request.error); request.onblocked = () => reject(new Error(`blocked deleting ${name}`)); });
}

export async function generateStoreOperations(write: boolean): Promise<void> {
  await deleteDatabase('slouch-tracker');
  const { DatasetStorage } = await import('../../src/services/dataset/storage');
  const { FrameLabel } = await import('../../src/services/dataset/types');
  const { DatasetOperations } = await import('../../src/services/dataset/operations');
  const { Keypoint } = await import('../../src/services/posture/Keypoint');
  const originalNow = Date.now;
  let now = 1735689600000;
  Date.now = () => (now += 1000);
  try {
    const storage = new DatasetStorage();
    const trace: unknown[] = [];
    const empty = await storage.loadDataset();
    trace.push({ operation: 'load-empty', version: empty.version, lastModified: empty.lastModified, frameIds: [] });
    const makeFrame = (id: string, label: typeof FrameLabel[keyof typeof FrameLabel], offset: number) => ({
      id, timestamp: now += 1000, label, thumbnail: new Blob([Uint8Array.from([82, 73, 70, 70, offset])], { type: 'image/webp' }),
      keypoints: Array.from({ length: 17 }, (_, index) => new Keypoint(index / 20, index / 25, 0.9)),
      bbox: { x1: 0.1, y1: 0.2, x2: 0.8, y2: 0.9, score: 0.95, width: 0.7, height: 0.7 },
      features: Object.fromEntries(Object.entries(dimensions).map(([id, length]) => [id, vector(length, offset)])),
    });
    const first = makeFrame('frame-a', FrameLabel.GOOD, 1);
    await storage.saveFrame(first);
    let dataset = await storage.loadDataset();
    trace.push({ operation: 'save-frame-a', version: dataset.version, lastModified: dataset.lastModified, frameIds: dataset.frames.map((frame) => frame.id), featureHashes: Object.fromEntries(Object.entries(first.features).map(([id, values]) => [id, digest(values)])) });
    const second = makeFrame('frame-b', FrameLabel.BAD, 2);
    await storage.saveFrame(second);
    trace.push({ operation: 'save-frame-b', stats: await storage.getStats(), frameById: (await storage.getFrameById('frame-b'))?.id, badIds: (await storage.getFramesByLabel(FrameLabel.BAD)).map((frame) => frame.id) });
    const overwrite = makeFrame('frame-a', FrameLabel.AWAY, 3);
    await storage.saveFrame(overwrite);
    dataset = await storage.loadDataset();
    trace.push({ operation: 'overwrite-frame-a', version: dataset.version, labels: Object.fromEntries(dataset.frames.map((frame) => [frame.id, frame.label])), stats: await storage.getStats() });
    await storage.updateFrameLabel('frame-b', FrameLabel.GOOD);
    trace.push({ operation: 'update-frame-b-good', stats: await storage.getStats() });
    await storage.updateFrameLabel('frame-b', FrameLabel.UNUSED);
    dataset = await storage.loadDataset();
    trace.push({ operation: 'update-unused-deletes', version: dataset.version, frameIds: dataset.frames.map((frame) => frame.id) });
    await storage.clearDataset();
    dataset = await storage.loadDataset();
    trace.push({ operation: 'clear-dataset', version: dataset.version, lastModified: dataset.lastModified, frameIds: [] });

    const operations = new DatasetOperations(storage);
    const operationCases: unknown[] = [];
    operationCases.push({ id: 'missing-id-get', result: await operations.getFrameById('missing') });
    try {
      await operations.updateFrameLabel('missing', FrameLabel.GOOD);
      operationCases.push({ id: 'missing-id-update', ok: true });
    } catch (error) {
      operationCases.push({ id: 'missing-id-update', ok: false, error: error instanceof Error ? error.message : String(error) });
    }
    await storage.saveFrame(makeFrame('bulk-a', FrameLabel.GOOD, 4));
    await storage.saveFrame(makeFrame('bulk-b', FrameLabel.BAD, 5));
    await storage.saveFrame(makeFrame('bulk-c', FrameLabel.AWAY, 6));
    operationCases.push({ id: 'delete-bulk-all-success', result: await operations.deleteBulk(['bulk-a', 'bulk-b']), nativeResult: { deleted: 2, success: true } });
    operationCases.push({ id: 'delete-bulk-partial-failure', result: await operations.deleteBulk(['bulk-c', 'missing']), nativeResult: { deleted: 1, success: true } });
    operationCases.push({ id: 'delete-bulk-all-failure', result: await operations.deleteBulk(['missing-a', 'missing-b']), nativeResult: { deleted: 0, success: false } });
    operationCases.push({ id: 'delete-bulk-empty', result: await operations.deleteBulk([]), nativeResult: { deleted: 0, success: false } });
    operationCases.push({ id: 'cleanup-unused-zero', result: await operations.cleanupUnused() });
    operationCases.push({ id: 'delete-by-label-zero', result: await operations.deleteByLabel(FrameLabel.BAD) });
    await storage.saveFrame(makeFrame('label-good-a', FrameLabel.GOOD, 7));
    await storage.saveFrame(makeFrame('label-good-b', FrameLabel.GOOD, 8));
    operationCases.push({ id: 'delete-by-label-success', result: await operations.deleteByLabel(FrameLabel.GOOD) });
    operationCases.push({ id: 'needs-retraining-without-model', result: await operations.needsRetraining() });
    await operations.resetDataset();
    operationCases.push({ id: 'reset-dataset', frameIds: (await operations.loadDataset()).frames.map((frame) => frame.id) });
    writeOrCheck(path, envelope('store/store-operations-v1', generator, [
      'src/services/dataset/storage.ts', 'src/services/dataset/operations.ts', 'src/services/dataset/types.ts',
      'src/services/dataset/featureRegistry.ts', 'src/services/dataset/__tests__/storage.test.ts', 'src/services/dataset/__tests__/operations.test.ts',
    ], 'localforage 1.10.0 over fake-indexeddb 6.2.3', trace, {
      fixedClock: { startMs: 1735689600000, incrementMs: 1000 }, dimensions,
      nativePolicy: { applicationId: 1397506888, userVersion: 1, foreignKeys: true, journalMode: 'WAL', busyTimeoutMs: 5000, synchronous: 'NORMAL' },
      dualExpectations: [
        { case: 'arbitrary-feature-length', typescript: 'accepted', native: 'reject InvalidFeatureDimension' },
        { case: 'nonfinite-feature', typescript: 'legacy paths may accept', native: 'reject non-finite' },
        { case: 'duck-typed-blob', typescript: 'fast guard may accept', native: 'reject bytes and MIME required' },
        { case: 'malformed-boundary', typescript: 'weak legacy paths may filter', native: 'reject' },
        { case: 'schema-version-mismatch', typescript: 'clear all data', native: 'transactional migration or future-version rejection' },
      ],
      operationCases,
      operationContracts: { deleteBulk: ['all-success', 'all-failure', 'partial-failure', 'empty'], resetDataset: 'preserves settings and model pair', resetAllData: 'clears frames reservoir models settings', needsRetrainingOnStorageFailure: true, defaultDimReductionConfig: { method: 'none', components: 64 } },
    }), write);
  } finally {
    Date.now = originalNow;
  }
}

if (isMain(import.meta.url)) await generateStoreOperations(parseWriteFlag());
