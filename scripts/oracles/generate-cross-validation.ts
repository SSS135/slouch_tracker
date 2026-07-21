import { createStratifiedKFold, createTemporalBlockKFold, detectTemporalBlocks, validateFolds } from '../../src/services/ml/crossValidation';
import { envelope, isMain, parseWriteFlag, writeOrCheck } from './common';

const generator = 'scripts/oracles/generate-cross-validation.ts';
export function generateCrossValidation(write: boolean): void {
  const labels = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1]);
  const timestamps = [1000, 1100, 1200, 8000, 8100, 15000, 15100, 30000];
  const stratified = createStratifiedKFold(labels, 4, 42);
  const temporalBlocks = detectTemporalBlocks(timestamps, labels);
  const temporal = createTemporalBlockKFold(timestamps, labels, 4, 42);
  const cases = [{ id: 'stratified-seed42', labels: Array.from(labels), folds: stratified, valid: validateFolds(stratified, labels.length) }, { id: 'temporal-blocks-seed42', labels: Array.from(labels), timestamps, blocks: temporalBlocks, folds: temporal, valid: validateFolds(temporal, labels.length) }];
  writeOrCheck('src-tauri/fixtures/math/cross-validation-v1.json', envelope('math/cross-validation-v1', generator, ['src/services/ml/crossValidation.ts', 'src/services/ml/__tests__/crossValidation.test.ts'], 'pure TypeScript', cases), write);
}
if (isMain(import.meta.url)) generateCrossValidation(parseWriteFlag());
