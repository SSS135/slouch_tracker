import { envelope, isMain, parseWriteFlag, writeOrCheck } from './common';

const generator = 'scripts/oracles/generate-inventory.ts';
export function generateInventory(write: boolean): void {
  const rows = [
    { id: 'kmeans', fixture: 'src-tauri/fixtures/math/kmeans-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'pca', fixture: 'src-tauri/fixtures/math/pca-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'random-projection', fixture: 'src-tauri/fixtures/math/random-projection-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'cross-validation', fixture: 'src-tauri/fixtures/math/cross-validation-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'adamw', fixture: 'src-tauri/fixtures/classifiers/adamw-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'mlp', fixture: 'src-tauri/fixtures/classifiers/mlp-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'svm', fixture: 'src-tauri/fixtures/classifiers/svm-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'kmeans-logistic', fixture: 'src-tauri/fixtures/classifiers/kmeans-logistic-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'kmeans-prototype', fixture: 'src-tauri/fixtures/classifiers/kmeans-prototype-v1.json', rust: 'slouch-ml::compatibility_oracles' },
    { id: 'model-envelopes', fixture: 'src-tauri/fixtures/models/model-envelope-v1.json', rust: 'slouch-store::compatibility_oracles' },
    { id: 'store-operations', fixture: 'src-tauri/fixtures/store/store-operations-v1.json', rust: 'slouch-store::compatibility_oracles' },
    { id: 'vision-preprocessing-inference', fixture: 'src-tauri/fixtures/vision/vision-inference-v1.json', rust: 'slouch-vision::inference_parity' },
  ];
  writeOrCheck('src-tauri/fixtures/compatibility-oracle-inventory-v1.json', envelope(
    'compatibility-oracle-inventory-v1', generator,
    ['PORTING.md', '.plans/rust-typescript-refactor-plan.md', '.pi-subagents/artifacts/outputs/4ca31fdf-fecf-4c54-9426-d0185d9575b6/oracles/oracle-audit.md'],
    'inventory', rows,
    { gate: 'node scripts/run-gate.mjs compatibility-oracles', completeness: 'Every frozen audit row has a TypeScript generator, committed fixture, and Rust consumer.' },
  ), write);
}
if (isMain(import.meta.url)) generateInventory(parseWriteFlag());
