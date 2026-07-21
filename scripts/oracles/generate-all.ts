import { readdirSync, statSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { parseWriteFlag, root } from './common';
import { generateKmeans } from './generate-kmeans';
import { generatePca } from './generate-pca';
import { generateRandomProjection } from './generate-random-projection';
import { generateCrossValidation } from './generate-cross-validation';
import { generateAdamw } from './generate-adamw';
import { generateMlp } from './generate-mlp';
import { generateSvm } from './generate-svm';
import { generateKmeansLogistic } from './generate-kmeans-logistic';
import { generateKmeansPrototype } from './generate-kmeans-prototype';
import { generateModelEnvelopes } from './generate-model-envelopes';
import { generateStoreOperations } from './generate-store-operations';
import { generateVisionInference } from './generate-vision-inference';
import { generateInventory } from './generate-inventory';

const write = parseWriteFlag();
generateKmeans(write);
generatePca(write);
await generateRandomProjection(write);
generateCrossValidation(write);
await generateAdamw(write);
await generateMlp(write);
await generateSvm(write);
await generateKmeansLogistic(write);
generateKmeansPrototype(write);
await generateModelEnvelopes(write);
await generateStoreOperations(write);
await generateVisionInference(write);
generateInventory(write);

const expected = new Set([
  'src-tauri/fixtures/compatibility-oracle-inventory-v1.json',
  'src-tauri/fixtures/classifiers/adamw-v1.json', 'src-tauri/fixtures/classifiers/gaussian-nb-v1.json',
  'src-tauri/fixtures/classifiers/kmeans-logistic-v1.json', 'src-tauri/fixtures/classifiers/kmeans-prototype-v1.json',
  'src-tauri/fixtures/classifiers/mlp-v1.json', 'src-tauri/fixtures/classifiers/svm-v1.json',
  'src-tauri/fixtures/domain/keypoint-v1.json', 'src-tauri/fixtures/math/cross-validation-v1.json',
  'src-tauri/fixtures/math/kmeans-v1.json', 'src-tauri/fixtures/math/pca-v1.json',
  'src-tauri/fixtures/math/random-projection-v1.json', 'src-tauri/fixtures/math/rtmpose-pooling-v1.json',
  'src-tauri/fixtures/models/model-envelope-v1-binaries.json', 'src-tauri/fixtures/models/model-envelope-v1.json',
  'src-tauri/fixtures/models/posture-svm-v1.bin', 'src-tauri/fixtures/models/presence-svm-v1.bin',
  'src-tauri/fixtures/models/training-config-v1.bin', 'src-tauri/fixtures/models/nonsquare-mlp-v1.bin',
  'src-tauri/fixtures/store/store-operations-v1.json',
  'src-tauri/fixtures/vision/vision-inference-v1.json',
  ...['boundary-crop-silhouette', 'edge-clipped-silhouette', 'empty-landscape', 'multiple-silhouettes', 'noise-portrait', 'odd-gradient-boundary', 'single-silhouette-square']
    .map((id) => `src-tauri/fixtures/vision/frames/${id}.rgba`),
]);
const found: string[] = [];
const walk = (directory: string): void => {
  for (const name of readdirSync(directory).sort()) {
    const path = resolve(directory, name);
    if (statSync(path).isDirectory()) walk(path);
    else found.push(relative(root, path).replaceAll('\\', '/'));
  }
};
walk(resolve(root, 'src-tauri/fixtures'));
const extra = found.filter((path) => !expected.has(path));
const missing = [...expected].filter((path) => !found.includes(path));
if (extra.length || missing.length) throw new Error(`oracle inventory mismatch: missing=${missing.join(',')} extra=${extra.join(',')}`);
console.log(JSON.stringify({ ok: true, mode: write ? 'write' : 'check', fixtureFamilies: 13, fixtureFiles: found.length }));
