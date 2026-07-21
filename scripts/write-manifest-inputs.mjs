import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';

const paths = [
  '.plans/rust-typescript-refactor-plan.md',
  'PORTING.md',
  'package-lock.json',
  'src-tauri/Cargo.lock',
  'src-tauri/tauri.conf.json',
  'src-tauri/resource-lock.json',
  'src-tauri/model-format-v1.md',
  'src-tauri/schema/live-v1.sql',
  'src-tauri/schema/archive-v1.sql',
];
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const manifestPath = 'src-tauri/port-manifest.json';
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
if (manifest.version !== 2) throw new Error('manifest v2 is required before refreshing verification inputs');
manifest.verificationInputs = Object.fromEntries(paths.map((path) => [path, sha256(path)]));
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ ok: true, manifestPath, verificationInputs: manifest.verificationInputs }, null, 2));
