import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const snapshotPath = '.pi-subagents/verification/starting-tree.json';
const startSnapshot = spawnSync(`node scripts/snapshot-tree.mjs --write ${snapshotPath}`, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
const commands = [
  ['acceptance-schema', 'node scripts/validate-acceptance.mjs'],
  ['manifest', 'node scripts/validate-port-manifest.mjs'],
  ['compatibility-oracles', 'node scripts/run-gate.mjs compatibility-oracles'],
  ['rust-fmt', 'cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check'],
  ['rust-clippy', 'cargo clippy --manifest-path src-tauri/Cargo.toml --workspace --all-targets --all-features -- -D warnings'],
  ['rust-test', 'cargo test --manifest-path src-tauri/Cargo.toml --workspace'],
  ['typecheck', 'npm run typecheck'],
  ['lint', 'npm run lint'],
  ['vitest', 'npm run test:fast'],
  ['frontend-build', 'npm run build:web'],
  ['bindings', 'npm run bindings:check'],
  ['browser-e2e', 'npm run test:e2e:web'],
  ['native-e2e', 'npm run test:e2e:native'],
  ['security-data', 'npm run test:security'],
  ['windows-package', 'npm run tauri:build:win'],
  ['package-inspect', 'npm run package:inspect'],
  ['cleanup', 'node scripts/check-cleanup.mjs'],
  ['manifest-final', 'node scripts/validate-port-manifest.mjs'],
];
const results = commands.map(([id, command]) => {
  const run = spawnSync(command, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  return {
    id,
    command,
    ok: run.status === 0,
    status: run.status,
    stdoutTail: (run.stdout ?? '').trim().split(/\r?\n/).slice(-20),
    stderrTail: (run.stderr ?? '').trim().split(/\r?\n/).slice(-20),
  };
});

const manifest = (() => {
  try { return JSON.parse(readFileSync('src-tauri/port-manifest.json', 'utf8')); }
  catch { return null; }
})();
const entries = Array.isArray(manifest?.queue) ? manifest.queue : [];
const unresolved = entries.filter((entry) => entry.status !== 'verified').map((entry) => entry.id ?? entry.source ?? 'unknown');
const openBlockers = entries.flatMap((entry) => entry.blockers ?? []);
const residualRisks = entries.flatMap((entry) => entry.residualRisks ?? []);

const forbiddenPaths = [
  'src/workers',
  'src/services/ml',
  'src/services/dataset',
  'src/services/validation',
  'src/services/posture',
  'public/rtmdet-nano.onnx',
  'public/rtmpose-m.onnx',
  'public/rtmpose-s.onnx',
];
const presentForbiddenPaths = forbiddenPaths.filter((path) => existsSync(path));
const forbiddenDependencies = [
  'react', 'react-dom', '@vitejs/plugin-react', '@mantine/core', '@mantine/hooks', '@mantine/notifications',
  '@emotion/react', '@emotion/styled', '@tanstack/react-query', '@testing-library/react', '@types/react', '@types/react-dom',
  '@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-wasm',
  'onnxruntime-web', 'localforage', 'zod', 'seedrandom', 'ml-pca', 'jszip', 'fflate', 'file-saver',
];
const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
const allDependencies = { ...(packageJson.dependencies ?? {}), ...(packageJson.devDependencies ?? {}) };
const presentForbiddenDependencies = forbiddenDependencies.filter((name) => name in allDependencies);

function walk(directory, predicate, output = []) {
  if (!existsSync(directory)) return output;
  for (const name of readdirSync(directory).sort()) {
    const path = resolve(directory, name);
    const info = statSync(path);
    if (info.isDirectory()) walk(path, predicate, output);
    else if (predicate(path)) output.push(path.replaceAll('\\', '/'));
  }
  return output;
}
const rustFiles = walk('src-tauri', (path) => path.endsWith('.rs'));
const placeholderFindings = [];
for (const path of rustFiles) {
  const text = readFileSync(path, 'utf8');
  if (/\b(?:todo|unimplemented)!\s*\(/.test(text)) placeholderFindings.push(`${path}: placeholder macro`);
  if (/#!?\s*\[allow\s*\(/.test(text)) placeholderFindings.push(`${path}: allow attribute`);
  if (/\bTODO\b/.test(text)) placeholderFindings.push(`${path}: TODO`);
}
for (const path of walk('src', (path) => /\.(?:test|spec)\.[jt]sx?$/.test(path))) {
  const text = readFileSync(path, 'utf8');
  if (/\b(?:it|test|describe)\.skip\s*\(/.test(text)) placeholderFindings.push(`${path}: skipped test`);
}

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const resourceLock = JSON.parse(readFileSync('src-tauri/resource-lock.json', 'utf8'));
const resourceErrors = [];
for (const item of resourceLock.resources.filter((entry) => entry.packagedPath)) {
  const path = resolve('src-tauri', item.packagedPath);
  if (!existsSync(path)) resourceErrors.push(`${item.id}: missing ${path}`);
  else {
    if (statSync(path).size !== item.bytes) resourceErrors.push(`${item.id}: byte length mismatch`);
    if (sha256(path) !== item.sha256) resourceErrors.push(`${item.id}: sha256 mismatch`);
  }
}

const forbiddenJsonBinary = [];
for (const path of walk('src', (file) => /\.[jt]sx?$/.test(file))) {
  const text = readFileSync(path, 'utf8');
  if (/Array\.from\s*\([^)]*(?:pixel|rgba|feature|weight)/i.test(text) || /JSON\.stringify\s*\([^)]*(?:pixel|rgba|feature|weight)/i.test(text)) forbiddenJsonBinary.push(path);
}

const endSnapshot = spawnSync(`node scripts/snapshot-tree.mjs --check ${snapshotPath}`, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
const snapshot = {
  startOk: startSnapshot.status === 0,
  endOk: endSnapshot.status === 0,
  startOutput: (startSnapshot.stdout || startSnapshot.stderr || '').trim(),
  endOutput: (endSnapshot.stdout || endSnapshot.stderr || '').trim(),
};

const ok = snapshot.startOk && snapshot.endOk
  && results.every((result) => result.ok)
  && unresolved.length === 0
  && openBlockers.length === 0
  && residualRisks.length === 0
  && presentForbiddenPaths.length === 0
  && presentForbiddenDependencies.length === 0
  && placeholderFindings.length === 0
  && resourceErrors.length === 0
  && forbiddenJsonBinary.length === 0;

console.log(JSON.stringify({
  ok,
  results,
  manifest: { entries: entries.length, unresolved, openBlockers, residualRisks },
  cleanup: { presentForbiddenPaths, presentForbiddenDependencies, placeholderFindings, forbiddenJsonBinary },
  resources: { errors: resourceErrors },
  snapshot,
}, null, 2));
if (!ok) process.exit(1);
