import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative } from 'node:path';

const root = process.cwd();
const sourceRoot = join(root, 'src');
const manifestPath = join(root, '.plans', 'per-file-port-manifest.json');
const workflowPlan = '.plans/bun-style-per-file-rewrite-plan.md';
const sourcePattern = /\.(?:ts|tsx|js|jsx)$/;

const toPosix = (value) => value.replaceAll('\\', '/');
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const snake = (value) => value
  .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
  .replace(/[^A-Za-z0-9]+/g, '_')
  .replace(/^_+|_+$/g, '')
  .toLowerCase();

function walk(dir, output = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, output);
    else if (sourcePattern.test(entry.name)) output.push(path);
  }
  return output;
}

function rustTarget(source) {
  let crate;
  if (source.startsWith('src/services/ml/')) crate = 'slouch-ml';
  else if (source.startsWith('src/services/dataset/')) crate = 'slouch-store';
  else if (source.startsWith('src/workers/inference') || source.includes('/createInferenceWorker')) crate = 'slouch-vision';
  else if (source.startsWith('src/workers/training')) crate = 'slouch-ml';
  else crate = 'slouch-domain';

  const isTest = source.includes('/__tests__/') || /\.test\.[jt]sx?$/.test(source);
  const relativeSource = source
    .replace(/^src\/(?:services\/(?:ml|dataset|posture|validation|image)|workers)\//, '')
    .replace(/\.[^.]+$/, '');
  const segments = relativeSource.split('/').filter((part) => part !== '__tests__').map(snake);
  const file = `${segments.pop() || 'port'}.rs`;
  return isTest
    ? `src-tauri/crates/${crate}/tests/ported/${[...segments, file].join('/')}`
    : `src-tauri/crates/${crate}/src/ported/${[...segments, file].join('/')}`;
}

function svelteTarget(source) {
  const rel = source.replace(/^src\//, '');
  if (source === 'src/main.tsx') return 'src-svelte/main.ts';
  if (source === 'src/App.tsx') return 'src-svelte/App.svelte';
  if (source.endsWith('.d.ts')) return `src-svelte/${rel}`;
  if (/\.test\.tsx$/.test(source)) return `src-svelte/${rel.replace(/\.test\.tsx$/, '.test.ts')}`;
  if (source.startsWith('src/contexts/') && source.endsWith('.tsx')) return `src-svelte/${rel.replace(/\.tsx$/, '.svelte.ts')}`;
  if (source.startsWith('src/hooks/') && source.endsWith('.ts')) return `src-svelte/${rel.replace(/\.ts$/, '.svelte.ts')}`;
  if (source.endsWith('.tsx')) return `src-svelte/${rel.replace(/\.tsx$/, '.svelte')}`;
  return `src-svelte/${rel}`;
}

function classify(source) {
  if (source.startsWith('src/services/ml/')) return 'rust';
  if (source.startsWith('src/services/posture/')) return 'rust';
  if (source.startsWith('src/services/validation/')) return 'rust';
  if (source === 'src/services/types.ts') return 'rust';
  if (source.startsWith('src/services/dataset/') && !source.includes('thumbnailGenerator')) return 'rust';
  if (source.startsWith('src/workers/')) return 'rust';
  if (source.endsWith('featureCloning.ts') || source.endsWith('featureCloning.test.ts')) return 'rust';
  return 'svelte';
}

function imports(content) {
  const found = new Set();
  const regex = /(?:from\s+|import\s*\()(['"])([^'"]+)\1/g;
  for (const match of content.matchAll(regex)) found.add(match[2]);
  return [...found].sort();
}

function baseLayer(source) {
  if (source.startsWith('src/services/posture/') || source.startsWith('src/services/validation/') || source === 'src/services/types.ts') return { order: 0, name: 'domain-contracts' };
  if (source.startsWith('src/services/ml/') && /(?:constants|types|asyncUtils|backend|adamw|sgd|layerNorm|binning|pca|randomProjection|kmeans|classWeights|featureValidation|tensorCleanup)/.test(source)) return { order: 1, name: 'numeric-primitives' };
  if (source.startsWith('src/services/ml/') && /(?:Features|feature|Extraction|Extractor)/i.test(source)) return { order: 2, name: 'feature-pipeline' };
  if (source.startsWith('src/services/ml/')) return { order: 3, name: 'classifiers-training' };
  if (source.startsWith('src/services/dataset/')) return { order: 4, name: 'storage-archive' };
  if (source.startsWith('src/workers/')) return { order: 5, name: 'workers-native-boundary' };
  if (source.startsWith('src/hooks/') || source.startsWith('src/contexts/') || source.startsWith('src/providers/')) return { order: 7, name: 'svelte-state-lifecycle' };
  if (source.startsWith('src/components/unified/') || source.startsWith('src/pages/') || source === 'src/App.tsx' || source === 'src/main.tsx') return { order: 9, name: 'svelte-composition-entry' };
  if (source.startsWith('src/components/')) return { order: 8, name: 'svelte-components' };
  return { order: 6, name: 'shared-media-utilities' };
}

function resolveImport(source, specifier, sourceSet) {
  if (!specifier.startsWith('.')) return undefined;
  const base = toPosix(join(dirname(source), specifier));
  const candidates = [
    base,
    `${base}.ts`, `${base}.tsx`, `${base}.js`, `${base}.jsx`,
    `${base}/index.ts`, `${base}/index.tsx`, `${base}/index.js`, `${base}/index.jsx`,
  ];
  return candidates.find((candidate) => sourceSet.has(candidate));
}

const files = walk(sourceRoot).map((path) => toPosix(relative(root, path))).sort();
const sourceSet = new Set(files);
const entries = files.map((source) => {
  const content = readFileSync(join(root, source));
  const contentText = content.toString('utf8');
  const category = classify(source);
  const target = category === 'rust' ? rustTarget(source) : svelteTarget(source);
  const suffix = target.endsWith('.svelte.ts') ? '.svelte.ts'
    : target.endsWith('.d.ts') ? '.d.ts'
      : extname(target);
  const sourceHash = sha256(content);
  const id = `${snake(source.replace(/^src\//, '').replace(/\.[^.]+$/, ''))}-${sourceHash.slice(0, 8)}`;
  const highRisk = /(?:inference-worker|training-worker|classifier|model|storage|archive|import|export|camera|frameProcessor|datasetOperations)/i.test(source);
  const layer = baseLayer(source);
  const importSpecifiers = imports(contentText);
  const dependencySources = importSpecifiers
    .map((specifier) => resolveImport(source, specifier, sourceSet))
    .filter(Boolean);
  return {
    id,
    source,
    sourceSha256: sourceHash,
    category,
    target,
    stagingPath: `.plans/port-output/${id}/target${suffix}`,
    imports: importSpecifiers,
    dependencySources,
    baseLayer: layer.name,
    baseLayerOrder: layer.order,
    risk: highRisk ? 'high' : 'normal',
    model: 'openai-codex/gpt-5.6-luna:high',
  };
});

const bySource = new Map(entries.map((entry) => [entry.source, entry]));
for (const entry of entries) {
  entry.dependencyEntryIds = entry.dependencySources.map((source) => bySource.get(source)?.id).filter(Boolean);
}

// Kahn-style dependency ordering with architectural priority. All entries stay
// in one logical rewrite wave, but later batches can read completed dependency
// staging outputs. Cycles are broken deterministically one entry at a time.
const remaining = new Set(entries.map((entry) => entry.id));
let batch = 0;
while (remaining.size > 0) {
  let ready = entries.filter((entry) => remaining.has(entry.id)
    && entry.dependencyEntryIds.every((dependencyId) => !remaining.has(dependencyId)));
  if (ready.length === 0) {
    ready = entries
      .filter((entry) => remaining.has(entry.id))
      .sort((a, b) => a.baseLayerOrder - b.baseLayerOrder || a.source.localeCompare(b.source))
      .slice(0, 1);
  } else {
    const minimumLayer = Math.min(...ready.map((entry) => entry.baseLayerOrder));
    ready = ready.filter((entry) => entry.baseLayerOrder === minimumLayer);
  }
  ready.sort((a, b) => Number(a.source.includes('/__tests__/')) - Number(b.source.includes('/__tests__/')) || a.source.localeCompare(b.source));
  for (const entry of ready) {
    entry.rewriteBatch = batch;
    remaining.delete(entry.id);
  }
  batch += 1;
}
entries.sort((a, b) => a.rewriteBatch - b.rewriteBatch || a.baseLayerOrder - b.baseLayerOrder || a.source.localeCompare(b.source));

const targetSet = new Set();
const errors = [];
for (const entry of entries) {
  if (targetSet.has(entry.target)) errors.push(`duplicate target: ${entry.target}`);
  targetSet.add(entry.target);
}
if (errors.length) throw new Error(errors.join('\n'));

const manifest = {
  version: 1,
  workflowPlan,
  sourceGlob: 'src/**/*.{ts,tsx,js,jsx}',
  sourceCount: entries.length,
  rewriteBatchCount: Math.max(...entries.map((entry) => entry.rewriteBatch)) + 1,
  ordering: 'internal-import dependency order, then domain → numeric → features → classifiers → storage → workers → shared UI → state → components → entry',
  model: 'openai-codex/gpt-5.6-luna:high',
  concurrency: 8,
  maxAgents: 800,
  entries,
};
const serialized = `${JSON.stringify(manifest, null, 2)}\n`;

if (process.argv.includes('--check')) {
  if (!existsSync(manifestPath)) throw new Error(`missing ${toPosix(relative(root, manifestPath))}`);
  const current = readFileSync(manifestPath, 'utf8');
  if (current !== serialized) throw new Error('per-file port manifest is stale; regenerate before workflow launch');
  console.log(JSON.stringify({ ok: true, sourceCount: entries.length, manifest: toPosix(relative(root, manifestPath)) }));
} else {
  mkdirSync(dirname(manifestPath), { recursive: true });
  writeFileSync(manifestPath, serialized);
  console.log(JSON.stringify({ ok: true, sourceCount: entries.length, manifest: toPosix(relative(root, manifestPath)) }));
}
