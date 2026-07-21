import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const excludedDirs = new Set([
  '.git',
  '.pi-subagents',
  'node_modules',
  'target',
  'web-dist',
  'web-dist-svelte',
  'web-dist-svelte-harness',
  'test-results',
  'playwright-report',
  'dist',
  'coverage',
]);
const excludedFiles = new Set(['.DS_Store']);
const excludedEvidence = new Set(['.plans/rust-refactor-acceptance.json']);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const mode = process.argv[2];
const file = process.argv[3] ? resolve(root, process.argv[3]) : null;

function walk(directory, output = {}) {
  for (const name of readdirSync(directory).sort()) {
    if (excludedDirs.has(name) || excludedFiles.has(name)) continue;
    const absolute = resolve(directory, name);
    const projectPath = relative(root, absolute).replaceAll('\\', '/');
    if (file && absolute === file) continue;
    // src-tauri/gen is tauri-build output whose content depends on enabled cargo
    // features (devbuild adds the WDIO plugins to the ACL manifests), so builds
    // with different feature sets rewrite it back and forth; fingerprinting it
    // would make gate drift-detection nondeterministic.
    if (excludedEvidence.has(projectPath) || projectPath === 'src-tauri/reviews' || projectPath.startsWith('src-tauri/reviews/') || projectPath === 'src-tauri/gen' || projectPath.startsWith('src-tauri/gen/')) continue;
    const info = statSync(absolute);
    if (info.isDirectory()) walk(absolute, output);
    else if (info.isFile()) output[relative(root, absolute).replaceAll('\\', '/')] = sha256(readFileSync(absolute));
  }
  return output;
}

const files = walk(root);
const gitPath = resolve(root, '.git');
if (existsSync(gitPath)) {
  const gitInfo = statSync(gitPath);
  if (gitInfo.isFile()) files['$git/worktree-link'] = sha256(readFileSync(gitPath));
  else {
    for (const name of ['HEAD', 'index', 'config', 'packed-refs']) {
      const path = resolve(gitPath, name);
      if (existsSync(path) && statSync(path).isFile()) files[`$git/${name}`] = sha256(readFileSync(path));
    }
    const refs = resolve(gitPath, 'refs');
    if (existsSync(refs)) {
      const refFiles = {};
      walk(refs, refFiles);
      for (const [path, hash] of Object.entries(refFiles)) {
        const refPath = relative(refs, resolve(root, path)).replaceAll('\\', '/');
        if (refPath === 'pi-checkpoints' || refPath.startsWith('pi-checkpoints/')) continue;
        files[`$git/${refPath}`] = hash;
      }
    }
  }
}
const digest = sha256(Buffer.from(JSON.stringify(files)));

if (mode === '--write' && file) {
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, `${JSON.stringify({ version: 1, digest, files }, null, 2)}\n`);
  console.log(JSON.stringify({ ok: true, mode: 'write', file, digest, count: Object.keys(files).length }));
} else if (mode === '--check' && file) {
  if (!existsSync(file)) {
    console.error(JSON.stringify({ ok: false, error: `snapshot missing: ${file}` }));
    process.exit(1);
  }
  const expected = JSON.parse(readFileSync(file, 'utf8'));
  const changed = [...new Set([...Object.keys(expected.files ?? {}), ...Object.keys(files)])]
    .filter((path) => expected.files?.[path] !== files[path]);
  console.log(JSON.stringify({ ok: changed.length === 0, mode: 'check', expected: expected.digest, actual: digest, changed }, null, 2));
  if (changed.length) process.exit(1);
} else {
  console.log(JSON.stringify({ ok: true, digest, count: Object.keys(files).length }));
}
