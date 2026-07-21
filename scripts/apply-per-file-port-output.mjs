import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const root = process.cwd();
const manifestPath = join(root, '.plans', 'per-file-port-manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const errors = [];
const targets = new Set();

for (const entry of manifest.entries) {
  const sourcePath = join(root, entry.source);
  const stagingPath = join(root, entry.stagingPath);
  const targetPath = join(root, entry.target);
  if (!existsSync(sourcePath)) {
    errors.push(`missing source: ${entry.source}`);
    continue;
  }
  if (sha256(readFileSync(sourcePath)) !== entry.sourceSha256) errors.push(`source changed: ${entry.source}`);
  if (targets.has(entry.target)) errors.push(`duplicate target: ${entry.target}`);
  targets.add(entry.target);
  if (!existsSync(stagingPath)) {
    errors.push(`missing staged output: ${entry.stagingPath}`);
    continue;
  }
  const output = readFileSync(stagingPath, 'utf8');
  if (!output.trim()) errors.push(`empty staged output: ${entry.stagingPath}`);
  if (/\b(?:TODO|FIXME)\b|\btodo!\s*\(|\bunimplemented!\s*\(/.test(output)) errors.push(`placeholder in staged output: ${entry.stagingPath}`);
  if (entry.category === 'svelte' && /from\s+['"](?:react|react-dom|@mantine\/|@emotion\/|@tanstack\/react-query)/.test(output)) errors.push(`forbidden React import: ${entry.stagingPath}`);
  if (/JSON\.stringify\s*\([^)]*(?:pixel|rgba|feature|weight)/i.test(output)) errors.push(`possible JSON binary transport: ${entry.stagingPath}`);
  entry.outputSha256 = sha256(output);
  entry.resolvedTargetPath = targetPath;
}

if (errors.length) {
  console.error(JSON.stringify({ ok: false, errors }, null, 2));
  process.exit(1);
}

if (process.argv.includes('--check')) {
  console.log(JSON.stringify({ ok: true, entries: manifest.entries.length, mode: 'check' }));
  process.exit(0);
}

for (const entry of manifest.entries) {
  const targetPath = join(root, entry.target);
  mkdirSync(dirname(targetPath), { recursive: true });
  copyFileSync(join(root, entry.stagingPath), targetPath);
}

console.log(JSON.stringify({ ok: true, entries: manifest.entries.length, mode: 'apply' }));
