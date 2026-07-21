import { readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join, relative } from 'node:path';

const root = 'src-svelte';
const excludedDirectories = new Set(['__tests__', 'harness']);
const sourceExtensions = new Set(['.js', '.ts', '.svelte']);
const checks = [
  ['React or retired component framework import', /(?:from\s+|import\s*)['"](?:react(?:-dom)?|@mantine\/|@emotion\/|@tanstack\/react-query)/],
  ['browser persistence fallback', /\b(?:localStorage|sessionStorage|indexedDB|localforage)\b/],
  ['browser ML runtime', /(?:onnxruntime-web|@tensorflow\/)/],
  ['browser worker runtime', /\bnew\s+(?:Shared)?Worker\s*\(|(?:from\s+|import\s*)['"][^'"]*\/workers(?:\/|['"])/],
  ['legacy React/core source import', /(?:from\s+|import\s*)['"][^'"]*(?:\.\.\/)+src\//],
  ['retired browser archive/download runtime', /(?:from\s+|import\s*)['"](?:jszip|file-saver|fflate)['"]/],
  ['placeholder native operation', /\b(?:unsupported|unimplemented|placeholder)\b/i],
];

function collect(directory, files = []) {
  for (const name of readdirSync(directory).sort()) {
    const path = join(directory, name);
    const info = statSync(path);
    if (info.isDirectory()) {
      if (!excludedDirectories.has(name)) collect(path, files);
    } else if (
      info.isFile()
      && sourceExtensions.has(extname(name))
      && !name.includes('.test.')
    ) {
      files.push(path);
    }
  }
  return files;
}

const findings = [];
for (const file of collect(root)) {
  const source = readFileSync(file, 'utf8');
  for (const [description, pattern] of checks) {
    if (pattern.test(source)) findings.push({ file: relative('.', file).replaceAll('\\', '/'), description });
  }
}

const packageJson = readFileSync('package.json', 'utf8');
if (/@sveltejs\/kit/.test(packageJson)) findings.push({ file: 'package.json', description: 'SvelteKit is forbidden' });

if (findings.length > 0) {
  console.error(JSON.stringify({ ok: false, findings }, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({ ok: true, scannedFiles: collect(root).length }, null, 2));
