import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const lock = JSON.parse(readFileSync(resolve(root, 'src-tauri/resource-lock.json'), 'utf8'));
const archive = lock.resources.find((item) => item.id === 'onnxruntime-archive');
const members = lock.resources.filter((item) => item.archiveMember);
const temp = mkdtempSync(resolve(tmpdir(), 'slouch-ort-'));
const zip = resolve(temp, 'onnxruntime.zip');
const extracted = resolve(temp, 'extracted');
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

try {
  execFileSync('curl.exe', ['-L', '--fail', '--silent', '--show-error', '-o', zip, archive.source], { stdio: 'inherit' });
  if (statSync(zip).size !== archive.bytes || sha256(zip) !== archive.sha256) throw new Error('ONNX Runtime archive does not match resource lock');
  execFileSync('powershell.exe', ['-NoProfile', '-Command', `Expand-Archive -LiteralPath '${zip.replaceAll("'", "''")}' -DestinationPath '${extracted.replaceAll("'", "''")}' -Force`], { stdio: 'inherit' });
  for (const item of members) {
    const source = resolve(extracted, item.archiveMember);
    if (!existsSync(source) || statSync(source).size !== item.bytes || sha256(source) !== item.sha256) throw new Error(`resource mismatch: ${item.id}`);
    const destination = resolve(root, 'src-tauri', item.packagedPath);
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(source, destination);
  }
  for (const item of lock.resources.filter((entry) => entry.source?.startsWith('public/'))) {
    const source = resolve(root, item.source);
    if (statSync(source).size !== item.bytes || sha256(source) !== item.sha256) throw new Error(`model mismatch: ${item.id}`);
    const destination = resolve(root, 'src-tauri', item.packagedPath);
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(source, destination);
  }
  console.log(JSON.stringify({ ok: true, copied: lock.resources.filter((item) => item.packagedPath).map((item) => item.packagedPath) }, null, 2));
} finally {
  rmSync(temp, { recursive: true, force: true });
}
