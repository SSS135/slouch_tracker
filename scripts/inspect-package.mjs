import { createHash } from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, resolve } from 'node:path';

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const sleep = (ms) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
function walk(directory, output = []) {
  if (!existsSync(directory)) return output;
  for (const name of readdirSync(directory).sort()) {
    const path = resolve(directory, name);
    const info = statSync(path);
    if (info.isDirectory()) walk(path, output);
    else output.push(path);
  }
  return output;
}

const lock = JSON.parse(readFileSync('src-tauri/resource-lock.json', 'utf8'));
const config = JSON.parse(readFileSync('src-tauri/tauri.conf.json', 'utf8'));
const errors = [];
const targets = config.bundle?.targets;
if (!(targets === 'nsis' || (Array.isArray(targets) && targets.length === 1 && targets[0] === 'nsis'))) errors.push('Windows migration package target must be exactly nsis');

const nsisRoot = resolve('src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis');
const installers = walk(nsisRoot).filter((path) => path.toLowerCase().endsWith('.exe'));
if (installers.length !== 1) errors.push(`expected exactly one NSIS installer, found ${installers.length}`);

const installDir = mkdtempSync(resolve(tmpdir(), 'slouch-package-'));
const appDataDir = mkdtempSync(resolve(tmpdir(), 'slouch-package-data-'));
let installedFiles = [];
try {
  if (installers.length === 1) {
    const install = spawnSync(installers[0], ['/S', `/D=${installDir}`], { encoding: 'utf8', timeout: 180000 });
    if (install.status !== 0) errors.push(`silent NSIS install failed: ${install.status} ${(install.stderr ?? '').trim()}`);
    installedFiles = walk(installDir);
  }

  for (const item of lock.resources.filter((entry) => entry.packagedPath)) {
    const installedPath = resolve(installDir, item.packagedPath);
    if (!existsSync(installedPath)) errors.push(`${item.id}: missing installed path ${item.packagedPath}`);
    else {
      if (statSync(installedPath).size !== item.bytes) errors.push(`${item.id}: installed byte length mismatch`);
      if (sha256(installedPath) !== item.sha256) errors.push(`${item.id}: installed hash mismatch`);
    }
  }

  const productionBinaries = installedFiles.filter((path) => /\.(?:exe|dll)$/i.test(path) && !/^uninstall/i.test(basename(path)));
  for (const path of productionBinaries) {
    const bytes = readFileSync(path);
    const ascii = bytes.toString('latin1').toLowerCase();
    const utf16 = bytes.toString('utf16le').toLowerCase();
    for (const marker of ['tauri_plugin_wdio', 'remote-debugging-port=', 'slouch-devbuild-marker']) if (ascii.includes(marker) || utf16.includes(marker)) errors.push(`${basename(path)} contains forbidden production marker ${marker}`);
  }

  const appExe = productionBinaries.find((path) => path.toLowerCase().endsWith('.exe'));
  if (!appExe) errors.push('installed application executable missing');
  else {
    const systemRoot = process.env.SystemRoot || 'C:\\Windows';
    const child = spawn(appExe, [], { env: { ...process.env, PATH: `${systemRoot}\\System32`, SLOUCH_APP_DATA_DIR: appDataDir }, stdio: 'ignore' });
    await sleep(5000);
    if (child.exitCode !== null) errors.push(`installed app exited during minimal-PATH smoke: ${child.exitCode}`);
    else child.kill();
  }

  const uninstall = installedFiles.find((path) => /^uninstall.*\.exe$/i.test(basename(path)));
  if (!uninstall) errors.push('NSIS uninstaller missing');
  else {
    const removed = spawnSync(uninstall, ['/S'], { encoding: 'utf8', timeout: 120000 });
    if (removed.status !== 0) errors.push(`silent uninstall failed: ${removed.status} ${(removed.stderr ?? '').trim()}`);
    for (let attempt = 0; attempt < 20 && walk(installDir).length > 0; attempt += 1) await sleep(500);
    const leftovers = walk(installDir);
    if (leftovers.length > 0) errors.push(`silent uninstall left ${leftovers.length} installed files`);
  }
} finally {
  rmSync(installDir, { recursive: true, force: true });
  rmSync(appDataDir, { recursive: true, force: true });
}

console.log(JSON.stringify({ ok: errors.length === 0, installer: installers[0] ?? null, installedFiles: installedFiles.length, errors }, null, 2));
if (errors.length) process.exit(1);
