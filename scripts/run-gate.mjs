import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const COMMUNITY_VCVARS =
  'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat';

// Locate vcvars64.bat portably: an explicit VCVARS64 override wins; otherwise
// ask vswhere for the latest install that ships the x64 MSVC toolset; finally
// fall back to the well-known Community install path.
function resolveVcvars() {
  const override = (process.env.VCVARS64 ?? '').trim();
  if (override) {
    return override;
  }
  const programFilesX86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
  const vswhere = `${programFilesX86}\\Microsoft Visual Studio\\Installer\\vswhere.exe`;
  if (existsSync(vswhere)) {
    const query = spawnSync(
      vswhere,
      [
        '-latest',
        '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath',
      ],
      { encoding: 'utf8' },
    );
    const installationPath = (query.stdout ?? '').trim().split(/\r?\n/)[0]?.trim();
    if (query.status === 0 && installationPath) {
      const candidate = `${installationPath}\\VC\\Auxiliary\\Build\\vcvars64.bat`;
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return COMMUNITY_VCVARS;
}

const VCVARS64 = resolveVcvars();

const vs2022X64 = (command) =>
  `call "${VCVARS64}" >nul && set "CARGO_INCREMENTAL=0" && ${command}`;

const sveltePreCutoverCommands = [
  'npm run check:svelte',
  'npm run check:svelte:plumbing',
  'npm run lint:svelte',
  'npm run test:svelte',
  'npm run build:svelte',
  'npm run build:svelte:harness',
  'npm run test:e2e:web',
  vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p app --test bindings_freshness -- --exact generated_bindings_are_fresh'),
  'npm run scan:svelte:runtime',
];

const gates = {
  'plan-approval': ['node scripts/validate-plan-approval.mjs'],
  'lockfiles': [
    'npm install --package-lock-only --ignore-scripts',
    'cargo generate-lockfile --manifest-path src-tauri/Cargo.toml',
  ],
  'dependencies': [
    'npm ci',
    'npx playwright install chromium',
    'cargo fetch --manifest-path src-tauri/Cargo.toml --locked',
  ],
  'refresh-inputs': ['node scripts/write-manifest-inputs.mjs'],
  'manifest': ['node scripts/validate-port-manifest.mjs'],
  'trial': [
    'cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check',
    'cargo clippy --manifest-path src-tauri/Cargo.toml -p slouch-domain -p slouch-ml --all-targets --all-features -- -D warnings',
    'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-domain -p slouch-ml --all-features',
    'npx tsx scripts/check-wave1-oracles.ts',
    'npm run test:fast -- src/services/ml/__tests__/rtmposeFeatures.test.ts src/services/ml/__tests__/naiveBayesClassifier.test.ts src/services/ml/__tests__/kmeansLogisticClassifier.test.ts src/services/ml/__tests__/classifierFactory.test.ts src/services/validation/__tests__/schemas.test.ts src/services/dataset/__tests__/featureRegistry.test.ts src/services/validation/__tests__/guards.test.ts src/services/posture/__tests__/detection.test.ts',
  ],
  // The TypeScript-side oracle regeneration check ('npx tsx scripts/oracles/generate-all.ts --check')
  // was removed at migration cutover: it imports the deleted src/services TS sources and the
  // cleanup-forbidden tfjs/seedrandom/onnxruntime-web dependencies, so it is definitionally a
  // pre-cutover check (its last pre-cutover run was green). The committed fixtures remain frozen;
  // the cargo tests below validate the native implementation and fixture provenance against them.
  'compatibility-oracles': [
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-ml --test compatibility_oracles'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-store --test compatibility_oracles'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-vision --test inference_parity --all-features'),
  ],
  'native-runtime': [
    'node scripts/fetch-onnxruntime.mjs',
    'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-vision',
    'npm run tauri:build:dev:win',
    'npm run package:inspect',
  ],
  'smoke': [
    'cargo test --manifest-path src-tauri/Cargo.toml -p app --test smoke',
  ],
  'golden-parity': [
    'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-ml --test golden_parity',
    'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-vision --test inference_parity',
  ],
  'rust-tests': [
    'cargo test --manifest-path src-tauri/Cargo.toml --workspace',
  ],
  'rust-workspace': [
    vs2022X64('cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check'),
    vs2022X64('cargo check --locked --manifest-path src-tauri/Cargo.toml --workspace --all-targets --all-features'),
    vs2022X64('cargo clippy --locked --manifest-path src-tauri/Cargo.toml --workspace --all-targets --all-features -- -D warnings'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml --workspace --all-targets --all-features'),
  ],
  'native-api': [
    vs2022X64('cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check'),
    'npx tsc --noEmit --target ES2022 --module ESNext --moduleResolution Bundler --strict --skipLibCheck --types vitest/globals,node,vite/client src/generated/bindings.ts src/generated/bindings.generated.ts src/generated/bindings.contract.test.ts',
    'npx vitest run src/generated/bindings.contract.test.ts --pool=forks --maxWorkers=1',
    vs2022X64('cargo check --locked --manifest-path src-tauri/Cargo.toml -p app -p slouch-store -p slouch-vision --all-targets --all-features'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-store --lib ported::storage::tests::'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-store --test native_model_format'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p slouch-vision --lib'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p app --lib api::tests::'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p app --lib actors::tests::'),
    vs2022X64('cargo test --locked --manifest-path src-tauri/Cargo.toml -p app --test bindings_freshness -- --exact generated_bindings_are_fresh'),
    vs2022X64('cargo clippy --locked --manifest-path src-tauri/Cargo.toml -p app -p slouch-store -p slouch-vision --all-targets --all-features -- -D warnings'),
  ],
  'frontend': [
    'npm run typecheck',
    'npm run lint',
    'npm run test:fast',
    'npm run build:web',
    'npm run bindings:check',
  ],
  'svelte-pre-cutover': sveltePreCutoverCommands,
  // Compatibility alias for evidence produced before the acceptance matrix name was restored.
  'svelte-integration': sveltePreCutoverCommands,
  'browser-e2e': ['npm run test:e2e:web'],
  'native-e2e': ['npm run test:e2e:native'],
  'security-data': ['npm run test:security'],
  'cleanup': ['node scripts/check-cleanup.mjs'],
  'windows-package': ['npm run tauri:build:win', 'npm run package:inspect'],
  'acceptance': ['npm run verify:migration'],
  'acceptance-final': ['npm run acceptance:check', 'npm run verify:migration'],
};
const allowedChanges = {
  lockfiles: new Set(['package-lock.json', 'src-tauri/Cargo.lock']),
  'refresh-inputs': new Set(['src-tauri/port-manifest.json']),
  'native-runtime': new Set([
    'src-tauri/resources/onnxruntime/windows-x86_64/onnxruntime.dll',
    'src-tauri/resources/onnxruntime/notices/LICENSE',
    'src-tauri/resources/onnxruntime/notices/Privacy.md',
    'src-tauri/resources/onnxruntime/notices/ThirdPartyNotices.txt',
    'src-tauri/resources/models/rtmdet-nano.onnx',
    'src-tauri/resources/models/rtmpose-m.onnx',
  ]),
};

const gate = process.argv[2];
if (!gate || !(gate in gates) || process.argv.length !== 3) {
  console.error(JSON.stringify({ ok: false, error: 'unknown or malformed gate id', allowed: Object.keys(gates) }, null, 2));
  process.exit(2);
}

const snapshotPath = `.pi-subagents/gates/${gate}-before.json`;
const before = spawnSync(`node scripts/snapshot-tree.mjs --write ${snapshotPath}`, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
const results = [];
for (const command of gates[gate]) {
  const run = spawnSync(command, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  results.push({
    command,
    ok: run.status === 0,
    status: run.status,
    stdoutTail: (run.stdout ?? '').trim().split(/\r?\n/).slice(-30),
    stderrTail: (run.stderr ?? '').trim().split(/\r?\n/).slice(-30),
  });
}
const after = spawnSync(`node scripts/snapshot-tree.mjs --check ${snapshotPath}`, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
let changed = [];
let inputFingerprint = '';
let outputFingerprint = '';
try { inputFingerprint = JSON.parse(before.stdout || before.stderr).digest ?? ''; } catch { changed.push('starting snapshot output unreadable'); }
try {
  const snapshot = JSON.parse(after.stdout || after.stderr);
  changed.push(...(snapshot.changed ?? []));
  outputFingerprint = snapshot.actual ?? '';
} catch { changed.push('ending snapshot output unreadable'); }
const allowed = allowedChanges[gate] ?? new Set();
const unexpectedChanges = changed.filter((path) => !allowed.has(path));
const expectedChanges = changed.filter((path) => allowed.has(path));
const ok = before.status === 0 && results.every((result) => result.ok) && unexpectedChanges.length === 0;
console.log(JSON.stringify({ ok, gate, inputFingerprint, outputFingerprint, results, expectedChanges, unexpectedChanges }, null, 2));
if (!ok) process.exit(1);
