import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Packaged-native E2E harness (Windows x64).
 *
 * Drives the devbuild binary (`npm run tauri:build:dev:win`) through
 * `@wdio/tauri-service` with the `embedded` driver provider: the app itself
 * hosts a W3C WebDriver server via the devbuild-gated
 * `tauri-plugin-wdio-webdriver` crate, so no tauri-driver or msedgedriver is
 * required and driver/browser version management disappears entirely.
 *
 * App data is isolated into a per-run directory via SLOUCH_APP_DATA_DIR.
 * After every spec file the app process is terminated (gracefully first),
 * which both isolates specs and makes the persistence pair
 * (30-persistence-setup -> 31-persistence-restart-verify) run against a
 * genuine process restart: the service health-check relaunches the binary
 * with the same environment before the next session starts.
 */

const rootDir = dirname(fileURLToPath(import.meta.url));
const appBinary = join(
  rootDir,
  'src-tauri',
  'target',
  'x86_64-pc-windows-msvc',
  'release',
  'app.exe',
);
const dataDir = join(rootDir, 'test-results', 'native-e2e-data');

const psQuotedBinary = appBinary.replace(/'/g, "''");

function runPowerShell(script: string): string {
  try {
    return execFileSync(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', script],
      { encoding: 'utf8', timeout: 30_000 },
    );
  } catch {
    return '';
  }
}

function devAppPids(): number[] {
  const output = runPowerShell(
    `Get-Process | Where-Object { $_.Path -eq '${psQuotedBinary}' } | ForEach-Object { $_.Id }`,
  );
  return output
    .split(/\r?\n/)
    .map((line) => Number.parseInt(line.trim(), 10))
    .filter((pid) => Number.isInteger(pid) && pid > 0);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Close the devbuild app: WM_CLOSE first (clean SQLite shutdown), then force. */
async function terminateDevApp(): Promise<void> {
  if (devAppPids().length === 0) {
    return;
  }
  runPowerShell(
    `Get-Process | Where-Object { $_.Path -eq '${psQuotedBinary}' } | ForEach-Object { $null = $_.CloseMainWindow() }`,
  );
  const deadline = Date.now() + 8_000;
  while (Date.now() < deadline) {
    if (devAppPids().length === 0) {
      return;
    }
    await sleep(250);
  }
  runPowerShell(
    `Get-Process | Where-Object { $_.Path -eq '${psQuotedBinary}' } | Stop-Process -Force`,
  );
  const forceDeadline = Date.now() + 5_000;
  while (Date.now() < forceDeadline && devAppPids().length > 0) {
    await sleep(250);
  }
}

export const config: WebdriverIO.Config = {
  runner: 'local',

  // Explicit order: the persistence pair (30 -> 31) must run in sequence.
  specs: [
    './e2e/native/10-launch-readiness.e2e.ts',
    './e2e/native/20-raw-ipc-inference.e2e.ts',
    './e2e/native/30-persistence-setup.e2e.ts',
    './e2e/native/31-persistence-restart-verify.e2e.ts',
    './e2e/native/40-dialog-substitute-lifecycle-errors.e2e.ts',
    './e2e/native/50-real-camera-smoke.e2e.ts',
  ],
  maxInstances: 1,

  capabilities: [
    {
      browserName: 'tauri',
      'tauri:options': {
        application: appBinary,
      },
    },
  ],

  services: [
    [
      '@wdio/tauri-service',
      {
        driverProvider: 'embedded',
        // First launch loads a 54 MiB pose model synchronously during setup.
        startTimeout: 180_000,
        statusPollTimeout: 5_000,
        env: {
          SLOUCH_APP_DATA_DIR: dataDir,
        },
      },
    ],
  ],

  logLevel: 'warn',
  waitforTimeout: 30_000,
  connectionRetryTimeout: 180_000,
  connectionRetryCount: 2,

  framework: 'mocha',
  reporters: ['spec'],
  mochaOpts: {
    ui: 'bdd',
    timeout: 600_000,
  },

  onPrepare: async () => {
    if (!existsSync(appBinary)) {
      throw new Error(
        `Devbuild binary not found at ${appBinary}. Run "npm run tauri:build:dev:win" first.`,
      );
    }
    await terminateDevApp();
    rmSync(dataDir, { recursive: true, force: true });
    mkdirSync(dataDir, { recursive: true });
  },

  // Kill the app between spec files so every spec starts against a freshly
  // launched process (the tauri-service health check relaunches it) and the
  // persistence pair proves durability across a real restart.
  onWorkerEnd: async () => {
    await terminateDevApp();
  },

  onComplete: async () => {
    await terminateDevApp();
  },

  before: async () => {
    // Generous script budget: raw infer_frame calls block on native inference.
    await browser.setTimeout({ script: 120_000, pageLoad: 60_000, implicit: 0 });
  },
};
