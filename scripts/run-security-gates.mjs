import { spawnSync } from 'node:child_process';

const commands = [
  'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-store --test schema_security',
  'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-store --test archive_security',
  'cargo test --manifest-path src-tauri/Cargo.toml -p app --test ipc_security',
  'cargo test --manifest-path src-tauri/Cargo.toml -p app --test actor_contracts',
  'cargo test --manifest-path src-tauri/Cargo.toml -p slouch-ml --test model_format_security',
];
const results = commands.map((command) => {
  const run = spawnSync(command, { shell: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  return { command, ok: run.status === 0, status: run.status, stderrTail: (run.stderr ?? '').trim().split(/\r?\n/).slice(-20) };
});
console.log(JSON.stringify({ ok: results.every((result) => result.ok), results }, null, 2));
if (results.some((result) => !result.ok)) process.exit(1);
