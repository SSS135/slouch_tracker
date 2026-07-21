import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

if (!existsSync('src/generated/bindings.ts')) {
  console.error('src/generated/bindings.ts is missing');
  process.exit(1);
}
const run = spawnSync('cargo test --manifest-path src-tauri/Cargo.toml -p app --test bindings_freshness -- --exact generated_bindings_are_fresh', { shell: true, stdio: 'inherit' });
process.exit(run.status ?? 1);
