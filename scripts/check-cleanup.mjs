import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const forbiddenPaths = [
  'src/workers', 'src/services/ml', 'src/services/dataset', 'src/services/validation', 'src/services/posture',
  'public/rtmdet-nano.onnx', 'public/rtmpose-m.onnx', 'public/rtmpose-s.onnx',
];
const forbiddenDependencies = [
  'react', 'react-dom', '@vitejs/plugin-react',
  '@mantine/core', '@mantine/hooks', '@mantine/notifications', '@emotion/react', '@emotion/styled',
  '@tanstack/react-query', '@testing-library/react', '@types/react', '@types/react-dom',
  '@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-wasm',
  'onnxruntime-web', 'localforage', 'zod', 'seedrandom', 'ml-pca', 'jszip', 'fflate', 'file-saver',
];
const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
const dependencies = { ...(packageJson.dependencies ?? {}), ...(packageJson.devDependencies ?? {}) };
const errors = [];
for (const path of forbiddenPaths) if (existsSync(path)) errors.push(`forbidden path remains: ${path}`);
for (const name of forbiddenDependencies) if (name in dependencies) errors.push(`forbidden dependency remains: ${name}`);
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
for (const path of walk('src-tauri').filter((file) => file.endsWith('.rs'))) {
  const text = readFileSync(path, 'utf8');
  if (/\b(?:todo|unimplemented)!\s*\(|#!?\s*\[allow\s*\(|\bTODO\b/.test(text)) errors.push(`placeholder/suppression remains: ${path}`);
}
const frontendFiles = [...walk('src'), ...walk('src-svelte')].filter((file) => /\.(?:[jt]sx?|svelte)$/.test(file));
for (const path of frontendFiles) {
  const text = readFileSync(path, 'utf8');
  if (path.endsWith('.tsx') || /from\s+['"](?:react|react-dom|@mantine\/|@emotion\/|@tanstack\/react-query)/.test(text)) errors.push(`React/Mantine source remains: ${path}`);
  if (/Array\.from\s*\([^)]*(?:pixel|rgba|feature|weight)/i.test(text) || /JSON\.stringify\s*\([^)]*(?:pixel|rgba|feature|weight)/i.test(text)) errors.push(`possible JSON binary transport: ${path}`);
  if (/\b(?:it|test|describe)\.skip\s*\(/.test(text)) errors.push(`skipped test remains: ${path}`);
}
console.log(JSON.stringify({ ok: errors.length === 0, errors }, null, 2));
if (errors.length) process.exit(1);
