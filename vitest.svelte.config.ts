import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

const rootDirectory = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [svelte({ preprocess: vitePreprocess({ script: true }) })],
  resolve: {
    conditions: ['browser'],
    alias: {
      '@': path.resolve(rootDirectory, 'src-svelte'),
      '@generated': path.resolve(rootDirectory, 'src/generated'),
    },
  },
  test: {
    globals: true,
    isolate: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.svelte.setup.ts'],
    include: [
      'src-svelte/**/*.test.ts',
      'src-svelte/**/*.test.svelte.ts',
      'src/generated/**/*.test.ts',
    ],
    exclude: ['node_modules/**', 'web-dist*/**'],
    pool: 'forks',
    maxWorkers: 1,
    fileParallelism: false,
    testTimeout: 5000,
    hookTimeout: 5000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'web-dist*/**',
        '**/*.test.ts',
        '**/*.test.svelte.ts',
        '**/__tests__/**',
      ],
    },
  },
});
