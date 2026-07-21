import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

const rootDirectory = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [svelte({ preprocess: vitePreprocess({ script: true }) })],
  publicDir: 'public-svelte',
  resolve: {
    alias: {
      '@': path.resolve(rootDirectory, 'src-svelte'),
      '@generated': path.resolve(rootDirectory, 'src/generated'),
    },
  },
  build: {
    outDir: 'web-dist-svelte-harness',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        plumbing: path.resolve(rootDirectory, 'index.svelte-harness.html'),
        application: path.resolve(rootDirectory, 'index.svelte-app-harness.html'),
      },
    },
  },
  base: './',
  preview: {
    port: 4174,
  },
});
