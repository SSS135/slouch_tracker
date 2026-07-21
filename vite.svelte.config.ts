import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { svelte, vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

const rootDirectory = path.dirname(fileURLToPath(import.meta.url));
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [svelte({ preprocess: vitePreprocess({ script: true }), prebundleSvelteLibraries: false })],
  publicDir: 'public-svelte',
  resolve: {
    alias: {
      '@': path.resolve(rootDirectory, 'src-svelte'),
      '@generated': path.resolve(rootDirectory, 'src/generated'),
    },
  },
  optimizeDeps: {
    entries: ['index.svelte.html'],
    include: [
      '@msgpack/msgpack',
      '@tauri-apps/api/core',
      '@tauri-apps/api/event',
      '@tauri-apps/api/window',
    ],
  },
  build: {
    outDir: 'web-dist-svelte',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: path.resolve(rootDirectory, 'index.svelte.html'),
      output: {
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  base: './',
  clearScreen: false,
  server: {
    host: host || '127.0.0.1',
    port: 5174,
    strictPort: true,
    open: false,
    watch: { ignored: ['**/src-tauri/**'] },
  },
  preview: {
    port: 5174,
  },
});
