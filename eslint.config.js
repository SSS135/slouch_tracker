import tseslint from '@typescript-eslint/eslint-plugin';
import tsparser from '@typescript-eslint/parser';
import sveltePlugin from 'eslint-plugin-svelte';

export default [
  {
    ignores: [
      'dist/**',
      'web-dist-svelte/**',
      'web-dist-svelte-harness/**',
      'node_modules/**',
      'src-tauri/**',
      'coverage/**',
      'src/generated/bindings.generated.ts',
      '**/*.config.js',
      '**/*.config.ts',
    ],
  },
  {
    files: ['**/*.{js,mjs,cjs,ts}'],
    languageOptions: {
      parser: tsparser,
      ecmaVersion: 'latest',
      sourceType: 'module',
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
    },
  },
  ...sveltePlugin.configs['flat/recommended'],
  {
    files: ['src-svelte/**/*.svelte'],
    languageOptions: {
      parserOptions: {
        parser: tsparser,
        extraFileExtensions: ['.svelte'],
      },
    },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
    },
  },
  {
    files: ['src-svelte/**/*.svelte.ts'],
    languageOptions: {
      parser: tsparser,
      parserOptions: {
        sourceType: 'module',
      },
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
    },
  },
];
