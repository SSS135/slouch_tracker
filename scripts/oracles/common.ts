import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import os from 'node:os';

export const root = resolve(import.meta.dirname, '../..');
export const packageLockSha256 = sha256File('package-lock.json');

export function sha256(data: Uint8Array | string): string {
  return createHash('sha256').update(data).digest('hex');
}

export function sha256File(path: string): string {
  return sha256(readFileSync(resolve(root, path)));
}

export function source(path: string): { path: string; sha256: string } {
  return { path, sha256: sha256File(path) };
}

export type Fixture = Record<string, unknown>;

export function envelope(
  fixtureId: string,
  generatorPath: string,
  sources: string[],
  backend: string,
  cases: unknown[],
  extra: Record<string, unknown> = {},
): Fixture {
  return {
    schemaVersion: 1,
    fixtureId,
    sources: sources.map(source),
    generator: {
      path: generatorPath,
      sha256: sha256File(generatorPath),
      command: `npx tsx ${generatorPath} --write`,
    },
    environment: {
      os: os.platform(),
      arch: os.arch(),
      node: process.version.slice(1),
      packageLockSha256,
      backend,
      endianness: os.endianness().toLowerCase(),
      threads: 1,
      graphOptimization: 'all',
      rng: 'seedrandom 3.0.5 ARC4 where applicable',
      seedEncoding: 'JavaScript Number.prototype.toString',
    },
    tolerances: {
      numeric: { absolute: 0.000002, relative: 0.000002 },
      iterative: { absolute: 0.0002 },
      exact: ['ids', 'labels', 'shapes', 'assignments', 'decisionsAwayFromThreshold'],
    },
    ...extra,
    cases,
  };
}

export function jsonBytes(value: unknown): Buffer {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

export function writeOrCheck(path: string, value: unknown, write: boolean): void {
  const absolute = resolve(root, path);
  const bytes = Buffer.isBuffer(value) ? value : jsonBytes(value);
  if (write) {
    mkdirSync(dirname(absolute), { recursive: true });
    writeFileSync(absolute, bytes);
    return;
  }
  if (!existsSync(absolute)) throw new Error(`missing committed oracle: ${path}`);
  const committed = readFileSync(absolute);
  if (!committed.equals(bytes)) throw new Error(`stale committed oracle: ${path}`);
}

export function f32(values: Iterable<number>): number[] {
  return Array.from(values, Math.fround);
}

export function bits(values: Iterable<number>): number[] {
  const array = Float32Array.from(values);
  return Array.from(new Uint32Array(array.buffer));
}

export function encodeObserved(value: unknown): unknown {
  if (typeof value === 'number' && !Number.isFinite(value)) {
    if (Number.isNaN(value)) return 'NaN';
    return value > 0 ? '+Infinity' : '-Infinity';
  }
  if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number>, encodeObserved);
  if (Array.isArray(value)) return value.map(encodeObserved);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, encodeObserved(entry)]));
  }
  return value;
}

export function observe(action: () => unknown): Record<string, unknown> {
  try {
    return { ok: true, value: encodeObserved(action()) };
  } catch (error) {
    return {
      ok: false,
      errorName: error instanceof Error ? error.name : typeof error,
      errorMessage: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function observeAsync(action: () => Promise<unknown>): Promise<Record<string, unknown>> {
  try {
    return { ok: true, value: encodeObserved(await action()) };
  } catch (error) {
    return {
      ok: false,
      errorName: error instanceof Error ? error.name : typeof error,
      errorMessage: error instanceof Error ? error.message : String(error),
    };
  }
}

export function isMain(metaUrl: string): boolean {
  return Boolean(process.argv[1]) && metaUrl === pathToFileURL(resolve(process.argv[1])).href;
}

export function parseWriteFlag(): boolean {
  const args = process.argv.slice(2);
  if (args.length !== 1 || !['--write', '--check'].includes(args[0])) {
    throw new Error('expected exactly --write or --check');
  }
  return args[0] === '--write';
}
