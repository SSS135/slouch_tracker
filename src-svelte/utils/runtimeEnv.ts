type ElectronBridge = {
  isElectron?: boolean;
  platform?: string;
};

type GlobalWithElectron = typeof globalThis & {
  electron?: ElectronBridge;
  ELECTRON_MODE?: boolean;
};

interface RuntimeEnv {
  readonly isElectron: boolean;
  readonly platform: string | null;
  readonly baseUrl: string;
}

const FALLBACK_LOCATION = 'http://localhost/';

function getRuntimeEnv(): RuntimeEnv {
  const globalScope = globalThis as GlobalWithElectron;
  const bridge = globalScope.electron;
  const isElectron = Boolean(globalScope.ELECTRON_MODE ?? bridge?.isElectron);
  const platform = bridge?.platform ?? null;

  let baseUrl = '/';
  const importMeta = typeof window !== 'undefined' && (window as any).__VITE_BASE_URL__;
  if (importMeta) {
    baseUrl = importMeta;
  }

  return {
    isElectron,
    platform,
    baseUrl: baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`,
  };
}

/**
 * Resolves asset URLs against the configured Vite base URL for DOM builds.
 */
export function resolveAssetUrl(relativePath: string): string {
  const normalizedPath = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath;
  const { baseUrl } = getRuntimeEnv();
  const locationHref =
    (typeof self !== 'undefined' && typeof self.location?.href === 'string'
      ? self.location.href
      : FALLBACK_LOCATION) as string;

  return new URL(normalizedPath, new URL(baseUrl, locationHref)).toString();
}
