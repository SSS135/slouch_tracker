import type { CapturedFrame } from './useFrameSampler';

type Thumbnail = CapturedFrame['thumbnail'] | string | null;
type ReactiveValue<T> = T | (() => T);

function readReactive<T>(value: ReactiveValue<T>): T {
  return typeof value === 'function' ? (value as () => T)() : value;
}

export interface ThumbnailUrlState {
  readonly url: string | undefined;
}

/**
 * Manages thumbnail URL generation and cleanup.
 *
 * Blob URLs are created only after the Blob is ready and are revoked when the
 * thumbnail changes or the owning component is destroyed. Pass a getter when
 * the thumbnail is reactive Svelte state.
 */
export function useThumbnailUrl(
  thumbnail: ReactiveValue<Thumbnail>,
): ThumbnailUrlState {
  let url = $state<string | undefined>(undefined);

  $effect(() => {
    const currentThumbnail = readReactive(thumbnail);

    if (!currentThumbnail) {
      url = undefined;
      return;
    }

    if (typeof currentThumbnail === 'string') {
      url = currentThumbnail;
      return;
    }

    if (currentThumbnail.size === 0) {
      url = undefined;
      return;
    }

    const objectUrl = URL.createObjectURL(currentThumbnail);
    url = objectUrl;

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  });

  return {
    get url() {
      return url;
    },
  };
}
