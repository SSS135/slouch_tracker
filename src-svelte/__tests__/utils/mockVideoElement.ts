import { vi } from 'vitest';
import { THUMBNAIL_RESOLUTION } from '@/services/ml/constants';

export function createMockVideoElement(overrides?: Partial<HTMLVideoElement>): HTMLVideoElement {
  const mockVideo = {
    videoWidth: THUMBNAIL_RESOLUTION.width,
    videoHeight: THUMBNAIL_RESOLUTION.height,
    width: THUMBNAIL_RESOLUTION.width,
    height: THUMBNAIL_RESOLUTION.height,
    currentTime: 0,
    duration: 100,
    paused: false,
    ended: false,
    muted: false,
    volume: 1,
    playbackRate: 1,
    readyState: 4,
    networkState: 2,
    src: 'mock://video.mp4',
    // HTMLMediaElement readyState constants
    HAVE_NOTHING: 0,
    HAVE_METADATA: 1,
    HAVE_CURRENT_DATA: 2,
    HAVE_FUTURE_DATA: 3,
    HAVE_ENOUGH_DATA: 4,
    play: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
    pause: vi.fn(),
    load: vi.fn(),
    canPlayType: vi.fn((type: string) => {
      if (type.includes('video/')) return 'probably';
      return '';
    }),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
    ...overrides,
  } as unknown as HTMLVideoElement;

  return mockVideo;
}

export function createMockCanvasElement(width: number = 160, height: number = 120): HTMLCanvasElement {
  const mockContext = {
    canvas: null as any,
    fillRect: vi.fn(),
    clearRect: vi.fn(),
    getImageData: vi.fn(),
    putImageData: vi.fn(),
    createImageData: vi.fn(),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
    save: vi.fn(),
    fillText: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
    transform: vi.fn(),
    rect: vi.fn(),
    clip: vi.fn(),
    fillStyle: '#000000',
    strokeStyle: '#000000',
    lineWidth: 1,
    lineCap: 'butt',
    lineJoin: 'miter',
    globalAlpha: 1,
    globalCompositeOperation: 'source-over',
  };

  const mockCanvas = {
    width,
    height,
    getContext: vi.fn((type: string) => {
      if (type === '2d') {
        mockContext.canvas = mockCanvas;
        return mockContext;
      }
      return null;
    }),
    toDataURL: vi.fn(() => 'data:image/webp;base64,mockBase64Data'),
    toBlob: vi.fn((callback: (blob: Blob | null) => void) => {
      const mockBlob = new Blob(['mock data'], { type: 'image/webp' });
      callback(mockBlob);
    }),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as HTMLCanvasElement;

  return mockCanvas;
}

export function mockDocument() {
  const originalCreateElement = document.createElement.bind(document);

  vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
    if (tagName.toLowerCase() === 'canvas') {
      return createMockCanvasElement() as any;
    }
    if (tagName.toLowerCase() === 'video') {
      return createMockVideoElement() as any;
    }
    return originalCreateElement(tagName);
  });
}

export function restoreDocument() {
  if (vi.isMockFunction(document.createElement)) {
    (document.createElement as any).mockRestore();
  }
}
