import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  cleanup();
});

if (typeof globalThis.structuredClone === 'undefined') {
  globalThis.structuredClone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
}

if (typeof globalThis.ImageData === 'undefined') {
  globalThis.ImageData = class ImageData {
    readonly colorSpace: PredefinedColorSpace = 'srgb';
    readonly data: Uint8ClampedArray;
    readonly height: number;
    readonly width: number;

    constructor(
      dataOrWidth: number | Uint8ClampedArray,
      widthOrHeight: number,
      height?: number,
    ) {
      if (typeof dataOrWidth === 'number') {
        this.width = dataOrWidth;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      } else {
        this.data = dataOrWidth;
        this.width = widthOrHeight;
        this.height = height ?? 0;
      }
    }
  } as typeof ImageData;
}

Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  value: vi.fn((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  arc: vi.fn(),
  beginPath: vi.fn(),
  clearRect: vi.fn(),
  closePath: vi.fn(),
  drawImage: vi.fn(),
  fill: vi.fn(),
  fillRect: vi.fn(),
  fillText: vi.fn(),
  getImageData: vi.fn(() => new ImageData(1, 1)),
  lineTo: vi.fn(),
  measureText: vi.fn(() => ({ width: 0 })),
  moveTo: vi.fn(),
  putImageData: vi.fn(),
  restore: vi.fn(),
  save: vi.fn(),
  scale: vi.fn(),
  setTransform: vi.fn(),
  stroke: vi.fn(),
  translate: vi.fn(),
})) as unknown as typeof HTMLCanvasElement.prototype.getContext;

HTMLCanvasElement.prototype.toBlob = vi.fn((callback) => {
  callback(new Blob(['mock-thumbnail'], { type: 'image/webp' }));
});

if (typeof HTMLDialogElement !== 'undefined' && !HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function showModal(): void {
    this.setAttribute('open', '');
  };
  HTMLDialogElement.prototype.close = function close(): void {
    this.removeAttribute('open');
    this.dispatchEvent(new Event('close'));
  };
}
