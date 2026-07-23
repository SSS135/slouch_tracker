import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InferenceUiResult } from '@generated/bindings';
import PostureCamera from '../PostureCamera.svelte';
import * as useNativeCameraModule from '../../hooks/useNativeCamera.svelte';
import * as useCanvasRendererModule from '../../hooks/useCanvasRenderer.svelte';
import * as useWindowAspectModule from '../../hooks/useWindowAspect.svelte';
import { createMockNativeInferenceResult } from '../../__tests__/utils/mockNativeInferenceResult';
import { drawHumanLikeSkeleton, drawKeypointOverlay, drawDetectionBox } from '@/utils/canvasDrawing';

vi.mock('../../hooks/useNativeCamera.svelte');
vi.mock('../../hooks/useCanvasRenderer.svelte');
vi.mock('../../hooks/useWindowAspect.svelte');
vi.mock('@/utils/canvasDrawing', () => ({
  drawHumanLikeSkeleton: vi.fn(),
  drawKeypointOverlay: vi.fn(),
  drawDetectionBox: vi.fn(),
}));
vi.mock('../../services/logging', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

const mockCanvasRef = { current: document.createElement('canvas') };
const mockOnInferenceResult = vi.fn();
const mockOnFps = vi.fn();
const mockRetry = vi.fn().mockResolvedValue(undefined);

let capturedOnResult: ((result: InferenceUiResult) => void) | undefined;

function mockCamera(overrides: { error?: string | null; detectionFps?: number } = {}): void {
  vi.spyOn(useNativeCameraModule, 'useNativeCamera').mockImplementation((options) => {
    capturedOnResult = options.onResult;
    return {
      ready: true,
      error: overrides.error ?? null,
      detectionFps: overrides.detectionFps ?? 0,
      retry: mockRetry,
    };
  });
}

function renderCamera() {
  return render(PostureCamera, {
    props: {
      onInferenceResult: mockOnInferenceResult,
      onFps: mockOnFps,
    },
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(useWindowAspectModule, 'useWindowAspect').mockReturnValue(undefined);
  vi.spyOn(useCanvasRendererModule, 'useCanvasRenderer').mockReturnValue({
    canvasRef: mockCanvasRef,
    isRendering: false,
    isCanvasReady: false,
    isForeground: true,
  });
  mockCamera();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('PostureCamera native camera view', () => {
  it('shows no error banner during normal operation', () => {
    renderCamera();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('surfaces a camera error with a working retry', async () => {
    mockCamera({ error: 'no capture device' });
    renderCamera();
    expect(screen.getByRole('alert')).toHaveTextContent('Camera unavailable');
    expect(screen.getByRole('alert')).toHaveTextContent('no capture device');
    await fireEvent.click(screen.getByRole('button', { name: 'Retry camera' }));
    expect(mockRetry).toHaveBeenCalledTimes(1);
  });

  it('forwards pushed inference results to the consumer', async () => {
    renderCamera();
    const result = createMockNativeInferenceResult({ requestId: 5, token: 55 });
    capturedOnResult?.(result);
    await waitFor(() => expect(mockOnInferenceResult).toHaveBeenCalledWith(result));
  });

  it('reports the detection FPS to the consumer', () => {
    mockCamera({ detectionFps: 1.5 });
    renderCamera();
    expect(mockOnFps).toHaveBeenCalledWith(1.5);
  });

  it('invokes onBackgroundClick when the bare video area is clicked', async () => {
    const onBackgroundClick = vi.fn();
    const { container } = render(PostureCamera, {
      props: { onInferenceResult: mockOnInferenceResult, onFps: mockOnFps, onBackgroundClick },
    });
    const canvasContainer = container.querySelector('.canvas-container');
    expect(canvasContainer).not.toBeNull();
    await fireEvent.click(canvasContainer!);
    expect(onBackgroundClick).toHaveBeenCalledTimes(1);
  });

  it('passes the processed-view flag and both protocol URLs to the renderer', () => {
    render(PostureCamera, {
      props: {
        onInferenceResult: mockOnInferenceResult,
        onFps: mockOnFps,
        processedView: true,
      },
    });
    const rendererMock = vi.mocked(useCanvasRendererModule.useCanvasRenderer);
    expect(rendererMock).toHaveBeenCalled();
    const options = rendererMock.mock.calls[0][0];
    expect(options.processedView).toBe(true);
    expect(options.frameUrl).toMatch(/\/frame$/);
    expect(options.processedFrameUrl).toMatch(/\/processed$/);
  });

  it('defaults the processed view off', () => {
    renderCamera();
    const rendererMock = vi.mocked(useCanvasRendererModule.useCanvasRenderer);
    expect(rendererMock.mock.calls[0][0].processedView).toBe(false);
  });

  it('passes the detection-overlay flag to the renderer', () => {
    render(PostureCamera, {
      props: { onInferenceResult: mockOnInferenceResult, onFps: mockOnFps, showDetectionOverlay: true },
    });
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    expect(options.showDetectionOverlay).toBe(true);
  });

  it('defaults the detection overlay off', () => {
    renderCamera();
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    expect(options.showDetectionOverlay).toBe(false);
  });

  it('passes the debug-tiles URL and preprocessing debug flag to the renderer', () => {
    render(PostureCamera, {
      props: {
        onInferenceResult: mockOnInferenceResult,
        onFps: mockOnFps,
        processedView: true,
        preprocessingDebugView: true,
      },
    });
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    expect(options.debugTilesFrameUrl).toMatch(/\/debug-tiles$/);
    expect(options.preprocessingDebugView).toBe(true);
  });

  it('defaults the preprocessing debug view off', () => {
    renderCamera();
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    expect(options.preprocessingDebugView).toBe(false);
  });

  it('passes the inferred-frame URL and a detection sequence to the renderer', () => {
    render(PostureCamera, {
      props: { onInferenceResult: mockOnInferenceResult, onFps: mockOnFps, showDetectionOverlay: true },
    });
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    expect(options.inferredFrameUrl).toMatch(/\/inferred$/);
    expect(typeof options.detectionSequence).toBe('function');
  });

  it('advances the detection sequence once per inference result (detection-cadence display)', () => {
    render(PostureCamera, {
      props: { onInferenceResult: mockOnInferenceResult, onFps: mockOnFps, showDetectionOverlay: true },
    });
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    const seq0 = options.detectionSequence!();
    capturedOnResult?.(createMockNativeInferenceResult({ requestId: 1, token: 1 }));
    const seq1 = options.detectionSequence!();
    capturedOnResult?.(createMockNativeInferenceResult({ requestId: 2, token: 2 }));
    const seq2 = options.detectionSequence!();
    expect(seq1).toBe(seq0 + 1);
    expect(seq2).toBe(seq1 + 1);
  });

  it('draws raw keypoint dots+lines and the box (not the avatar) when overlay on, privacy off', () => {
    render(PostureCamera, {
      props: {
        onInferenceResult: mockOnInferenceResult,
        onFps: mockOnFps,
        showDetectionOverlay: true,
        privacyMode: false,
      },
    });
    capturedOnResult?.(createMockNativeInferenceResult());
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    const canvas = { width: 640, height: 480 } as HTMLCanvasElement;
    options.onDraw?.({} as unknown as CanvasRenderingContext2D, canvas);
    expect(drawKeypointOverlay).toHaveBeenCalled();
    expect(drawDetectionBox).toHaveBeenCalled();
    expect(drawHumanLikeSkeleton).not.toHaveBeenCalled();
  });

  it('draws nothing over the video when the overlay is off and privacy is off', () => {
    renderCamera();
    capturedOnResult?.(createMockNativeInferenceResult());
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    const canvas = { width: 640, height: 480 } as HTMLCanvasElement;
    options.onDraw?.({} as unknown as CanvasRenderingContext2D, canvas);
    expect(drawKeypointOverlay).not.toHaveBeenCalled();
    expect(drawHumanLikeSkeleton).not.toHaveBeenCalled();
    expect(drawDetectionBox).not.toHaveBeenCalled();
  });

  it('keeps the human-like avatar in privacy mode and adds only the box when overlay on', () => {
    render(PostureCamera, {
      props: {
        onInferenceResult: mockOnInferenceResult,
        onFps: mockOnFps,
        showDetectionOverlay: true,
        privacyMode: true,
      },
    });
    capturedOnResult?.(createMockNativeInferenceResult());
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    const canvas = { width: 640, height: 480 } as HTMLCanvasElement;
    options.onDraw?.({} as unknown as CanvasRenderingContext2D, canvas);
    expect(drawHumanLikeSkeleton).toHaveBeenCalled();
    expect(drawDetectionBox).toHaveBeenCalled();
    expect(drawKeypointOverlay).not.toHaveBeenCalled();
  });

  it('keeps the avatar only (no box, no dots) in privacy mode with the overlay off', () => {
    render(PostureCamera, {
      props: {
        onInferenceResult: mockOnInferenceResult,
        onFps: mockOnFps,
        showDetectionOverlay: false,
        privacyMode: true,
      },
    });
    capturedOnResult?.(createMockNativeInferenceResult());
    const options = vi.mocked(useCanvasRendererModule.useCanvasRenderer).mock.calls[0][0];
    const canvas = { width: 640, height: 480 } as HTMLCanvasElement;
    options.onDraw?.({} as unknown as CanvasRenderingContext2D, canvas);
    expect(drawHumanLikeSkeleton).toHaveBeenCalled();
    expect(drawKeypointOverlay).not.toHaveBeenCalled();
    expect(drawDetectionBox).not.toHaveBeenCalled();
  });
});
