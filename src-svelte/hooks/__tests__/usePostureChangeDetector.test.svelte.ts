import { flushSync } from "svelte";
import { describe, expect, it, vi } from "vitest";
import {
  usePostureChangeDetector,
  type PostureChangeDetectorConfig,
} from "../usePostureChangeDetector";
import type { ClassificationResult } from "@/services/types";
import { FrameLabel } from "@/services/dataset/types";

type PostureLabel = FrameLabel.GOOD | FrameLabel.BAD | FrameLabel.AWAY;
type DetectorState = PostureChangeDetectorConfig & {
  classification: ClassificationResult | null;
};
type DetectorHarness = {
  rerender: (changes: Partial<DetectorState>) => void;
  unmount: () => void;
};

const mountedHarnesses: DetectorHarness[] = [];

function mountDetector(
  classification: ClassificationResult | null,
  config: Omit<PostureChangeDetectorConfig, "onCapture"> & {
    onCapture: PostureChangeDetectorConfig["onCapture"];
  },
): DetectorHarness {
  const state = $state<DetectorState>({ classification, ...config });
  let disposed = false;

  const dispose = $effect.root(() => {
    usePostureChangeDetector(() => state.classification, state);
  });
  flushSync();

  const harness: DetectorHarness = {
    rerender: (changes) => {
      Object.assign(state, changes);
      flushSync();
    },
    unmount: () => {
      if (!disposed) {
        disposed = true;
        dispose();
        flushSync();
      }
    },
  };

  mountedHarnesses.push(harness);
  return harness;
}

function createClassification(
  prediction: FrameLabel.GOOD | FrameLabel.BAD,
): ClassificationResult {
  return {
    goodProbability: prediction === FrameLabel.GOOD ? 0.9 : 0.1,
    presentProbability: 0.95,
  };
}

function createAwayClassification(): ClassificationResult {
  return {
    goodProbability: null,
    presentProbability: 0.1,
  };
}

describe("usePostureChangeDetector", () => {
  let mockNow = 0;
  let mockOnCapture: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    mockNow = 0;
    vi.spyOn(Date, "now").mockImplementation(() => mockNow);
    mockOnCapture = vi.fn<(label: PostureLabel) => void>();
  });

  afterEach(() => {
    while (mountedHarnesses.length > 0) {
      mountedHarnesses.pop()?.unmount();
    }
    vi.runOnlyPendingTimers();
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("should not trigger on initial classification", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.unmount();
  });

  it("should not trigger when classification is null", () => {
    const harness = mountDetector(null, {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: null });
    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.unmount();
  });

  it("should trigger on good -> bad transition", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    harness.unmount();
  });

  it("should trigger on bad -> good transition", () => {
    const harness = mountDetector(createClassification(FrameLabel.BAD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    harness.unmount();
  });

  it("should trigger on good -> away transition", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);
    harness.unmount();
  });

  it("should trigger on bad -> away transition", () => {
    const harness = mountDetector(createClassification(FrameLabel.BAD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);
    harness.unmount();
  });

  it("should trigger on away -> good transition", () => {
    const harness = mountDetector(createAwayClassification(), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    harness.unmount();
  });

  it("should trigger on away -> bad transition", () => {
    const harness = mountDetector(createAwayClassification(), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    harness.unmount();
  });

  it("should enforce per-posture-type cooldown - same posture type blocked", () => {
    const cooldownMs = 2000;
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    mockNow += cooldownMs - 200 + 100;
    vi.advanceTimersByTime(cooldownMs - 200 + 100);

    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);

    harness.unmount();
  });

  it("should respect enabled flag", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      enabled: false,
      onCapture: mockOnCapture,
    });

    harness.rerender({
      classification: createClassification(FrameLabel.BAD),
      enabled: false,
    });
    expect(mockOnCapture).not.toHaveBeenCalled();

    harness.rerender({
      classification: createClassification(FrameLabel.BAD),
      enabled: true,
    });
    harness.rerender({
      classification: createClassification(FrameLabel.GOOD),
      enabled: true,
    });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);

    harness.unmount();
  });

  it("should only trigger once per transition (reset after trigger)", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    mockOnCapture.mockClear();

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    harness.unmount();
  });

  it("should allow rapid transitions between DIFFERENT posture types", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);

    harness.unmount();
  });

  it("should use default config when not provided", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);

    harness.unmount();
  });

  it("should handle classification going from value to null", () => {
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: null });
    expect(mockOnCapture).not.toHaveBeenCalled();

    harness.unmount();
  });

  it("should handle classification going from null to value", () => {
    const harness = mountDetector(null, {
      cooldownMs: 2000,
      onCapture: mockOnCapture,
    });

    expect(mockOnCapture).not.toHaveBeenCalled();
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledTimes(1);
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);

    harness.unmount();
  });

  it("should handle complex multi-state sequence with per-posture cooldowns", () => {
    const cooldownMs = 2000;
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    mockOnCapture.mockClear();

    mockNow += 500;
    vi.advanceTimersByTime(500);
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);
    mockOnCapture.mockClear();

    mockNow += 500;
    vi.advanceTimersByTime(500);
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    mockNow += 500;
    vi.advanceTimersByTime(500);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    mockOnCapture.mockClear();

    mockNow += 600;
    vi.advanceTimersByTime(600);
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);

    harness.unmount();
  });

  it("should maintain independent cooldowns for all three posture types", () => {
    const cooldownMs = 2000;
    const harness = mountDetector(createClassification(FrameLabel.GOOD), {
      cooldownMs,
      onCapture: mockOnCapture,
    });

    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.BAD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);
    mockOnCapture.mockClear();

    mockNow += 100;
    vi.advanceTimersByTime(100);
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.AWAY);
    mockOnCapture.mockClear();

    mockNow += 300;
    vi.advanceTimersByTime(300);
    harness.rerender({ classification: createClassification(FrameLabel.BAD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    mockNow += 200;
    vi.advanceTimersByTime(200);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).not.toHaveBeenCalled();

    mockNow += 200;
    vi.advanceTimersByTime(200);
    harness.rerender({ classification: createAwayClassification() });
    expect(mockOnCapture).not.toHaveBeenCalled();

    mockNow += 1200;
    vi.advanceTimersByTime(1200);
    harness.rerender({ classification: createClassification(FrameLabel.GOOD) });
    expect(mockOnCapture).toHaveBeenCalledWith(FrameLabel.GOOD);

    harness.unmount();
  });
});
