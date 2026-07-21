import { flushSync } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import type { InferenceUiResult } from '@generated/bindings';
import { useMultiTaskDetection } from '../useMultiTaskDetection';

const disposers: Array<() => void> = [];
function result(goodProbability: number | null, personFound = true): InferenceUiResult {
  return {
    requestId: 1,
    token: 2,
    personFound,
    bbox: null,
    keypoints: null,
    classification: { presentProbability: personFound ? 0.9 : 0.1, goodProbability },
  };
}
function mount(initial: InferenceUiResult | null) {
  const state = $state({ result: initial });
  let hook!: ReturnType<typeof useMultiTaskDetection>;
  const dispose = $effect.root(() => {
    hook = useMultiTaskDetection(() => state.result);
  });
  disposers.push(dispose);
  flushSync();
  return { hook, rerender(next: InferenceUiResult | null) { state.result = next; flushSync(); } };
}
afterEach(() => { while (disposers.length) disposers.pop()?.(); });

const allFalse = {
  person_found: false,
  slouching: false,
  forward_neck_tilt: false,
  hand_near_face: false,
  mouth_open: false,
};

function expectCompleteBooleanShape(value: ReturnType<typeof useMultiTaskDetection>['detection']): void {
  expect(Object.keys(value ?? {}).sort()).toEqual([
    'forward_neck_tilt',
    'hand_near_face',
    'mouth_open',
    'person_found',
    'slouching',
  ]);
  for (const field of Object.values(value ?? {})) expect(typeof field).toBe('boolean');
}

describe('useMultiTaskDetection native UI mapping', () => {
  it('initializes null input to the complete safe-default shape', () => {
    const { hook } = mount(null);
    expect(hook.detection).toEqual(allFalse);
    expectCompleteBooleanShape(hook.detection);
  });

  it('does not fabricate dedicated cues from posture probability', () => {
    const { hook } = mount(result(0.49));
    expect(hook.detection).toMatchObject({
      person_found: true,
      slouching: false,
      forward_neck_tilt: false,
    });
    expectCompleteBooleanShape(hook.detection);
  });

  it('keeps dedicated cue flags false regardless of posture probability (no threshold exists)', () => {
    // The cue flags are not derived from goodProbability; there is no >=/< 0.5 rule.
    // Probe both sides of a hypothetical tie plus the extreme to prove no derivation flips them.
    for (const p of [0, 0.5, 1]) {
      const { hook } = mount(result(p));
      expect(hook.detection).toMatchObject({
        person_found: true,
        slouching: false,
        forward_neck_tilt: false,
        hand_near_face: false,
        mouth_open: false,
      });
      expectCompleteBooleanShape(hook.detection);
    }
  });

  it('does not invent posture output without a person or probability', () => {
    const { hook } = mount(result(0.2, false));
    expect(hook.detection).toEqual(allFalse);
    expectCompleteBooleanShape(hook.detection);
  });

  it('reacts to successive native results and clears stale flags on null', () => {
    const harness = mount(result(0.8));
    expect(harness.hook.detection?.person_found).toBe(true);
    expect(harness.hook.detection?.slouching).toBe(false);
    harness.rerender(result(0.2));
    expect(harness.hook.detection?.person_found).toBe(true);
    expect(harness.hook.detection?.slouching).toBe(false);
    expectCompleteBooleanShape(harness.hook.detection);
    harness.rerender(null);
    expect(harness.hook.detection).toEqual(allFalse);
    expectCompleteBooleanShape(harness.hook.detection);
  });

  it('maps a changed personFound field to person_found across live updates', () => {
    const harness = mount(result(0.8, true));
    expect(harness.hook.detection?.person_found).toBe(true);
    harness.rerender(result(0.2, false));
    expect(harness.hook.detection).toEqual(allFalse);
    expectCompleteBooleanShape(harness.hook.detection);
    harness.rerender(result(0.9, true));
    expect(harness.hook.detection?.person_found).toBe(true);
    expectCompleteBooleanShape(harness.hook.detection);
  });
});
