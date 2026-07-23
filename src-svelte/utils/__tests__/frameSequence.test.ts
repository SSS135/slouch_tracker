import { describe, expect, it } from 'vitest';
import { isNewerFrame, MonotonicFrameGate } from '../frameSequence';

describe('isNewerFrame', () => {
  it('accepts a strictly greater sequence', () => {
    expect(isNewerFrame(4, 5)).toBe(true);
  });

  it('rejects an equal sequence', () => {
    expect(isNewerFrame(5, 5)).toBe(false);
  });

  it('rejects an older sequence', () => {
    expect(isNewerFrame(5, 4)).toBe(false);
  });
});

describe('MonotonicFrameGate', () => {
  it('admits the first frame regardless of value', () => {
    const gate = new MonotonicFrameGate();
    expect(gate.admit(0)).toBe(true);
    expect(gate.last).toBe(0);
  });

  it('admits strictly increasing sequences and advances', () => {
    const gate = new MonotonicFrameGate();
    expect(gate.admit(1)).toBe(true);
    expect(gate.admit(2)).toBe(true);
    expect(gate.admit(3)).toBe(true);
    expect(gate.last).toBe(3);
  });

  it('rejects a repeated sequence without advancing', () => {
    const gate = new MonotonicFrameGate();
    gate.admit(7);
    expect(gate.admit(7)).toBe(false);
    expect(gate.last).toBe(7);
  });

  it('rejects an out-of-order (older) sequence and keeps the newest committed', () => {
    const gate = new MonotonicFrameGate();
    gate.admit(10);
    gate.admit(11);
    // A stale sequence arriving after a newer one is dropped — no older frame commits.
    expect(gate.admit(9)).toBe(false);
    expect(gate.last).toBe(11);
    // The gate resumes on the next strictly-newer sequence.
    expect(gate.admit(12)).toBe(true);
    expect(gate.last).toBe(12);
  });

  it('never regresses across an interleaved stale/fresh stream', () => {
    const gate = new MonotonicFrameGate();
    const stream = [1, 2, 3, 2, 4, 3, 5, 5, 6];
    const committed: number[] = [];
    for (const seq of stream) {
      if (gate.admit(seq)) committed.push(seq);
    }
    expect(committed).toEqual([1, 2, 3, 4, 5, 6]);
    for (let i = 1; i < committed.length; i += 1) {
      expect(committed[i]).toBeGreaterThan(committed[i - 1]);
    }
  });
});
