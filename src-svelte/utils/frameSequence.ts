/**
 * Frontend mirror of the native processed-frame ordering guard.
 *
 * The Rust preview cell is authoritative: it tags every processed frame with a
 * monotonic capture sequence and drops any out-of-order write, so `slouchcam`
 * never serves an older frame after a newer one (echoed as `x-slouch-frame-seq`).
 * The native `<img>` fast path is additionally monotonic-by-construction — it
 * chains each request on `img.decode()`, so only one load is ever in flight and
 * commits happen strictly in fetch order without reading any header.
 *
 * This gate is the small, shared decision the sequence-driven paths use so a
 * stale or reset sequence can never re-commit a frame. It is pure and cheap
 * (no fetch, no header read), so it adds no per-frame cost to the preview.
 */

/** True iff `candidateSeq` is strictly newer than the last committed sequence. */
export function isNewerFrame(committedSeq: number, candidateSeq: number): boolean {
  return candidateSeq > committedSeq;
}

/**
 * Tracks the newest committed sequence and admits only strictly-newer ones.
 * Starts below every real sequence so the first frame always commits.
 */
export class MonotonicFrameGate {
  #last: number;

  constructor(initial: number = Number.NEGATIVE_INFINITY) {
    this.#last = initial;
  }

  /** Advances and returns true iff `seq` is strictly newer than the last committed. */
  admit(seq: number): boolean {
    if (isNewerFrame(this.#last, seq)) {
      this.#last = seq;
      return true;
    }
    return false;
  }

  get last(): number {
    return this.#last;
  }
}
