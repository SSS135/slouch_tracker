# Task 2025-10-31: Fix Hover Frame Bug
**STATUS:** COMPLETED

## User Request
frames are added to frame list when i hover over it, fix it. look at tasks, there was one with this requirement.

## Critical Discoveries (Non-Obvious)

**1. Count-based baseline breaks with circular buffer:**
When buffer is full (30/30), new frames evict old frames from beginning. `slice(0, count)` still shows same positions, which now include newly added frames. Count-based approach fundamentally incompatible with circular buffer eviction.

**2. First attempt was wrong direction:**
Tried ref pattern to stabilize callbacks, but this wasn't the real bug. Real issue: need to freeze actual frame array, not just count. Frames should be neither added NOR removed during hover.

## Solution

**Root Cause:** Count-based baseline (`baselineFrameCount`) breaks when buffer reaches capacity. After eviction, `slice(0, count)` includes newly added frames that replaced evicted ones.

**Fix Applied:**
Replaced count-based approach with snapshot-based approach (matches CollectTab pattern from `2025-10-26-refactor-unified-page.md`):

1. **State variable** (line 67): `const [frozenFrames, setFrozenFrames] = useState<CapturedFrame[] | null>(null);`
2. **Hover start** (line 425): `setFrozenFrames([...recentFrames]);` - captures snapshot
3. **Hover end** (line 430): `setFrozenFrames(null);` - clears snapshot
4. **Visible frames** (lines 435-437): `return frozenFrames ?? recentFrames;` - shows frozen or live
5. **Queued count** (lines 440-444): `Math.max(0, recentFrames.length - frozenFrames.length)` - diff between live and frozen

**Why This Works:**
Frozen snapshot is completely independent of buffer changes. New frames and evictions don't affect frozen array. Works correctly when buffer is full or not full.

## Lessons

**Circular buffer + count-based baseline = bug:** When buffer evicts old elements, count-based slicing shows wrong elements. Need actual snapshot, not position-based view.

**Pattern recognition saves time:** Recognized this matched CollectTab's hover-pause mechanism from Oct 26 refactor task. Reused proven snapshot pattern instead of inventing new approach.

## Related

- `tasks/2025-10-26-refactor-unified-page.md` - Original snapshot-based hover pattern from CollectTab

## Files Modified

- `src/pages/UnifiedPosturePage.tsx` (lines 67, 423-444)

## Impact

Hover-pause correctly freezes frame list - no additions or removals during hover. All 790 tests pass.
