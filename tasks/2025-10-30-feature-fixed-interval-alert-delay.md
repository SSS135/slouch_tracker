# Task 2025-10-30: Fixed Capture Interval Values and Time-Based Alert Delay
**STATUS:** COMPLETED

## User Request
Make capture interval slider have fixed values of [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. 0.5 by default. Make consecutive bad frame measure in seconds instead of frames [1, 2..15] values, 5 is default.

## Critical Discoveries

**1. Slider fixed values implementation:**
Maps array indices to actual values internally (`fixedValues[Math.round(sliderPosition)]`), consistent with exponential scale pattern. Clean UI without visual marks - position conveys value. Full backward compatibility with continuous sliders.

**2. Time-based alerting with timestamp tracking:**
Replaced frame counter with `Date.now()` timestamp tracking. Bad posture starts timer (`badPostureStartTimeRef`), calculates elapsed seconds, triggers when `>= alertDelaySeconds`. Hardware-independent - works same at 10 FPS or 60 FPS.

**3. Backward compatibility for renamed setting:**
Migration from `consecutiveBadFramesThreshold` (frames) → `alertDelaySeconds` (seconds). Fallback chain: `parsed.alertDelaySeconds ?? parsed.consecutiveBadFramesThreshold ?? 5`. Assumes ~1 frame/second for old values.

**4. Fixed values provide better granularity:**
19 discrete values vs 10 continuous values. Finer control at low end (0.1s increments for 0.1-1s). Slider UI better than RadioGroup for this many options - maintains visual position, click-to-edit, tooltips without vertical scroll.

## Solution

**Slider Component Enhancement** (Slider.tsx):
Added `fixedValues?: number[]` prop. When provided, uses internal positions 0 to length-1, maps to actual values via array indexing. Snaps manual input to nearest value. No visual marks (clean design). Works with all existing features (exponential/linear scales, tooltips, formatting).

**Settings Migration** (useCameraSettings.ts):
Renamed `consecutiveBadFramesThreshold` → `alertDelaySeconds`. Changed defaults: `captureIntervalSeconds: 1s → 0.5s`, `alertDelaySeconds: 5s` (was 4 frames). Backward compatibility: `parsed.alertDelaySeconds ?? parsed.consecutiveBadFramesThreshold ?? 5`.

**RuntimeTab UI** (RuntimeTab.tsx):
Capture Interval slider: `fixedValues={[0.1, 0.2, ..., 1, 2, ..., 10]}` (19 values), default 0.5s. Alert Delay slider: `fixedValues={[1..15]}`, label "Alert Delay" (was "Consecutive Bad Frames"), help text clarifies time-based measurement.

**Time-Based Alert Logic** (usePostureSound.ts):
Replaced `consecutiveBadFramesRef` (frame counter) with `badPostureStartTimeRef` (timestamp). On bad posture: start tracking time if null, calculate `elapsedSeconds = (Date.now() - startTime) / 1000`, trigger when `>= alertDelaySeconds`. On good posture: reset timestamp to null.

**Tests** (43 new tests, all 738 passing):
- `Slider.test.tsx`: Fixed values behavior, continuous mode, edge cases, snapping
- `usePostureSound.test.ts`: Time-based logic with mocked Date.now(), backward compatibility
- `RuntimeTab.test.tsx`: Component integration, slider configurations
- `jest.config.js`: Updated JSX transform

## Files Modified
- src/components/ui/Slider.tsx
- src/hooks/useCameraSettings.ts
- src/components/unified/RuntimeTab.tsx
- src/hooks/usePostureSound.ts
- src/pages/UnifiedPosturePage.tsx
- src/components/ui/__tests__/Slider.test.tsx (NEW)
- src/hooks/__tests__/usePostureSound.test.ts (NEW)
- src/components/unified/__tests__/RuntimeTab.test.tsx (NEW)
- jest.config.js

## Impact
**Better UX:** Time-based alerts predictable across hardware (5s consistent vs 4 frames = 0.25-0.067s depending on FPS). Fixed capture intervals provide finer control (19 values vs 10).

**Hardware-Independent:** Alert behavior no longer varies with inference speed. Same experience on slow/fast machines.

**Improved Granularity:** Capture interval 0.1s increments at low end enables fast data collection scenarios while maintaining coarse 1s increments at high end.
