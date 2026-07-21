# Task 2025-10-31: Fixed Camera Constraints and Removed Resolution Picker
**STATUS:** COMPLETED

## User Request
```
// Conservative approach
video: {
  width: { ideal: 1280, max: 1280 },   // Never exceed ideal
  height: { ideal: 720, max: 720 },
  frameRate: { ideal: 30, max: 30 }
}

use this, remove resolution picker from runtime settings
```

## Problem Context
User was concerned that browser might auto-select extreme resolutions/framerates (e.g., 4K @ 120fps or 640×480 @ 15fps) which would either ruin performance or provide poor quality. Dynamic resolution picker added user complexity without clear benefit for ONNX-based posture detection workload.

## Solution
Applied fixed conservative camera constraints (1280×720 @ 30fps max) directly in `useCameraStream.ts` and removed resolution picker UI from RuntimeTab. This ensures consistent camera negotiation across all devices while protecting performance for RTMDet + RTMW3D inference.

**Key changes:**
1. **useCameraStream.ts**: Hardcoded constraints with `max` values to prevent high-end cameras from exceeding limits. Added missing `frameRate: { ideal: 30, max: 30 }` constraint.
2. **RuntimeTab.tsx**: Removed `OptionCard` component, resolution picker section, and unused imports (Badge, Card). Updated FPS display help text to mention "fixed at 720p @ 30fps max".

**Backward compatibility**: Resolution fields remain in `CameraSettings` interface. Existing localStorage data loads without migration. Old resolution values stored but ignored.

**Code snippet (useCameraStream.ts):**
```typescript
// Before: dynamic constraints
width: { ideal: width }

// After: fixed conservative constraints
width: { ideal: 1280, max: 1280 },
height: { ideal: 720, max: 720 },
frameRate: { ideal: 30, max: 30 }
```

## Files Modified
- `src/hooks/useCameraStream.ts` (~10 lines modified, added frameRate constraint)
- `src/components/unified/RuntimeTab.tsx` (~90 lines removed, simplified Camera Settings section)

## Impact
**Performance**: Prevents excessive resolution/framerate that would degrade ONNX inference performance.
**UX**: Eliminates user confusion about resolution selection. Single "it just works" configuration.
**Consistency**: All devices now negotiate to same target resolution, improving reproducibility.

## Lessons
1. Fixed constraints with `max` values prevent browser from auto-selecting extreme capabilities
2. Missing `frameRate` constraint could allow high-framerate cameras to use 60fps+ unnecessarily
3. Backward compatible refactoring: keep interface fields, ignore values in implementation
