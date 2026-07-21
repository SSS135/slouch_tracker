# Task 0011: Persist Auto-Capture State and Update Logistic Regression Default

**STATUS:** ✅ COMPLETED

## User Request
make auto-capture enabled by default and saved so it persists afer page reload. Make default regularisation strengh of logistic regression 0.01

## General Description
Enhance UX by persisting auto-capture state in localStorage (following existing useCameraSettings pattern) and updating the logistic regression regularization default to 0.01 (lower C value = stronger regularization = more conservative probabilities).

## Action Plan
1. Add autoCaptureEnabled to CameraSettings interface (useCameraSettings.ts)
2. Update DEFAULT_SETTINGS to set autoCaptureEnabled: true
3. Update app/index.tsx to use settings.autoCaptureEnabled instead of local state
4. Update classifierRegistry.ts logistic regression C default from 1.0 to 0.01

## Rationale
**Auto-Capture Persistence:**
- Current: autoCaptureEnabled is local component state (line 136 in app/index.tsx), resets on reload
- Pattern exists: useCameraSettings already persists cameraWidth, cameraHeight, captureIntervalSeconds, alertVolume
- Consistency: All user preferences should persist across sessions (established pattern)
- UX improvement: Users won't need to re-enable auto-capture after every page load

**Regularization Default Change:**
- Current: C = 1.0 (standard regularization)
- New: C = 0.01 (100x stronger regularization)
- Effect: Lower C produces more conservative probability estimates, less prone to overconfidence
- Benefit: More reliable confidence scores for posture detection alerts

## Files to Modify
- src/hooks/useCameraSettings.ts - Add autoCaptureEnabled to interface and defaults
- app/index.tsx - Remove local state, use settings.autoCaptureEnabled
- src/services/ml/classifierRegistry.ts - Change logistic regression C default to 0.01

## Implementation Details

**Auto-Capture Persistence:**
- Added `autoCaptureEnabled: boolean` to CameraSettings interface
- Set default to `true` in DEFAULT_SETTINGS constant
- Updated app/index.tsx to use `settings.autoCaptureEnabled` instead of local useState
- Removed local state management (15 lines changed in app/index.tsx)
- Auto-capture state now persists via localStorage across page reloads
- Backward compatibility: Old settings without this field default to true

**Logistic Regression Default Update:**
- Changed C default from 1.0 to 0.01 in classifierRegistry.ts
- Produces more conservative probability estimates (stronger regularization)
- Updated corresponding tests to expect new default value

**Test Coverage:**
- Created comprehensive test suite for useCameraSettings hook (11 tests, 206 lines)
- Tests cover: initialization, persistence, updates, backward compatibility, error handling
- Updated classifierRegistry tests to reflect new C default
- Full test suite: 1210/1210 passing ✓

**Files Modified:**
- app/index.tsx (15 lines changed)
- src/hooks/useCameraSettings.ts (3 lines added)
- src/services/ml/classifierRegistry.ts (4 lines changed)
- src/services/ml/__tests__/classifierRegistry.test.ts (6 lines changed)

**Files Created:**
- src/hooks/__tests__/useCameraSettings.test.ts (206 lines)

**Key Benefits:**
- Improved UX: Auto-capture preference persists across sessions
- Default enabled: New users get optimal experience out-of-box
- Conservative ML: Lower C value reduces overconfident probability predictions
- Well-tested: Comprehensive test coverage ensures reliability
