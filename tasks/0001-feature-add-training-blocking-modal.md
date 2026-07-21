# Task 0001: Add Training Blocking Modal

**STATUS:** ✅ COMPLETED

## User Request
Add a full-screen modal blocking popup for model training with these requirements:
1. Full-screen modal that appears in center of screen when training starts
2. Dark backdrop that dims everything behind it
3. Displays training progress/spinner (currently in TrainingTab)
4. Automatically closes when training completes
5. Blocks all interaction while training is in progress

**Additional requirement:**
Pause camera feed and detection while training is in progress

## General Description

The TrainingTab component displays training progress inline within the tab content. This creates suboptimal UX where users can still interact with UI elements during training, and the progress indication is not prominent enough for long-running operations. The goal is to replace inline progress with a full-screen blocking modal that prevents all interaction and provides clear visual feedback.

The codebase already has a TrainingBlockingSpinner component designed for this purpose. It's currently used in TrainingPanel.tsx but NOT in the new TrainingTab.tsx. This task involves integrating the existing modal into TrainingTab and rendering it at root level to cover the entire viewport (both camera feed and settings panel).

## Action Plan

1. Add training state management to root level (app/index.tsx)
   - Add state for isTraining, trainingProgress, and trainingStage
   - Create handleTrainingStatusChange callback handler to receive state updates from TrainingTab
   - Pass onTrainingStatusChange callback prop to TrainingTab component

2. Render TrainingBlockingSpinner at root level (app/index.tsx)
   - Add TrainingBlockingSpinner component after the main 60/40 split container
   - Pass training state props: isVisible (from isTraining), progress, and stage
   - Modal will cover entire viewport (both camera feed and settings panel)

3. Update TrainingTab to report training state via callback
   - Add optional onTrainingStatusChange prop to TrainingTab
   - Add useEffect to call callback whenever training state changes
   - TrainingTab maintains training logic but delegates full-screen UI to parent
   - Remove local modal rendering (if previously added)

4. Remove inline training progress UI
   - Remove renderTrainingProgress function entirely
   - Remove renderTrainingStatus function entirely
   - Replace with simplified renderError function that only shows error messages
   - Keep "Train Model" button and training results display
   - Modal now provides all progress feedback

5. Update integration tests
   - Update TrainingTab.test.tsx to test callback invocation (not modal rendering)
   - Verify callback is called with correct state when training starts/stops
   - Verify different training stages are reported via callback

6. Run test suite
   - Check for regressions in existing TrainingTab tests

## Rationale

### Why TrainingBlockingSpinner is the Right Choice
1. **Already exists and is tested** - The component was built specifically for this use case with comprehensive test coverage (25 test cases)
2. **Matches training state interface** - Props align perfectly with `useModelTraining` state (isTraining, progress, stage)
3. **Proven architecture** - Already used successfully in TrainingPanel.tsx
4. **Non-dismissible by design** - No close button or escape handlers, correct for blocking operations
5. **Prominent visual feedback** - Dark backdrop and centered modal ensure users understand training is in progress

### Why Inline Progress Display Should Be Replaced
1. **Better UX** - Full-screen modal is more prominent for long operations (training can take 10-30 seconds)
2. **Prevents user errors** - Blocking overlay prevents users from changing settings or navigating away
3. **Cleaner UI** - Removes clutter from the tab interface
4. **Consistent with existing patterns** - TrainingPanel already uses this approach

### Why Root-Level Rendering is Required
Modal must be rendered in app/index.tsx (root level) rather than inside TrainingTab to cover the entire viewport. When rendered inside TrainingTab (within the 40% settings panel), its absolute positioning only covers that panel, not the full screen. Root-level rendering ensures the modal covers both the camera feed (60%) and settings panel (40%).

### Why Pause Camera and Detection
1. **Resource optimization** - Prevents conflicts between training and inference operations
2. **Faster training** - Frees CPU/GPU resources for model training
3. **Cleaner UX** - No confusing camera updates during training
4. **Automatic control** - Pauses when training starts, resumes when complete

## Alternative Approaches Considered

### Alternative 1: Wrap in React Native Modal Component
**Rejected because:**
- TrainingBlockingSpinner already works with absolute positioning
- Adding Modal wrapper adds unnecessary complexity
- Current approach is simpler and already tested
- Modal component may have quirks in web target (Expo React Native web)

### Alternative 2: Create New Modal Component
**Rejected because:**
- Duplicates existing functionality
- TrainingBlockingSpinner already has everything needed
- Would require new tests
- Violates DRY principle

### Alternative 3: Keep Inline Progress + Add Modal
**Rejected because:**
- Creates redundant UI elements
- Confusing to show progress in two places
- Inline progress is hidden behind modal anyway

### Alternative 4: Portal-Based Modal System
**Rejected because:**
- Overengineered for this simple use case
- React Native doesn't have built-in Portal (would need third-party library)
- Absolute positioning works fine for this layout

## Files to Modify

### Primary Changes
1. **app/index.tsx**
   - Add training status state (isTraining, trainingProgress, trainingStage)
   - Add handleTrainingStatusChange callback handler
   - Render TrainingBlockingSpinner at root level
   - Pass onTrainingStatusChange to TrainingTab
   - Add paused prop to camera component based on training state

2. **src/components/unified/TrainingTab.tsx**
   - Add optional onTrainingStatusChange callback prop
   - Add useEffect to call callback when training state changes
   - Remove renderTrainingProgress and renderTrainingStatus functions
   - Add renderError function for error-only display

3. **src/components/camera/RTMW3DCameraWeb.tsx**
   - Add optional paused prop to pause camera and detection

### Test Updates
4. **src/components/unified/__tests__/TrainingTab.test.tsx**
   - Update tests to verify callback invocation
   - Remove tests for modal rendering in TrainingTab

5. **src/components/camera/__tests__/RTMW3DCameraWeb.test.tsx**
   - Add tests for paused prop behavior

## Implementation Details

### Root-Level Modal Integration
The TrainingBlockingSpinner is rendered at the root level in app/index.tsx after the 60/40 split container. This ensures the modal covers the entire viewport (both camera feed and settings panel) during training. Training state is lifted to the root level and passed down to the modal via props (isVisible, progress, stage).

TrainingTab reports training status changes via an onTrainingStatusChange callback prop. When training state changes (isTraining, progress, or stage), the callback is invoked, updating root-level state and triggering modal visibility updates.

The inline progress UI (renderTrainingProgress and renderTrainingStatus functions) was removed from TrainingTab. Error handling was consolidated into a simpler renderError function that only displays error messages.

### Camera and Detection Pause
The RTMW3DCameraWeb component was enhanced with an optional paused prop. When paused=true:
- Camera stream is disabled (useCameraStream enabled prop set to false)
- Frame processor is disabled (useFrameProcessor enabled prop includes paused check)
- Canvas renderer is disabled (useCanvasRenderer enabled prop includes paused check)

At the root level (app/index.tsx), the camera's paused prop is tied to the isTraining state. When training starts, the camera automatically pauses. When training completes, it automatically resumes. This prevents resource conflicts and ensures optimal training performance.

### Sound Alert Pause
The usePostureSound hook was modified to accept an optional paused parameter. When paused=true, the sound playback is disabled regardless of posture detection state. At the root level, the isTraining state is passed to usePostureSound, automatically pausing sound alerts during training and resuming them when complete.

### Stale Classification Data Handling
The handleTrainingStatusChange callback clears the classification state when training completes (setClassification(null)). This prevents sound alerts from playing on stale posture data from before training. When training finishes, classification is cleared, ensuring the first frame after training determines the alert state (not outdated pre-training data).

### Testing
All tests pass (1,091 total tests). New test coverage includes:
- TrainingTab callback invocation tests (verifies state reporting)
- RTMW3DCameraWeb pause behavior tests (camera, processor, renderer disabled when paused)
- usePostureSound pause parameter tests (sound stops when paused)
- App-level integration tests (camera pauses during training)

No regressions were introduced. Test coverage was maintained across all modified components.

## Related Code References

### Similar Patterns in Codebase
1. **TrainingPanel.tsx** - Shows correct usage of TrainingBlockingSpinner
2. **ConfirmationModal.tsx** - Different modal pattern (dismissible, for confirmations)

### Training State Management
- **useModelTraining.ts** - Source of truth for training state
  - Training starts, state initialized to isTraining=true
  - Progress updates continuously throughout training
  - Training completes, state reset to isTraining=false

### Component Architecture
- TrainingTab is wrapped in ErrorBoundary in app/index.tsx
- TrainingConfigProvider context provides shared configuration (wraps entire app)
- Training state is local to TrainingTab via useModelTraining hook
- Parent component (app/index.tsx) owns viewport-level rendering concerns
- Child component (TrainingTab) reports state changes via callback
