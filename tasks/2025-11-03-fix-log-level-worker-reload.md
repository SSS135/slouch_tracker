# Task 2025-11-03: Fix Log Level Not Applying to Workers Without Reload

**STATUS:** COMPLETED

## User Request
when i change log level in ui, it applies to workers only after page reload. fix. find task that implemented it.

## Critical Discoveries

**1. replaceState() doesn't trigger popstate:**
`window.history.replaceState()` modifies URL without triggering `popstate` or `hashchange` events. These events only fire on browser back/forward navigation or hash changes.

**2. Workers had listening infrastructure:**
Both worker hooks (useWebWorkerInference, useModelTraining) were already listening for URL changes via `popstate`/`hashchange`, but these events never fired when LoggerSettings changed the URL programmatically.

**3. Worker message handlers already working:**
Workers already had `setLogLevel` message handlers implemented and working correctly. The only issue was the notification mechanism.

**4. Empty string converted to undefined:**
When switching to 'Off', LoggerSettings dispatches event with `logParam: ''` (empty string). Worker hooks used `logParam || undefined` which converted empty string to `undefined`, causing worker to ignore the message because it checks `payload?.logParam !== undefined`. This is why Debug→Off didn't work but Debug→All→Off did (All sets INFO level, so DEBUG messages already stopped).

## Solution

**Phase 1:** Replaced browser event listening with custom event pattern.

**LoggerSettings.tsx:** Dispatches `logLevelChanged` custom event after updating logger, passing `logParam` in event detail.

**useWebWorkerInference.ts & useModelTraining.ts:** Changed from listening to `popstate`/`hashchange` events to listening for `logLevelChanged` custom event. Handler extracts `logParam` from event detail and sends `setLogLevel` message to worker.

**Phase 2:** Fixed empty string handling to ensure workers receive 'Off' updates.

**Both hooks (4 locations total):** Changed `logParam || undefined` to `logParam || 'none'` in both initial setup and handleLogLevelChange, ensuring workers always receive valid string value instead of undefined when log level is 'none'.

**Phase 3:** Simplified UI to three essential log levels.

**LoggerSettings.tsx:** Reduced LOGGER_OPTIONS from 7 options (Off, Detection, Training, Worker, Storage, All, Debug) to 3 options (Error, Info, Debug). Removed category-specific options as they added complexity without clear user benefit.

Workers now receive log level updates immediately without page reload, including proper handling of 'Off' state. UI is simplified to essential log levels.

## Related

- `tasks/0003-fix-excessive-test-output.md` - Implemented the logger system with URL parameter control
