# Task 2025-11-18: Fix Privacy Mode for Thumbnails

**STATUS:** COMPLETED

## User Request
there was a recent task that removed bbox and keypoint from thumbnails. it broke privacy mode, now thumbnails never use it when enabled. fix

## Critical Discoveries

**Privacy mode never applied to thumbnails**
Task 2025-11-14 removed overlays from thumbnails by using preprocessed ImageData instead of display canvas. This fixed the overlay issue but broke privacy mode completely - thumbnails always showed real video frames regardless of privacy setting. Privacy mode worked for live display (skeleton on bicubic background) but was never passed to thumbnail generation.

**Three-layer parameter threading required**
Privacy mode needed to flow through: PostureTrackerApp → useFrameSampler → generateThumbnail. Missing any layer meant thumbnails ignored privacy setting.

**Manual capture bypassed useFrameSampler**
`handleCaptureWithLabel` in PostureTrackerApp called `generateThumbnail` directly without privacy parameters. Auto-capture used the hook (which we fixed), but manual capture (G/B/A shortcuts) bypassed it entirely.

**Skeleton rendering requires opacity property**
`drawHumanLikeSkeleton` expects `SmoothedKeypoint[]` with `opacity: number` property. Creating `Keypoint3D[]` without opacity causes all keypoints to fail the `isValid` check (`kp.opacity > 0.01`), preventing skeleton rendering. Background rendered correctly because bicubic grid doesn't depend on keypoints.

## Solution

**thumbnailGenerator.ts** - Added `privacyMode`, `keypoints`, `videoWidth`, `videoHeight` parameters. When privacy mode enabled: samples 4×4 color grid from video, renders bicubic background, draws skeleton overlay if keypoints available with opacity > 0.01.

**useFrameSampler.ts** - Accepts `privacyMode` parameter, creates keypoints with opacity property (`kp.score > 0.3 ? 1.0 : 0.0`), passes to `generateThumbnail` with video dimensions.

**PostureTrackerApp.tsx** - Threads `settings.privacyMode` to `useFrameSampler` hook. Fixed manual capture path to extract keypoints with opacity, pass privacy mode + keypoints + video dimensions to `generateThumbnail`.

**Tests** - Added 14 new privacy mode tests covering thumbnail generation, parameter passing, edge cases. All 31 thumbnailGenerator + 20 useFrameSampler tests pass.

**Dataset impact** - Keypoints stored in dataset now include `opacity` property. This is backward compatible - existing frames without opacity will still load (opacity defaults to 0 when missing). Privacy mode thumbnails are NOT stored in dataset - they're generated on-the-fly during capture based on current privacy setting.

## Related

- `tasks/2025-11-14-fix-remove-overlays-from-dataset-images.md` - Task that removed overlays, introducing privacy mode regression
- `tasks/2025-11-03-feature-privacy-mode.md` - Original privacy mode implementation (live display only)
