# Task 2025-11-18: Save Keypoints, Bounding Boxes, and Scores with Dataset
**STATUS:** COMPLETED

## User Request
make keypoints, found bboxes and scores to be saved with dataset

## Critical Discoveries

**Schema Definition Order (Zod):**
Circular dependency issue when `Keypoint3DSchema` and `BoundingBoxSchema` were defined after `PostureFrameSchema`. Must define primitive schemas before composite schemas that reference them.

**Duplicate Type Definitions:**
- `BoundingBox` duplicated in `cropUtils.ts` and `inference-worker.ts`
- `ClassifierConfig` duplicated in `types.ts`, `model.ts`, `classifierFactory.ts`
- `ClassificationResult` duplicated in `types.ts` and `inference-worker.ts`
Consolidated to single source of truth by importing from shared locations.

## Implementation Details

**Type System:**
- Added `keypoints?: Keypoint3D[]` and `bbox?: BoundingBox` to `PostureFrame` and `CapturedFrame`
- Added `hasKeypoints`, `hasBbox` flags to `FrameMetadata` for export manifest
- Reused existing `BoundingBox` from `cropUtils.ts` (removed duplicate from worker)

**Storage (v8→v9):**
- Incremented `STORAGE_VERSION` to 9 (clean break, clears existing datasets)
- Extended `reconstructFrame()` to deserialize Keypoint3D class instances from IndexedDB
- Save only original bbox (not expanded) - saves ~56 bytes/frame

**Capture Flow:**
- `useFrameSampler`: Extract keypoints/bbox from `InferenceResult`, clone to prevent reference issues
- Support all keypoint groups (body/face/hands) for future model compatibility

**Export/Import:**
- Export: Serialize keypoints/bbox to JSON files in ZIP (`keypoints.json`, `bbox.json`)
- Manifest includes `hasKeypoints`/`hasBbox` boolean flags
- Import: Deserialize and reconstruct Keypoint3D class instances (not plain objects)

**Testing (29 new tests):**
- Validation: BoundingBox schema, PostureFrame with optional keypoints/bbox
- Storage: Save/load frames, Keypoint3D reconstruction
- Export/Import: Round-trip integrity, backwards compatibility
- All 150 tests passing (except 11 pre-existing useCameraStream timing failures)

**Code Cleanup:**
- Removed duplicate `BoundingBox`/`BboxData` from `inference-worker.ts` → use `BoundingBox`/`ExpandedBbox` from `cropUtils.ts`
- Removed duplicate `ClassifierConfig`/`TrainingResult` from `model.ts`/`classifierFactory.ts` → use from `types.ts`
- Removed duplicate `ClassificationResult` from `inference-worker.ts` → use from `types.ts`

## Related
- `tasks/2025-11-09-*` - Storage simplification (single-key architecture, split-key removal)
- `tasks/2025-10-*` - Feature system evolution (unified features dict)
