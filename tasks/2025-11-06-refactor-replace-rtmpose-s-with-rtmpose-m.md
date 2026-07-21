# Task 2025-11-06: Replace RTMPose-S with RTMPose-M
**STATUS:** COMPLETED

## User Request
replace rtmpose-s with rtmpose-m, it is located in same folder

## Critical Discoveries

**Centralized config architecture eliminated complexity:**
The `RTMPOSE_MODEL_CONFIG` constant in `constants.ts` acts as single source of truth for all model dimensions. Updating this one object (name, path, backboneChannels, gauFeatures) automatically propagated through the entire inference pipeline - worker, hooks, components, feature extractors. No hardcoded dimension values found anywhere.

**Automatic dimension derivation:**
All raw feature dimensions (backbone: 36,864 dims, GAU: 6,528 dims) are computed automatically from config parameters. The architecture makes model swaps remarkably simple compared to manual find-and-replace of dimension constants.

## Solution

**Model Configuration Update (`src/services/ml/constants.ts`):**
Updated `RTMPOSE_MODEL_CONFIG` object:
- Model name: 'RTMPose-S' → 'RTMPose-M'
- Model path: 'rtmpose-s.onnx' → 'rtmpose-m.onnx'
- Backbone channels: 512 → 768 (50% increase in feature capacity)
- GAU features: 256 → 384 (50% increase in keypoint-aware features)

**Model Files:**
- Added `public/rtmpose-m.onnx` (54.4 MB) - new pose estimation model
- Kept `public/rtmpose-s.onnx` (21.9 MB) - preserved for backward compatibility
- Updated `vite.config.ts` to copy both models to build output

**Automatic Propagation:**
All dependent code automatically adapted without changes:
- `src/workers/inference-worker.ts` - Uses config dimensions for feature extraction
- `src/hooks/useWebWorkerInference.ts` - Loads model via `RTMPOSE_MODEL_CONFIG.path`
- `src/components/PostureCamera.tsx` - Imports config for model awareness
- Feature extractors - Derive dimensions from config (no hardcoded values)

**Result:** Fully functional RTMPose-M integration with improved feature capacity (768-dim backbone vs 512-dim, 384-dim GAU vs 256-dim). All pooled and raw features scale accordingly.

## Related
- `tasks/2025-10-24-refactor-replace-rtmw3d-with-rtmpose-s.md` - Previous model replacement (RTMW3D→RTMPose-S) using same config-driven architecture
- `tasks/2025-10-27-feature-update-worker-model-outputs.md` - Worker updates for model outputs, pooling strategies, storage versioning
- `tasks/2025-10-28-fix-neck-p5-extraction.md` - Feature extraction validation patterns in inference worker
- `tasks/2025-11-06-feature-rtmdet-user-selectable.md` - Feature registry extension patterns for user-selectable model outputs
- `tasks/2025-11-06-feature-opencv-clahe-preprocessing.md` - Preprocessing pipeline affecting feature extraction quality
