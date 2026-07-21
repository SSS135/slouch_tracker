# Task 2025-11-27: Add Engineered RTMPose Binned Features
**STATUS:** IN PROGRESS

## User Request
Add engineered RTMPose features: neck length, neck len / shoulder width, neck len / inter eye dist, neck len / inter ear dist. Each feature binned into 5-10 soft Gaussian bins with data-driven bin ranges (computed from training data percentiles). Bin probabilities multiplied by min confidence of keypoints used.

## General Description
Add geometric features computed from RTMPose keypoints that capture posture-relevant body proportions. Features are soft-binned using Gaussian probability distributions with data-driven bin edges learned at training time.

**Features (4 total):**
- `neck_len` - Distance from mid-shoulders to mid-eyes
- `neck_shoulder_ratio` - neck_len / shoulder_width
- `neck_eye_ratio` - neck_len / inter_eye_dist
- `neck_ear_ratio` - neck_len / inter_ear_dist

**Output:** 4 features × 5 bins = 20 dimensions (default)

## Action Plan
1. Create `keypointIndices.ts` - COCO keypoint index constants
2. Create `keypointGeometry.ts` - Geometric feature extraction functions
3. Create `softBinning.ts` - SoftBinningTransformer class
4. Create `keypointBinningTransformer.ts` - Orchestrator class
5. Update `constants.ts` - Add keypoint geometry constants
6. Update `types.ts` - Add serialization types
7. Update `schemas.ts` - Add Zod validation schemas
8. Integrate with `featureExtractor.ts`
9. Update `TrainingTab.tsx` - Add UI controls
10. Add unit tests

## Rationale
- **Soft binning**: Provides smooth gradients for ML training vs hard bins
- **Data-driven edges**: Adapts to each user's body proportions via percentiles
- **Confidence weighting**: Handles uncertain keypoint detections gracefully
- **Computed on-demand**: No storage overhead, requires keypoints saved with frames

## Files to Modify
**New:**
- `src/services/posture/keypointIndices.ts`
- `src/services/ml/keypointGeometry.ts`
- `src/services/ml/softBinning.ts`
- `src/services/ml/keypointBinningTransformer.ts`

**Modified:**
- `src/services/ml/featureExtractor.ts`
- `src/services/ml/constants.ts`
- `src/services/ml/types.ts`
- `src/services/validation/schemas.ts`
- `src/components/unified/TrainingTab.tsx`
