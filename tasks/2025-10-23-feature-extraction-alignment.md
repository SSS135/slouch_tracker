# Feature Extraction Alignment with Python Reference

## User Request

Compare the Python feature extraction implementation (`tracker_lib/feature_extraction.py`) with the TypeScript implementation and make the TypeScript version match it. Achieve complete feature parity (~92 features total, excluding hand shape features).

**User Preferences:**
- Complete feature parity (all ~108 features from Python, minus hand shape)
- Implement harmful feature filtering
- Force retrain approach (no model versioning)
- Defer hand shape features (not core to posture tracking)

## Scope

**Target**: Expand from 36 → 92 engineered features

**Include:**
- Advanced posture features (~20 new)
- Advanced hand-to-face features (~15 new)
- Advanced mouth-open features (~21 new)
- Harmful feature filtering mechanism

**Exclude:**
- Hand shape features (15 features - deferred per user request)
- Temporal features (YAGNI)
- Model versioning/migration logic
- UI warnings for incompatible models

## Current State

**TypeScript Implementation (36 features)**:
- File: `src/services/posture/detection.ts:extractAllFeatures()`
- Interface: `src/services/posture/types.ts:UnifiedFeatures`
- Categories:
  - Detection meta: 1 feature (`detectionConfidence`)
  - Posture: 24 features (angles, z-deltas, dimensions, tilt/asymmetry, rotation)
  - Hand-face: 5 features (min distances, elevation, proximity score)
  - Mouth: 6 features (lip opening, spread, aspect ratio, jaw angle, volume, score)

## Missing Features Inventory

### Batch 1: Advanced Posture - Head & Neck (5 features)
- [x] `theta_neck` - Angle between (ear_mid - shoulder_mid) and vertical
- [x] `forward_head` - Horizontal displacement of nose relative to shoulders
- [x] `ear_shoulder_ratio` - Distance from ear to shoulder normalized by shoulder width
- [x] `craniovertebral_angle` - CVA (ear-shoulder vector to vertical angle)
- [x] `head_forward_distance` - Sagittal plane head forward distance

### Batch 2: Advanced Posture - Head Positioning (5 features)
- [x] `ear_vertical_offset` - Ear Y-position relative to shoulder
- [x] `nose_ear_shoulder_angle` - Angle between (nose-ear) and (ear-shoulder) vectors
- [x] `eye_ear_z_delta` - Z-depth difference between eyes and ears
- [x] `head_vertical_displacement` - Nose Y-position relative to ears
- [x] `shoulder_elevation_asymmetry` - Bilateral shoulder height difference

### Batch 3: Advanced Posture - Spinal & Torso (5 features)
- [x] `hip_rotation_z` - Z-depth asymmetry between hips
- [x] `shoulder_hip_x_offset` - Lateral offset between shoulder and hip midpoints
- [x] `torso_centerline_deviation` - Nose X-position relative to hip center
- [x] `torso_width_height_ratio` - Shoulder width divided by torso length
- [x] `upper_lower_torso_z_diff` - Z-depth gradient from shoulders to hips

### Batch 4: Advanced Posture - Shoulder Girdle (4 features)
- [x] `shoulder_protraction` - Average shoulder Z-position relative to ears
- [x] `shoulder_elevation` - Shoulder Y-position relative to hips
- [x] `bilateral_shoulder_z_symmetry` - Symmetry of left/right shoulder Z-depth
- [x] `shoulder_rotation_indicator` - Angle of shoulder line from horizontal

### Batch 5: Advanced Posture - Chest & Stability (5 features)
- [x] `shoulder_blade_distance` - Shoulder width normalized by torso length
- [x] `chest_collapse_ratio` - Anterior chest compression (shoulder-nose Z distance)
- [x] `head_stability_score` - Deviation from ideal vertical ear-shoulder alignment
- [x] `neck_shoulder_z_gradient` - Z-axis slope from ear to shoulder
- [x] `shoulder_height_symmetry` - Bilateral shoulder height symmetry (1.0 = perfect)

### Batch 6: Advanced Posture - Arms & Lower Body (5 features)
- [x] `arm_shoulder_activity_left` - Left elbow-shoulder distance
- [x] `arm_shoulder_activity_right` - Right elbow-shoulder distance
- [x] `elbow_distance_ratio` - Elbow spacing (wider = arms extended sideways)
- [x] `wrist_hip_distance_left` - Left wrist to hip distance
- [x] `wrist_hip_distance_right` - Right wrist to hip distance

### Batch 7: Advanced Posture - Arms & Alignment (5 features)
- [x] `arm_extension_left` - Left wrist-elbow-shoulder angle
- [x] `arm_extension_right` - Right wrist-elbow-shoulder angle
- [x] `nose_shoulder_frontal_offset` - Lateral head shift from shoulder midline
- [x] `hip_shoulder_z_correlation` - Z-depth alignment between hips and shoulders
- [x] `leg_presence_indicator` - Lower body visibility score

### Batch 8: Advanced Posture - Symmetry & Refinement (4 features)
- [x] `head_inclination` - Nose-eye vector angle to horizontal
- [x] `shoulder_vertical_displacement` - Shoulder-ear vertical distance
- [x] `posture_symmetry_index` - Combined shoulder and hip symmetry measure
- [x] `vertical_alignment_score` - Vertical stacking alignment (nose-shoulder-hip)

### Batch 9: Hand-to-Face - Finger Distances (4 features)
- [x] `left_index_nose_distance` - Left index finger to nose distance
- [x] `right_index_nose_distance` - Right index finger to nose distance
- [x] `left_thumb_nose_distance` - Left thumb to nose distance
- [x] `right_thumb_nose_distance` - Right thumb to nose distance

### Batch 10: Hand-to-Face - Advanced Proximity (6 features)
- [x] `left_hand_above_mouth` - Binary flag: left hand above mouth
- [x] `left_hand_above_nose` - Binary flag: left hand above nose
- [x] `left_hand_crosses_center` - Binary flag: left hand crossing face centerline
- [x] `left_wrist_min_face_distance` - Left wrist minimum distance to any face point
- [x] `right_wrist_min_face_distance` - Right wrist minimum distance to any face point
- [x] `hand_face_overlap_score` - Combined proximity score (min of left/right)

### Batch 11: Mouth - Basic Features (5 features)
- [x] `mouth_height_proxy` - Mouth height relative to face width
- [x] `mouth_to_nose_distance` - Mouth center to nose distance
- [x] `mouth_vertical_position` - Mouth Y-position relative to nose
- [x] `mouth_z_depth` - Mouth Z-depth relative to nose
- [x] `left_mouth_nose_distance` - Left mouth corner to nose distance

### Batch 12: Mouth - Angles & Geometry (5 features)
- [x] `right_mouth_nose_distance` - Right mouth corner to nose distance
- [x] `mouth_opening_angle` - 3D angle between mouth corners and nose
- [x] `mouth_area_proxy` - Mouth area (width × height / face area)
- [x] `mouth_corners_z_diff` - Z-depth difference between mouth corners
- [x] `nose_mouth_angle` - Nose-to-mouth vector angle to vertical

### Batch 13: FaceMesh - Lip Landmarks (5 features)
- [x] `facemesh_lip_vertical_opening` - Upper-to-lower lip distance
- [x] `facemesh_lip_horizontal_spread` - Mouth corner distance
- [x] `facemesh_lip_aspect_ratio` - Lip vertical / horizontal ratio
- [x] `facemesh_mouth_volume_proxy` - 3D mouth volume approximation
- [x] `facemesh_jaw_opening_angle` - Chin to mouth corners angle

### Batch 14: FaceMesh - Advanced Mouth (5 features)
- [x] `facemesh_lip_center_to_nose` - Lip center to nose distance
- [x] `facemesh_mouth_depth_variation` - Z-axis variation between mouth corners
- [x] `facemesh_upper_lip_displacement` - Upper lip vertical displacement from neutral
- [x] `facemesh_lower_lip_displacement` - Lower lip vertical displacement from neutral
- [x] `facemesh_mouth_score` - Combined FaceMesh mouth opening score

## Implementation Progress

### Phase 1: Setup ✅
- [x] Created `featureConfig.ts` with harmful feature filter lists
- [x] Created this task file with feature inventory

### Phase 2: Interface Extension ✅
- [x] Extend `UnifiedFeatures` interface with all 56 new field names

### Phase 3: Feature Implementation (14 batches) ✅
Implementation status tracked in batches above.

**Current: All 14 batches complete (56 features implemented)**

### Phase 4: Integration ✅
- [x] Update `featureExtractors.ts` - Add all new features to array conversion
- [x] Update `featureExtractor.ts` - Add all new features to array conversion
- [x] Update `featureRegistry.ts` - Change ENGINEERED dimensions from 36 → 92
- [ ] Apply harmful feature filtering in extraction pipeline (deferred - not critical)

### Phase 5: Testing
- [ ] Update test dimension assertions (36 → 92)
- [ ] Add test cases for harmful feature filtering
- [ ] Run full test suite and fix failures
- [ ] Manual end-to-end test (capture → train → infer)

## Files to Modify

1. `src/services/posture/featureConfig.ts` ✅ (Created)
2. `src/services/posture/types.ts` - Extend `UnifiedFeatures` interface
3. `src/services/posture/detection.ts` - Add ~56 feature computations to `extractAllFeatures()`
4. `src/services/ml/featureExtractors.ts` - Update array conversion (add 56 features)
5. `src/services/ml/featureExtractor.ts` - Update array conversion (add 56 features)
6. `src/services/dataset/featureRegistry.ts` - Update dimensions (36 → 92)
7. `src/services/ml/__tests__/featureExtractors.test.ts` - Update dimension checks
8. `src/services/ml/__tests__/featureExtractor.test.ts` - Update dimension checks

## Breaking Changes

**Model Incompatibility**: All existing trained models will be invalidated (36 → 92 dimensions).

**Migration Strategy**: Force retrain (no UI warnings, no model versioning per user request).

## Success Criteria

- [ ] 92 total engineered features (36 existing + 56 new)
- [ ] All features match Python reference implementation
- [ ] Harmful feature filtering implemented
- [ ] All tests passing
- [ ] All 14 feature batches marked complete
- [ ] Performance maintained (<5ms extraction per frame)

## Notes

- Hand shape features (15 features) deferred - not implemented per user request
- No model versioning or UI warnings needed
- Breaking change to trained models is acceptable

## Hand Feature Alignment (October 23, 2025)

**Issue**: TypeScript implementation had 5 extra hand features not in Python reference, and was missing 5 Python features.

**Features Removed (not in Python)**:
1. `left_pinky_nose_distance` - Python only tracks index/thumb, not pinky
2. `right_pinky_nose_distance` - Python only tracks index/thumb, not pinky
3. `right_hand_above_mouth` - Python only has LEFT hand version
4. `right_hand_above_nose` - Python only has LEFT hand version
5. `right_hand_crosses_center` - Python only has LEFT hand version

**Features Added (from Python reference)**:
1. `left_hand_face_distance_normalized` - Left hand to face distance / head_size
2. `right_hand_face_distance_normalized` - Right hand to face distance / head_size
3. `left_hand_to_shoulder_ratio` - Left wrist to shoulder distance / torso_length
4. `right_hand_to_shoulder_ratio` - Right wrist to shoulder distance / torso_length
5. `hands_near_midline` - Average of left/right hand distance to body centerline

**Result**: Still 19 hand features total (matching Python), but now exactly the correct 19 features.

**Files Modified**:
- `src/services/posture/types.ts` - Updated UnifiedFeatures interface
- `src/services/posture/detection.ts` - Removed 5 old computations, added 5 new ones
- `src/services/ml/featureExtractors.ts` - Updated hand section in array conversion
- `src/services/ml/featureExtractor.ts` - Updated hand section in array conversion
- `src/services/ml/__tests__/featureExtractor.test.ts` - Added new features to mock, fixed snake_case names
