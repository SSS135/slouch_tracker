use slouch_domain::{BoundingBox, Keypoint};
use slouch_ml::ported::{
    constants::{NUM_SOFT_BINS, RTMDET_ENGINEERED_DIMS},
    rtmdet_engineered_features::{
        compute_avg_keypoint_group_score_for_logging, compute_face_visibility_ratio_for_logging,
        compute_torso_height_ratio_for_logging, compute_upper_body_ratio_for_logging,
        extract_keypoint_scores_feature, extract_rtm_det_engineered_features, BODY_INDICES,
        FACE_INDICES, HEAD_SHOULDERS_INDICES, LEGS_FEET_INDICES, RTMDET_FIXED_BIN_EDGES,
    },
};

const NUM_KEYPOINTS: usize = 17;
const GRID_SIZE: usize = 9;

fn create_mock_keypoints(scores: Option<&[f64]>) -> Vec<Keypoint> {
    let positions = [
        (0.5, 0.1),   // NOSE
        (0.48, 0.08), // LEFT_EYE
        (0.52, 0.08), // RIGHT_EYE
        (0.45, 0.1),  // LEFT_EAR
        (0.55, 0.1),  // RIGHT_EAR
        (0.4, 0.25),  // LEFT_SHOULDER
        (0.6, 0.25),  // RIGHT_SHOULDER
        (0.35, 0.4),  // LEFT_ELBOW
        (0.65, 0.4),  // RIGHT_ELBOW
        (0.3, 0.5),   // LEFT_WRIST
        (0.7, 0.5),   // RIGHT_WRIST
        (0.45, 0.55), // LEFT_HIP
        (0.55, 0.55), // RIGHT_HIP
        (0.42, 0.75), // LEFT_KNEE
        (0.58, 0.75), // RIGHT_KNEE
        (0.4, 0.95),  // LEFT_ANKLE
        (0.6, 0.95),  // RIGHT_ANKLE
    ];

    positions
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| {
            Keypoint::new(
                x,
                y,
                scores
                    .and_then(|values| values.get(index).copied())
                    .unwrap_or(0.9),
            )
        })
        .collect()
}

fn create_mock_bbox(x1: f64, y1: f64, x2: f64, y2: f64, score: f64) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        score,
        width: x2 - x1,
        height: y2 - y1,
    }
}

fn default_bbox() -> BoundingBox {
    create_mock_bbox(0.2, 0.1, 0.8, 0.9, 0.95)
}

#[test]
fn dimension_consistency_returns_135_f32_values() {
    let keypoints = create_mock_keypoints(None);
    let bbox = default_bbox();

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert_eq!(result.len(), 135);
}

#[test]
fn dimension_consistency_returns_135_zeros_when_bbox_is_missing() {
    let keypoints = create_mock_keypoints(None);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), None).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| *value == 0.0));
}

#[test]
fn dimension_consistency_returns_135_zeros_when_bbox_and_keypoints_are_missing() {
    let result = extract_rtm_det_engineered_features(None, None).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| *value == 0.0));
}

#[test]
fn dimension_consistency_handles_bbox_without_keypoints() {
    let bbox = default_bbox();

    let result = extract_rtm_det_engineered_features(None, Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().any(|value| *value != 0.0));
}

#[test]
fn coverage_grid_is_zero_when_bbox_is_missing() {
    let result = extract_rtm_det_engineered_features(None, None).unwrap();

    let grid_values = &result[..GRID_SIZE * GRID_SIZE];
    assert!(grid_values.iter().all(|value| *value == 0.0));
}

#[test]
fn coverage_grid_has_nonzero_values_for_covered_cells() {
    let bbox = create_mock_bbox(0.3, 0.3, 0.7, 0.7, 0.95);

    let result = extract_rtm_det_engineered_features(None, Some(&bbox)).unwrap();

    let grid_values = &result[..GRID_SIZE * GRID_SIZE];
    assert!(grid_values.iter().any(|value| *value > 0.0));
}

#[test]
fn coverage_grid_has_expected_overlap_for_centered_bbox() {
    let bbox = create_mock_bbox(0.333, 0.333, 0.667, 0.667, 0.95);

    let result = extract_rtm_det_engineered_features(None, Some(&bbox)).unwrap();
    let grid_values = &result[..GRID_SIZE * GRID_SIZE];

    assert!(grid_values[4 * GRID_SIZE + 4] > 0.0);
    assert_eq!(grid_values[0], 0.0);
}

#[test]
fn coverage_grid_is_fully_covered_by_full_frame_bbox() {
    let bbox = create_mock_bbox(0.0, 0.0, 1.0, 1.0, 0.95);

    let result = extract_rtm_det_engineered_features(None, Some(&bbox)).unwrap();
    let grid_values = &result[..GRID_SIZE * GRID_SIZE];

    assert!(grid_values
        .iter()
        .all(|value| f64::from((*value - 1.0).abs()) < 0.01));
}

#[test]
fn keypoint_scores_extracts_all_17_scores() {
    let scores = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
    ];
    let keypoints = create_mock_keypoints(Some(&scores));

    let result = extract_keypoint_scores_feature(Some(&keypoints)).unwrap();

    assert_eq!(result.len(), NUM_KEYPOINTS);
    for (actual, expected) in result.iter().zip(scores) {
        assert!((f64::from(*actual) - expected).abs() < 5e-6);
    }
}

#[test]
fn keypoint_scores_returns_none_for_missing_keypoints() {
    assert!(extract_keypoint_scores_feature(None).is_none());
}

#[test]
fn keypoint_scores_returns_none_for_partial_keypoints() {
    let keypoints = create_mock_keypoints(None);
    let partial_keypoints = &keypoints[..10];

    assert!(extract_keypoint_scores_feature(Some(partial_keypoints)).is_none());
}

#[test]
fn soft_binning_groups_each_sum_to_one() {
    let keypoints = create_mock_keypoints(None);
    let bbox = default_bbox();
    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();
    let offsets = [
        GRID_SIZE * GRID_SIZE,
        GRID_SIZE * GRID_SIZE + NUM_SOFT_BINS,
        GRID_SIZE * GRID_SIZE + 2 * NUM_SOFT_BINS,
        GRID_SIZE * GRID_SIZE + 3 * NUM_SOFT_BINS,
        GRID_SIZE * GRID_SIZE + 4 * NUM_SOFT_BINS,
        GRID_SIZE * GRID_SIZE + 5 * NUM_SOFT_BINS,
    ];

    for offset in offsets {
        let sum: f64 = result[offset..offset + NUM_SOFT_BINS]
            .iter()
            .map(|value| f64::from(*value))
            .sum();
        assert!((sum - 1.0).abs() < 5e-5);
    }
}

#[test]
fn soft_binning_handles_zero_confidence() {
    let scores = [0.0; NUM_KEYPOINTS];
    let keypoints = create_mock_keypoints(Some(&scores));
    let bbox = create_mock_bbox(0.2, 0.1, 0.8, 0.9, 0.0);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| value.is_finite()));
}

#[test]
fn soft_binning_low_confidence_is_nearly_uniform() {
    let scores = [0.1; NUM_KEYPOINTS];
    let keypoints = create_mock_keypoints(Some(&scores));
    let bbox = create_mock_bbox(0.3, 0.2, 0.7, 0.8, 0.1);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();
    let offset = GRID_SIZE * GRID_SIZE + 3 * NUM_SOFT_BINS;
    let values = &result[offset..offset + NUM_SOFT_BINS];
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);

    assert!(max - min < 0.5);
}

#[test]
fn edge_cases_handle_low_score_keypoints() {
    let scores = [
        0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.05, 0.05, 0.02, 0.02, 0.9, 0.9, 0.1, 0.1, 0.05, 0.05,
    ];
    let keypoints = create_mock_keypoints(Some(&scores));
    let bbox = default_bbox();

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| value.is_finite()));
}

#[test]
fn edge_cases_handle_bbox_at_frame_origin() {
    let bbox = create_mock_bbox(0.0, 0.0, 0.5, 0.5, 0.95);
    let keypoints = create_mock_keypoints(None);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| value.is_finite()));
    assert!(result[0] > 0.0);
}

#[test]
fn edge_cases_handle_bbox_covering_entire_frame() {
    let bbox = create_mock_bbox(0.0, 0.0, 1.0, 1.0, 0.95);
    let keypoints = create_mock_keypoints(None);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();
    let grid_values = &result[..GRID_SIZE * GRID_SIZE];

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(grid_values
        .iter()
        .all(|value| f64::from((*value - 1.0).abs()) < 0.01));
}

#[test]
fn edge_cases_handle_very_small_bbox() {
    let bbox = create_mock_bbox(0.49, 0.49, 0.51, 0.51, 0.95);
    let keypoints = create_mock_keypoints(None);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result.iter().all(|value| value.is_finite()));
}

#[test]
fn edge_cases_handle_bbox_at_bottom_right_corner() {
    let bbox = create_mock_bbox(0.5, 0.5, 1.0, 1.0, 0.95);
    let keypoints = create_mock_keypoints(None);

    let result = extract_rtm_det_engineered_features(Some(&keypoints), Some(&bbox)).unwrap();

    assert_eq!(result.len(), RTMDET_ENGINEERED_DIMS);
    assert!(result[GRID_SIZE * GRID_SIZE - 1] > 0.0);
}

#[test]
fn logging_torso_height_ratio_returns_valid_feature_value() {
    let keypoints = create_mock_keypoints(None);
    let result = compute_torso_height_ratio_for_logging(&keypoints, 0.8).unwrap();

    assert!(result.valid);
    assert!(result.value > 0.0);
    assert!(result.confidence > 0.0);
}

#[test]
fn logging_torso_height_ratio_rejects_zero_height() {
    let keypoints = create_mock_keypoints(None);
    let result = compute_torso_height_ratio_for_logging(&keypoints, 0.0).unwrap();

    assert!(!result.valid);
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn logging_upper_body_ratio_returns_valid_feature_value() {
    let keypoints = create_mock_keypoints(None);
    let bbox = default_bbox();
    let result = compute_upper_body_ratio_for_logging(&keypoints, &bbox).unwrap();

    assert!(result.valid);
    assert!(result.value > 0.0);
    assert!(result.value < 1.0);
}

#[test]
fn logging_average_keypoint_group_score_computes_the_expected_average() {
    let scores: Vec<f64> = (0..NUM_KEYPOINTS)
        .map(|index| index as f64 * 0.05)
        .collect();
    let keypoints = create_mock_keypoints(Some(&scores));
    let actual = compute_avg_keypoint_group_score_for_logging(&keypoints, &HEAD_SHOULDERS_INDICES);
    let expected = HEAD_SHOULDERS_INDICES
        .iter()
        .map(|index| scores[*index])
        .sum::<f64>()
        / HEAD_SHOULDERS_INDICES.len() as f64;

    assert!((actual - expected).abs() < 5e-6);
}

#[test]
fn logging_face_visibility_ratio_returns_valid_ratio() {
    let keypoints = create_mock_keypoints(None);
    let result = compute_face_visibility_ratio_for_logging(&keypoints).unwrap();

    assert!(result.valid);
    assert!(result.value > 0.0);
}

#[test]
fn logging_face_visibility_ratio_rejects_zero_body_scores() {
    let scores = [0.0; NUM_KEYPOINTS];
    let keypoints = create_mock_keypoints(Some(&scores));
    let result = compute_face_visibility_ratio_for_logging(&keypoints).unwrap();

    assert!(!result.valid);
}

#[test]
fn exported_head_shoulders_indices_match_the_source_contract() {
    for index in [0, 1, 2, 5, 6] {
        assert!(HEAD_SHOULDERS_INDICES.contains(&index));
    }
    assert_eq!(HEAD_SHOULDERS_INDICES.len(), 7);
}

#[test]
fn exported_legs_feet_indices_match_the_source_contract() {
    for index in [13, 14, 15, 16] {
        assert!(LEGS_FEET_INDICES.contains(&index));
    }
    assert_eq!(LEGS_FEET_INDICES.len(), 4);
}

#[test]
fn exported_face_indices_match_the_source_contract() {
    for index in [0, 1, 2, 3, 4] {
        assert!(FACE_INDICES.contains(&index));
    }
    assert_eq!(FACE_INDICES.len(), 5);
}

#[test]
fn exported_body_indices_match_the_source_contract() {
    for index in [5, 6, 11, 12] {
        assert!(BODY_INDICES.contains(&index));
    }
    assert_eq!(BODY_INDICES.len(), 4);
}

#[test]
fn fixed_bin_edges_contain_all_expected_feature_keys() {
    for key in [
        "torso_height_ratio",
        "head_shoulders_avg_score",
        "legs_feet_avg_score",
        "bbox_aspect_ratio",
        "upper_body_ratio",
        "face_visibility_ratio",
    ] {
        assert!(RTMDET_FIXED_BIN_EDGES
            .iter()
            .any(|(feature, _)| *feature == key));
    }
}

#[test]
fn fixed_bin_edges_all_have_nine_values() {
    assert!(RTMDET_FIXED_BIN_EDGES
        .iter()
        .all(|(_, edges)| edges.len() == NUM_SOFT_BINS));
}

#[test]
fn fixed_bin_edges_are_sorted_in_ascending_order() {
    assert!(RTMDET_FIXED_BIN_EDGES
        .iter()
        .all(|(_, edges)| { edges.windows(2).all(|window| window[1] > window[0]) }));
}
