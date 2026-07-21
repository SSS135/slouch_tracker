use slouch_domain::ported::detect_multi_task;
use slouch_domain::{BoundingBox, ExpandedBbox, InferenceResult};

fn create_mock_bbox(score: f64) -> ExpandedBbox {
    let original = BoundingBox {
        x1: 0.1,
        y1: 0.1,
        x2: 0.9,
        y2: 0.9,
        score,
        width: 0.8,
        height: 0.8,
    };

    ExpandedBbox {
        original,
        expanded: original,
    }
}

#[test]
fn detects_person_when_bbox_score_is_above_threshold() {
    let bbox = create_mock_bbox(0.9);
    assert!(detect_multi_task(Some(&bbox)).person_found);
}

#[test]
fn does_not_detect_person_when_bbox_score_is_below_threshold() {
    let bbox = create_mock_bbox(0.2);
    assert!(!detect_multi_task(Some(&bbox)).person_found);
}

#[test]
fn detects_person_when_bbox_score_is_above_threshold_at_zero_point_four() {
    let bbox = create_mock_bbox(0.4);
    assert!(detect_multi_task(Some(&bbox)).person_found);
}

#[test]
fn does_not_detect_person_when_bbox_is_missing() {
    assert!(!detect_multi_task(None).person_found);
}

#[test]
fn detects_person_when_bbox_score_equals_threshold() {
    let bbox = create_mock_bbox(0.3);
    assert!(detect_multi_task(Some(&bbox)).person_found);
}

#[test]
fn slouching_is_false_without_the_ml_classifier() {
    let bbox = create_mock_bbox(0.9);
    assert!(!detect_multi_task(Some(&bbox)).slouching);
}

#[test]
fn forward_neck_tilt_is_false_without_the_ml_classifier() {
    let bbox = create_mock_bbox(0.9);
    assert!(!detect_multi_task(Some(&bbox)).forward_neck_tilt);
}

#[test]
fn hand_near_face_is_false_without_hand_keypoints() {
    let bbox = create_mock_bbox(0.9);
    assert!(!detect_multi_task(Some(&bbox)).hand_near_face);
}

#[test]
fn mouth_open_is_false_without_face_keypoints() {
    let bbox = create_mock_bbox(0.9);
    assert!(!detect_multi_task(Some(&bbox)).mouth_open);
}

#[test]
fn bbox_score_still_controls_person_detection_with_empty_keypoints() {
    let result = InferenceResult {
        features: Default::default(),
        keypoints: Vec::new(),
        bbox: create_mock_bbox(0.9),
        classification: None,
    };

    let detection = detect_multi_task(Some(&result.bbox));

    assert!(detection.person_found);
    assert!(!detection.slouching);
    assert!(!detection.forward_neck_tilt);
    assert!(!detection.hand_near_face);
    assert!(!detection.mouth_open);
}
