use slouch_domain::{detect_multi_task, BoundingBox, ExpandedBbox};

fn bbox(score: f64) -> ExpandedBbox {
    let box_value = BoundingBox {
        x1: 0.1,
        y1: 0.1,
        x2: 0.9,
        y2: 0.9,
        score,
        width: 0.8,
        height: 0.8,
    };
    ExpandedBbox {
        original: box_value,
        expanded: box_value,
    }
}

#[test]
fn detects_score_above_threshold() {
    assert!(detect_multi_task(Some(&bbox(0.9))).person_found);
}
#[test]
fn rejects_score_below_threshold() {
    assert!(!detect_multi_task(Some(&bbox(0.2))).person_found);
}
#[test]
fn detects_second_score_above_threshold() {
    assert!(detect_multi_task(Some(&bbox(0.4))).person_found);
}
#[test]
fn rejects_missing_bbox() {
    assert!(!detect_multi_task(None).person_found);
}
#[test]
fn includes_exact_threshold() {
    assert!(detect_multi_task(Some(&bbox(0.3))).person_found);
}
#[test]
fn slouching_defaults_false() {
    assert!(!detect_multi_task(Some(&bbox(0.9))).slouching);
}
#[test]
fn forward_neck_tilt_defaults_false() {
    assert!(!detect_multi_task(Some(&bbox(0.9))).forward_neck_tilt);
}
#[test]
fn hand_near_face_defaults_false() {
    assert!(!detect_multi_task(Some(&bbox(0.9))).hand_near_face);
}
#[test]
fn mouth_open_defaults_false() {
    assert!(!detect_multi_task(Some(&bbox(0.9))).mouth_open);
}
#[test]
fn nonfinite_score_is_safe_default() {
    assert_eq!(
        detect_multi_task(Some(&bbox(f64::NAN))),
        detect_multi_task(None)
    );
}
