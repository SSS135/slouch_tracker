use slouch_domain::Keypoint;
use slouch_ml::ported::engineered_features::extract_posture_geometry_features;

const POSTURE_GEOMETRY_DIMS: usize = 10;

fn base_keypoints() -> Vec<Keypoint> {
    // COCO order; only indices 0..=6 (nose, eyes, ears, shoulders) drive the feature set.
    let mut keypoints = vec![Keypoint::new(0.5, 0.7, 0.9); 17];
    keypoints[0] = Keypoint::new(0.50, 0.30, 0.95);
    keypoints[1] = Keypoint::new(0.47, 0.28, 0.90);
    keypoints[2] = Keypoint::new(0.53, 0.28, 0.92);
    keypoints[3] = Keypoint::new(0.44, 0.30, 0.80);
    keypoints[4] = Keypoint::new(0.56, 0.30, 0.85);
    keypoints[5] = Keypoint::new(0.40, 0.50, 0.88);
    keypoints[6] = Keypoint::new(0.60, 0.50, 0.87);
    keypoints
}

// Uniform scale about a center followed by a constant translation: an affine
// map that both cancelling differences and shoulder-width division must ignore.
fn scale_and_translate(keypoints: &[Keypoint], scale: f64, offset: f64) -> Vec<Keypoint> {
    let (center_x, center_y) = (0.5, 0.5);
    keypoints
        .iter()
        .map(|keypoint| {
            Keypoint::new(
                (keypoint.x - center_x) * scale + center_x + offset,
                (keypoint.y - center_y) * scale + center_y + offset,
                keypoint.score,
            )
        })
        .collect()
}

#[test]
fn posture_geometry_has_expected_dimension() {
    let features = extract_posture_geometry_features(&base_keypoints())
        .unwrap()
        .expect("features present for a full keypoint set");

    assert_eq!(features.len(), POSTURE_GEOMETRY_DIMS);
}

#[test]
fn posture_geometry_is_scale_and_translation_invariant() {
    let base = base_keypoints();
    let transformed = scale_and_translate(&base, 2.0, 0.15);

    let base_features = extract_posture_geometry_features(&base).unwrap().unwrap();
    let transformed_features = extract_posture_geometry_features(&transformed)
        .unwrap()
        .unwrap();

    assert_eq!(base_features.len(), transformed_features.len());
    for (index, (original, moved)) in base_features.iter().zip(&transformed_features).enumerate() {
        assert!(
            (original - moved).abs() < 1e-4,
            "feature {index} not invariant: {original} vs {moved}"
        );
    }
}

#[test]
fn posture_geometry_returns_none_for_too_few_keypoints() {
    let too_few = vec![Keypoint::new(0.5, 0.5, 0.9); 6];

    assert_eq!(extract_posture_geometry_features(&too_few).unwrap(), None);
    assert_eq!(extract_posture_geometry_features(&[]).unwrap(), None);
}
