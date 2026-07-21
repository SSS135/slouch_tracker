use slouch_domain::{
    Keypoint, LEFT_EAR, LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_HIP, RIGHT_SHOULDER,
};
use slouch_ml::ported::engineered_features::{
    extract_posture_geometry_features, extract_torso_invariant_features,
};

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

// ===========================================================================
// torso_invariant
// ===========================================================================

const TORSO_INVARIANT_DIMS: usize = 7;
// Torso-invariant output layout (fixed order).
const TORSO_INCLINATION: usize = 0;
const NECK_VS_TORSO_ANGLE: usize = 1;
const FORWARD_HEAD_RATIO: usize = 2;
const HEAD_DROP_TORSO_NORM: usize = 3;
const SHOULDER_HIP_TILT: usize = 4;
const LEAN_IN_RATIO: usize = 5;
const MIN_CONF: usize = 6;

// An upright, well-detected 17-keypoint skeleton with hips present. Shoulders sit
// above hips (smaller image-y), head above shoulders, everything vertically aligned:
// shoulder_center = hip_center = (0.50, x) so the trunk is plumb and centred.
fn torso_base() -> Vec<Keypoint> {
    let mut keypoints = vec![Keypoint::new(0.5, 0.7, 0.9); 17];
    keypoints[NOSE] = Keypoint::new(0.50, 0.30, 0.95);
    keypoints[LEFT_EAR] = Keypoint::new(0.44, 0.30, 0.80);
    keypoints[RIGHT_EAR] = Keypoint::new(0.56, 0.30, 0.85);
    keypoints[LEFT_SHOULDER] = Keypoint::new(0.40, 0.50, 0.88);
    keypoints[RIGHT_SHOULDER] = Keypoint::new(0.60, 0.50, 0.87);
    keypoints[LEFT_HIP] = Keypoint::new(0.43, 0.80, 0.75);
    keypoints[RIGHT_HIP] = Keypoint::new(0.57, 0.80, 0.76);
    keypoints
}

fn torso_features(keypoints: &[Keypoint]) -> Vec<f32> {
    extract_torso_invariant_features(keypoints)
        .unwrap()
        .expect("torso-invariant features present for a full keypoint set")
}

fn translate(keypoints: &[Keypoint], dx: f64, dy: f64) -> Vec<Keypoint> {
    keypoints
        .iter()
        .map(|keypoint| Keypoint::new(keypoint.x + dx, keypoint.y + dy, keypoint.score))
        .collect()
}

fn scale_about(keypoints: &[Keypoint], center_x: f64, center_y: f64, scale: f64) -> Vec<Keypoint> {
    keypoints
        .iter()
        .map(|keypoint| {
            Keypoint::new(
                (keypoint.x - center_x) * scale + center_x,
                (keypoint.y - center_y) * scale + center_y,
                keypoint.score,
            )
        })
        .collect()
}

fn assert_all_finite(features: &[f32]) {
    assert!(
        features.iter().all(|value| value.is_finite()),
        "non-finite feature in {features:?}"
    );
}

// The math is exactly invariant in f64; the API returns f32, so the residual is pure
// f32 quantization (~1e-7 near unit magnitude). 1e-9 is not representable in f32.
const INVARIANCE_TOLERANCE: f32 = 1e-5;

fn assert_vectors_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() < INVARIANCE_TOLERANCE,
            "dim {index} differs: {a} vs {b}"
        );
    }
}

#[test]
fn torso_invariant_has_expected_dimension_and_base_values() {
    let features = torso_features(&torso_base());
    assert_eq!(features.len(), TORSO_INVARIANT_DIMS);
    assert_all_finite(&features);

    // Concrete anchors for the plumb, centred base skeleton:
    //   trunk vertical -> inclination 0; head centred over shoulders -> forward 0;
    //   head 0.20 above shoulders over a 0.30 torso -> drop 0.6667;
    //   inter-ear 0.12 over shoulder-width 0.20 -> lean-in 0.6; min score 0.75.
    assert!(features[TORSO_INCLINATION].abs() < 1e-6);
    assert!(features[FORWARD_HEAD_RATIO].abs() < 1e-6);
    assert!((features[HEAD_DROP_TORSO_NORM] - 0.6667).abs() < 1e-3);
    assert!((features[LEAN_IN_RATIO] - 0.6).abs() < 1e-3);
    assert!((features[MIN_CONF] - 0.75).abs() < 1e-6);
}

#[test]
fn torso_invariant_is_translation_invariant() {
    let base = torso_features(&torso_base());
    let moved = torso_features(&translate(&torso_base(), 0.21, -0.13));
    assert_vectors_close(&base, &moved);
}

#[test]
fn torso_invariant_is_uniform_scale_invariant_about_any_point() {
    let base = torso_features(&torso_base());
    // Scale about an off-centre point to prove invariance is not tied to the centroid.
    let scaled = torso_features(&scale_about(&torso_base(), 0.2, 0.9, 2.5));
    assert_vectors_close(&base, &scaled);
}

#[test]
fn torso_invariant_forward_head_ratio_grows_when_head_pushes_forward() {
    let base = torso_features(&torso_base());
    // Slide the whole head anchor (both ears) forward in +x.
    let mut forward = torso_base();
    forward[LEFT_EAR].x += 0.12;
    forward[RIGHT_EAR].x += 0.12;
    let forward = torso_features(&forward);
    assert!(
        forward[FORWARD_HEAD_RATIO] > base[FORWARD_HEAD_RATIO],
        "forward-head ratio did not increase: {} -> {}",
        base[FORWARD_HEAD_RATIO],
        forward[FORWARD_HEAD_RATIO]
    );
}

#[test]
fn torso_invariant_inclination_grows_when_trunk_tilts() {
    let base = torso_features(&torso_base());
    // Shift both shoulders sideways so the shoulder-hip axis leans off vertical.
    let mut tilted = torso_base();
    tilted[LEFT_SHOULDER].x += 0.14;
    tilted[RIGHT_SHOULDER].x += 0.14;
    let tilted = torso_features(&tilted);
    assert!(
        tilted[TORSO_INCLINATION].abs() > base[TORSO_INCLINATION].abs() + 0.05,
        "trunk tilt did not raise |inclination|: {} -> {}",
        base[TORSO_INCLINATION],
        tilted[TORSO_INCLINATION]
    );
}

#[test]
fn torso_invariant_disambiguates_head_flexion_from_trunk_lean() {
    // Head flexion only: move the head anchor (ears) while shoulders and hips — and
    // therefore the torso axis — stay fixed. neck_vs_torso_angle must react; the
    // trunk inclination must not. This is the feature's whole reason to exist.
    let base = torso_features(&torso_base());
    let mut flexed = torso_base();
    flexed[LEFT_EAR].x += 0.10;
    flexed[RIGHT_EAR].x += 0.10;
    flexed[LEFT_EAR].y += 0.06;
    flexed[RIGHT_EAR].y += 0.06;
    let flexed = torso_features(&flexed);

    assert!(
        (flexed[TORSO_INCLINATION] - base[TORSO_INCLINATION]).abs() < INVARIANCE_TOLERANCE,
        "torso inclination moved under head-only flexion: {} -> {}",
        base[TORSO_INCLINATION],
        flexed[TORSO_INCLINATION]
    );
    assert!(
        (flexed[NECK_VS_TORSO_ANGLE] - base[NECK_VS_TORSO_ANGLE]).abs() > 0.1,
        "neck-vs-torso angle failed to track head flexion: {} -> {}",
        base[NECK_VS_TORSO_ANGLE],
        flexed[NECK_VS_TORSO_ANGLE]
    );
}

#[test]
fn torso_invariant_returns_none_without_hip_keypoints() {
    // Fewer than 13 keypoints cannot address the hips (indices 11/12).
    let too_few = vec![Keypoint::new(0.5, 0.5, 0.9); 12];
    assert_eq!(extract_torso_invariant_features(&too_few).unwrap(), None);
    assert_eq!(extract_torso_invariant_features(&[]).unwrap(), None);
}

#[test]
fn torso_invariant_neutralizes_degenerate_zero_length_torso() {
    // Hips coincide with shoulders: the torso axis collapses to zero length.
    let mut degenerate = torso_base();
    degenerate[LEFT_HIP] = Keypoint::new(0.40, 0.50, 0.90);
    degenerate[RIGHT_HIP] = Keypoint::new(0.60, 0.50, 0.90);
    let features = torso_features(&degenerate);

    assert_all_finite(&features);
    for dim in [
        TORSO_INCLINATION,
        NECK_VS_TORSO_ANGLE,
        HEAD_DROP_TORSO_NORM,
        SHOULDER_HIP_TILT,
    ] {
        assert_eq!(features[dim], 0.0, "dim {dim} should be neutral");
    }
}

#[test]
fn torso_invariant_neutralizes_absent_hips() {
    // Hips present in the array but scored as absent (below the confidence guard).
    let mut absent = torso_base();
    absent[LEFT_HIP].score = 0.0;
    absent[RIGHT_HIP].score = 0.0;
    let features = torso_features(&absent);

    assert_all_finite(&features);
    for dim in [
        TORSO_INCLINATION,
        NECK_VS_TORSO_ANGLE,
        HEAD_DROP_TORSO_NORM,
        SHOULDER_HIP_TILT,
    ] {
        assert_eq!(
            features[dim], 0.0,
            "dim {dim} should be neutral for absent hips"
        );
    }
}

#[test]
fn torso_invariant_stays_finite_under_uniform_low_confidence() {
    // Every used keypoint is below the confidence guard: torso dims neutralize, the
    // ear anchor falls back to the nose, and min_conf faithfully reports the low score.
    let mut low = torso_base();
    for keypoint in &mut low {
        keypoint.score = 0.05;
    }
    let features = torso_features(&low);

    assert_all_finite(&features);
    assert_eq!(features[TORSO_INCLINATION], 0.0);
    assert_eq!(features[NECK_VS_TORSO_ANGLE], 0.0);
    assert_eq!(features[SHOULDER_HIP_TILT], 0.0);
    assert!((features[MIN_CONF] - 0.05).abs() < 1e-6);
}
