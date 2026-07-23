//! Computed 3D posture features built on the hidden `raw_keypoints_3d` substrate.
//!
//! VISION extracts the substrate once per frame from the NLF-L `coords3d_rel` tensor
//! (`extract_raw_keypoints_3d`): 17 COCO keypoints, hip-centered and torso-normalized so
//! `|SC − HC| = 1`, still in camera axes (translated + scaled, NOT rotated). It is stored
//! (hidden from the selector) and consumed by the three computed features here, which
//! rebuild the body-intrinsic orthonormal frame `(T̂, Ŝ, F̂)` via the shared
//! `body_frame::build_body_frame` and read calibrated keypoint scores for their validity dim.
//!
//! Substrate axes: `x` = right, `y` = down, `z` = depth-away-from-camera.
//!
//! All geometric dims collapse to a neutral `0.0` (and validity to `0.0`) when the body
//! frame is degenerate; every output is finite. `torso_invariant_3d` dims 1-2 are
//! *intentional* camera-frame quantities (raw substrate axes), so they change under a 3D
//! rotation while the body-frame dims do not — that is by design, not a bug.

use std::fmt;

use slouch_domain::{
    Keypoint, COCO_KEYPOINT_COUNT, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_EAR,
    RIGHT_EYE, RIGHT_HIP, RIGHT_SHOULDER,
};

use super::body_frame::{build_body_frame, midpoint, BodyFrame, Vec3, MIN_TORSO_LEN};
use super::constants::{
    NLF_COCO17_CANONICAL, NLF_NUM_CANONICAL, POSTURE_GEOMETRY_3D_DIMS, POSTURE_RAW_3D_DIMS,
    RAW_KEYPOINTS_3D_DIMS, TORSO_INVARIANT_3D_DIMS,
};

/// A keypoint at or below this calibrated confidence is treated as absent for the ear
/// fallback, matching the 2D posture extractors (`engineered_features`). Kept low: an
/// absence guard, not a quality threshold.
const MIN_KEYPOINT_CONFIDENCE: f64 = 0.1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Keypoints3dError {
    InvalidCoordsLength { expected: usize, actual: usize },
    InvalidSubstrateLength { expected: usize, actual: usize },
    InvalidKeypointCount { expected: usize, actual: usize },
}

impl fmt::Display for Keypoints3dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCoordsLength { expected, actual } => write!(
                formatter,
                "keypoints_3d coords3d_rel length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidSubstrateLength { expected, actual } => write!(
                formatter,
                "keypoints_3d substrate length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidKeypointCount { expected, actual } => write!(
                formatter,
                "keypoints_3d keypoint count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for Keypoints3dError {}

/// Builds the hidden 3D substrate from the raw NLF-L `coords3d_rel` tensor.
///
/// Picks the 17 COCO keypoints out of the canonical output (COCO order), centers them on
/// the hip midpoint, and scales by the shoulder-hip torso length. Returns `Ok(None)` for a
/// collapsed torso (mirrors `extract_nlf_depth_features`, so VISION skips the insert) and
/// `Err` only when the input tensor has the wrong length. The result is exactly
/// `RAW_KEYPOINTS_3D_DIMS` finite values with the hip midpoint at the origin.
pub fn extract_raw_keypoints_3d(
    coords3d_rel: &[f32],
) -> Result<Option<Vec<f32>>, Keypoints3dError> {
    let expected = NLF_NUM_CANONICAL * 3;
    if coords3d_rel.len() != expected {
        return Err(Keypoints3dError::InvalidCoordsLength {
            expected,
            actual: coords3d_rel.len(),
        });
    }

    let point = |coco_index: usize| -> Vec3 {
        let base = NLF_COCO17_CANONICAL[coco_index] * 3;
        Vec3 {
            x: f64::from(coords3d_rel[base]),
            y: f64::from(coords3d_rel[base + 1]),
            z: f64::from(coords3d_rel[base + 2]),
        }
    };

    let shoulder_center = midpoint(point(LEFT_SHOULDER), point(RIGHT_SHOULDER));
    let hip_center = midpoint(point(LEFT_HIP), point(RIGHT_HIP));
    let torso_len = shoulder_center.sub(hip_center).norm();
    if torso_len < MIN_TORSO_LEN {
        return Ok(None);
    }

    let mut substrate = vec![0.0_f32; RAW_KEYPOINTS_3D_DIMS];
    for coco_index in 0..COCO_KEYPOINT_COUNT {
        let joint = point(coco_index);
        substrate[coco_index * 3] = ((joint.x - hip_center.x) / torso_len) as f32;
        substrate[coco_index * 3 + 1] = ((joint.y - hip_center.y) / torso_len) as f32;
        substrate[coco_index * 3 + 2] = ((joint.z - hip_center.z) / torso_len) as f32;
    }
    scrub(&mut substrate);
    Ok(Some(substrate))
}

/// `posture_raw_3d` (6 dims): body-intrinsic 3D analogues of the raw 2D posture cues.
pub fn extract_posture_raw_3d(
    substrate: Option<&[f32]>,
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, Keypoints3dError> {
    let Some(substrate) = guard(substrate, keypoints)? else {
        return Ok(None);
    };
    let frame = rebuild_frame(substrate);
    let ear_center = ear_center(substrate, keypoints);
    let eye_center = midpoint(joint(substrate, LEFT_EYE), joint(substrate, RIGHT_EYE));
    let ear_axis = joint(substrate, RIGHT_EAR).sub(joint(substrate, LEFT_EAR));

    let geometric = frame.frame.as_ref().map(|basis| {
        let head = ear_center.sub(frame.shoulder_center);
        [
            ear_center.sub(eye_center).dot(basis.trunk_hat), // 1 ear_eye_vert_trunk
            ear_axis
                .dot(basis.forward_hat)
                .atan2(ear_axis.dot(basis.shoulder_hat)), // 2 head_yaw_3d
            head.norm(),                                     // 3 neck_len_ratio
            ear_axis.norm(),                                 // 4 inter_ear_ratio
            head.dot(basis.trunk_hat),                       // 5 head_up_trunk
        ]
    });

    let geometric = geometric.unwrap_or([0.0; 5]);
    let validity = validity(
        frame.frame.is_some(),
        keypoints,
        &[NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE],
    );

    let mut features = vec![
        geometric[0] as f32,
        geometric[1] as f32,
        geometric[2] as f32,
        geometric[3] as f32,
        geometric[4] as f32,
        validity as f32,
    ];
    scrub(&mut features);
    debug_assert_eq!(features.len(), POSTURE_RAW_3D_DIMS);
    Ok(Some(features))
}

/// `posture_geometry_3d` (10 dims): body-intrinsic 3D analogue of `posture_geometry`.
pub fn extract_posture_geometry_3d(
    substrate: Option<&[f32]>,
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, Keypoints3dError> {
    let Some(substrate) = guard(substrate, keypoints)? else {
        return Ok(None);
    };
    let frame = rebuild_frame(substrate);
    let ear_center = ear_center(substrate, keypoints);
    let nose = joint(substrate, NOSE);
    let lear = joint(substrate, LEFT_EAR);
    let rear = joint(substrate, RIGHT_EAR);
    let lsho = joint(substrate, LEFT_SHOULDER);
    let rsho = joint(substrate, RIGHT_SHOULDER);

    let geometric = frame.frame.as_ref().map(|basis| {
        let head = ear_center.sub(frame.shoulder_center);
        let nose_rel = nose.sub(frame.shoulder_center);
        let ear_axis = rear.sub(lear);
        [
            head.dot(basis.trunk_hat),       // 1 head_up_offset
            nose_rel.dot(basis.trunk_hat),   // 2 nose_up_offset
            nose_rel.dot(basis.forward_hat), // 3 nose_fwd_offset
            ear_axis
                .dot(basis.forward_hat)
                .atan2(ear_axis.dot(basis.shoulder_hat)), // 4 head_yaw_3d
            rsho.sub(lsho).dot(basis.trunk_hat), // 5 shoulder_axis_tilt
            lear.sub(lsho).dot(basis.trunk_hat), // 6 lear_up_offset
            rear.sub(rsho).dot(basis.trunk_hat), // 7 rear_up_offset
            head.norm(),                     // 8 neck_len_ratio
            head.cross(basis.trunk_hat)
                .norm()
                .atan2(head.dot(basis.trunk_hat)), // 9 neck_trunk_angle
        ]
    });

    let geometric = geometric.unwrap_or([0.0; 9]);
    let validity = validity(
        frame.frame.is_some(),
        keypoints,
        &[
            NOSE,
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_EAR,
            RIGHT_EAR,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
        ],
    );

    let mut features = geometric
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    features.push(validity as f32);
    scrub(&mut features);
    debug_assert_eq!(features.len(), POSTURE_GEOMETRY_3D_DIMS);
    Ok(Some(features))
}

/// `torso_invariant_3d` (9 dims): trunk-anchored 3D posture, separating head flexion from
/// trunk lean. Dims 3-6 recompute `nlf_depth`'s body-intrinsic concepts from the substrate;
/// dims 1-2 are deliberate camera-frame lean angles (not rotation-invariant); dims 7-8 are
/// 3D-only axial twist and signed sagittal flex.
pub fn extract_torso_invariant_3d(
    substrate: Option<&[f32]>,
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, Keypoints3dError> {
    let Some(substrate) = guard(substrate, keypoints)? else {
        return Ok(None);
    };
    let frame = rebuild_frame(substrate);
    let ear_center = ear_center(substrate, keypoints);
    let lsho = joint(substrate, LEFT_SHOULDER);
    let rsho = joint(substrate, RIGHT_SHOULDER);
    let lhip = joint(substrate, LEFT_HIP);
    let rhip = joint(substrate, RIGHT_HIP);
    let trunk = frame.trunk;

    let geometric = frame.frame.as_ref().map(|basis| {
        let head = ear_center.sub(frame.shoulder_center);
        let shoulder_axis = rsho.sub(lsho);
        let hip_axis = rhip.sub(lhip);
        let fwd_head = head.dot(basis.forward_hat);
        [
            trunk.x.atan2(-trunk.y), // 1 coronal_trunk_lean [camera]
            trunk.z.atan2(-trunk.y), // 2 sagittal_trunk_lean [camera]
            head.cross(basis.trunk_hat)
                .norm()
                .atan2(head.dot(basis.trunk_hat)), // 3 neck_trunk_angle
            fwd_head,                // 4 fwd_head
            head.dot(basis.shoulder_hat), // 5 lat_head
            head.dot(basis.trunk_hat), // 6 head_drop
            shoulder_axis
                .cross(hip_axis)
                .dot(basis.trunk_hat)
                .atan2(shoulder_axis.dot(hip_axis)), // 7 shoulder_hip_twist
            fwd_head.atan2(head.dot(basis.trunk_hat)), // 8 head_sagittal_flex
        ]
    });

    let geometric = geometric.unwrap_or([0.0; 8]);
    let validity = validity(
        frame.frame.is_some(),
        keypoints,
        &[
            NOSE,
            LEFT_EAR,
            RIGHT_EAR,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        ],
    );

    let mut features = geometric
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    features.push(validity as f32);
    scrub(&mut features);
    debug_assert_eq!(features.len(), TORSO_INVARIANT_3D_DIMS);
    Ok(Some(features))
}

/// The rebuilt body frame plus the shoulder-center and trunk vectors it was derived from.
struct RebuiltFrame {
    shoulder_center: Vec3,
    trunk: Vec3,
    frame: Option<BodyFrame>,
}

fn rebuild_frame(substrate: &[f32]) -> RebuiltFrame {
    let lsho = joint(substrate, LEFT_SHOULDER);
    let rsho = joint(substrate, RIGHT_SHOULDER);
    let lhip = joint(substrate, LEFT_HIP);
    let rhip = joint(substrate, RIGHT_HIP);
    let shoulder_center = midpoint(lsho, rsho);
    let hip_center = midpoint(lhip, rhip);
    let trunk = shoulder_center.sub(hip_center);
    let shoulder_axis = rsho.sub(lsho);
    let frame = build_body_frame(trunk, trunk.norm(), shoulder_axis, shoulder_axis.norm());
    RebuiltFrame {
        shoulder_center,
        trunk,
        frame,
    }
}

/// Validates the substrate/keypoints pair and returns the substrate slice, or `Ok(None)`
/// when the substrate is absent.
fn guard<'a>(
    substrate: Option<&'a [f32]>,
    keypoints: &[Keypoint],
) -> Result<Option<&'a [f32]>, Keypoints3dError> {
    let Some(substrate) = substrate else {
        return Ok(None);
    };
    if substrate.len() != RAW_KEYPOINTS_3D_DIMS {
        return Err(Keypoints3dError::InvalidSubstrateLength {
            expected: RAW_KEYPOINTS_3D_DIMS,
            actual: substrate.len(),
        });
    }
    if keypoints.len() < COCO_KEYPOINT_COUNT {
        return Err(Keypoints3dError::InvalidKeypointCount {
            expected: COCO_KEYPOINT_COUNT,
            actual: keypoints.len(),
        });
    }
    Ok(Some(substrate))
}

fn joint(substrate: &[f32], coco_index: usize) -> Vec3 {
    let base = coco_index * 3;
    Vec3 {
        x: f64::from(substrate[base]),
        y: f64::from(substrate[base + 1]),
        z: f64::from(substrate[base + 2]),
    }
}

/// Head anchor: ear centre, falling back to the nose only when BOTH ears are unconfident
/// (a single reliable ear still yields a usable centre).
fn ear_center(substrate: &[f32], keypoints: &[Keypoint]) -> Vec3 {
    if keypoints[LEFT_EAR].score < MIN_KEYPOINT_CONFIDENCE
        && keypoints[RIGHT_EAR].score < MIN_KEYPOINT_CONFIDENCE
    {
        joint(substrate, NOSE)
    } else {
        midpoint(joint(substrate, LEFT_EAR), joint(substrate, RIGHT_EAR))
    }
}

/// Validity dim: the worst calibrated keypoint score over the feature's joints, clamped to
/// `[0, 1]`, or `0.0` when the body frame is degenerate.
fn validity(has_frame: bool, keypoints: &[Keypoint], joints: &[usize]) -> f64 {
    if !has_frame {
        return 0.0;
    }
    joints
        .iter()
        .map(|&index| keypoints[index].score)
        .fold(f64::INFINITY, f64::min)
        .clamp(0.0, 1.0)
}

fn scrub(values: &mut [f32]) {
    for value in values {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ported::constants::NLF_NUM_CANONICAL;
    use crate::ported::nlf_features::extract_nlf_depth_features;

    /// A plausible near-frontal seated pose in COCO-17 order (x right, y down, z depth-away).
    fn coco_pose() -> [[f64; 3]; 17] {
        [
            [0.00, -0.20, 1.12],  // 0 nose
            [-0.04, -0.25, 1.15], // 1 left eye
            [0.04, -0.25, 1.15],  // 2 right eye
            [-0.07, -0.22, 1.19], // 3 left ear
            [0.07, -0.22, 1.19],  // 4 right ear
            [-0.20, 0.00, 1.24],  // 5 left shoulder
            [0.20, 0.00, 1.24],   // 6 right shoulder
            [-0.30, 0.30, 1.26],  // 7 left elbow
            [0.30, 0.30, 1.26],   // 8 right elbow
            [-0.32, 0.55, 1.28],  // 9 left wrist
            [0.32, 0.55, 1.28],   // 10 right wrist
            [-0.15, 0.50, 1.30],  // 11 left hip
            [0.15, 0.50, 1.30],   // 12 right hip
            [-0.13, 0.95, 1.32],  // 13 left knee
            [0.13, 0.95, 1.32],   // 14 right knee
            [-0.13, 1.30, 1.34],  // 15 left ankle
            [0.13, 1.30, 1.34],   // 16 right ankle
        ]
    }

    /// Places a COCO-17 pose at the matching canonical NLF indices so both the substrate
    /// extractor and `extract_nlf_depth_features` read identical points.
    fn canonical_from_coco(coco: &[[f64; 3]; 17]) -> Vec<f32> {
        let mut coords = vec![0.0_f32; NLF_NUM_CANONICAL * 3];
        for (coco_index, point) in coco.iter().enumerate() {
            let canonical = NLF_COCO17_CANONICAL[coco_index];
            coords[canonical * 3] = point[0] as f32;
            coords[canonical * 3 + 1] = point[1] as f32;
            coords[canonical * 3 + 2] = point[2] as f32;
        }
        coords
    }

    fn confident_keypoints() -> Vec<Keypoint> {
        (0..COCO_KEYPOINT_COUNT)
            .map(|_| Keypoint::new(0.0, 0.0, 0.9))
            .collect()
    }

    fn substrate_of(coco: &[[f64; 3]; 17]) -> Vec<f32> {
        extract_raw_keypoints_3d(&canonical_from_coco(coco))
            .expect("valid coords length")
            .expect("non-degenerate torso")
    }

    fn rotation(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
        let (sa, ca) = alpha.sin_cos();
        let (sb, cb) = beta.sin_cos();
        let (sg, cg) = gamma.sin_cos();
        // Rz(gamma) · Ry(beta) · Rx(alpha).
        [
            [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
            [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
            [-sb, cb * sa, cb * ca],
        ]
    }

    fn rotate_substrate(substrate: &[f32], matrix: [[f64; 3]; 3]) -> Vec<f32> {
        let mut out = substrate.to_vec();
        for index in 0..COCO_KEYPOINT_COUNT {
            let x = f64::from(substrate[index * 3]);
            let y = f64::from(substrate[index * 3 + 1]);
            let z = f64::from(substrate[index * 3 + 2]);
            out[index * 3] = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z) as f32;
            out[index * 3 + 1] = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z) as f32;
            out[index * 3 + 2] = (matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z) as f32;
        }
        out
    }

    #[test]
    fn substrate_is_fifty_one_finite_hip_centered_dims() {
        let coords = canonical_from_coco(&coco_pose());
        let substrate = extract_raw_keypoints_3d(&coords)
            .expect("valid length")
            .expect("non-degenerate");
        assert_eq!(substrate.len(), RAW_KEYPOINTS_3D_DIMS);
        assert!(substrate.iter().all(|value| value.is_finite()));
        // The hip midpoint sits at the origin by construction.
        let hip_mid_x = (substrate[LEFT_HIP * 3] + substrate[RIGHT_HIP * 3]) / 2.0;
        let hip_mid_y = (substrate[LEFT_HIP * 3 + 1] + substrate[RIGHT_HIP * 3 + 1]) / 2.0;
        let hip_mid_z = (substrate[LEFT_HIP * 3 + 2] + substrate[RIGHT_HIP * 3 + 2]) / 2.0;
        assert!(hip_mid_x.abs() < 1e-5 && hip_mid_y.abs() < 1e-5 && hip_mid_z.abs() < 1e-5);
        // The shoulder-hip torso length is normalized to 1.
        let sc_y = (substrate[LEFT_SHOULDER * 3 + 1] + substrate[RIGHT_SHOULDER * 3 + 1]) / 2.0;
        let sc_z = (substrate[LEFT_SHOULDER * 3 + 2] + substrate[RIGHT_SHOULDER * 3 + 2]) / 2.0;
        assert!(((sc_y * sc_y + sc_z * sc_z).sqrt() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn substrate_rejects_bad_length_and_skips_collapsed_torso() {
        assert!(matches!(
            extract_raw_keypoints_3d(&[0.0; 10]),
            Err(Keypoints3dError::InvalidCoordsLength { .. })
        ));
        // Shoulders and hips coincide -> zero torso -> skip (Ok(None)).
        let mut collapsed = coco_pose();
        collapsed[LEFT_HIP] = collapsed[LEFT_SHOULDER];
        collapsed[RIGHT_HIP] = collapsed[RIGHT_SHOULDER];
        assert!(extract_raw_keypoints_3d(&canonical_from_coco(&collapsed))
            .expect("valid length")
            .is_none());
    }

    #[test]
    fn valid_pose_yields_correctly_sized_finite_features() {
        let substrate = substrate_of(&coco_pose());
        let keypoints = confident_keypoints();
        for (features, dims) in [
            (
                extract_posture_raw_3d(Some(&substrate), &keypoints),
                POSTURE_RAW_3D_DIMS,
            ),
            (
                extract_posture_geometry_3d(Some(&substrate), &keypoints),
                POSTURE_GEOMETRY_3D_DIMS,
            ),
            (
                extract_torso_invariant_3d(Some(&substrate), &keypoints),
                TORSO_INVARIANT_3D_DIMS,
            ),
        ] {
            let features = features.expect("no error").expect("substrate present");
            assert_eq!(features.len(), dims);
            assert!(features.iter().all(|value| value.is_finite()));
            // Validity dim is the last entry and high for a confident pose.
            assert!(*features.last().expect("validity") > 0.85);
        }
    }

    #[test]
    fn body_frame_dims_are_invariant_under_3d_rotation() {
        let substrate = substrate_of(&coco_pose());
        let keypoints = confident_keypoints();
        let rotated = rotate_substrate(&substrate, rotation(0.3, 0.5, 0.7));

        let raw = extract_posture_raw_3d(Some(&substrate), &keypoints)
            .unwrap()
            .unwrap();
        let raw_rot = extract_posture_raw_3d(Some(&rotated), &keypoints)
            .unwrap()
            .unwrap();
        // posture_raw_3d dims 1-5 are all body-intrinsic; only validity (dim 6) is excluded
        // as it is score-derived (and identical).
        for dim in 0..5 {
            assert!(
                (raw[dim] - raw_rot[dim]).abs() < 1e-4,
                "posture_raw_3d dim {} not rotation-invariant: {} vs {}",
                dim + 1,
                raw[dim],
                raw_rot[dim]
            );
        }

        let geom = extract_posture_geometry_3d(Some(&substrate), &keypoints)
            .unwrap()
            .unwrap();
        let geom_rot = extract_posture_geometry_3d(Some(&rotated), &keypoints)
            .unwrap()
            .unwrap();
        for dim in 0..9 {
            assert!(
                (geom[dim] - geom_rot[dim]).abs() < 1e-4,
                "posture_geometry_3d dim {} not rotation-invariant: {} vs {}",
                dim + 1,
                geom[dim],
                geom_rot[dim]
            );
        }

        let torso = extract_torso_invariant_3d(Some(&substrate), &keypoints)
            .unwrap()
            .unwrap();
        let torso_rot = extract_torso_invariant_3d(Some(&rotated), &keypoints)
            .unwrap()
            .unwrap();
        // Dims 3-8 (indices 2-7) are body-intrinsic and must survive the rotation.
        for dim in 2..8 {
            assert!(
                (torso[dim] - torso_rot[dim]).abs() < 1e-4,
                "torso_invariant_3d body dim {} not rotation-invariant: {} vs {}",
                dim + 1,
                torso[dim],
                torso_rot[dim]
            );
        }
        // Dims 1-2 (indices 0-1) are INTENTIONAL camera-frame non-invariants: they must
        // change under a 3D rotation. Do NOT "fix" this.
        assert!(
            (torso[0] - torso_rot[0]).abs() > 1e-3,
            "torso_invariant_3d dim 1 (coronal, camera-frame) must change under rotation: {} vs {}",
            torso[0],
            torso_rot[0]
        );
        assert!(
            (torso[1] - torso_rot[1]).abs() > 1e-3,
            "torso_invariant_3d dim 2 (sagittal, camera-frame) must change under rotation: {} vs {}",
            torso[1],
            torso_rot[1]
        );
    }

    #[test]
    fn absent_substrate_and_degenerate_frame_stay_neutral_without_nan() {
        let keypoints = confident_keypoints();
        // None substrate -> Ok(None) for every extractor.
        assert!(extract_posture_raw_3d(None, &keypoints).unwrap().is_none());
        assert!(extract_posture_geometry_3d(None, &keypoints)
            .unwrap()
            .is_none());
        assert!(extract_torso_invariant_3d(None, &keypoints)
            .unwrap()
            .is_none());

        // Wrong substrate length -> Err.
        assert!(matches!(
            extract_posture_geometry_3d(Some(&[0.0; 10]), &keypoints),
            Err(Keypoints3dError::InvalidSubstrateLength { .. })
        ));

        // All-zero substrate -> collapsed torso -> neutral geometry + zero validity, all finite.
        let zero = vec![0.0_f32; RAW_KEYPOINTS_3D_DIMS];
        for (features, dims) in [
            (
                extract_posture_raw_3d(Some(&zero), &keypoints),
                POSTURE_RAW_3D_DIMS,
            ),
            (
                extract_posture_geometry_3d(Some(&zero), &keypoints),
                POSTURE_GEOMETRY_3D_DIMS,
            ),
            (
                extract_torso_invariant_3d(Some(&zero), &keypoints),
                TORSO_INVARIANT_3D_DIMS,
            ),
        ] {
            let features = features.expect("no error").expect("substrate present");
            assert_eq!(features.len(), dims);
            assert!(features.iter().all(|value| value.is_finite()));
            assert!(
                features.iter().all(|value| *value == 0.0),
                "must be neutral"
            );
        }
    }

    #[test]
    fn torso_invariant_3d_body_dims_match_nlf_depth_on_the_same_pose() {
        // The substrate rebuilds nlf_depth's body-intrinsic concepts. Dims 3/4/5/6 of
        // torso_invariant_3d must equal nlf_depth dims 4/1/2/3 (neck-trunk angle, forward-,
        // lateral-, and vertical-head offsets) on an identical synthetic pose.
        let coords = canonical_from_coco(&coco_pose());
        let uncertainty = vec![0.03_f32; NLF_NUM_CANONICAL];
        let nlf = extract_nlf_depth_features(&coords, &uncertainty)
            .expect("valid lengths")
            .expect("valid pose");

        let substrate = extract_raw_keypoints_3d(&coords)
            .expect("valid length")
            .expect("non-degenerate");
        let torso = extract_torso_invariant_3d(Some(&substrate), &confident_keypoints())
            .unwrap()
            .unwrap();

        // torso dim 3 (index 2) == nlf dim 4 (index 3): neck-trunk angle.
        assert!(
            (torso[2] - nlf[3]).abs() < 1e-3,
            "neck_trunk_angle mismatch: {} vs {}",
            torso[2],
            nlf[3]
        );
        // torso dim 4 (index 3) == nlf dim 1 (index 0): forward-head offset.
        assert!(
            (torso[3] - nlf[0]).abs() < 1e-3,
            "fwd_head mismatch: {} vs {}",
            torso[3],
            nlf[0]
        );
        // torso dim 5 (index 4) == nlf dim 2 (index 1): lateral-head offset.
        assert!(
            (torso[4] - nlf[1]).abs() < 1e-3,
            "lat_head mismatch: {} vs {}",
            torso[4],
            nlf[1]
        );
        // torso dim 6 (index 5) == nlf dim 3 (index 2): vertical-head offset.
        assert!(
            (torso[5] - nlf[2]).abs() < 1e-3,
            "head_drop mismatch: {} vs {}",
            torso[5],
            nlf[2]
        );
    }
}
