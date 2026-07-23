//! NLF-L 3D depth posture feature (14 dims), a stored posture feature.
//!
//! NLF-L emits, per crop, `coords3d_rel` (`[867, 3]`, meters, box/root-relative;
//! larger z = farther from the camera) and `uncertainty` (`[867]`, meters,
//! softplus-based, LOWER = more confident). This module distills those two raw
//! tensors into a fixed 14-dim posture descriptor built in a body-intrinsic frame
//! so the per-user classifier can separate forward-head flexion from trunk lean.
//!
//! Body frame: `SC = mid(lsho, rsho)`, `HC = mid(lhip, rhip)`, `EC = mid(lear, rear)`
//! (falling back to the nose when both ears are unconfident). `T = SC - HC` is trunk-up;
//! `Ŝ` is the shoulder axis re-orthonormalized against `T̂` (Gram-Schmidt); `F̂ = T̂ × Ŝ`
//! is the body sagittal-forward normal. Offsets are normalized by the torso length `|T|`,
//! angles are radians. All output values are finite; degenerate geometry (collapsed torso,
//! shoulder line parallel to the trunk, zero shoulder width) yields a neutral `0.0` for the
//! ten geometric dims and drives the validity dim to `0.0` — never a NaN or infinity.
//!
//! Honesty note on camera geometry (pilot Phase-4 finding): dims 7-9 read raw camera-frame
//! depth and COLLAPSE under an elevated/above camera (looking down makes the nose always
//! closer than the shoulders regardless of posture) — a geometry limit, not a model one.
//! Dims 1-6 are body-intrinsic 3D angles/offsets that survive camera rotation; the per-user
//! classifier learns which cues to trust for its own camera placement. `NLF_DEPTH_DIMS` is
//! FROZEN at 14 (see constants) — the dim count and order must not change.
//!
//! The 14 dims (1-based):
//!   1. fwd_head_offset  = (H·F̂)/|T|      — forward-head, camera-pitch-invariant (primary)
//!   2. lat_head_offset  = (H·Ŝ)/|T|      — head lateral lean
//!   3. vert_head_offset = (H·T̂)/|T|      — head raise/drop along the trunk
//!   4. neck_trunk_angle = atan2(|H×T|, H·T) — unsigned neck↔trunk flexion angle
//!   5. nose_fwd_offset  = ((nose−SC)·F̂)/|T| — sharper nose-specific forward cue
//!   6. gaze_pitch       = angle(nose−EC, F̂)  — face-forward vs looking-down
//!   7. dz_nose_sho      = (z_nose − z_SC)/|T| — near-frontal forward-head depth
//!   8. dz_ear_sho       = (z_EC − z_SC)/|T|
//!   9. dz_sho_hip       = (z_SC − z_HC)/|T|   — trunk lean toward/away camera
//!  10. shoulder_width_ratio = |rsho−lsho|/|T| — yaw/scale sanity
//!  11. head_uncert  = mean uncert{nose, lear, rear, leye, reye}
//!  12. torso_uncert = mean uncert{lsho, rsho, lhip, rhip}
//!  13. trunc_ratio  = mean uncert{lkne, lank, rkne, rank} / max(torso_uncert, eps)
//!  14. validity ∈ [0,1] — AND-style abstain dim (torso non-degenerate AND posture-joint
//!      confidence AND a head joint present), degrading toward 0 otherwise.

use std::fmt;

use super::constants::{
    NLF_DEPTH_DIMS, NLF_JOINT_LANK, NLF_JOINT_LEAR, NLF_JOINT_LEYE, NLF_JOINT_LHIP, NLF_JOINT_LKNE,
    NLF_JOINT_LSHO, NLF_JOINT_NOSE, NLF_JOINT_RANK, NLF_JOINT_REAR, NLF_JOINT_REYE, NLF_JOINT_RHIP,
    NLF_JOINT_RKNE, NLF_JOINT_RSHO, NLF_NUM_CANONICAL,
};
// Vec3, midpoint/mean, the body frame, and its degeneracy floors are shared with
// `keypoints_3d_features` through `body_frame`; MIN_TORSO_LEN lives there (used only by
// `build_body_frame`), while MIN_AXIS_LEN is still referenced directly below.
use super::body_frame::{build_body_frame, mean, midpoint, BodyFrame, Vec3, MIN_AXIS_LEN};

/// Uncertainty (meters; LOWER = more confident) mapped to full validity confidence.
const UNCERT_CONFIDENT: f64 = 0.05;
/// Uncertainty at or above which a joint contributes zero validity confidence.
const UNCERT_UNUSABLE: f64 = 0.25;
/// Guards the truncation-ratio denominator against a zero torso uncertainty.
const UNCERT_EPS: f64 = 1e-6;

/// Maps a raw NLF joint uncertainty (meters; LOWER = more confident) to a calibrated
/// keypoint confidence in `[0, 1]`, reusing the same band as the depth-feature
/// validity dim: `u <= UNCERT_CONFIDENT` (0.05) is fully confident (1.0) and
/// `u >= UNCERT_UNUSABLE` (0.25) is unusable (0.0), linear in between. Non-finite
/// uncertainty is treated as fully unusable so a pathological model output can never
/// fabricate a confident keypoint.
///
/// Pilot band: confident upper-body `u ~= 0.026-0.03` -> 1.0; truncated legs
/// `u ~= 0.09-0.12` -> ~0.80-0.65; `u >= 0.23` -> < 0.1 (treated absent by the
/// downstream `MIN_KEYPOINT_CONFIDENCE` guard); `u >= 0.25` -> 0.
pub fn uncertainty_to_keypoint_score(u: f64) -> f64 {
    if !u.is_finite() {
        return 0.0;
    }
    ((UNCERT_UNUSABLE - u) / (UNCERT_UNUSABLE - UNCERT_CONFIDENT)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlfFeatureError {
    InvalidCoordsLength { expected: usize, actual: usize },
    InvalidUncertaintyLength { expected: usize, actual: usize },
}

impl fmt::Display for NlfFeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCoordsLength { expected, actual } => write!(
                formatter,
                "nlf coords3d_rel length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidUncertaintyLength { expected, actual } => write!(
                formatter,
                "nlf uncertainty length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for NlfFeatureError {}

/// Extracts the 14-dim NLF depth posture feature from raw NLF-L crop outputs.
///
/// `coords3d_rel` must be `NLF_NUM_CANONICAL * 3` long and `uncertainty`
/// `NLF_NUM_CANONICAL` long, else an `NlfFeatureError` is returned. On valid input
/// this always returns `Some` with exactly `NLF_DEPTH_DIMS` finite values; the
/// geometric dims and the validity dim collapse to a neutral `0.0` when the body
/// frame is degenerate.
pub fn extract_nlf_depth_features(
    coords3d_rel: &[f32],
    uncertainty: &[f32],
) -> Result<Option<Vec<f32>>, NlfFeatureError> {
    let expected_coords = NLF_NUM_CANONICAL * 3;
    if coords3d_rel.len() != expected_coords {
        return Err(NlfFeatureError::InvalidCoordsLength {
            expected: expected_coords,
            actual: coords3d_rel.len(),
        });
    }
    if uncertainty.len() != NLF_NUM_CANONICAL {
        return Err(NlfFeatureError::InvalidUncertaintyLength {
            expected: NLF_NUM_CANONICAL,
            actual: uncertainty.len(),
        });
    }

    let joint = |index: usize| -> Vec3 {
        let base = index * 3;
        Vec3 {
            x: f64::from(coords3d_rel[base]),
            y: f64::from(coords3d_rel[base + 1]),
            z: f64::from(coords3d_rel[base + 2]),
        }
    };
    let uncert = |index: usize| -> f64 { f64::from(uncertainty[index]) };

    let lsho = joint(NLF_JOINT_LSHO);
    let rsho = joint(NLF_JOINT_RSHO);
    let lhip = joint(NLF_JOINT_LHIP);
    let rhip = joint(NLF_JOINT_RHIP);
    let lear = joint(NLF_JOINT_LEAR);
    let rear = joint(NLF_JOINT_REAR);
    let nose = joint(NLF_JOINT_NOSE);

    let shoulder_center = midpoint(lsho, rsho);
    let hip_center = midpoint(lhip, rhip);

    // Head anchor: ear centre, falling back to the nose only when BOTH ears are
    // unconfident (a single reliable ear still yields a usable centre).
    let ears_absent =
        uncert(NLF_JOINT_LEAR) > UNCERT_UNUSABLE && uncert(NLF_JOINT_REAR) > UNCERT_UNUSABLE;
    let ear_center = if ears_absent {
        nose
    } else {
        midpoint(lear, rear)
    };

    // Uncertainty guards are frame-independent and always computable.
    let head_uncerts = [
        uncert(NLF_JOINT_NOSE),
        uncert(NLF_JOINT_LEAR),
        uncert(NLF_JOINT_REAR),
        uncert(NLF_JOINT_LEYE),
        uncert(NLF_JOINT_REYE),
    ];
    let torso_uncert = mean(&[
        uncert(NLF_JOINT_LSHO),
        uncert(NLF_JOINT_RSHO),
        uncert(NLF_JOINT_LHIP),
        uncert(NLF_JOINT_RHIP),
    ]);
    let lower_uncert = mean(&[
        uncert(NLF_JOINT_LKNE),
        uncert(NLF_JOINT_LANK),
        uncert(NLF_JOINT_RKNE),
        uncert(NLF_JOINT_RANK),
    ]);
    let head_uncert = mean(&head_uncerts);
    let trunc_ratio = lower_uncert / torso_uncert.max(UNCERT_EPS);
    // Best (minimum) head-joint confidence encodes "ears OR nose present".
    let head_best_uncert = head_uncerts.into_iter().fold(f64::INFINITY, f64::min);

    // Build the body-intrinsic orthonormal frame (T̂, Ŝ, F̂) with explicit degeneracy guards.
    let trunk = shoulder_center.sub(hip_center);
    let torso_len = trunk.norm();
    let shoulder_axis = rsho.sub(lsho);
    let shoulder_width = shoulder_axis.norm();

    let frame = build_body_frame(trunk, torso_len, shoulder_axis, shoulder_width);

    let geometry = frame.map(
        |BodyFrame {
             trunk_hat,
             shoulder_hat,
             forward_hat,
         }| {
            let head = ear_center.sub(shoulder_center);
            let neck_trunk_angle = head.cross(trunk).norm().atan2(head.dot(trunk));

            let nose_rel = nose.sub(shoulder_center);
            let gaze = nose.sub(ear_center);
            let gaze_pitch = if gaze.norm() >= MIN_AXIS_LEN {
                gaze.cross(forward_hat).norm().atan2(gaze.dot(forward_hat))
            } else {
                0.0
            };

            [
                head.dot(forward_hat) / torso_len,              // 1 fwd_head_offset
                head.dot(shoulder_hat) / torso_len,             // 2 lat_head_offset
                head.dot(trunk_hat) / torso_len,                // 3 vert_head_offset
                neck_trunk_angle,                               // 4 neck_trunk_angle
                nose_rel.dot(forward_hat) / torso_len,          // 5 nose_fwd_offset
                gaze_pitch,                                     // 6 gaze_pitch
                (nose.z - shoulder_center.z) / torso_len,       // 7 dz_nose_sho
                (ear_center.z - shoulder_center.z) / torso_len, // 8 dz_ear_sho
                (shoulder_center.z - hip_center.z) / torso_len, // 9 dz_sho_hip
                shoulder_width / torso_len,                     // 10 shoulder_width_ratio
            ]
        },
    );

    let geometric = geometry.unwrap_or([0.0; 10]);

    // Validity: 0 when the frame is degenerate, otherwise the worst-case (AND-style)
    // confidence over the head and torso joints, mapped from uncertainty to [0, 1].
    let validity = if geometry.is_some() {
        let worst_uncert = head_best_uncert.max(torso_uncert);
        ((UNCERT_UNUSABLE - worst_uncert) / (UNCERT_UNUSABLE - UNCERT_CONFIDENT)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut features = vec![
        geometric[0] as f32,
        geometric[1] as f32,
        geometric[2] as f32,
        geometric[3] as f32,
        geometric[4] as f32,
        geometric[5] as f32,
        geometric[6] as f32,
        geometric[7] as f32,
        geometric[8] as f32,
        geometric[9] as f32,
        head_uncert as f32,
        torso_uncert as f32,
        trunc_ratio as f32,
        validity as f32,
    ];
    // Belt-and-suspenders: a supplementary feature tap must never inject NaN/inf into
    // stored training data even if the model emits a pathological value.
    for value in &mut features {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    debug_assert_eq!(features.len(), NLF_DEPTH_DIMS);

    Ok(Some(features))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ported::constants::{NLF_JOINT_NECK, NLF_JOINT_PELV};

    fn make_input(
        joints: &[(usize, [f64; 3])],
        base_uncert: f64,
        uncert_overrides: &[(usize, f64)],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut coords = vec![0.0_f32; NLF_NUM_CANONICAL * 3];
        let mut uncertainty = vec![base_uncert as f32; NLF_NUM_CANONICAL];
        for (index, point) in joints {
            coords[index * 3] = point[0] as f32;
            coords[index * 3 + 1] = point[1] as f32;
            coords[index * 3 + 2] = point[2] as f32;
        }
        for (index, value) in uncert_overrides {
            uncertainty[*index] = *value as f32;
        }
        (coords, uncertainty)
    }

    /// A plausible near-frontal seated pose (x right, y down, z depth-away).
    fn upright_pose() -> Vec<(usize, [f64; 3])> {
        vec![
            (NLF_JOINT_LSHO, [-0.20, 0.00, 1.24]),
            (NLF_JOINT_RSHO, [0.20, 0.00, 1.24]),
            (NLF_JOINT_LHIP, [-0.15, 0.50, 1.30]),
            (NLF_JOINT_RHIP, [0.15, 0.50, 1.30]),
            (NLF_JOINT_LEAR, [-0.07, -0.22, 1.19]),
            (NLF_JOINT_REAR, [0.07, -0.22, 1.19]),
            (NLF_JOINT_LEYE, [-0.04, -0.25, 1.15]),
            (NLF_JOINT_REYE, [0.04, -0.25, 1.15]),
            (NLF_JOINT_NOSE, [0.00, -0.20, 1.12]),
            (NLF_JOINT_NECK, [0.00, -0.05, 1.23]),
            (NLF_JOINT_PELV, [0.00, 0.50, 1.30]),
            (NLF_JOINT_LKNE, [-0.13, 0.95, 1.32]),
            (NLF_JOINT_LANK, [-0.13, 1.30, 1.34]),
            (NLF_JOINT_RKNE, [0.13, 0.95, 1.32]),
            (NLF_JOINT_RANK, [0.13, 1.30, 1.34]),
        ]
    }

    fn rotate_x(coords: &[f32], phi: f64) -> Vec<f32> {
        let (sin, cos) = phi.sin_cos();
        let mut out = coords.to_vec();
        for index in 0..NLF_NUM_CANONICAL {
            let y = f64::from(coords[index * 3 + 1]);
            let z = f64::from(coords[index * 3 + 2]);
            out[index * 3 + 1] = (y * cos - z * sin) as f32;
            out[index * 3 + 2] = (y * sin + z * cos) as f32;
        }
        out
    }

    /// Applies the proper rotation `Rz(gamma) · Ry(beta) · Rx(alpha)` to every point.
    fn rotate_xyz(coords: &[f32], alpha: f64, beta: f64, gamma: f64) -> Vec<f32> {
        let (sa, ca) = alpha.sin_cos();
        let (sb, cb) = beta.sin_cos();
        let (sg, cg) = gamma.sin_cos();
        // R = Rz * Ry * Rx.
        let r = [
            [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
            [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
            [-sb, cb * sa, cb * ca],
        ];
        let mut out = coords.to_vec();
        for index in 0..NLF_NUM_CANONICAL {
            let x = f64::from(coords[index * 3]);
            let y = f64::from(coords[index * 3 + 1]);
            let z = f64::from(coords[index * 3 + 2]);
            out[index * 3] = (r[0][0] * x + r[0][1] * y + r[0][2] * z) as f32;
            out[index * 3 + 1] = (r[1][0] * x + r[1][1] * y + r[1][2] * z) as f32;
            out[index * 3 + 2] = (r[2][0] * x + r[2][1] * y + r[2][2] * z) as f32;
        }
        out
    }

    fn extract(coords: &[f32], uncertainty: &[f32]) -> Vec<f32> {
        extract_nlf_depth_features(coords, uncertainty)
            .expect("valid lengths")
            .expect("valid pose")
    }

    #[test]
    fn upright_pose_yields_fourteen_finite_dims() {
        let (coords, uncertainty) = make_input(&upright_pose(), 0.03, &[]);
        let features = extract(&coords, &uncertainty);
        assert_eq!(features.len(), NLF_DEPTH_DIMS);
        assert!(features.iter().all(|value| value.is_finite()));
        // Validity should be high for a confident, well-formed pose.
        assert!(features[13] > 0.9, "validity {}", features[13]);
    }

    #[test]
    fn dims_one_to_six_are_invariant_under_3d_rotation() {
        let (coords, uncertainty) = make_input(&upright_pose(), 0.03, &[]);
        let base = extract(&coords, &uncertainty);
        let rotated_coords = rotate_xyz(&coords, 0.3, 0.5, 0.7);
        let rotated = extract(&rotated_coords, &uncertainty);

        for dim in 0..6 {
            assert!(
                (base[dim] - rotated[dim]).abs() < 1e-4,
                "dim {} not rotation-invariant: {} vs {}",
                dim + 1,
                base[dim],
                rotated[dim]
            );
        }
        // The camera-frame depth dims are NOT invariant — a rotation that moves z must
        // change at least one of dims 7-9, proving the intrinsic/camera split is real.
        assert!(
            (0..3).any(|offset| (base[6 + offset] - rotated[6 + offset]).abs() > 1e-3),
            "expected a camera-frame depth dim to change under rotation"
        );
    }

    #[test]
    fn raising_head_uncertainty_raises_dim11_and_lowers_validity() {
        let pose = upright_pose();
        let (coords, low) = make_input(&pose, 0.03, &[]);
        let confident = extract(&coords, &low);

        let head_joints = [
            NLF_JOINT_NOSE,
            NLF_JOINT_LEAR,
            NLF_JOINT_REAR,
            NLF_JOINT_LEYE,
            NLF_JOINT_REYE,
        ];
        let overrides = head_joints
            .iter()
            .map(|&index| (index, 0.20))
            .collect::<Vec<_>>();
        let (_, high) = make_input(&pose, 0.03, &overrides);
        let uncertain = extract(&coords, &high);

        assert!(
            uncertain[10] > confident[10],
            "head_uncert must rise: {} -> {}",
            confident[10],
            uncertain[10]
        );
        assert!(
            uncertain[13] < confident[13],
            "validity must drop: {} -> {}",
            confident[13],
            uncertain[13]
        );
    }

    #[test]
    fn lower_body_truncation_lands_in_the_pilot_band() {
        let pose = upright_pose();
        // Torso joints confident (~0.03), lower body ~3.6x less confident (~0.108),
        // mirroring the pilot's observed 3.4-4.4x truncation ratio.
        let overrides = [
            (NLF_JOINT_LSHO, 0.03),
            (NLF_JOINT_RSHO, 0.03),
            (NLF_JOINT_LHIP, 0.03),
            (NLF_JOINT_RHIP, 0.03),
            (NLF_JOINT_LKNE, 0.108),
            (NLF_JOINT_LANK, 0.108),
            (NLF_JOINT_RKNE, 0.108),
            (NLF_JOINT_RANK, 0.108),
        ];
        let (coords, uncertainty) = make_input(&pose, 0.03, &overrides);
        let features = extract(&coords, &uncertainty);
        let trunc_ratio = features[12];
        assert!(
            (3.0..=4.5).contains(&trunc_ratio),
            "trunc_ratio {trunc_ratio} outside the pilot band"
        );
    }

    /// Places the head at `SC + h_up·T̂ + f·F̂` for a chosen forward-head amount `f` in a
    /// canonical body orientation whose axes are the world axes (T̂ = +Y, Ŝ = +X, F̂ = −Z).
    fn forward_head_pose(forward: f64) -> Vec<(usize, [f64; 3])> {
        let torso = 0.5;
        let h_up = 0.25;
        let ear_half = 0.06;
        // SC = origin, HC below along -Y (so T = SC - HC = +Y). F̂ = T̂ × Ŝ = -Z.
        // Head centre EC = SC + h_up·(+Y) + f·(-Z) = (0, h_up, -f).
        let head = [0.0, h_up, -forward];
        vec![
            (NLF_JOINT_LSHO, [-0.20, 0.00, 0.00]),
            (NLF_JOINT_RSHO, [0.20, 0.00, 0.00]),
            (NLF_JOINT_LHIP, [-0.15, -torso, 0.00]),
            (NLF_JOINT_RHIP, [0.15, -torso, 0.00]),
            (NLF_JOINT_LEAR, [-ear_half, h_up, -forward]),
            (NLF_JOINT_REAR, [ear_half, h_up, -forward]),
            (NLF_JOINT_LEYE, head),
            (NLF_JOINT_REYE, head),
            (NLF_JOINT_NOSE, head),
            (NLF_JOINT_LKNE, [-0.15, -0.95, 0.00]),
            (NLF_JOINT_LANK, [-0.15, -1.30, 0.00]),
            (NLF_JOINT_RKNE, [0.15, -0.95, 0.00]),
            (NLF_JOINT_RANK, [0.15, -1.30, 0.00]),
        ]
    }

    #[test]
    fn above_view_collapses_camera_depth_while_body_intrinsic_survives() {
        // Frontal (canonical): both the body-intrinsic forward-head dim (1) and the
        // camera-frame depth dim (7) separate good (small forward-head) from bad (large).
        let (good_coords, uncertainty) = make_input(&forward_head_pose(0.0), 0.03, &[]);
        let (bad_coords, _) = make_input(&forward_head_pose(0.10), 0.03, &[]);
        let good = extract(&good_coords, &uncertainty);
        let bad = extract(&bad_coords, &uncertainty);

        let dim1_sep = bad[0] - good[0];
        let dim7_sep = bad[6] - good[6];
        assert!(
            dim1_sep.abs() > 0.1,
            "frontal dim1 must separate: {dim1_sep}"
        );
        assert!(
            dim7_sep.abs() > 0.1,
            "frontal dim7 must separate: {dim7_sep}"
        );

        // Above-view: a steep camera pitch about the shoulder axis. The body-intrinsic
        // forward-head dim is rotation-invariant, so its separation is preserved; the
        // camera-frame depth dim collapses toward the shoulder value.
        let phi = 80.0_f64.to_radians();
        let good_above = extract(&rotate_x(&good_coords, phi), &uncertainty);
        let bad_above = extract(&rotate_x(&bad_coords, phi), &uncertainty);

        let dim1_sep_above = bad_above[0] - good_above[0];
        let dim7_sep_above = bad_above[6] - good_above[6];
        assert!(
            (dim1_sep_above - dim1_sep).abs() < 1e-3,
            "dim1 (body-intrinsic) must survive the above view: {dim1_sep} -> {dim1_sep_above}"
        );
        assert!(
            dim7_sep_above.abs() < 0.3 * dim7_sep.abs(),
            "dim7 (camera depth) must collapse: {dim7_sep} -> {dim7_sep_above}"
        );
    }

    #[test]
    fn degenerate_frame_neutralizes_geometry_without_nan() {
        // Zero torso: hips coincide with shoulders (missing/collapsed torso).
        let pose = vec![
            (NLF_JOINT_LSHO, [-0.20, 0.00, 1.24]),
            (NLF_JOINT_RSHO, [0.20, 0.00, 1.24]),
            (NLF_JOINT_LHIP, [-0.20, 0.00, 1.24]),
            (NLF_JOINT_RHIP, [0.20, 0.00, 1.24]),
            (NLF_JOINT_NOSE, [0.00, -0.20, 1.12]),
            (NLF_JOINT_LEAR, [-0.07, -0.22, 1.19]),
            (NLF_JOINT_REAR, [0.07, -0.22, 1.19]),
        ];
        let (coords, uncertainty) = make_input(&pose, 0.03, &[]);
        let features = extract(&coords, &uncertainty);
        assert_eq!(features.len(), NLF_DEPTH_DIMS);
        assert!(features.iter().all(|value| value.is_finite()));
        for (dim, value) in features.iter().take(10).enumerate() {
            assert_eq!(*value, 0.0, "geometric dim {} must be neutral", dim + 1);
        }
        assert_eq!(features[13], 0.0, "validity must be zero on degeneracy");

        // All-zero coordinates (e.g. a wholly absent skeleton) must also stay finite.
        let zeros = vec![0.0_f32; NLF_NUM_CANONICAL * 3];
        let uncertainty = vec![0.03_f32; NLF_NUM_CANONICAL];
        let features = extract(&zeros, &uncertainty);
        assert!(features.iter().all(|value| value.is_finite()));
        assert_eq!(features[13], 0.0);
    }

    #[test]
    fn uncertainty_to_keypoint_score_matches_the_pinned_calibration_band() {
        // Confident upper-body uncertainty saturates at full confidence.
        assert_eq!(uncertainty_to_keypoint_score(0.03), 1.0);
        // The confident-band edge is exactly full confidence.
        assert_eq!(uncertainty_to_keypoint_score(UNCERT_CONFIDENT), 1.0);
        // Anything more confident than the edge is still clamped to 1.0.
        assert_eq!(uncertainty_to_keypoint_score(0.0), 1.0);
        // Mid-band truncated legs land near 0.75.
        assert!((uncertainty_to_keypoint_score(0.10) - 0.75).abs() < 1e-9);
        // The unusable edge and beyond are zero.
        assert_eq!(uncertainty_to_keypoint_score(UNCERT_UNUSABLE), 0.0);
        assert_eq!(uncertainty_to_keypoint_score(0.30), 0.0);
        // Non-finite uncertainty is fully unusable.
        assert_eq!(uncertainty_to_keypoint_score(f64::NAN), 0.0);
        assert_eq!(uncertainty_to_keypoint_score(f64::INFINITY), 0.0);
        assert_eq!(uncertainty_to_keypoint_score(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn uncertainty_to_keypoint_score_is_monotonically_non_increasing() {
        let mut previous = f64::INFINITY;
        for step in 0..=40 {
            let score = uncertainty_to_keypoint_score(f64::from(step) * 0.01);
            assert!((0.0..=1.0).contains(&score));
            assert!(score <= previous + 1e-12, "score rose at step {step}");
            previous = score;
        }
    }

    #[test]
    fn invalid_input_lengths_are_rejected() {
        let uncertainty = vec![0.03_f32; NLF_NUM_CANONICAL];
        let short_coords = vec![0.0_f32; NLF_NUM_CANONICAL * 3 - 1];
        assert!(matches!(
            extract_nlf_depth_features(&short_coords, &uncertainty),
            Err(NlfFeatureError::InvalidCoordsLength { .. })
        ));

        let coords = vec![0.0_f32; NLF_NUM_CANONICAL * 3];
        let short_uncertainty = vec![0.03_f32; NLF_NUM_CANONICAL - 1];
        assert!(matches!(
            extract_nlf_depth_features(&coords, &short_uncertainty),
            Err(NlfFeatureError::InvalidUncertaintyLength { .. })
        ));
    }
}
