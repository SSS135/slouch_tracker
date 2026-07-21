use std::fmt;

use serde::Serialize;
use slouch_domain::{
    Keypoint, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_EYE, RIGHT_HIP,
    RIGHT_SHOULDER,
};

use super::binning::{soft_bin_with_fixed_edges, BinningError};
use super::constants::{
    ENGINEERED_1D_DIMS, ENGINEERED_2D_DIMS, ENGINEERED_3D_DIMS, ENGINEERED_4D_DIMS, NUM_SOFT_BINS,
    NUM_SOFT_BINS_5, RAW_KEYPOINTS_DIMS,
};

const MIN_DENOMINATOR: f64 = 0.001;
const MIN_KEYPOINTS_REQUIRED: usize = 7;
const RAW_KEYPOINT_COUNT: usize = 17;

/// Absent-keypoint guard for the torso-invariant feature. NLF emits all 17 COCO
/// keypoints every frame, each carrying a calibrated confidence in [0, 1] (see
/// `nlf_features::uncertainty_to_keypoint_score`); an out-of-frame joint comes back
/// with a low score. A keypoint at or below this bar is treated as absent so it
/// cannot fabricate a bogus torso axis; residual quality is still surfaced through
/// the feature's `min_conf` dim rather than being filtered here. Kept low on purpose
/// — this is an absence guard, not a quality threshold — so genuinely weak-but-present
/// joints still inform the model. Note: unlike SimCC, NLF reports confident
/// coordinates for occluded hips, so `torso_reliable` is now often true for seated
/// users (accepted).
const MIN_KEYPOINT_CONFIDENCE: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GeometricFeatureResult {
    pub value: f64,
    pub min_confidence: f64,
    pub valid: bool,
}

/// The aggregate geometry record in the same insertion order as the source
/// TypeScript object.  A fixed struct keeps keyed access while making
/// iteration and serialization deterministic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeometricFeatureMap {
    pub neck_length: GeometricFeatureResult,
    pub ear_eye_vertical: GeometricFeatureResult,
    pub head_rotation: GeometricFeatureResult,
    pub neck_shoulder_ratio: GeometricFeatureResult,
    pub neck_eye_ratio: GeometricFeatureResult,
    pub neck_ear_ratio: GeometricFeatureResult,
}

impl GeometricFeatureMap {
    pub fn get(&self, name: &str) -> Option<&GeometricFeatureResult> {
        match name {
            "neck_length" => Some(&self.neck_length),
            "ear_eye_vertical" => Some(&self.ear_eye_vertical),
            "head_rotation" => Some(&self.head_rotation),
            "neck_shoulder_ratio" => Some(&self.neck_shoulder_ratio),
            "neck_eye_ratio" => Some(&self.neck_eye_ratio),
            "neck_ear_ratio" => Some(&self.neck_ear_ratio),
            _ => None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &GeometricFeatureResult)> {
        [
            ("neck_length", &self.neck_length),
            ("ear_eye_vertical", &self.ear_eye_vertical),
            ("head_rotation", &self.head_rotation),
            ("neck_shoulder_ratio", &self.neck_shoulder_ratio),
            ("neck_eye_ratio", &self.neck_eye_ratio),
            ("neck_ear_ratio", &self.neck_ear_ratio),
        ]
        .into_iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineeredFeaturesError {
    InsufficientKeypoints { required: usize, actual: usize },
    Binning(BinningError),
    DimensionMismatch { expected: usize, actual: usize },
    EdgeShapeMismatch { first: usize, second: usize },
    UnknownFeature { name: &'static str },
}

impl fmt::Display for EngineeredFeaturesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientKeypoints { required, actual } => write!(
                formatter,
                "expected at least {required} keypoints, got {actual}"
            ),
            Self::Binning(error) => error.fmt(formatter),
            Self::DimensionMismatch { expected, actual } => {
                write!(formatter, "expected {expected} dims, got {actual}")
            }
            Self::EdgeShapeMismatch { first, second } => {
                write!(
                    formatter,
                    "joint histogram edge counts differ: {first} and {second}"
                )
            }
            Self::UnknownFeature { name } => write!(formatter, "unknown geometric feature: {name}"),
        }
    }
}

impl std::error::Error for EngineeredFeaturesError {}

impl From<BinningError> for EngineeredFeaturesError {
    fn from(error: BinningError) -> Self {
        Self::Binning(error)
    }
}

#[derive(Debug, Clone, Copy)]
struct Midpoint {
    x: f64,
    y: f64,
}

pub const FIXED_BIN_EDGES: [(&str, [f64; 9]); 6] = [
    (
        "neck_shoulder_ratio",
        [
            0.670, 0.712, 0.736, 0.755, 0.769, 0.781, 0.793, 0.810, 0.883,
        ],
    ),
    (
        "neck_eye_ratio",
        [
            2.591, 2.826, 2.980, 3.095, 3.193, 3.273, 3.394, 3.584, 6.197,
        ],
    ),
    (
        "neck_ear_ratio",
        [
            1.063, 1.146, 1.197, 1.243, 1.288, 1.326, 1.367, 1.420, 1.545,
        ],
    ),
    (
        "ear_eye_vertical",
        [
            -0.014, -0.002, 0.002, 0.005, 0.006, 0.008, 0.010, 0.013, 0.016,
        ],
    ),
    (
        "head_rotation",
        [
            -3.091, 0.258, 2.979, 3.038, 3.058, 3.077, 3.092, 3.105, 3.124,
        ],
    ),
    (
        "neck_length",
        [
            0.073, 0.183, 0.196, 0.204, 0.210, 0.214, 0.218, 0.223, 0.228,
        ],
    ),
];

pub const FIXED_BIN_EDGES_5: [(&str, [f64; 5]); 4] = [
    ("ear_eye_vertical", [-0.014, 0.002, 0.006, 0.010, 0.016]),
    ("head_rotation", [-3.091, 2.979, 3.058, 3.092, 3.124]),
    ("neck_length", [0.073, 0.196, 0.210, 0.218, 0.228]),
    ("inter_ear_distance", [0.038, 0.160, 0.165, 0.171, 0.179]),
];

const GEOMETRY_FEATURE_ORDER: [&str; 6] = [
    "neck_shoulder_ratio",
    "neck_eye_ratio",
    "neck_ear_ratio",
    "ear_eye_vertical",
    "head_rotation",
    "neck_length",
];

pub fn compute_neck_length(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_SHOULDER + 1)?;
    let left_shoulder = keypoints[LEFT_SHOULDER];
    let right_shoulder = keypoints[RIGHT_SHOULDER];
    let left_eye = keypoints[LEFT_EYE];
    let right_eye = keypoints[RIGHT_EYE];

    let neck_base = midpoint(left_shoulder, right_shoulder);
    let neck_top = midpoint(left_eye, right_eye);
    let dx = neck_top.x - neck_base.x;
    let dy = neck_top.y - neck_base.y;

    Ok(GeometricFeatureResult {
        value: (dx * dx + dy * dy).sqrt(),
        min_confidence: min_confidence(&[
            left_shoulder.score,
            right_shoulder.score,
            left_eye.score,
            right_eye.score,
        ]),
        valid: true,
    })
}

pub fn compute_ear_eye_vertical(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_EAR + 1)?;
    let left_ear = keypoints[LEFT_EAR];
    let right_ear = keypoints[RIGHT_EAR];
    let left_eye = keypoints[LEFT_EYE];
    let right_eye = keypoints[RIGHT_EYE];

    let average_ear_y = (left_ear.y + right_ear.y) / 2.0;
    let average_eye_y = (left_eye.y + right_eye.y) / 2.0;

    Ok(GeometricFeatureResult {
        value: average_ear_y - average_eye_y,
        min_confidence: min_confidence(&[
            left_ear.score,
            right_ear.score,
            left_eye.score,
            right_eye.score,
        ]),
        valid: true,
    })
}

pub fn compute_head_rotation(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_EAR + 1)?;
    let left_ear = keypoints[LEFT_EAR];
    let right_ear = keypoints[RIGHT_EAR];
    let left_eye = keypoints[LEFT_EYE];
    let right_eye = keypoints[RIGHT_EYE];

    let average_dx = ((right_ear.x - left_ear.x) + (right_eye.x - left_eye.x)) / 2.0;
    let average_dy = ((right_ear.y - left_ear.y) + (right_eye.y - left_eye.y)) / 2.0;

    Ok(GeometricFeatureResult {
        value: average_dy.atan2(average_dx),
        min_confidence: min_confidence(&[
            left_ear.score,
            right_ear.score,
            left_eye.score,
            right_eye.score,
        ]),
        valid: true,
    })
}

pub fn compute_shoulder_width(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_SHOULDER + 1)?;
    let left_shoulder = keypoints[LEFT_SHOULDER];
    let right_shoulder = keypoints[RIGHT_SHOULDER];

    Ok(GeometricFeatureResult {
        value: distance_2d(left_shoulder, right_shoulder),
        min_confidence: min_confidence(&[left_shoulder.score, right_shoulder.score]),
        valid: true,
    })
}

pub fn compute_inter_eye_distance(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_EYE + 1)?;
    let left_eye = keypoints[LEFT_EYE];
    let right_eye = keypoints[RIGHT_EYE];

    Ok(GeometricFeatureResult {
        value: distance_2d(left_eye, right_eye),
        min_confidence: min_confidence(&[left_eye.score, right_eye.score]),
        valid: true,
    })
}

pub fn compute_inter_ear_distance(
    keypoints: &[Keypoint],
) -> Result<GeometricFeatureResult, EngineeredFeaturesError> {
    ensure_keypoints_for(keypoints, RIGHT_EAR + 1)?;
    let left_ear = keypoints[LEFT_EAR];
    let right_ear = keypoints[RIGHT_EAR];

    Ok(GeometricFeatureResult {
        value: distance_2d(left_ear, right_ear),
        min_confidence: min_confidence(&[left_ear.score, right_ear.score]),
        valid: true,
    })
}

pub fn extract_all_geometric_features(
    keypoints: &[Keypoint],
) -> Result<Option<GeometricFeatureMap>, EngineeredFeaturesError> {
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }

    let neck_length = compute_neck_length(keypoints)?;
    let shoulder_width = compute_shoulder_width(keypoints)?;
    let inter_eye_distance = compute_inter_eye_distance(keypoints)?;
    let inter_ear_distance = compute_inter_ear_distance(keypoints)?;
    let ear_eye_vertical = compute_ear_eye_vertical(keypoints)?;
    let head_rotation = compute_head_rotation(keypoints)?;

    Ok(Some(GeometricFeatureMap {
        neck_length,
        ear_eye_vertical,
        head_rotation,
        neck_shoulder_ratio: compute_ratio(neck_length, shoulder_width),
        neck_eye_ratio: compute_ratio(neck_length, inter_eye_distance),
        neck_ear_ratio: compute_ratio(neck_length, inter_ear_distance),
    }))
}

pub fn extract_engineered_features(
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(geometry) = extract_all_geometric_features(keypoints)? else {
        return Ok(None);
    };

    let mut result = vec![0.0_f32; ENGINEERED_1D_DIMS];
    let mut offset = 0;
    for name in GEOMETRY_FEATURE_ORDER {
        let feature = geometry
            .get(name)
            .ok_or(EngineeredFeaturesError::UnknownFeature { name })?;
        let edges = fixed_edges(name)?;
        let probabilities =
            soft_bin_with_fixed_edges(feature.value, feature.min_confidence, edges)?;
        let end = offset + probabilities.len();
        if end > result.len() {
            return Err(EngineeredFeaturesError::DimensionMismatch {
                expected: result.len(),
                actual: end,
            });
        }
        result[offset..end].copy_from_slice(&probabilities);
        offset = end;
    }

    if offset != ENGINEERED_1D_DIMS {
        return Err(EngineeredFeaturesError::DimensionMismatch {
            expected: ENGINEERED_1D_DIMS,
            actual: offset,
        });
    }
    Ok(Some(result))
}

pub fn extract_joint_2d_features(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }
    let Some(geometry) = extract_all_geometric_features(keypoints)? else {
        return Ok(None);
    };

    let ear_eye_vertical = geometry_value(&geometry, "ear_eye_vertical")?;
    let neck_length = geometry_value(&geometry, "neck_length")?;
    if !ear_eye_vertical.valid || !neck_length.valid {
        return Ok(None);
    }

    let result = compute_joint_histogram_2d(
        ear_eye_vertical.value,
        ear_eye_vertical.min_confidence,
        fixed_edges("ear_eye_vertical")?,
        neck_length.value,
        neck_length.min_confidence,
        fixed_edges("neck_length")?,
    )?;
    ensure_dimension(result.len(), ENGINEERED_2D_DIMS)?;
    Ok(Some(result))
}

pub fn extract_joint_3d_features(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }
    let Some(geometry) = extract_all_geometric_features(keypoints)? else {
        return Ok(None);
    };

    let ear_eye_vertical = geometry_value(&geometry, "ear_eye_vertical")?;
    let neck_length = geometry_value(&geometry, "neck_length")?;
    let head_rotation = geometry_value(&geometry, "head_rotation")?;
    if !ear_eye_vertical.valid || !neck_length.valid || !head_rotation.valid {
        return Ok(None);
    }

    let result = compute_joint_histogram_3d([
        (
            ear_eye_vertical.value,
            ear_eye_vertical.min_confidence,
            fixed_edges_5("ear_eye_vertical")?,
        ),
        (
            neck_length.value,
            neck_length.min_confidence,
            fixed_edges_5("neck_length")?,
        ),
        (
            head_rotation.value,
            head_rotation.min_confidence,
            fixed_edges_5("head_rotation")?,
        ),
    ])?;
    ensure_dimension(result.len(), ENGINEERED_3D_DIMS)?;
    Ok(Some(result))
}

pub fn extract_joint_4d_features(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }
    let Some(geometry) = extract_all_geometric_features(keypoints)? else {
        return Ok(None);
    };

    let ear_eye_vertical = geometry_value(&geometry, "ear_eye_vertical")?;
    let head_rotation = geometry_value(&geometry, "head_rotation")?;
    let neck_length = geometry_value(&geometry, "neck_length")?;
    let inter_ear_distance = compute_inter_ear_distance(keypoints)?;
    if !ear_eye_vertical.valid || !head_rotation.valid || !neck_length.valid {
        return Ok(None);
    }

    let result = compute_joint_histogram_4d([
        (
            ear_eye_vertical.value,
            ear_eye_vertical.min_confidence,
            fixed_edges_5("ear_eye_vertical")?,
        ),
        (
            head_rotation.value,
            head_rotation.min_confidence,
            fixed_edges_5("head_rotation")?,
        ),
        (
            neck_length.value,
            neck_length.min_confidence,
            fixed_edges_5("neck_length")?,
        ),
        (
            inter_ear_distance.value,
            inter_ear_distance.min_confidence,
            fixed_edges_5("inter_ear_distance")?,
        ),
    ])?;
    ensure_dimension(result.len(), ENGINEERED_4D_DIMS)?;
    Ok(Some(result))
}

pub fn extract_posture_raw_features(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }
    let Some(geometry) = extract_all_geometric_features(keypoints)? else {
        return Ok(None);
    };

    let ear_eye_vertical = geometry_value(&geometry, "ear_eye_vertical")?;
    let head_rotation = geometry_value(&geometry, "head_rotation")?;
    let neck_length = geometry_value(&geometry, "neck_length")?;
    let inter_ear_distance = compute_inter_ear_distance(keypoints)?;
    let average_shoulder_y = (keypoints[LEFT_SHOULDER].y + keypoints[RIGHT_SHOULDER].y) / 2.0;

    if !ear_eye_vertical.valid || !head_rotation.valid || !neck_length.valid {
        return Ok(None);
    }

    Ok(Some(vec![
        ear_eye_vertical.value as f32,
        head_rotation.value as f32,
        neck_length.value as f32,
        inter_ear_distance.value as f32,
        average_shoulder_y as f32,
    ]))
}

pub fn extract_posture_geometry_features(
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    if keypoints.len() < MIN_KEYPOINTS_REQUIRED {
        return Ok(None);
    }

    let nose = keypoints[NOSE];
    let left_eye = keypoints[LEFT_EYE];
    let right_eye = keypoints[RIGHT_EYE];
    let left_ear = keypoints[LEFT_EAR];
    let right_ear = keypoints[RIGHT_EAR];
    let left_shoulder = keypoints[LEFT_SHOULDER];
    let right_shoulder = keypoints[RIGHT_SHOULDER];

    let shoulder_mid = midpoint(left_shoulder, right_shoulder);
    let ear_mid = midpoint(left_ear, right_ear);
    let shoulder_width = compute_shoulder_width(keypoints)?;

    let min_conf = min_confidence(&[
        nose.score,
        left_eye.score,
        right_eye.score,
        left_ear.score,
        right_ear.score,
        left_shoulder.score,
        right_shoulder.score,
    ]);

    // Each span is divided by shoulder_width through compute_ratio so a
    // degenerate (near-zero) shoulder width yields a defined 0.0 rather than a
    // NaN; the span differences cancel translation and the division cancels
    // uniform scale, leaving every entry scale/translation-invariant.
    let normalized = |span: f64| {
        compute_ratio(
            GeometricFeatureResult {
                value: span,
                min_confidence: min_conf,
                valid: true,
            },
            shoulder_width,
        )
        .value as f32
    };

    Ok(Some(vec![
        normalized(shoulder_mid.y - ear_mid.y),
        normalized(shoulder_mid.y - nose.y),
        normalized(nose.x - shoulder_mid.x),
        compute_head_rotation(keypoints)?.value as f32,
        normalized(left_shoulder.y - right_shoulder.y),
        normalized(left_shoulder.y - left_ear.y),
        normalized(right_shoulder.y - right_ear.y),
        compute_ratio(compute_neck_length(keypoints)?, shoulder_width).value as f32,
        compute_ratio(compute_ear_eye_vertical(keypoints)?, shoulder_width).value as f32,
        min_conf as f32,
    ]))
}

/// Torso-anchored posture geometry (7 dims), scale/translation-invariant.
///
/// Every other posture feature measures the head against image-vertical spans, so
/// head flexion ("looking down") and trunk slouch alias onto the same numbers. This
/// feature is the only one that uses the hips: it builds the shoulder-hip torso axis
/// and expresses head pose *relative to the trunk*, letting a classifier separate the
/// two failure modes.
///
/// Axis-mixing note: keypoints arrive image-normalized and ANISOTROPIC (`x` by width,
/// `y` by height; see `inference_worker::transform_keypoints`), so any cross-axis
/// quantity carries a constant per-camera W/H warp. Same-axis ratios (dims 3, 4, 5, 6)
/// are warp-free by construction; the two angle dims (1, 2) accept the constant warp
/// exactly like the existing `head_rotation` feature — it is fixed per camera and
/// absorbed by z-score normalization.
///
/// Robustness: the four torso-dependent dims (1, 2, 4, 5) emit a neutral `0.0` when the
/// shoulders or hips are absent (low confidence) or the axis is geometrically
/// degenerate, via the same `compute_ratio`/guard convention used elsewhere; no dim can
/// be NaN or infinite.
pub fn extract_torso_invariant_features(
    keypoints: &[Keypoint],
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    // Hips (indices 11/12) are required, so demand the full lower-body index range
    // rather than the 7-keypoint head/shoulder minimum the other extractors use.
    if keypoints.len() < RIGHT_HIP + 1 {
        return Ok(None);
    }

    let nose = keypoints[NOSE];
    let left_ear = keypoints[LEFT_EAR];
    let right_ear = keypoints[RIGHT_EAR];
    let left_shoulder = keypoints[LEFT_SHOULDER];
    let right_shoulder = keypoints[RIGHT_SHOULDER];
    let left_hip = keypoints[LEFT_HIP];
    let right_hip = keypoints[RIGHT_HIP];

    let shoulder_center = midpoint(left_shoulder, right_shoulder);
    let hip_center = midpoint(left_hip, right_hip);

    // Head anchor E: ear centre, falling back to the nose only when BOTH ears are
    // absent (a single reliable ear still yields a usable centre).
    let ear_center =
        if left_ear.score < MIN_KEYPOINT_CONFIDENCE && right_ear.score < MIN_KEYPOINT_CONFIDENCE {
            Midpoint {
                x: nose.x,
                y: nose.y,
            }
        } else {
            midpoint(left_ear, right_ear)
        };

    let min_conf = min_confidence(&[
        nose.score,
        left_ear.score,
        right_ear.score,
        left_shoulder.score,
        right_shoulder.score,
        left_hip.score,
        right_hip.score,
    ]);

    // The torso axis is trustworthy only when both shoulders and both hips are present;
    // an absent hip would otherwise fabricate a large, meaningless axis.
    let torso_reliable = left_shoulder.score >= MIN_KEYPOINT_CONFIDENCE
        && right_shoulder.score >= MIN_KEYPOINT_CONFIDENCE
        && left_hip.score >= MIN_KEYPOINT_CONFIDENCE
        && right_hip.score >= MIN_KEYPOINT_CONFIDENCE;

    // Torso vector T = shoulder_center - hip_center (mostly vertical). |T| is computed
    // normally despite the cross-axis warp; it only gates degeneracy, not a stored dim.
    let torso_x = shoulder_center.x - hip_center.x;
    let torso_y = shoulder_center.y - hip_center.y;
    let torso_len = (torso_x * torso_x + torso_y * torso_y).sqrt();
    // Same-axis vertical extent of the torso; the y/y denominator for dims 4 and 5.
    let torso_len_y = as_feature((shoulder_center.y - hip_center.y).abs(), min_conf);

    let shoulder_width = compute_shoulder_width(keypoints)?;
    let inter_ear = compute_inter_ear_distance(keypoints)?;

    // Dim 1 — trunk lean vs image vertical (signed). The torso vector T = SC - HC
    // points up-image, i.e. toward negative y (image coordinates increase downward),
    // so the deviation from the vertical axis is atan2(T.x, -T.y): 0 is a plumb trunk
    // and the magnitude grows as it tilts. Depends only on shoulders + hips, never E,
    // which is exactly what lets it stay fixed while dim 2 tracks head flexion.
    let torso_inclination = if torso_reliable && torso_len >= MIN_DENOMINATOR {
        torso_x.atan2(-torso_y)
    } else {
        0.0
    };

    // Dim 2 — head flexion RELATIVE to the trunk: the signed angle of the neck vector
    // (shoulder_center -> E) against the torso axis T. This is the disambiguator —
    // "looking down" moves this while leaving dim 1 (which ignores E) unchanged.
    let neck_x = ear_center.x - shoulder_center.x;
    let neck_y = ear_center.y - shoulder_center.y;
    let neck_len = (neck_x * neck_x + neck_y * neck_y).sqrt();
    let neck_vs_torso_angle =
        if torso_reliable && torso_len >= MIN_DENOMINATOR && neck_len >= MIN_DENOMINATOR {
            // atan2(cross, dot) of the neck vector relative to the torso axis; signed.
            (neck_x * torso_y - neck_y * torso_x).atan2(neck_x * torso_x + neck_y * torso_y)
        } else {
            0.0
        };

    // Dim 3 — forward-head horizontal offset, same-axis (x span / x-dominant width).
    let forward_head_ratio = compute_ratio(
        as_feature(ear_center.x - shoulder_center.x, min_conf),
        shoulder_width,
    )
    .value;

    // Dim 4 — head drop normalized by the torso's vertical extent, same-axis (y / y);
    // yaw-robust because torso height barely changes as the trunk rotates in yaw.
    let head_drop_torso_norm = if torso_reliable {
        compute_ratio(
            as_feature(shoulder_center.y - ear_center.y, min_conf),
            torso_len_y,
        )
        .value
    } else {
        0.0
    };

    // Dim 5 — shoulder-line tilt relative to the hip-line, normalized same-axis (y / y).
    let shoulder_hip_tilt = if torso_reliable {
        compute_ratio(
            as_feature(
                (left_shoulder.y - right_shoulder.y) - (left_hip.y - right_hip.y),
                min_conf,
            ),
            torso_len_y,
        )
        .value
    } else {
        0.0
    };

    // Dim 6 — dimensionless lean-in / face-proximity proxy, same-axis (x-dominant / x).
    let lean_in_ratio = compute_ratio(inter_ear, shoulder_width).value;

    Ok(Some(vec![
        torso_inclination as f32,
        neck_vs_torso_angle as f32,
        forward_head_ratio as f32,
        head_drop_torso_norm as f32,
        shoulder_hip_tilt as f32,
        lean_in_ratio as f32,
        min_conf as f32,
    ]))
}

pub fn extract_raw_keypoints(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, EngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < RAW_KEYPOINT_COUNT {
        return Ok(None);
    }

    let mut result = vec![0.0_f32; RAW_KEYPOINTS_DIMS];
    for index in 0..RAW_KEYPOINT_COUNT {
        result[index * 2] = keypoints[index].x as f32;
        result[index * 2 + 1] = keypoints[index].y as f32;
    }
    Ok(Some(result))
}

fn distance_2d(first: Keypoint, second: Keypoint) -> f64 {
    let dx = first.x - second.x;
    let dy = first.y - second.y;
    (dx * dx + dy * dy).sqrt()
}

fn midpoint(first: Keypoint, second: Keypoint) -> Midpoint {
    Midpoint {
        x: (first.x + second.x) / 2.0,
        y: (first.y + second.y) / 2.0,
    }
}

/// Wraps a raw span/length into a `GeometricFeatureResult` so it can flow through
/// `compute_ratio`'s degenerate-denominator guard with a shared confidence.
fn as_feature(value: f64, min_confidence: f64) -> GeometricFeatureResult {
    GeometricFeatureResult {
        value,
        min_confidence,
        valid: true,
    }
}

fn compute_ratio(
    numerator: GeometricFeatureResult,
    denominator: GeometricFeatureResult,
) -> GeometricFeatureResult {
    let denominator_too_small = denominator.value < MIN_DENOMINATOR;
    GeometricFeatureResult {
        value: if denominator_too_small {
            0.0
        } else {
            numerator.value / denominator.value
        },
        min_confidence: min_confidence(&[numerator.min_confidence, denominator.min_confidence]),
        valid: !denominator_too_small,
    }
}

fn ensure_keypoints_for(
    keypoints: &[Keypoint],
    required: usize,
) -> Result<(), EngineeredFeaturesError> {
    if keypoints.len() < required {
        Err(EngineeredFeaturesError::InsufficientKeypoints {
            required,
            actual: keypoints.len(),
        })
    } else {
        Ok(())
    }
}

fn ensure_dimension(actual: usize, expected: usize) -> Result<(), EngineeredFeaturesError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EngineeredFeaturesError::DimensionMismatch { expected, actual })
    }
}

fn min_confidence(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, js_min)
}

fn js_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.min(right)
    }
}

fn geometry_value<'a>(
    geometry: &'a GeometricFeatureMap,
    name: &'static str,
) -> Result<&'a GeometricFeatureResult, EngineeredFeaturesError> {
    geometry
        .get(name)
        .ok_or(EngineeredFeaturesError::UnknownFeature { name })
}

fn fixed_edges(name: &'static str) -> Result<&'static [f64], EngineeredFeaturesError> {
    FIXED_BIN_EDGES
        .iter()
        .find(|(feature, _)| *feature == name)
        .map(|(_, edges)| edges.as_slice())
        .ok_or(EngineeredFeaturesError::UnknownFeature { name })
}

fn fixed_edges_5(name: &'static str) -> Result<&'static [f64], EngineeredFeaturesError> {
    FIXED_BIN_EDGES_5
        .iter()
        .find(|(feature, _)| *feature == name)
        .map(|(_, edges)| edges.as_slice())
        .ok_or(EngineeredFeaturesError::UnknownFeature { name })
}

fn compute_joint_histogram_2d(
    value1: f64,
    confidence1: f64,
    edges1: &[f64],
    value2: f64,
    confidence2: f64,
    edges2: &[f64],
) -> Result<Vec<f32>, EngineeredFeaturesError> {
    if edges1.len() != edges2.len() {
        return Err(EngineeredFeaturesError::EdgeShapeMismatch {
            first: edges1.len(),
            second: edges2.len(),
        });
    }
    let probabilities1 = soft_bin_with_fixed_edges(value1, confidence1, edges1)?;
    let probabilities2 = soft_bin_with_fixed_edges(value2, confidence2, edges2)?;
    let bins = edges1.len();
    let mut joint = vec![0.0_f32; bins * bins];
    for i in 0..bins {
        for j in 0..bins {
            joint[i * bins + j] =
                (f64::from(probabilities1[i]) * f64::from(probabilities2[j])) as f32;
        }
    }
    Ok(joint)
}

fn compute_joint_histogram_3d(
    axes: [(f64, f64, &[f64]); 3],
) -> Result<Vec<f32>, EngineeredFeaturesError> {
    let [(value1, confidence1, edges1), (value2, confidence2, edges2), (value3, confidence3, edges3)] =
        axes;
    ensure_equal_edge_counts(&[edges1, edges2, edges3])?;
    let probabilities1 = soft_bin_with_fixed_edges(value1, confidence1, edges1)?;
    let probabilities2 = soft_bin_with_fixed_edges(value2, confidence2, edges2)?;
    let probabilities3 = soft_bin_with_fixed_edges(value3, confidence3, edges3)?;
    let bins = edges1.len();
    let mut joint = vec![0.0_f32; bins * bins * bins];
    for (i, probability1) in probabilities1.iter().enumerate() {
        for (j, probability2) in probabilities2.iter().enumerate() {
            for (k, probability3) in probabilities3.iter().enumerate() {
                joint[i * bins * bins + j * bins + k] = (f64::from(*probability1)
                    * f64::from(*probability2)
                    * f64::from(*probability3))
                    as f32;
            }
        }
    }
    Ok(joint)
}

fn compute_joint_histogram_4d(
    axes: [(f64, f64, &[f64]); 4],
) -> Result<Vec<f32>, EngineeredFeaturesError> {
    let [(value1, confidence1, edges1), (value2, confidence2, edges2), (value3, confidence3, edges3), (value4, confidence4, edges4)] =
        axes;
    ensure_equal_edge_counts(&[edges1, edges2, edges3, edges4])?;
    let probabilities1 = soft_bin_with_fixed_edges(value1, confidence1, edges1)?;
    let probabilities2 = soft_bin_with_fixed_edges(value2, confidence2, edges2)?;
    let probabilities3 = soft_bin_with_fixed_edges(value3, confidence3, edges3)?;
    let probabilities4 = soft_bin_with_fixed_edges(value4, confidence4, edges4)?;
    let bins = edges1.len();
    let mut joint = vec![0.0_f32; bins * bins * bins * bins];
    for (i, probability1) in probabilities1.iter().enumerate() {
        for (j, probability2) in probabilities2.iter().enumerate() {
            for (k, probability3) in probabilities3.iter().enumerate() {
                for (l, probability4) in probabilities4.iter().enumerate() {
                    let index = i * bins * bins * bins + j * bins * bins + k * bins + l;
                    joint[index] = (f64::from(*probability1)
                        * f64::from(*probability2)
                        * f64::from(*probability3)
                        * f64::from(*probability4)) as f32;
                }
            }
        }
    }
    Ok(joint)
}

fn ensure_equal_edge_counts(edges: &[&[f64]]) -> Result<(), EngineeredFeaturesError> {
    let first = edges.first().map_or(0, |values| values.len());
    if let Some(values) = edges.iter().find(|values| values.len() != first) {
        return Err(EngineeredFeaturesError::EdgeShapeMismatch {
            first,
            second: values.len(),
        });
    }
    Ok(())
}

const _: () = {
    assert!(NUM_SOFT_BINS == 9);
    assert!(NUM_SOFT_BINS_5 == 5);
    assert!(ENGINEERED_1D_DIMS == 54);
    assert!(ENGINEERED_2D_DIMS == 81);
    assert!(ENGINEERED_3D_DIMS == 125);
    assert!(ENGINEERED_4D_DIMS == 625);
    assert!(RAW_KEYPOINTS_DIMS == RAW_KEYPOINT_COUNT * 2);
};
