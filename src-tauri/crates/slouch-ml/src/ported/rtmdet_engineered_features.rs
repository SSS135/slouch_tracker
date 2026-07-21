use std::fmt;

use slouch_domain::{
    BoundingBox, Keypoint, LEFT_ANKLE, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
    NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
};

use super::binning::{soft_bin_with_fixed_edges, BinningError};
use super::constants::NUM_SOFT_BINS;
pub use super::constants::RTMDET_ENGINEERED_DIMS;

const GRID_SIZE: usize = 9;
const NUM_KEYPOINTS: usize = 17;

pub const HEAD_SHOULDERS_INDICES: [usize; 7] = [
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
];
pub const LEGS_FEET_INDICES: [usize; 4] = [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE];
pub const FACE_INDICES: [usize; 5] = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR];
pub const BODY_INDICES: [usize; 4] = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP];

pub const RTMDET_FIXED_BIN_EDGES: [(&str, [f64; 9]); 6] = [
    (
        "torso_height_ratio",
        [
            0.310, 0.481, 0.488, 0.491, 0.493, 0.495, 0.498, 0.501, 0.508,
        ],
    ),
    (
        "head_shoulders_avg_score",
        [
            0.727, 0.773, 0.799, 0.816, 0.831, 0.840, 0.850, 0.858, 0.866,
        ],
    ),
    (
        "legs_feet_avg_score",
        [
            0.190, 0.201, 0.207, 0.213, 0.219, 0.224, 0.230, 0.240, 0.501,
        ],
    ),
    (
        "bbox_aspect_ratio",
        [
            0.464, 0.669, 0.717, 0.744, 0.767, 0.793, 0.810, 0.841, 0.887,
        ],
    ),
    (
        "upper_body_ratio",
        [
            0.510, 0.997, 0.998, 0.999, 1.000, 1.001, 1.002, 1.003, 1.004,
        ],
    ),
    (
        "face_visibility_ratio",
        [
            1.198, 1.337, 1.368, 1.385, 1.397, 1.410, 1.421, 1.438, 1.473,
        ],
    ),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureValue {
    pub value: f64,
    pub confidence: f64,
    pub valid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundingBoxValidationError {
    NonFinite,
    InvalidScore,
    ReversedCoordinates,
    NegativeDimensions,
}

impl fmt::Display for BoundingBoxValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("bounding box values must be finite"),
            Self::InvalidScore => formatter.write_str("bounding box score must be between 0 and 1"),
            Self::ReversedCoordinates => {
                formatter.write_str("bounding box coordinates must be ordered")
            }
            Self::NegativeDimensions => {
                formatter.write_str("bounding box dimensions must be non-negative")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeypointValueValidationError {
    NonFiniteCoordinate,
    NonFiniteScore,
}

impl fmt::Display for KeypointValueValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCoordinate => formatter.write_str("keypoint coordinates must be finite"),
            Self::NonFiniteScore => formatter.write_str("keypoint scores must be finite"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RtmDetEngineeredFeaturesError {
    InsufficientKeypoints {
        required: usize,
        actual: usize,
    },
    InvalidBoundingBox(BoundingBoxValidationError),
    InvalidKeypoint {
        index: usize,
        reason: KeypointValueValidationError,
    },
    Binning(BinningError),
}

impl fmt::Display for RtmDetEngineeredFeaturesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientKeypoints { required, actual } => write!(
                formatter,
                "expected at least {required} keypoints, got {actual}"
            ),
            Self::InvalidBoundingBox(error) => error.fmt(formatter),
            Self::InvalidKeypoint { index, reason } => {
                write!(formatter, "invalid keypoint {index}: {reason}")
            }
            Self::Binning(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RtmDetEngineeredFeaturesError {}

impl From<BinningError> for RtmDetEngineeredFeaturesError {
    fn from(error: BinningError) -> Self {
        Self::Binning(error)
    }
}

fn compute_bbox_coverage_grid(bbox: &BoundingBox) -> Vec<f32> {
    let mut grid = vec![0.0_f32; GRID_SIZE * GRID_SIZE];
    let cell_size = 1.0 / GRID_SIZE as f64;

    for j in 0..GRID_SIZE {
        for i in 0..GRID_SIZE {
            let cell_left = i as f64 * cell_size;
            let cell_right = (i + 1) as f64 * cell_size;
            let cell_top = j as f64 * cell_size;
            let cell_bottom = (j + 1) as f64 * cell_size;

            let overlap_width = js_max(
                0.0,
                js_min(bbox.x2, cell_right) - js_max(bbox.x1, cell_left),
            );
            let overlap_height = js_max(
                0.0,
                js_min(bbox.y2, cell_bottom) - js_max(bbox.y1, cell_top),
            );
            let cell_area = cell_size * cell_size;
            grid[j * GRID_SIZE + i] = (overlap_width * overlap_height / cell_area) as f32;
        }
    }

    grid
}

pub fn extract_keypoint_scores_feature_checked(
    keypoints: Option<&[Keypoint]>,
) -> Result<Option<Vec<f32>>, RtmDetEngineeredFeaturesError> {
    let Some(keypoints) = keypoints else {
        return Ok(None);
    };
    if keypoints.len() < NUM_KEYPOINTS {
        return Ok(None);
    }

    validate_keypoint_values(&keypoints[..NUM_KEYPOINTS])?;
    Ok(Some(
        keypoints
            .iter()
            .take(NUM_KEYPOINTS)
            .map(|keypoint| keypoint.score as f32)
            .collect(),
    ))
}

pub fn extract_keypoint_scores_feature(keypoints: Option<&[Keypoint]>) -> Option<Vec<f32>> {
    extract_keypoint_scores_feature_checked(keypoints)
        .ok()
        .flatten()
}

fn compute_avg_keypoint_group_score(keypoints: &[Keypoint], indices: &[usize]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    for &index in indices {
        if index < keypoints.len() {
            sum += keypoints[index].score;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn compute_torso_height_ratio(
    keypoints: &[Keypoint],
    bbox_height: f64,
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    if bbox_height <= 0.0 {
        return Ok(FeatureValue {
            value: 0.0,
            confidence: 0.0,
            valid: false,
        });
    }
    ensure_keypoints(keypoints)?;

    let left_shoulder = keypoints[LEFT_SHOULDER];
    let right_shoulder = keypoints[RIGHT_SHOULDER];
    let left_hip = keypoints[LEFT_HIP];
    let right_hip = keypoints[RIGHT_HIP];
    let shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0;
    let hip_mid_y = (left_hip.y + right_hip.y) / 2.0;
    let torso_height = (hip_mid_y - shoulder_mid_y).abs();

    Ok(FeatureValue {
        value: torso_height / bbox_height,
        confidence: min_js(&[
            left_shoulder.score,
            right_shoulder.score,
            left_hip.score,
            right_hip.score,
        ]),
        valid: true,
    })
}

fn compute_upper_body_ratio(
    keypoints: &[Keypoint],
    bbox: &BoundingBox,
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    let bbox_height = bbox.height;
    if bbox_height <= 0.0 {
        return Ok(FeatureValue {
            value: 0.0,
            confidence: 0.0,
            valid: false,
        });
    }
    ensure_keypoints(keypoints)?;

    let left_hip = keypoints[LEFT_HIP];
    let right_hip = keypoints[RIGHT_HIP];
    let hip_mid_y = (left_hip.y + right_hip.y) / 2.0;

    Ok(FeatureValue {
        value: (hip_mid_y - bbox.y1) / bbox_height,
        confidence: min_js(&[left_hip.score, right_hip.score]),
        valid: true,
    })
}

fn compute_face_visibility_ratio(
    keypoints: &[Keypoint],
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    let face_avg = compute_avg_keypoint_group_score(keypoints, &FACE_INDICES);
    let body_avg = compute_avg_keypoint_group_score(keypoints, &BODY_INDICES);

    if body_avg <= 0.001 {
        return Ok(FeatureValue {
            value: 0.0,
            confidence: 0.0,
            valid: false,
        });
    }

    Ok(FeatureValue {
        value: face_avg / body_avg,
        confidence: min_js(&[face_avg, body_avg]),
        valid: true,
    })
}

pub fn extract_rtm_det_engineered_features(
    keypoints: Option<&[Keypoint]>,
    bbox: Option<&BoundingBox>,
) -> Result<Vec<f32>, RtmDetEngineeredFeaturesError> {
    let mut result = vec![0.0_f32; RTMDET_ENGINEERED_DIMS];
    let Some(bbox) = bbox else {
        return Ok(result);
    };

    validate_bbox(bbox)?;
    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        validate_keypoint_values(&keypoints[..NUM_KEYPOINTS])?;
    }

    let coverage_grid = compute_bbox_coverage_grid(bbox);
    result[..GRID_SIZE * GRID_SIZE].copy_from_slice(&coverage_grid);
    let mut offset = GRID_SIZE * GRID_SIZE;

    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        let torso_ratio = compute_torso_height_ratio(keypoints, bbox.height)?;
        if torso_ratio.valid {
            write_binned(
                &mut result,
                &mut offset,
                torso_ratio.value,
                torso_ratio.confidence,
                fixed_bin_edges("torso_height_ratio"),
            )?;
        }
    }
    offset += NUM_SOFT_BINS;

    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        let average_score = compute_avg_keypoint_group_score(keypoints, &HEAD_SHOULDERS_INDICES);
        if average_score > 0.0 {
            write_binned(
                &mut result,
                &mut offset,
                average_score,
                average_score,
                fixed_bin_edges("head_shoulders_avg_score"),
            )?;
        }
    }
    offset += NUM_SOFT_BINS;

    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        let average_score = compute_avg_keypoint_group_score(keypoints, &LEGS_FEET_INDICES);
        if average_score > 0.0 {
            write_binned(
                &mut result,
                &mut offset,
                average_score,
                average_score,
                fixed_bin_edges("legs_feet_avg_score"),
            )?;
        }
    }
    offset += NUM_SOFT_BINS;

    if bbox.height > 0.0 {
        let aspect_ratio = bbox.width / bbox.height;
        write_binned(
            &mut result,
            &mut offset,
            aspect_ratio,
            bbox.score,
            fixed_bin_edges("bbox_aspect_ratio"),
        )?;
    }
    offset += NUM_SOFT_BINS;

    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        let upper_body_ratio = compute_upper_body_ratio(keypoints, bbox)?;
        if upper_body_ratio.valid {
            write_binned(
                &mut result,
                &mut offset,
                upper_body_ratio.value,
                upper_body_ratio.confidence,
                fixed_bin_edges("upper_body_ratio"),
            )?;
        }
    }
    offset += NUM_SOFT_BINS;

    if let Some(keypoints) = keypoints.filter(|keypoints| keypoints.len() >= NUM_KEYPOINTS) {
        let face_visibility = compute_face_visibility_ratio(keypoints)?;
        if face_visibility.valid {
            write_binned(
                &mut result,
                &mut offset,
                face_visibility.value,
                face_visibility.confidence,
                fixed_bin_edges("face_visibility_ratio"),
            )?;
        }
    }
    offset += NUM_SOFT_BINS;

    debug_assert_eq!(offset, RTMDET_ENGINEERED_DIMS);
    Ok(result)
}

pub fn compute_torso_height_ratio_for_logging(
    keypoints: &[Keypoint],
    bbox_height: f64,
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    compute_torso_height_ratio(keypoints, bbox_height)
}

pub fn compute_upper_body_ratio_for_logging(
    keypoints: &[Keypoint],
    bbox: &BoundingBox,
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    compute_upper_body_ratio(keypoints, bbox)
}

pub fn compute_avg_keypoint_group_score_for_logging(
    keypoints: &[Keypoint],
    indices: &[usize],
) -> f64 {
    compute_avg_keypoint_group_score(keypoints, indices)
}

pub fn compute_face_visibility_ratio_for_logging(
    keypoints: &[Keypoint],
) -> Result<FeatureValue, RtmDetEngineeredFeaturesError> {
    compute_face_visibility_ratio(keypoints)
}

fn write_binned(
    result: &mut [f32],
    offset: &mut usize,
    value: f64,
    confidence: f64,
    edges: &[f64],
) -> Result<(), RtmDetEngineeredFeaturesError> {
    let binned = soft_bin_with_fixed_edges(value, confidence, edges)?;
    let end = *offset + binned.len();
    result[*offset..end].copy_from_slice(&binned);
    Ok(())
}

fn fixed_bin_edges(name: &str) -> &[f64] {
    RTMDET_FIXED_BIN_EDGES
        .iter()
        .find(|(feature, _)| *feature == name)
        .map_or(&RTMDET_FIXED_BIN_EDGES[0].1, |(_, edges)| edges.as_slice())
}

fn ensure_keypoints(keypoints: &[Keypoint]) -> Result<(), RtmDetEngineeredFeaturesError> {
    if keypoints.len() < NUM_KEYPOINTS {
        Err(RtmDetEngineeredFeaturesError::InsufficientKeypoints {
            required: NUM_KEYPOINTS,
            actual: keypoints.len(),
        })
    } else {
        Ok(())
    }
}

/// Bounding-box gate for engineered features. The validity contract lives in a
/// single place — [`slouch_domain::validate_bbox`] (finite fields, ordered
/// coordinates, non-negative extents, score in `[0, 1]`) — so the false
/// `width == x2 - x1` span check can never be reintroduced here. `width`/`height`
/// carry the UNCLAMPED detector extent while `x1..y2` are clamped to the frame,
/// so that identity legitimately fails at frame edges. The local granular error
/// kind is derived only to preserve this crate's error vocabulary.
fn validate_bbox(bbox: &BoundingBox) -> Result<(), RtmDetEngineeredFeaturesError> {
    if slouch_domain::validate_bbox(bbox).is_ok() {
        return Ok(());
    }
    let defect = if [
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.width,
        bbox.height,
        bbox.score,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        BoundingBoxValidationError::NonFinite
    } else if !(0.0..=1.0).contains(&bbox.score) {
        BoundingBoxValidationError::InvalidScore
    } else if bbox.x2 < bbox.x1 || bbox.y2 < bbox.y1 {
        BoundingBoxValidationError::ReversedCoordinates
    } else {
        BoundingBoxValidationError::NegativeDimensions
    };
    Err(RtmDetEngineeredFeaturesError::InvalidBoundingBox(defect))
}

fn validate_keypoint_values(keypoints: &[Keypoint]) -> Result<(), RtmDetEngineeredFeaturesError> {
    for (index, keypoint) in keypoints.iter().enumerate() {
        if !keypoint.x.is_finite() || !keypoint.y.is_finite() {
            return Err(RtmDetEngineeredFeaturesError::InvalidKeypoint {
                index,
                reason: KeypointValueValidationError::NonFiniteCoordinate,
            });
        }
        if !keypoint.score.is_finite() {
            return Err(RtmDetEngineeredFeaturesError::InvalidKeypoint {
                index,
                reason: KeypointValueValidationError::NonFiniteScore,
            });
        }
    }
    Ok(())
}

fn min_js(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, js_min)
}

fn js_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left < right {
        left
    } else {
        right
    }
}

fn js_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left > right {
        left
    } else {
        right
    }
}

const _: () = {
    assert!(NUM_SOFT_BINS == 9);
    assert!(RTMDET_ENGINEERED_DIMS == GRID_SIZE * GRID_SIZE + 6 * NUM_SOFT_BINS);
};
