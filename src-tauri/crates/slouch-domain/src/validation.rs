use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{
    BoundingBox, ClassificationResult, FeatureId, FeatureMap, InferenceResult, PostureFrame,
    COCO_KEYPOINT_COUNT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCode {
    Empty,
    NonFinite,
    OutOfRange,
    InvalidLength,
    InvalidDimensions,
    InconsistentGeometry,
    InvalidMimeType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub code: ValidationCode,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationError {
    pub issues: Vec<ValidationIssue>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = self
            .issues
            .iter()
            .map(|issue| format!("{}: {}", issue.path, issue.message))
            .collect::<Vec<_>>()
            .join(", ");
        formatter.write_str(&message)
    }
}

impl std::error::Error for ValidationError {}

fn issue(
    code: ValidationCode,
    path: impl Into<String>,
    message: impl Into<String>,
) -> ValidationIssue {
    ValidationIssue {
        code,
        path: path.into(),
        message: message.into(),
    }
}

pub fn is_feature_id(value: &str) -> bool {
    FeatureId::from_str(value).is_ok()
}

pub fn has_required_frame_shape(frame: &PostureFrame) -> bool {
    frame.keypoints.len() == COCO_KEYPOINT_COUNT
        && !frame.thumbnail.bytes.is_empty()
        && !frame.thumbnail.mime_type.is_empty()
}

pub fn validate_posture_frame(frame: &PostureFrame) -> Result<(), ValidationError> {
    let mut issues = Vec::new();

    if frame.id.trim().is_empty() {
        issues.push(issue(ValidationCode::Empty, "id", "must not be empty"));
    }
    if !frame.timestamp.is_finite() {
        issues.push(issue(
            ValidationCode::NonFinite,
            "timestamp",
            "must be finite",
        ));
    } else if frame.timestamp <= 0.0 {
        issues.push(issue(
            ValidationCode::OutOfRange,
            "timestamp",
            "must be positive",
        ));
    }
    validate_keypoints(&frame.keypoints, &mut issues);
    collect_bbox_issues(&frame.bbox, "bbox", &mut issues);
    validate_feature_map(&frame.features, &mut issues);

    if frame.thumbnail.bytes.is_empty() {
        issues.push(issue(
            ValidationCode::Empty,
            "thumbnail.bytes",
            "must not be empty",
        ));
    }
    if !frame.thumbnail.mime_type.starts_with("image/") {
        issues.push(issue(
            ValidationCode::InvalidMimeType,
            "thumbnail.mimeType",
            "must be an image MIME type",
        ));
    }

    finish(issues)
}

pub fn validate_inference_result(result: &InferenceResult) -> Result<(), ValidationError> {
    let mut issues = Vec::new();
    validate_keypoints(&result.keypoints, &mut issues);
    collect_bbox_issues(&result.bbox.original, "bbox.original", &mut issues);
    collect_bbox_issues(&result.bbox.expanded, "bbox.expanded", &mut issues);
    validate_feature_map(&result.features, &mut issues);
    if let Some(classification) = result.classification {
        validate_probability(
            classification.present_probability,
            "classification.presentProbability",
            &mut issues,
        );
        if let Some(probability) = classification.good_probability {
            validate_probability(probability, "classification.goodProbability", &mut issues);
        }
    }
    finish(issues)
}

pub fn validate_classification_result(
    result: &ClassificationResult,
) -> Result<(), ValidationError> {
    let mut issues = Vec::new();
    validate_probability(
        result.present_probability,
        "presentProbability",
        &mut issues,
    );
    if let Some(probability) = result.good_probability {
        validate_probability(probability, "goodProbability", &mut issues);
    }
    finish(issues)
}

fn validate_keypoints(keypoints: &[crate::Keypoint], issues: &mut Vec<ValidationIssue>) {
    if keypoints.len() != COCO_KEYPOINT_COUNT {
        issues.push(issue(
            ValidationCode::InvalidLength,
            "keypoints",
            format!("expected {COCO_KEYPOINT_COUNT}, got {}", keypoints.len()),
        ));
    }

    for (index, keypoint) in keypoints.iter().enumerate() {
        // Keypoint scores are SimCC activation means, not probabilities, so they
        // legitimately exceed 1 on real frames. Only finiteness is required.
        for (field, value) in [
            ("x", keypoint.x),
            ("y", keypoint.y),
            ("score", keypoint.score),
        ] {
            if !value.is_finite() {
                issues.push(issue(
                    ValidationCode::NonFinite,
                    format!("keypoints.{index}.{field}"),
                    "must be finite",
                ));
            }
        }
    }
}

/// The single authoritative bounding-box validity contract: finite fields, a
/// score in `[0, 1]`, ordered coordinates, and non-negative extents. Inference
/// clamps `x1..y2` to the frame while `width`/`height` keep the UNCLAMPED
/// detector extent (matching the frozen TS oracle), so `width == x2 - x1` is
/// deliberately NOT an invariant — it fails legitimately whenever the subject
/// clips a frame edge. All bbox validators across the workspace delegate here.
pub fn validate_bbox(bbox: &BoundingBox) -> Result<(), ValidationError> {
    let mut issues = Vec::new();
    collect_bbox_issues(bbox, "bbox", &mut issues);
    finish(issues)
}

fn collect_bbox_issues(bbox: &BoundingBox, path: &str, issues: &mut Vec<ValidationIssue>) {
    for (field, value) in [
        ("x1", bbox.x1),
        ("y1", bbox.y1),
        ("x2", bbox.x2),
        ("y2", bbox.y2),
        ("width", bbox.width),
        ("height", bbox.height),
    ] {
        if !value.is_finite() {
            issues.push(issue(
                ValidationCode::NonFinite,
                format!("{path}.{field}"),
                "must be finite",
            ));
        }
    }
    validate_probability(bbox.score, &format!("{path}.score"), issues);

    if bbox.x2 < bbox.x1 || bbox.y2 < bbox.y1 || bbox.width < 0.0 || bbox.height < 0.0 {
        issues.push(issue(
            ValidationCode::OutOfRange,
            path,
            "coordinates and dimensions must be ordered and non-negative",
        ));
    }
}

fn validate_feature_map(features: &FeatureMap, issues: &mut Vec<ValidationIssue>) {
    for (feature_id, values) in features {
        let expected = feature_id.metadata().dimensions;
        if values.len() != expected {
            issues.push(issue(
                ValidationCode::InvalidDimensions,
                format!("features.{feature_id}"),
                format!("expected {expected}, got {}", values.len()),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            issues.push(issue(
                ValidationCode::NonFinite,
                format!("features.{feature_id}"),
                "all values must be finite",
            ));
        }
    }
}

fn validate_probability(value: f64, path: &str, issues: &mut Vec<ValidationIssue>) {
    if !value.is_finite() {
        issues.push(issue(ValidationCode::NonFinite, path, "must be finite"));
    } else if !(0.0..=1.0).contains(&value) {
        issues.push(issue(
            ValidationCode::OutOfRange,
            path,
            "must be between 0 and 1",
        ));
    }
}

fn finish(issues: Vec<ValidationIssue>) -> Result<(), ValidationError> {
    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidationError { issues })
    }
}
