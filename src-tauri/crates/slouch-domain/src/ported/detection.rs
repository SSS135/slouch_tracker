//! Person presence detection based on RTMDet bounding-box confidence.
//!
//! The NLF-L pose model supplies the 17 body keypoints only; posture, neck-tilt,
//! hand, and mouth detections remain disabled until their classifier inputs are
//! wired.

use crate::ExpandedBbox;

use super::types::MultiTaskDetectionResult;

/// Minimum RTMDet confidence required to consider a person present.
pub const PERSON_DETECTION_CONFIDENCE: f64 = 0.3;

fn detect_person(bbox: Option<&ExpandedBbox>) -> bool {
    bbox.is_some_and(|value| value.original.score >= PERSON_DETECTION_CONFIDENCE)
}

/// Detects person presence from the original RTMDet bounding-box score.
///
/// The remaining task flags intentionally stay false: the neck-tilt, hand, and
/// mouth classifier outputs those tasks need are not wired.
pub fn detect_multi_task(bbox: Option<&ExpandedBbox>) -> MultiTaskDetectionResult {
    MultiTaskDetectionResult {
        person_found: detect_person(bbox),
        slouching: false,
        forward_neck_tilt: false,
        hand_near_face: false,
        mouth_open: false,
    }
}
