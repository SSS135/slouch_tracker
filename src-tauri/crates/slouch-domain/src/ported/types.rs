//! Multi-task posture detection result types.

use serde::{Deserialize, Serialize};

/// Result of the four simultaneous posture and proximity detection tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiTaskDetectionResult {
    pub person_found: bool,
    pub slouching: bool,
    pub forward_neck_tilt: bool,
    pub hand_near_face: bool,
    pub mouth_open: bool,
}
