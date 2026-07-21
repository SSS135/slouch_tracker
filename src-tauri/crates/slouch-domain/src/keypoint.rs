use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

impl Keypoint {
    pub const fn new(x: f64, y: f64, score: f64) -> Self {
        Self { x, y, score }
    }
}

pub const NOSE: usize = 0;
pub const LEFT_EYE: usize = 1;
pub const RIGHT_EYE: usize = 2;
pub const LEFT_EAR: usize = 3;
pub const RIGHT_EAR: usize = 4;
pub const LEFT_SHOULDER: usize = 5;
pub const RIGHT_SHOULDER: usize = 6;
pub const LEFT_ELBOW: usize = 7;
pub const RIGHT_ELBOW: usize = 8;
pub const LEFT_WRIST: usize = 9;
pub const RIGHT_WRIST: usize = 10;
pub const LEFT_HIP: usize = 11;
pub const RIGHT_HIP: usize = 12;
pub const LEFT_KNEE: usize = 13;
pub const RIGHT_KNEE: usize = 14;
pub const LEFT_ANKLE: usize = 15;
pub const RIGHT_ANKLE: usize = 16;
pub const COCO_KEYPOINT_COUNT: usize = 17;
