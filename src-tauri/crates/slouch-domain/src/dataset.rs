use serde::{Deserialize, Serialize};

use crate::{FrameLabel, PostureFrame};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureAction {
    pub frame_id: String,
    pub timestamp: f64,
    pub label: FrameLabel,
    pub thumbnail_url: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostureFrameMetadata {
    pub id: String,
    pub timestamp: f64,
    pub label: FrameLabel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostureDataset {
    pub frames: Vec<PostureFrame>,
    pub version: u64,
    pub last_modified: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct DatasetStats {
    #[specta(type = specta_typescript::Number)]
    pub total: usize,
    #[specta(type = specta_typescript::Number)]
    pub good: usize,
    #[specta(type = specta_typescript::Number)]
    pub bad: usize,
    #[specta(type = specta_typescript::Number)]
    pub away: usize,
    #[specta(type = specta_typescript::Number)]
    pub unused: usize,
    pub imbalance_ratio: f64,
    pub has_minimum_frames: bool,
    pub has_away_frames: bool,
}

impl DatasetStats {
    pub fn from_labels(labels: impl IntoIterator<Item = FrameLabel>) -> Self {
        let mut stats = Self {
            total: 0,
            good: 0,
            bad: 0,
            away: 0,
            unused: 0,
            imbalance_ratio: 0.0,
            has_minimum_frames: false,
            has_away_frames: false,
        };

        for label in labels {
            stats.total += 1;
            match label {
                FrameLabel::Good => stats.good += 1,
                FrameLabel::Bad => stats.bad += 1,
                FrameLabel::Away => stats.away += 1,
                FrameLabel::Unused => stats.unused += 1,
            }
        }

        let classified = stats.good + stats.bad;
        stats.imbalance_ratio = if classified == 0 {
            0.0
        } else {
            stats.good.abs_diff(stats.bad) as f64 / classified as f64
        };
        stats.has_minimum_frames = stats.good > 0 && stats.bad > 0;
        stats.has_away_frames = stats.away > 0;
        stats
    }
}
