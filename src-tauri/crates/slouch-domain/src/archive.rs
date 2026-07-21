use serde::{Deserialize, Serialize};

use crate::{FeatureId, FrameLabel};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub id: String,
    pub label: FrameLabel,
    pub timestamp: f64,
    pub features: Vec<FeatureId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReservoirManifest {
    pub count: usize,
    pub total_seen: u64,
    pub max_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetManifest {
    pub version: u32,
    pub exported_at: String,
    pub frame_count: usize,
    pub frames: Vec<FrameMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservoir: Option<ReservoirManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportResult {
    pub imported: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}
