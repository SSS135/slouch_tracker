use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{BboxAccessor, BoundingBox, ClassificationResult, ExpandedBbox, FeatureId, Keypoint};

pub type FeatureMap = BTreeMap<FeatureId, Vec<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum FrameLabel {
    Good,
    Bad,
    Away,
    Unused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Thumbnail {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceResult {
    pub features: FeatureMap,
    pub keypoints: Vec<Keypoint>,
    pub bbox: ExpandedBbox,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostureFrame {
    pub id: String,
    pub timestamp: f64,
    pub features: FeatureMap,
    pub thumbnail: Thumbnail,
    pub keypoints: Vec<Keypoint>,
    pub bbox: BoundingBox,
    pub label: FrameLabel,
}

pub trait FeatureSource {
    type Bbox: BboxAccessor;

    fn features(&self) -> &FeatureMap;
    fn keypoints(&self) -> &[Keypoint];
    fn bbox(&self) -> &Self::Bbox;
}

impl FeatureSource for InferenceResult {
    type Bbox = ExpandedBbox;

    fn features(&self) -> &FeatureMap {
        &self.features
    }

    fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    fn bbox(&self) -> &Self::Bbox {
        &self.bbox
    }
}

impl FeatureSource for PostureFrame {
    type Bbox = BoundingBox;

    fn features(&self) -> &FeatureMap {
        &self.features
    }

    fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    fn bbox(&self) -> &Self::Bbox {
        &self.bbox
    }
}
