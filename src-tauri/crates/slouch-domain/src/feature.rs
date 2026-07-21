use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, specta::Type,
)]
pub enum FeatureId {
    #[serde(rename = "backbone_features")]
    BackboneFeatures,
    #[serde(rename = "backbone_features_max")]
    BackboneFeaturesMax,
    #[serde(rename = "backbone_features_std")]
    BackboneFeaturesStd,
    #[serde(rename = "gau_features")]
    GauFeatures,
    #[serde(rename = "gau_features_max")]
    GauFeaturesMax,
    #[serde(rename = "gau_features_std")]
    GauFeaturesStd,
    #[serde(rename = "rtmdet_extracted")]
    RtmDetExtracted,
    #[serde(rename = "rtmdet_engineered")]
    RtmDetEngineered,
    #[serde(rename = "engineered_features")]
    EngineeredFeatures,
    #[serde(rename = "joint_2d")]
    Joint2d,
    #[serde(rename = "joint_3d")]
    Joint3d,
    #[serde(rename = "joint_4d")]
    Joint4d,
    #[serde(rename = "posture_raw")]
    PostureRaw,
    #[serde(rename = "keypoint_scores")]
    KeypointScores,
    #[serde(rename = "raw_keypoints")]
    RawKeypoints,
    #[serde(rename = "posture_geometry")]
    PostureGeometry,
}

impl FeatureId {
    pub const ALL: [Self; 16] = [
        Self::BackboneFeatures,
        Self::BackboneFeaturesMax,
        Self::BackboneFeaturesStd,
        Self::GauFeatures,
        Self::GauFeaturesMax,
        Self::GauFeaturesStd,
        Self::RtmDetExtracted,
        Self::RtmDetEngineered,
        Self::EngineeredFeatures,
        Self::Joint2d,
        Self::Joint3d,
        Self::Joint4d,
        Self::PostureRaw,
        Self::KeypointScores,
        Self::RawKeypoints,
        Self::PostureGeometry,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BackboneFeatures => "backbone_features",
            Self::BackboneFeaturesMax => "backbone_features_max",
            Self::BackboneFeaturesStd => "backbone_features_std",
            Self::GauFeatures => "gau_features",
            Self::GauFeaturesMax => "gau_features_max",
            Self::GauFeaturesStd => "gau_features_std",
            Self::RtmDetExtracted => "rtmdet_extracted",
            Self::RtmDetEngineered => "rtmdet_engineered",
            Self::EngineeredFeatures => "engineered_features",
            Self::Joint2d => "joint_2d",
            Self::Joint3d => "joint_3d",
            Self::Joint4d => "joint_4d",
            Self::PostureRaw => "posture_raw",
            Self::KeypointScores => "keypoint_scores",
            Self::RawKeypoints => "raw_keypoints",
            Self::PostureGeometry => "posture_geometry",
        }
    }

    pub const fn metadata(self) -> FeatureMetadata {
        match self {
      Self::BackboneFeatures => FeatureMetadata::stored(self, "Backbone Features (Avg Pool)", "Average pooled backbone features from RTMPose", 768, ModelCategory::Posture),
      Self::BackboneFeaturesMax => FeatureMetadata::stored(self, "Backbone Features (Max Pool)", "Max pooled backbone features from RTMPose - Captures peak spatial activations", 768, ModelCategory::Posture),
      Self::BackboneFeaturesStd => FeatureMetadata::stored(self, "Backbone Features (Std Pool)", "Std pooled backbone features from RTMPose - Captures spatial variation", 768, ModelCategory::Posture),
      Self::GauFeatures => FeatureMetadata::stored(self, "GAU Features (Avg Pool)", "Average pooled GAU features from RTMPose", 256, ModelCategory::Posture),
      Self::GauFeaturesMax => FeatureMetadata::stored(self, "GAU Features (Max Pool)", "Max pooled GAU features from RTMPose - Captures peak keypoint activations", 256, ModelCategory::Posture),
      Self::GauFeaturesStd => FeatureMetadata::stored(self, "GAU Features (Std Pool)", "Std pooled GAU features from RTMPose - Captures keypoint variation", 256, ModelCategory::Posture),
      Self::RtmDetExtracted => FeatureMetadata::stored(self, "RTMDet Extracted Features", "Pooled (avg/std/max) cls_p5 + reg_p5 features", 384, ModelCategory::Presence),
      Self::RtmDetEngineered => FeatureMetadata::computed(self, "Detection Features", "Geometric features from detection bbox and keypoints (135 dims)", 135, Some(ModelCategory::Presence), None),
      Self::EngineeredFeatures => FeatureMetadata::computed(self, "Posture Features (1D)", "Body proportion ratios with 1D soft binning (54 dims)", 54, Some(ModelCategory::Posture), Some(false)),
      Self::Joint2d => FeatureMetadata::computed(self, "Joint 2D Histogram", "ear_eye_vertical x neck_length (81 dims)", 81, Some(ModelCategory::Posture), Some(false)),
      Self::Joint3d => FeatureMetadata::computed(self, "Joint 3D Histogram", "ear_eye_vertical x neck_length x head_rotation (125 dims)", 125, Some(ModelCategory::Posture), Some(false)),
      Self::Joint4d => FeatureMetadata::computed(self, "Joint 4D Histogram", "All 4 raw posture features (625 dims)", 625, Some(ModelCategory::Posture), Some(false)),
      Self::PostureRaw => FeatureMetadata::computed(self, "Posture Raw Features", "Raw geometric features: ear_eye_vertical, head_rotation, neck_length, inter_ear_distance, avg_shoulder_y (5 dims)", 5, Some(ModelCategory::Posture), Some(false)),
      Self::KeypointScores => FeatureMetadata::computed(self, "Keypoint Scores", "Confidence scores for all 17 keypoints (17 dims)", 17, None, Some(false)),
      Self::RawKeypoints => FeatureMetadata::computed(self, "Raw Keypoints", "Raw x,y coordinates for all 17 keypoints (34 dims)", 34, Some(ModelCategory::Posture), Some(false)),
      Self::PostureGeometry => FeatureMetadata::computed(self, "Posture Geometry (invariant)", "10 scale/translation-invariant geometric posture features from head and shoulder keypoints", 10, Some(ModelCategory::Posture), Some(false)),
    }
    }
}

impl fmt::Display for FeatureId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFeatureId(pub String);

impl fmt::Display for UnknownFeatureId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown feature ID: {}", self.0)
    }
}

impl std::error::Error for UnknownFeatureId {}

impl FromStr for FeatureId {
    type Err = UnknownFeatureId;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|id| id.as_str() == value)
            .ok_or_else(|| UnknownFeatureId(value.to_owned()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    Posture,
    Presence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct FeatureMetadata {
    pub id: FeatureId,
    pub name: &'static str,
    pub description: &'static str,
    #[specta(type = specta_typescript::Number)]
    pub dimensions: usize,
    #[specta(type = specta_typescript::Number)]
    pub storage_cost: usize,
    pub computed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<ModelCategory>,
    pub user_selectable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_fitting: Option<bool>,
}

impl FeatureMetadata {
    const fn stored(
        id: FeatureId,
        name: &'static str,
        description: &'static str,
        dimensions: usize,
        model_type: ModelCategory,
    ) -> Self {
        Self {
            id,
            name,
            description,
            dimensions,
            storage_cost: dimensions * size_of::<f32>(),
            computed: false,
            model_type: Some(model_type),
            user_selectable: true,
            requires_fitting: None,
        }
    }

    const fn computed(
        id: FeatureId,
        name: &'static str,
        description: &'static str,
        dimensions: usize,
        model_type: Option<ModelCategory>,
        requires_fitting: Option<bool>,
    ) -> Self {
        Self {
            id,
            name,
            description,
            dimensions,
            storage_cost: 0,
            computed: true,
            model_type,
            user_selectable: true,
            requires_fitting,
        }
    }
}

pub fn feature_registry() -> [FeatureMetadata; 16] {
    FeatureId::ALL.map(FeatureId::metadata)
}
