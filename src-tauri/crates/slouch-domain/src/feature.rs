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
    #[serde(rename = "torso_invariant")]
    TorsoInvariant,
    #[serde(rename = "nlf_depth")]
    NlfDepth,
    #[serde(rename = "nlf_backbone")]
    NlfBackbone,
    #[serde(rename = "nlf_backbone_max")]
    NlfBackboneMax,
    #[serde(rename = "nlf_backbone_std")]
    NlfBackboneStd,
    #[serde(rename = "raw_keypoints_3d")]
    RawKeypoints3d,
    #[serde(rename = "posture_raw_3d")]
    PostureRaw3d,
    #[serde(rename = "posture_geometry_3d")]
    PostureGeometry3d,
    #[serde(rename = "torso_invariant_3d")]
    TorsoInvariant3d,
}

impl FeatureId {
    pub const ALL: [Self; 25] = [
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
        Self::TorsoInvariant,
        Self::NlfDepth,
        Self::NlfBackbone,
        Self::NlfBackboneMax,
        Self::NlfBackboneStd,
        Self::RawKeypoints3d,
        Self::PostureRaw3d,
        Self::PostureGeometry3d,
        Self::TorsoInvariant3d,
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
            Self::TorsoInvariant => "torso_invariant",
            Self::NlfDepth => "nlf_depth",
            Self::NlfBackbone => "nlf_backbone",
            Self::NlfBackboneMax => "nlf_backbone_max",
            Self::NlfBackboneStd => "nlf_backbone_std",
            Self::RawKeypoints3d => "raw_keypoints_3d",
            Self::PostureRaw3d => "posture_raw_3d",
            Self::PostureGeometry3d => "posture_geometry_3d",
            Self::TorsoInvariant3d => "torso_invariant_3d",
        }
    }

    pub const fn metadata(self) -> FeatureMetadata {
        match self {
      Self::BackboneFeatures => FeatureMetadata::stored_unavailable(self, "Backbone Features (Avg Pool)", "Average pooled backbone features from RTMPose", 768, ModelCategory::Posture),
      Self::BackboneFeaturesMax => FeatureMetadata::stored_unavailable(self, "Backbone Features (Max Pool)", "Max pooled backbone features from RTMPose - Captures peak spatial activations", 768, ModelCategory::Posture),
      Self::BackboneFeaturesStd => FeatureMetadata::stored_unavailable(self, "Backbone Features (Std Pool)", "Std pooled backbone features from RTMPose - Captures spatial variation", 768, ModelCategory::Posture),
      Self::GauFeatures => FeatureMetadata::stored_unavailable(self, "GAU Features (Avg Pool)", "Average pooled GAU features from RTMPose", 256, ModelCategory::Posture),
      Self::GauFeaturesMax => FeatureMetadata::stored_unavailable(self, "GAU Features (Max Pool)", "Max pooled GAU features from RTMPose - Captures peak keypoint activations", 256, ModelCategory::Posture),
      Self::GauFeaturesStd => FeatureMetadata::stored_unavailable(self, "GAU Features (Std Pool)", "Std pooled GAU features from RTMPose - Captures keypoint variation", 256, ModelCategory::Posture),
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
      Self::TorsoInvariant => FeatureMetadata::computed(self, "Torso-Invariant Geometry", "7 scale/translation-invariant torso-anchored features separating head flexion from trunk slouch (7 dims)", 7, Some(ModelCategory::Posture), Some(false)),
      // Dimension literal `14` must stay in lock-step with `slouch_ml::constants::NLF_DEPTH_DIMS`;
      // slouch-domain cannot depend on slouch-ml, so it is duplicated here (frozen at 14).
      Self::NlfDepth => FeatureMetadata::stored(self, "NLF Depth (3D)", "Body-intrinsic 3D depth and angle features from the NLF-L model that separate forward-head posture from trunk lean (14 dims)", 14, ModelCategory::Posture),
      // Dimension literal `512` must stay in lock-step with `slouch_ml::constants::NLF_BACKBONE_POOLED_DIMS`;
      // slouch-domain cannot depend on slouch-ml, so it is duplicated here (frozen at 512).
      Self::NlfBackbone => FeatureMetadata::stored(self, "NLF Backbone Features (Avg Pool)", "Average-pooled NLF-L backbone embedding (512 dims)", 512, ModelCategory::Posture),
      Self::NlfBackboneMax => FeatureMetadata::stored(self, "NLF Backbone Features (Max Pool)", "Max-pooled NLF-L backbone embedding - captures peak spatial activations (512 dims)", 512, ModelCategory::Posture),
      Self::NlfBackboneStd => FeatureMetadata::stored(self, "NLF Backbone Features (Std Pool)", "Std-pooled NLF-L backbone embedding - captures spatial variation (512 dims)", 512, ModelCategory::Posture),
      // Dimension literal `51` (storage cost 204) must stay in lock-step with
      // `slouch_ml::constants::RAW_KEYPOINTS_3D_DIMS` / `RAW_KEYPOINTS_3D_STORAGE_COST`;
      // slouch-domain cannot depend on slouch-ml, so it is duplicated here (frozen at 51).
      Self::RawKeypoints3d => FeatureMetadata::stored_hidden(self, "Raw Keypoints 3D", "17 torso-normalized, root-centered 3D COCO keypoints (51 dims) — hidden substrate for the 3D posture features", 51, ModelCategory::Posture),
      // Dimension literal `6` must stay in lock-step with `slouch_ml::constants::POSTURE_RAW_3D_DIMS`.
      Self::PostureRaw3d => FeatureMetadata::computed(self, "Posture Raw 3D", "Body-intrinsic 3D raw posture features (ear-eye vertical, head yaw, neck length, inter-ear distance, head-up trunk projection) + validity (6 dims)", 6, Some(ModelCategory::Posture), Some(false)),
      // Dimension literal `10` must stay in lock-step with `slouch_ml::constants::POSTURE_GEOMETRY_3D_DIMS`.
      Self::PostureGeometry3d => FeatureMetadata::computed(self, "Posture Geometry 3D", "10 body-intrinsic 3D geometric posture features from head and shoulder keypoints, derived from the 3D keypoint substrate", 10, Some(ModelCategory::Posture), Some(false)),
      // Dimension literal `9` must stay in lock-step with `slouch_ml::constants::TORSO_INVARIANT_3D_DIMS`.
      Self::TorsoInvariant3d => FeatureMetadata::computed(self, "Torso-Invariant 3D", "9 torso-anchored 3D posture features separating head flexion from trunk lean, including camera-frame coronal/sagittal lean and axial twist (9 dims)", 9, Some(ModelCategory::Posture), Some(false)),
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

    // Retired stored feature: its variant, discriminant, and dimensions persist so
    // older datasets/models keep deserializing, but the pose model that produced it
    // is gone, so it is hidden from the selector (`user_selectable: false`).
    const fn stored_unavailable(
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
            user_selectable: false,
            requires_fitting: None,
        }
    }

    // Present-by-design stored substrate, hidden from direct selection, consumed as a
    // dependency by the computed 3D posture features. Unlike `stored_unavailable` (retired /
    // no longer produced), this feature is still extracted every frame.
    const fn stored_hidden(
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
            user_selectable: false,
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

pub fn feature_registry() -> [FeatureMetadata; 25] {
    FeatureId::ALL.map(FeatureId::metadata)
}
