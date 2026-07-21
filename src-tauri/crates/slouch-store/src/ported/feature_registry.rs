//! Feature type registry and feature extraction dispatch for dataset frames.
//!
//! The source registry combines feature metadata with the extractor for each
//! feature kind. Rust keeps that metadata in a fixed registry and represents
//! the source callbacks with a small, explicit extractor enum.

use std::fmt;

use slouch_domain::{BboxAccessor, ModelCategory};
use slouch_ml::ported::constants::{
    ENGINEERED_1D_DIMS, JOINT_2D_DIMS, JOINT_3D_DIMS, JOINT_4D_DIMS, KEYPOINT_SCORES_DIMS,
    POSTURE_GEOMETRY_DIMS, POSTURE_RAW_DIMS, RAW_KEYPOINTS_DIMS, RTMDET_ENGINEERED_DIMS,
    RTMDET_EXTRACTED_DIMS, RTMDET_EXTRACTED_STORAGE_COST, RTMPOSE_BACKBONE_POOLED_DIMS,
    RTMPOSE_BACKBONE_POOLED_STORAGE_COST, RTMPOSE_GAU_POOLED_DIMS, RTMPOSE_GAU_POOLED_STORAGE_COST,
    TORSO_INVARIANT_DIMS,
};
use slouch_ml::ported::engineered_features::{
    extract_engineered_features, extract_joint_2d_features, extract_joint_3d_features,
    extract_joint_4d_features, extract_posture_geometry_features, extract_posture_raw_features,
    extract_raw_keypoints, extract_torso_invariant_features, EngineeredFeaturesError,
};
use slouch_ml::ported::rtmdet_engineered_features::{
    extract_keypoint_scores_feature, extract_rtm_det_engineered_features,
    RtmDetEngineeredFeaturesError,
};

use super::types::FeatureContainer;

/// The set of feature identifiers accepted by the dataset and training APIs.
///
/// `FeatureId` is the domain's serialized string enum; re-exporting it keeps
/// storage, validation, and native IPC on one identifier type.
pub use slouch_domain::FeatureId as FeatureType;

pub const FEATURE_BACKBONE_AVG: FeatureType = FeatureType::BackboneFeatures;
pub const FEATURE_BACKBONE_MAX: FeatureType = FeatureType::BackboneFeaturesMax;
pub const FEATURE_BACKBONE_STD: FeatureType = FeatureType::BackboneFeaturesStd;
pub const FEATURE_GAU_AVG: FeatureType = FeatureType::GauFeatures;
pub const FEATURE_GAU_MAX: FeatureType = FeatureType::GauFeaturesMax;
pub const FEATURE_GAU_STD: FeatureType = FeatureType::GauFeaturesStd;
pub const FEATURE_RTMDET_EXTRACTED: FeatureType = FeatureType::RtmDetExtracted;
pub const FEATURE_RTMDET_ENGINEERED: FeatureType = FeatureType::RtmDetEngineered;
pub const FEATURE_ENGINEERED: FeatureType = FeatureType::EngineeredFeatures;
pub const FEATURE_JOINT_2D: FeatureType = FeatureType::Joint2d;
pub const FEATURE_JOINT_3D: FeatureType = FeatureType::Joint3d;
pub const FEATURE_JOINT_4D: FeatureType = FeatureType::Joint4d;
pub const FEATURE_POSTURE_RAW: FeatureType = FeatureType::PostureRaw;
pub const FEATURE_KEYPOINT_SCORES: FeatureType = FeatureType::KeypointScores;
pub const FEATURE_RAW_KEYPOINTS: FeatureType = FeatureType::RawKeypoints;
pub const FEATURE_POSTURE_GEOMETRY: FeatureType = FeatureType::PostureGeometry;
pub const FEATURE_TORSO_INVARIANT: FeatureType = FeatureType::TorsoInvariant;

/// Valid feature type identifiers, in the same insertion order as the source
/// object registry.
pub const FEATURE_TYPES: [FeatureType; 17] = FeatureType::ALL;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractorKind {
    Stored,
    RtmDetEngineered,
    Engineered,
    Joint2d,
    Joint3d,
    Joint4d,
    PostureRaw,
    KeypointScores,
    RawKeypoints,
    PostureGeometry,
    TorsoInvariant,
}

/// Metadata and extraction behavior for one feature type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureDefinition {
    pub id: FeatureType,
    pub name: &'static str,
    pub description: &'static str,
    pub dimensions: usize,
    pub storage_cost: usize,
    pub computed: bool,
    pub model_type: Option<ModelCategory>,
    pub user_selectable: bool,
    pub requires_fitting: Option<bool>,
    extractor: ExtractorKind,
}

impl FeatureDefinition {
    const fn stored(
        id: FeatureType,
        name: &'static str,
        description: &'static str,
        dimensions: usize,
        storage_cost: usize,
        model_type: ModelCategory,
    ) -> Self {
        Self {
            id,
            name,
            description,
            dimensions,
            storage_cost,
            computed: false,
            model_type: Some(model_type),
            user_selectable: true,
            requires_fitting: None,
            extractor: ExtractorKind::Stored,
        }
    }

    const fn computed(
        id: FeatureType,
        name: &'static str,
        description: &'static str,
        dimensions: usize,
        model_type: Option<ModelCategory>,
        requires_fitting: Option<bool>,
        extractor: ExtractorKind,
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
            extractor,
        }
    }

    const fn from_feature_type(id: FeatureType) -> Self {
        match id {
            FeatureType::BackboneFeatures => Self::stored(
                id,
                "Backbone Features (Avg Pool)",
                "Average pooled backbone features from RTMPose",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::BackboneFeaturesMax => Self::stored(
                id,
                "Backbone Features (Max Pool)",
                "Max pooled backbone features from RTMPose - Captures peak spatial activations",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::BackboneFeaturesStd => Self::stored(
                id,
                "Backbone Features (Std Pool)",
                "Std pooled backbone features from RTMPose - Captures spatial variation",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeatures => Self::stored(
                id,
                "GAU Features (Avg Pool)",
                "Average pooled GAU features from RTMPose",
                RTMPOSE_GAU_POOLED_DIMS,
                RTMPOSE_GAU_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeaturesMax => Self::stored(
                id,
                "GAU Features (Max Pool)",
                "Max pooled GAU features from RTMPose - Captures peak keypoint activations",
                RTMPOSE_GAU_POOLED_DIMS,
                RTMPOSE_GAU_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeaturesStd => Self::stored(
                id,
                "GAU Features (Std Pool)",
                "Std pooled GAU features from RTMPose - Captures keypoint variation",
                RTMPOSE_GAU_POOLED_DIMS,
                RTMPOSE_GAU_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::RtmDetExtracted => Self::stored(
                id,
                "RTMDet Extracted Features",
                "Pooled (avg/std/max) cls_p5 + reg_p5 features",
                RTMDET_EXTRACTED_DIMS,
                RTMDET_EXTRACTED_STORAGE_COST,
                ModelCategory::Presence,
            ),
            FeatureType::RtmDetEngineered => Self::computed(
                id,
                "Detection Features",
                "Geometric features from detection bbox and keypoints (135 dims)",
                RTMDET_ENGINEERED_DIMS,
                Some(ModelCategory::Presence),
                None,
                ExtractorKind::RtmDetEngineered,
            ),
            FeatureType::EngineeredFeatures => Self::computed(
                id,
                "Posture Features (1D)",
                "Body proportion ratios with 1D soft binning (54 dims)",
                ENGINEERED_1D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::Engineered,
            ),
            FeatureType::Joint2d => Self::computed(
                id,
                "Joint 2D Histogram",
                "ear_eye_vertical x neck_length (81 dims)",
                JOINT_2D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::Joint2d,
            ),
            FeatureType::Joint3d => Self::computed(
                id,
                "Joint 3D Histogram",
                "ear_eye_vertical x neck_length x head_rotation (125 dims)",
                JOINT_3D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::Joint3d,
            ),
            FeatureType::Joint4d => Self::computed(
                id,
                "Joint 4D Histogram",
                "All 4 raw posture features (625 dims)",
                JOINT_4D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::Joint4d,
            ),
            FeatureType::PostureRaw => Self::computed(
                id,
                "Posture Raw Features",
                "Raw geometric features: ear_eye_vertical, head_rotation, neck_length, inter_ear_distance, avg_shoulder_y (5 dims)",
                POSTURE_RAW_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::PostureRaw,
            ),
            FeatureType::KeypointScores => Self::computed(
                id,
                "Keypoint Scores",
                "Confidence scores for all 17 keypoints (17 dims)",
                KEYPOINT_SCORES_DIMS,
                None,
                Some(false),
                ExtractorKind::KeypointScores,
            ),
            FeatureType::RawKeypoints => Self::computed(
                id,
                "Raw Keypoints",
                "Raw x,y coordinates for all 17 keypoints (34 dims)",
                RAW_KEYPOINTS_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::RawKeypoints,
            ),
            FeatureType::PostureGeometry => Self::computed(
                id,
                "Posture Geometry (invariant)",
                "10 scale/translation-invariant geometric posture features from head and shoulder keypoints",
                POSTURE_GEOMETRY_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::PostureGeometry,
            ),
            FeatureType::TorsoInvariant => Self::computed(
                id,
                "Torso-Invariant Geometry",
                "7 scale/translation-invariant torso-anchored features separating head flexion from trunk slouch (7 dims)",
                TORSO_INVARIANT_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::TorsoInvariant,
            ),
        }
    }

    /// Extract this feature from a container.
    ///
    /// Stored features preserve the source's distinction between a missing
    /// feature (`None`) and a present, possibly empty, vector (`Some(vec![])`).
    pub fn extract(
        &self,
        container: &impl FeatureContainer,
    ) -> Result<Option<Vec<f32>>, FeatureExtractionError> {
        match self.extractor {
            ExtractorKind::Stored => Ok(container.features().get(&self.id).cloned()),
            ExtractorKind::RtmDetEngineered => Ok(Some(extract_rtm_det_engineered_features(
                Some(container.keypoints()),
                Some(container.bbox().original_bbox()),
            )?)),
            ExtractorKind::Engineered => Ok(extract_engineered_features(container.keypoints())?),
            ExtractorKind::Joint2d => Ok(extract_joint_2d_features(Some(container.keypoints()))?),
            ExtractorKind::Joint3d => Ok(extract_joint_3d_features(Some(container.keypoints()))?),
            ExtractorKind::Joint4d => Ok(extract_joint_4d_features(Some(container.keypoints()))?),
            ExtractorKind::PostureRaw => {
                Ok(extract_posture_raw_features(Some(container.keypoints()))?)
            }
            ExtractorKind::KeypointScores => {
                Ok(extract_keypoint_scores_feature(Some(container.keypoints())))
            }
            ExtractorKind::RawKeypoints => Ok(extract_raw_keypoints(Some(container.keypoints()))?),
            ExtractorKind::PostureGeometry => {
                Ok(extract_posture_geometry_features(container.keypoints())?)
            }
            ExtractorKind::TorsoInvariant => {
                Ok(extract_torso_invariant_features(container.keypoints())?)
            }
        }
    }
}

/// Registry of all available feature types.
pub const FEATURE_REGISTRY: [FeatureDefinition; 17] = [
    FeatureDefinition::from_feature_type(FEATURE_BACKBONE_AVG),
    FeatureDefinition::from_feature_type(FEATURE_BACKBONE_MAX),
    FeatureDefinition::from_feature_type(FEATURE_BACKBONE_STD),
    FeatureDefinition::from_feature_type(FEATURE_GAU_AVG),
    FeatureDefinition::from_feature_type(FEATURE_GAU_MAX),
    FeatureDefinition::from_feature_type(FEATURE_GAU_STD),
    FeatureDefinition::from_feature_type(FEATURE_RTMDET_EXTRACTED),
    FeatureDefinition::from_feature_type(FEATURE_RTMDET_ENGINEERED),
    FeatureDefinition::from_feature_type(FEATURE_ENGINEERED),
    FeatureDefinition::from_feature_type(FEATURE_JOINT_2D),
    FeatureDefinition::from_feature_type(FEATURE_JOINT_3D),
    FeatureDefinition::from_feature_type(FEATURE_JOINT_4D),
    FeatureDefinition::from_feature_type(FEATURE_POSTURE_RAW),
    FeatureDefinition::from_feature_type(FEATURE_KEYPOINT_SCORES),
    FeatureDefinition::from_feature_type(FEATURE_RAW_KEYPOINTS),
    FeatureDefinition::from_feature_type(FEATURE_POSTURE_GEOMETRY),
    FeatureDefinition::from_feature_type(FEATURE_TORSO_INVARIANT),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFeatureType {
    pub requested: String,
}

impl fmt::Display for UnknownFeatureType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Feature type \"{}\" not found in registry. Available types: {}",
            self.requested,
            FEATURE_TYPES
                .iter()
                .map(|feature_type| feature_type.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl std::error::Error for UnknownFeatureType {}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureExtractionError {
    Engineered(EngineeredFeaturesError),
    RtmDet(RtmDetEngineeredFeaturesError),
}

impl fmt::Display for FeatureExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engineered(error) => error.fmt(formatter),
            Self::RtmDet(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FeatureExtractionError {}

impl From<EngineeredFeaturesError> for FeatureExtractionError {
    fn from(error: EngineeredFeaturesError) -> Self {
        Self::Engineered(error)
    }
}

impl From<RtmDetEngineeredFeaturesError> for FeatureExtractionError {
    fn from(error: RtmDetEngineeredFeaturesError) -> Self {
        Self::RtmDet(error)
    }
}

/// Get a feature definition by its serialized identifier.
pub fn require_feature_definition(
    feature_type: &str,
) -> Result<&'static FeatureDefinition, UnknownFeatureType> {
    FEATURE_REGISTRY
        .iter()
        .find(|definition| definition.id.as_str() == feature_type)
        .ok_or_else(|| UnknownFeatureType {
            requested: feature_type.to_owned(),
        })
}

/// Get the feature vector dimension for a serialized identifier.
pub fn get_feature_dimensions(feature_type: &str) -> Result<usize, UnknownFeatureType> {
    Ok(require_feature_definition(feature_type)?.dimensions)
}

/// Return all feature identifiers in registry order.
pub fn get_all_feature_types() -> [FeatureType; 17] {
    FEATURE_TYPES
}

/// Check whether a serialized identifier is computed on demand.
pub fn is_computed_feature(feature_type: &str) -> Result<bool, UnknownFeatureType> {
    Ok(require_feature_definition(feature_type)?.computed)
}

/// Check whether a string is a valid feature type.
pub fn is_feature_type(value: &str) -> bool {
    FEATURE_TYPES
        .iter()
        .any(|feature_type| feature_type.as_str() == value)
}

/// Return feature types exposed for user selection.
pub fn get_user_selectable_feature_types() -> Vec<FeatureType> {
    FEATURE_REGISTRY
        .iter()
        .filter(|definition| definition.user_selectable)
        .map(|definition| definition.id)
        .collect()
}

/// Extract a feature by serialized identifier.
pub fn extract_feature(
    feature_type: &str,
    container: &impl FeatureContainer,
) -> Result<Option<Vec<f32>>, FeatureRegistryError> {
    let definition = require_feature_definition(feature_type)?;
    Ok(definition.extract(container)?)
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureRegistryError {
    Unknown(UnknownFeatureType),
    Extraction(FeatureExtractionError),
}

impl fmt::Display for FeatureRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(error) => error.fmt(formatter),
            Self::Extraction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FeatureRegistryError {}

impl From<UnknownFeatureType> for FeatureRegistryError {
    fn from(error: UnknownFeatureType) -> Self {
        Self::Unknown(error)
    }
}

impl From<FeatureExtractionError> for FeatureRegistryError {
    fn from(error: FeatureExtractionError) -> Self {
        Self::Extraction(error)
    }
}
