//! Feature type registry and feature extraction dispatch for dataset frames.
//!
//! The source registry combines feature metadata with the extractor for each
//! feature kind. Rust keeps that metadata in a fixed registry and represents
//! the source callbacks with a small, explicit extractor enum.

use std::fmt;

use slouch_domain::{BboxAccessor, ModelCategory};
use slouch_ml::ported::constants::{
    ENGINEERED_1D_DIMS, JOINT_2D_DIMS, JOINT_3D_DIMS, JOINT_4D_DIMS, KEYPOINT_SCORES_DIMS,
    NLF_BACKBONE_POOLED_DIMS, NLF_BACKBONE_POOLED_STORAGE_COST, NLF_DEPTH_DIMS,
    NLF_DEPTH_STORAGE_COST, POSTURE_GEOMETRY_3D_DIMS, POSTURE_GEOMETRY_DIMS, POSTURE_RAW_3D_DIMS,
    POSTURE_RAW_DIMS, RAW_KEYPOINTS_3D_DIMS, RAW_KEYPOINTS_3D_STORAGE_COST, RAW_KEYPOINTS_DIMS,
    RTMDET_ENGINEERED_DIMS, RTMDET_EXTRACTED_DIMS, RTMDET_EXTRACTED_STORAGE_COST,
    RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_BACKBONE_POOLED_STORAGE_COST, RTMPOSE_GAU_POOLED_DIMS,
    RTMPOSE_GAU_POOLED_STORAGE_COST, TORSO_INVARIANT_3D_DIMS, TORSO_INVARIANT_DIMS,
};
use slouch_ml::ported::engineered_features::{
    extract_engineered_features, extract_joint_2d_features, extract_joint_3d_features,
    extract_joint_4d_features, extract_posture_geometry_features, extract_posture_raw_features,
    extract_raw_keypoints, extract_torso_invariant_features, EngineeredFeaturesError,
};
use slouch_ml::ported::keypoints_3d_features::{
    extract_posture_geometry_3d, extract_posture_raw_3d, extract_torso_invariant_3d,
    Keypoints3dError,
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
pub const FEATURE_NLF_DEPTH: FeatureType = FeatureType::NlfDepth;
pub const FEATURE_NLF_BACKBONE: FeatureType = FeatureType::NlfBackbone;
pub const FEATURE_NLF_BACKBONE_MAX: FeatureType = FeatureType::NlfBackboneMax;
pub const FEATURE_NLF_BACKBONE_STD: FeatureType = FeatureType::NlfBackboneStd;
pub const FEATURE_RAW_KEYPOINTS_3D: FeatureType = FeatureType::RawKeypoints3d;
pub const FEATURE_POSTURE_RAW_3D: FeatureType = FeatureType::PostureRaw3d;
pub const FEATURE_POSTURE_GEOMETRY_3D: FeatureType = FeatureType::PostureGeometry3d;
pub const FEATURE_TORSO_INVARIANT_3D: FeatureType = FeatureType::TorsoInvariant3d;

/// Valid feature type identifiers, in the same insertion order as the source
/// object registry.
pub const FEATURE_TYPES: [FeatureType; 25] = FeatureType::ALL;

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
    PostureRaw3d,
    PostureGeometry3d,
    TorsoInvariant3d,
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

    /// A retired stored feature: the variant/dimensions/storage persist for
    /// existing datasets, but it is hidden from user selection because RTMPose no
    /// longer produces it (NLF is the sole pose model).
    const fn stored_unavailable(
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
            user_selectable: false,
            requires_fitting: None,
            extractor: ExtractorKind::Stored,
        }
    }

    /// A present-by-design stored substrate, hidden from direct selection and
    /// consumed as a dependency by the computed 3D features. Unlike
    /// `stored_unavailable` (retired/not-produced), this variant is produced at
    /// capture time; it is merely hidden from the selector.
    const fn stored_hidden(
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
            user_selectable: false,
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
            FeatureType::BackboneFeatures => Self::stored_unavailable(
                id,
                "Backbone Features (Avg Pool)",
                "Average pooled backbone features from RTMPose",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::BackboneFeaturesMax => Self::stored_unavailable(
                id,
                "Backbone Features (Max Pool)",
                "Max pooled backbone features from RTMPose - Captures peak spatial activations",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::BackboneFeaturesStd => Self::stored_unavailable(
                id,
                "Backbone Features (Std Pool)",
                "Std pooled backbone features from RTMPose - Captures spatial variation",
                RTMPOSE_BACKBONE_POOLED_DIMS,
                RTMPOSE_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeatures => Self::stored_unavailable(
                id,
                "GAU Features (Avg Pool)",
                "Average pooled GAU features from RTMPose",
                RTMPOSE_GAU_POOLED_DIMS,
                RTMPOSE_GAU_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeaturesMax => Self::stored_unavailable(
                id,
                "GAU Features (Max Pool)",
                "Max pooled GAU features from RTMPose - Captures peak keypoint activations",
                RTMPOSE_GAU_POOLED_DIMS,
                RTMPOSE_GAU_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::GauFeaturesStd => Self::stored_unavailable(
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
            FeatureType::NlfDepth => Self::stored(
                id,
                "NLF Depth (3D)",
                "14 camera-robust 3D depth cues from NLF-L (forward-head, neck flexion, trunk lean, plus confidence/truncation guards)",
                NLF_DEPTH_DIMS,
                NLF_DEPTH_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::NlfBackbone => Self::stored(
                id,
                "NLF Backbone Features (Avg Pool)",
                "Average-pooled NLF-L backbone embedding (512 dims)",
                NLF_BACKBONE_POOLED_DIMS,
                NLF_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::NlfBackboneMax => Self::stored(
                id,
                "NLF Backbone Features (Max Pool)",
                "Max-pooled NLF-L backbone embedding - captures peak spatial activations (512 dims)",
                NLF_BACKBONE_POOLED_DIMS,
                NLF_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::NlfBackboneStd => Self::stored(
                id,
                "NLF Backbone Features (Std Pool)",
                "Std-pooled NLF-L backbone embedding - captures spatial variation (512 dims)",
                NLF_BACKBONE_POOLED_DIMS,
                NLF_BACKBONE_POOLED_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::RawKeypoints3d => Self::stored_hidden(
                id,
                "Raw Keypoints 3D",
                "17 torso-normalized, root-centered 3D COCO keypoints (51 dims) - hidden substrate for the 3D posture features",
                RAW_KEYPOINTS_3D_DIMS,
                RAW_KEYPOINTS_3D_STORAGE_COST,
                ModelCategory::Posture,
            ),
            FeatureType::PostureRaw3d => Self::computed(
                id,
                "Posture Raw Features (3D)",
                "6 root-centered 3D raw posture features rebuilt from the 3D keypoint substrate",
                POSTURE_RAW_3D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::PostureRaw3d,
            ),
            FeatureType::PostureGeometry3d => Self::computed(
                id,
                "Posture Geometry (3D)",
                "10 body-frame 3D geometric posture features rebuilt from the 3D keypoint substrate",
                POSTURE_GEOMETRY_3D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::PostureGeometry3d,
            ),
            FeatureType::TorsoInvariant3d => Self::computed(
                id,
                "Torso-Invariant Geometry (3D)",
                "9 torso-anchored 3D posture features (camera and body-frame) rebuilt from the 3D keypoint substrate",
                TORSO_INVARIANT_3D_DIMS,
                Some(ModelCategory::Posture),
                Some(false),
                ExtractorKind::TorsoInvariant3d,
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
            ExtractorKind::PostureRaw3d => Ok(extract_posture_raw_3d(
                container
                    .features()
                    .get(&FeatureType::RawKeypoints3d)
                    .map(Vec::as_slice),
                container.keypoints(),
            )?),
            ExtractorKind::PostureGeometry3d => Ok(extract_posture_geometry_3d(
                container
                    .features()
                    .get(&FeatureType::RawKeypoints3d)
                    .map(Vec::as_slice),
                container.keypoints(),
            )?),
            ExtractorKind::TorsoInvariant3d => Ok(extract_torso_invariant_3d(
                container
                    .features()
                    .get(&FeatureType::RawKeypoints3d)
                    .map(Vec::as_slice),
                container.keypoints(),
            )?),
        }
    }
}

/// Registry of all available feature types.
pub const FEATURE_REGISTRY: [FeatureDefinition; 25] = [
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
    FeatureDefinition::from_feature_type(FEATURE_NLF_DEPTH),
    FeatureDefinition::from_feature_type(FEATURE_NLF_BACKBONE),
    FeatureDefinition::from_feature_type(FEATURE_NLF_BACKBONE_MAX),
    FeatureDefinition::from_feature_type(FEATURE_NLF_BACKBONE_STD),
    FeatureDefinition::from_feature_type(FEATURE_RAW_KEYPOINTS_3D),
    FeatureDefinition::from_feature_type(FEATURE_POSTURE_RAW_3D),
    FeatureDefinition::from_feature_type(FEATURE_POSTURE_GEOMETRY_3D),
    FeatureDefinition::from_feature_type(FEATURE_TORSO_INVARIANT_3D),
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
    Keypoints3d(Keypoints3dError),
}

impl fmt::Display for FeatureExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engineered(error) => error.fmt(formatter),
            Self::RtmDet(error) => error.fmt(formatter),
            Self::Keypoints3d(error) => error.fmt(formatter),
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

impl From<Keypoints3dError> for FeatureExtractionError {
    fn from(error: Keypoints3dError) -> Self {
        Self::Keypoints3d(error)
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
pub fn get_all_feature_types() -> [FeatureType; 25] {
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
