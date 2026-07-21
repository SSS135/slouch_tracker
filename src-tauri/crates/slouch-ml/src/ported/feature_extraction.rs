//! Feature extraction utilities for training and inference.
//!
//! This is the native equivalent of `src/services/ml/featureExtraction.ts`.
//! Feature vectors are represented as `Vec<f32>` and containers use the shared
//! `slouch-domain::FeatureSource` contract implemented by both frames and
//! inference results.

use std::{fmt, str::FromStr};

use slouch_domain::{BboxAccessor, FeatureId, FeatureSource, PostureFrame};

use super::engineered_features::{
    extract_engineered_features, extract_joint_2d_features, extract_joint_3d_features,
    extract_joint_4d_features, extract_posture_geometry_features, extract_posture_raw_features,
    extract_raw_keypoints, extract_torso_invariant_features, EngineeredFeaturesError,
};
use super::rtmdet_engineered_features::{
    extract_keypoint_scores_feature, extract_rtm_det_engineered_features,
    RtmDetEngineeredFeaturesError,
};

/// Values accepted as feature identifiers by the extraction API.
///
/// The browser source accepts strings while native callers normally use the
/// shared `FeatureId` enum. Supporting both keeps the public helpers useful at
/// the boundary without duplicating the extraction implementation.
pub trait FeatureTypeValue {
    fn feature_id(&self) -> Result<FeatureId, String>;
    fn feature_name(&self) -> String;
}

impl FeatureTypeValue for FeatureId {
    fn feature_id(&self) -> Result<FeatureId, String> {
        Ok(*self)
    }

    fn feature_name(&self) -> String {
        self.as_str().to_owned()
    }
}

impl FeatureTypeValue for &str {
    fn feature_id(&self) -> Result<FeatureId, String> {
        FeatureId::from_str(self).map_err(|_| (*self).to_owned())
    }

    fn feature_name(&self) -> String {
        (*self).to_owned()
    }
}

impl FeatureTypeValue for String {
    fn feature_id(&self) -> Result<FeatureId, String> {
        FeatureId::from_str(self).map_err(|_| self.clone())
    }

    fn feature_name(&self) -> String {
        self.clone()
    }
}

/// Errors produced while extracting or assembling feature vectors.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureExtractionError {
    UnknownFeature {
        feature_type: String,
    },
    FeatureUnavailable {
        feature_type: String,
        available_features: Vec<String>,
        required_dependencies: Vec<String>,
    },
    DimensionMismatch {
        feature_type: String,
        expected: usize,
        actual: usize,
    },
    EmptyFeatureTypes,
    MissingFeaturesInDataset {
        errors: Vec<String>,
    },
    Engineered(EngineeredFeaturesError),
    RtmDetEngineered(RtmDetEngineeredFeaturesError),
}

impl fmt::Display for FeatureExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFeature { feature_type } => {
                write!(formatter, "Unknown feature type: {feature_type}")
            }
            Self::FeatureUnavailable {
                feature_type,
                available_features,
                required_dependencies,
            } => write!(
                formatter,
                "Feature type {feature_type} not available in this container. Available features: {}. Required dependencies: {}",
                if available_features.is_empty() {
                    "none".to_owned()
                } else {
                    available_features.join(", ")
                },
                if required_dependencies.is_empty() {
                    "none".to_owned()
                } else {
                    required_dependencies.join(", ")
                }
            ),
            Self::DimensionMismatch {
                feature_type,
                expected,
                actual,
            } => write!(
                formatter,
                "Feature \"{feature_type}\" dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::EmptyFeatureTypes => {
                formatter.write_str("At least one feature type must be specified")
            }
            Self::MissingFeaturesInDataset { errors } => {
                write!(formatter, "Missing features in dataset: {}", errors.join("; "))
            }
            Self::Engineered(error) => error.fmt(formatter),
            Self::RtmDetEngineered(error) => error.fmt(formatter),
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
        Self::RtmDetEngineered(error)
    }
}

/// Insertion-ordered missing-feature results matching JavaScript `Map`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MissingFeatureMap(Vec<(String, Vec<String>)>);

impl MissingFeatureMap {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, feature_type: &str) -> Option<&Vec<String>> {
        self.0
            .iter()
            .find_map(|(candidate, frames)| (candidate == feature_type).then_some(frames))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Vec<String>)> {
        self.0
            .iter()
            .map(|(feature_type, frames)| (feature_type.as_str(), frames))
    }

    fn insert(&mut self, feature_type: String, frames: Vec<String>) {
        self.0.push((feature_type, frames));
    }
}

/// Extracts one registered feature from a frame or inference result.
pub fn extract_features<S, T>(
    container: &S,
    feature_type: T,
) -> Result<Vec<f32>, FeatureExtractionError>
where
    S: FeatureSource,
    T: FeatureTypeValue,
{
    let feature_id = resolve_feature_type(&feature_type)?;
    let feature_name = feature_id.as_str();

    let extracted =
        match feature_id {
            FeatureId::BackboneFeatures
            | FeatureId::BackboneFeaturesMax
            | FeatureId::BackboneFeaturesStd
            | FeatureId::GauFeatures
            | FeatureId::GauFeaturesMax
            | FeatureId::GauFeaturesStd
            | FeatureId::RtmDetExtracted => Ok(container.features().get(&feature_id).cloned()),
            FeatureId::RtmDetEngineered => extract_rtm_det_engineered_features(
                Some(container.keypoints()),
                Some(container.bbox().original_bbox()),
            )
            .map(Some)
            .map_err(FeatureExtractionError::from),
            FeatureId::EngineeredFeatures => extract_engineered_features(container.keypoints())
                .map_err(FeatureExtractionError::from),
            FeatureId::Joint2d => extract_joint_2d_features(Some(container.keypoints()))
                .map_err(FeatureExtractionError::from),
            FeatureId::Joint3d => extract_joint_3d_features(Some(container.keypoints()))
                .map_err(FeatureExtractionError::from),
            FeatureId::Joint4d => extract_joint_4d_features(Some(container.keypoints()))
                .map_err(FeatureExtractionError::from),
            FeatureId::PostureRaw => extract_posture_raw_features(Some(container.keypoints()))
                .map_err(FeatureExtractionError::from),
            FeatureId::KeypointScores => {
                Ok(extract_keypoint_scores_feature(Some(container.keypoints())))
            }
            FeatureId::RawKeypoints => extract_raw_keypoints(Some(container.keypoints()))
                .map_err(FeatureExtractionError::from),
            FeatureId::PostureGeometry => extract_posture_geometry_features(container.keypoints())
                .map_err(FeatureExtractionError::from),
            FeatureId::TorsoInvariant => extract_torso_invariant_features(container.keypoints())
                .map_err(FeatureExtractionError::from),
        };

    let Some(feature_array) = extracted? else {
        let available_features = container
            .features()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        log_error(&format!(
            "[EXTRACT] Failed to extract \"{feature_name}\". Available features: {}",
            if available_features.is_empty() {
                "none".to_owned()
            } else {
                available_features.join(", ")
            }
        ));
        return Err(FeatureExtractionError::FeatureUnavailable {
            feature_type: feature_name.to_owned(),
            available_features,
            required_dependencies: Vec::new(),
        });
    };

    let expected = feature_id.metadata().dimensions;
    if feature_array.len() != expected {
        return Err(FeatureExtractionError::DimensionMismatch {
            feature_type: feature_name.to_owned(),
            expected,
            actual: feature_array.len(),
        });
    }

    Ok(feature_array)
}

/// Concatenates selected feature vectors in the requested order.
pub fn concatenate_features<S, T>(
    container: &S,
    feature_types: &[T],
) -> Result<Vec<f32>, FeatureExtractionError>
where
    S: FeatureSource,
    T: FeatureTypeValue + Clone,
{
    if feature_types.is_empty() {
        return Err(FeatureExtractionError::EmptyFeatureTypes);
    }

    let feature_arrays = feature_types
        .iter()
        .map(|feature_type| extract_features(container, feature_type.clone()))
        .collect::<Result<Vec<_>, _>>()?;

    if feature_arrays.len() == 1 {
        return Ok(feature_arrays[0].clone());
    }

    let total_length = feature_arrays.iter().map(Vec::len).sum();
    let mut concatenated = Vec::with_capacity(total_length);
    for feature_array in feature_arrays {
        concatenated.extend(feature_array);
    }
    Ok(concatenated)
}

/// Returns whether a registered feature can be extracted from a container.
///
/// Unknown features and genuinely absent dependencies are ordinary `false`
/// results. Malformed geometry or non-finite computed inputs remain typed
/// extraction errors instead of being hidden as feature absence.
pub fn is_feature_available<S, T>(
    container: &S,
    feature_type: T,
) -> Result<bool, FeatureExtractionError>
where
    S: FeatureSource,
    T: FeatureTypeValue,
{
    match extract_features(container, feature_type) {
        Ok(_) => Ok(true),
        Err(FeatureExtractionError::UnknownFeature { .. })
        | Err(FeatureExtractionError::FeatureUnavailable { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

/// Returns the IDs of frames that do not contain the requested feature.
pub fn validate_frames_have_feature<T>(
    frames: &[PostureFrame],
    feature_type: T,
) -> Result<Vec<String>, FeatureExtractionError>
where
    T: FeatureTypeValue + Clone,
{
    let mut missing = Vec::new();
    for frame in frames {
        if !is_feature_available(frame, feature_type.clone())? {
            missing.push(frame.id.clone());
        }
    }
    Ok(missing)
}

/// Returns missing frame IDs grouped by requested feature type.
pub fn validate_frames_have_features<T>(
    frames: &[PostureFrame],
    feature_types: &[T],
) -> Result<MissingFeatureMap, FeatureExtractionError>
where
    T: FeatureTypeValue + Clone,
{
    let mut missing_map = MissingFeatureMap::default();

    for feature_type in feature_types {
        let feature_name = feature_type.feature_name();
        let missing_frames = validate_frames_have_feature(frames, feature_type.clone())?;
        if !missing_frames.is_empty() {
            missing_map.insert(feature_name, missing_frames);
        }
    }

    Ok(missing_map)
}

/// Builds a row-major feature matrix from the supplied posture frames.
pub fn build_feature_matrix<T>(
    frames: &[PostureFrame],
    feature_types: &[T],
) -> Result<Vec<Vec<f32>>, FeatureExtractionError>
where
    T: FeatureTypeValue + Clone,
{
    log_debug(&format!(
        "[FEATURE_EXTRACT] Building feature matrix for {} frames",
        frames.len()
    ));
    log_debug(&format!(
        "[FEATURE_EXTRACT] Feature types: {}",
        feature_types
            .iter()
            .map(FeatureTypeValue::feature_name)
            .collect::<Vec<_>>()
            .join(", ")
    ));

    let missing_map = match validate_frames_have_features(frames, feature_types) {
        Ok(missing_map) => missing_map,
        Err(error) => {
            log_error(&format!("[FEATURE_EXTRACT] Extraction failed: {error}"));
            return Err(error);
        }
    };
    if !missing_map.is_empty() {
        let errors = missing_map
            .iter()
            .map(|(feature_type, frame_ids)| {
                format!(
                    "{} frame(s) missing {feature_type} features (e.g., {})",
                    frame_ids.len(),
                    frame_ids
                        .iter()
                        .take(3)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>();
        log_error(&format!(
            "[FEATURE_EXTRACT] Missing features: {}",
            errors.join("; ")
        ));
        return Err(FeatureExtractionError::MissingFeaturesInDataset { errors });
    }

    let expected_dimensions = get_expected_concatenated_dimensions(feature_types)?;
    log_debug(&format!(
        "[FEATURE_EXTRACT] Expected concatenated dimensions: {expected_dimensions}"
    ));
    let matrix = match frames
        .iter()
        .map(|frame| concatenate_features(frame, feature_types))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(matrix) => matrix,
        Err(error) => {
            log_error(&format!("[FEATURE_EXTRACT] Matrix build failed: {error}"));
            return Err(error);
        }
    };

    log_debug(&format!(
        "[FEATURE_EXTRACT] Built matrix: {} samples × {} features",
        matrix.len(),
        matrix.first().map_or(0, Vec::len)
    ));
    if let Some(first) = matrix.first() {
        if first.len() != expected_dimensions {
            log_warn(&format!(
                "[FEATURE_EXTRACT] Dimension mismatch! Expected {expected_dimensions}, got {}",
                first.len()
            ));
        }
    }

    Ok(matrix)
}

/// Returns the registered dimension for one feature type.
pub fn get_expected_dimensions<T>(feature_type: T) -> Result<usize, FeatureExtractionError>
where
    T: FeatureTypeValue,
{
    Ok(resolve_feature_type(&feature_type)?.metadata().dimensions)
}

/// Returns the sum of the registered dimensions for selected feature types.
pub fn get_expected_concatenated_dimensions<T>(
    feature_types: &[T],
) -> Result<usize, FeatureExtractionError>
where
    T: FeatureTypeValue + Clone,
{
    feature_types.iter().try_fold(0, |sum, feature_type| {
        get_expected_dimensions(feature_type.clone()).map(|dimensions| sum + dimensions)
    })
}

fn resolve_feature_type<T>(feature_type: &T) -> Result<FeatureId, FeatureExtractionError>
where
    T: FeatureTypeValue,
{
    feature_type
        .feature_id()
        .map_err(|feature_type| FeatureExtractionError::UnknownFeature { feature_type })
}

fn log_debug(message: &str) {
    eprintln!("[training][debug] {message}");
}

fn log_error(message: &str) {
    eprintln!("[training][error] {message}");
}

fn log_warn(message: &str) {
    eprintln!("[training][warn] {message}");
}
