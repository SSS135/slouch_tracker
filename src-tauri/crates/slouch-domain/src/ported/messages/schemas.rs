//! Typed Rust equivalents of the worker message schemas.
//!
//! The browser schemas validate structured-clone messages at the worker edge.
//! These types keep the same discriminators and camel-case wire fields while
//! using native domain DTOs for inference results and model state.

use std::collections::BTreeMap;
use std::fmt;

use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    validate_classification_result, validate_inference_result, ClassificationResult, ExpandedBbox,
    FeatureMap, InferenceResult, Keypoint,
};

fn deserialize_present_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    Option::<T>::deserialize(deserializer)?
        .map(Some)
        .ok_or_else(|| {
            serde::de::Error::custom("explicit null is not accepted; omit the optional field")
        })
}

/// Mirrors the oracle `z.string().min(1)` guard on worker path fields.
fn deserialize_nonempty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.is_empty() {
        return Err(serde::de::Error::custom("string must be non-empty"));
    }
    Ok(value)
}

/// JSON-compatible value used only for diagnostic response details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

/// Maximum accepted RGBA frame size at the native inference boundary.
pub const MAX_IMAGE_PIXELS: u64 = 1920 * 1080;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageDataError {
    EmptyDimensions,
    DimensionsOverflow,
    Oversized { pixels: u64, maximum: u64 },
    InvalidByteLength { expected: usize, actual: usize },
}

impl fmt::Display for ImageDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => formatter.write_str("image dimensions must be positive"),
            Self::DimensionsOverflow => formatter.write_str("image dimensions overflow"),
            Self::Oversized { pixels, maximum } => {
                write!(formatter, "image has {pixels} pixels, maximum is {maximum}")
            }
            Self::InvalidByteLength { expected, actual } => {
                write!(
                    formatter,
                    "image has {actual} RGBA bytes, expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for ImageDataError {}

/// Raw RGBA image data carried only by the in-process/binary inference boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl ImageData {
    pub fn try_new(data: Vec<u8>, width: u32, height: u32) -> Result<Self, ImageDataError> {
        let image = Self {
            data,
            width,
            height,
        };
        image.validate()?;
        Ok(image)
    }

    pub fn validate(&self) -> Result<(), ImageDataError> {
        if self.width == 0 || self.height == 0 {
            return Err(ImageDataError::EmptyDimensions);
        }
        let pixels = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .ok_or(ImageDataError::DimensionsOverflow)?;
        if pixels > MAX_IMAGE_PIXELS {
            return Err(ImageDataError::Oversized {
                pixels,
                maximum: MAX_IMAGE_PIXELS,
            });
        }
        let expected = pixels
            .checked_mul(4)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(ImageDataError::DimensionsOverflow)?;
        if self.data.len() != expected {
            return Err(ImageDataError::InvalidByteLength {
                expected,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

impl Serialize for ImageData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            return Err(serde::ser::Error::custom(
                "image pixels require the native binary transport",
            ));
        }
        (&self.data, self.width, self.height).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ImageData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            return Err(serde::de::Error::custom(
                "image pixels require the native binary transport",
            ));
        }
        let (data, width, height) = <(Vec<u8>, u32, u32)>::deserialize(deserializer)?;
        Self::try_new(data, width, height).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedModel {
    pub feature_extractor: SerializedFeatureExtractor,
    pub classifier: SerializedClassifier,
    pub trained_at: f64,
    pub version: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedFeatureExtractor {
    pub feature_types: Vec<String>,
    pub normalization_mode: NormalizationMode,
    pub dim_reduction_config: DimensionalityReductionConfig,
    pub concatenated_dimensions: usize,
    pub normalization_mean: Option<Vec<f64>>,
    pub normalization_std: Option<Vec<f64>>,
    pub dim_reduction_transformer: Option<DimensionalityReductionTransformer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    None,
    Layer,
    ZScore,
    Calibrated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DimensionalityReductionConfig {
    pub method: DimensionalityReductionMethod,
    pub components: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionalityReductionMethod {
    RandomProjection,
    Pca,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DimensionalityReductionTransformer {
    #[serde(rename = "random_projection")]
    RandomProjection(RandomProjectionState),
    #[serde(rename = "pca")]
    Pca(SerializedPca),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RandomProjectionState {
    pub projection_matrix: Vec<Vec<f64>>,
    pub n_components: usize,
    pub n_features: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedPca {
    pub components: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    pub n_components: usize,
    pub n_features: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explained_variance: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedClassifier {
    pub classifier_id: String,
    pub state: SerializedClassifierState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializedClassifierState {
    Mlp(SerializedMlp),
    Knn(SerializedKnn),
    Svm(SerializedSvm),
    KMeansPrototype(SerializedKMeansPrototype),
    GaussianNb(SerializedGaussianNb),
    KMeansLogistic(SerializedKMeansLogistic),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMlp {
    pub layer_weights: Vec<Vec<f64>>,
    pub layer_biases: Vec<Vec<f64>>,
    pub layer_shapes: Vec<usize>,
    pub hidden_layers: usize,
    pub hidden_size: usize,
    pub class_weights: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKnn {
    pub training_data: Vec<Vec<f64>>,
    pub training_labels: Vec<f64>,
    pub k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<KnnKernel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnKernel {
    Cosine,
    Rbf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedSvm {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub class_weights: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansPrototype {
    pub clusters: Vec<KMeansPrototypeCluster>,
    pub global_prototype_good: Vec<f64>,
    pub global_prototype_bad: Vec<f64>,
    pub temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KMeansPrototypeCluster {
    pub centroid: Vec<f64>,
    pub prototype_good: Option<Vec<f64>>,
    pub prototype_bad: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedGaussianNb {
    pub class_means: [Vec<f64>; 2],
    pub class_variances: [Vec<f64>; 2],
    pub class_priors: [f64; 2],
    pub epsilon: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansLogistic {
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<Option<SerializedMlp>>,
    pub global_model: SerializedMlp,
    pub temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InferenceWorkerMessage {
    #[serde(rename = "initialize")]
    Initialize { payload: InitializePayload },
    #[serde(rename = "process")]
    Process { payload: ProcessPayload },
    #[serde(rename = "loadPostureModel")]
    LoadPostureModel { payload: PostureModelPayload },
    #[serde(rename = "loadPresenceModel")]
    LoadPresenceModel { payload: PresenceModelPayload },
    #[serde(rename = "unloadClassifier")]
    UnloadClassifier,
    #[serde(rename = "setLogLevel")]
    SetLogLevel {
        #[serde(
            default,
            deserialize_with = "deserialize_present_option",
            skip_serializing_if = "Option::is_none"
        )]
        payload: Option<LogLevelPayload>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializePayload {
    #[serde(deserialize_with = "deserialize_nonempty_string")]
    pub rtmdet_path: String,
    #[serde(deserialize_with = "deserialize_nonempty_string")]
    pub rtmw3d_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessPayload {
    pub image_data: ImageData,
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostureModelPayload {
    pub posture_model: SerializedModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PresenceModelPayload {
    pub presence_model: SerializedModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogLevelPayload {
    #[serde(
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub log_param: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingWorkerMessage {
    #[serde(rename = "train")]
    Train {
        #[serde(
            default,
            deserialize_with = "deserialize_present_option",
            skip_serializing_if = "Option::is_none"
        )]
        payload: Option<TrainPayload>,
    },
    #[serde(rename = "setLogLevel")]
    SetLogLevel {
        #[serde(
            default,
            deserialize_with = "deserialize_present_option",
            skip_serializing_if = "Option::is_none"
        )]
        payload: Option<LogLevelPayload>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainPayload {
    #[serde(
        rename = "doCV",
        default,
        deserialize_with = "deserialize_present_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub do_cv: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InferenceWorkerResponse {
    #[serde(rename = "initialized")]
    Initialized { provider: String },
    #[serde(rename = "classifierLoaded")]
    ClassifierLoaded { success: bool },
    #[serde(rename = "classifierUnloaded")]
    ClassifierUnloaded,
    #[serde(rename = "error")]
    Error {
        error: String,
        #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
        request_id: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
    },
    #[serde(rename = "result")]
    Result {
        #[serde(rename = "requestId")]
        request_id: u64,
        result: InferenceResponseResult,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceResponseResult {
    pub person_found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ExpandedBbox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keypoints: Option<Vec<Keypoint>>,
    pub features: FeatureMap,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationResult>,
}

impl InferenceResponseResult {
    /// Enforces the stricter native result contract after wire decoding and
    /// before a response is published outside the worker boundary.
    pub fn validate_native(&self) -> Result<(), String> {
        if self.person_found {
            let bbox = self
                .bbox
                .ok_or_else(|| "person-found result is missing bbox".to_owned())?;
            let keypoints = self
                .keypoints
                .clone()
                .ok_or_else(|| "person-found result is missing keypoints".to_owned())?;
            return validate_inference_result(&InferenceResult {
                features: self.features.clone(),
                keypoints,
                bbox,
                classification: self.classification,
            })
            .map_err(|error| error.to_string());
        }

        if self.bbox.is_some() || self.keypoints.is_some() {
            return Err("no-person result must not contain bbox or keypoints".to_owned());
        }
        for (feature, values) in &self.features {
            let expected = feature.metadata().dimensions;
            if values.len() != expected {
                return Err(format!(
                    "features.{feature} expected {expected} values, got {}",
                    values.len()
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "features.{feature} must contain only finite values"
                ));
            }
        }
        if let Some(classification) = self.classification {
            validate_classification_result(&classification).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}
