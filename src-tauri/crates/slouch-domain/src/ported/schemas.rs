//! Runtime validation contracts ported from `src/services/validation/schemas.ts`.
//!
//! Zod validates untrusted storage and worker payloads at runtime.  Rust uses
//! serde-backed DTOs plus explicit validators instead of relying on a schema
//! object at each call site.  The validators intentionally mirror the source
//! constraints: feature vectors are only checked to be `Float32Array`-like
//! values, while model metadata and posture geometry receive field checks.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Deserializer, Serialize};

use crate::{PostureFrame, COCO_KEYPOINT_COUNT};

// ============================================================================
// Schema DTOs
// ============================================================================

/// Feature identifiers accepted by the source `z.enum(FEATURE_TYPES)` schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureTypeSchema {
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

impl FeatureTypeSchema {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimensionalityReductionMethodSchema {
    #[serde(rename = "random_projection")]
    RandomProjection,
    #[serde(rename = "pca")]
    Pca,
    #[serde(rename = "none")]
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionalityReductionConfigSchema {
    pub method: DimensionalityReductionMethodSchema,
    pub components: usize,
}

/// Values allowed in the source classifier `params` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SchemaParameterValue {
    Number(f64),
    String(String),
    Boolean(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierConfigSchema {
    pub classifier_id: String,
    pub params: BTreeMap<String, SchemaParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedMlpSchema {
    pub layer_weights: Vec<Vec<f64>>,
    pub layer_biases: Vec<Vec<f64>>,
    pub layer_shapes: Vec<f64>,
    pub hidden_layers: f64,
    pub hidden_size: f64,
    pub class_weights: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKnnSchema {
    pub training_data: Vec<Vec<f64>>,
    pub training_labels: Vec<f64>,
    pub k: f64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_non_null"
    )]
    pub kernel: Option<KnnKernelSchema>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_non_null"
    )]
    pub gamma: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnKernelSchema {
    Cosine,
    Rbf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedSvmSchema {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub class_weights: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansPrototypeSchema {
    pub clusters: Vec<KMeansPrototypeClusterSchema>,
    pub global_prototype_good: Vec<f64>,
    pub global_prototype_bad: Vec<f64>,
    pub temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KMeansPrototypeClusterSchema {
    pub centroid: Vec<f64>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub prototype_good: Option<Vec<f64>>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub prototype_bad: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedGaussianNbSchema {
    pub class_means: [Vec<f64>; 2],
    pub class_variances: [Vec<f64>; 2],
    pub class_priors: [f64; 2],
    pub epsilon: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansLogisticSchema {
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<Option<SerializedMlpSchema>>,
    pub global_model: SerializedMlpSchema,
    pub temperature: f64,
}

/// Rust equivalent of the source untagged classifier-state union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializedClassifierStateSchema {
    Mlp(SerializedMlpSchema),
    Knn(SerializedKnnSchema),
    Svm(SerializedSvmSchema),
    KMeansPrototype(SerializedKMeansPrototypeSchema),
    GaussianNb(SerializedGaussianNbSchema),
    KMeansLogistic(SerializedKMeansLogisticSchema),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RandomProjectionStateSchema {
    pub projection_matrix: Vec<Vec<f64>>,
    pub n_components: f64,
    pub n_features: f64,
    pub seed: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedPcaSchema {
    pub components: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    pub n_components: f64,
    pub n_features: f64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_non_null"
    )]
    pub explained_variance: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DimReductionTransformerSchema {
    #[serde(rename = "random_projection")]
    RandomProjection(RandomProjectionStateSchema),
    #[serde(rename = "pca")]
    Pca(SerializedPcaSchema),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedFeatureExtractorSchema {
    pub feature_types: Vec<FeatureTypeSchema>,
    pub normalization_mode: NormalizationModeSchema,
    pub dim_reduction_config: DimensionalityReductionConfigSchema,
    pub concatenated_dimensions: usize,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub normalization_mean: Option<Vec<f64>>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub normalization_std: Option<Vec<f64>>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub dim_reduction_transformer: Option<DimReductionTransformerSchema>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationModeSchema {
    None,
    Layer,
    ZScore,
    Calibrated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedModelSchema {
    pub feature_extractor: SerializedFeatureExtractorSchema,
    pub classifier: SerializedClassifierSchema,
    pub trained_at: f64,
    pub version: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedClassifierSchema {
    pub classifier_id: String,
    pub state: SerializedClassifierStateSchema,
}

/// Marker matching the exported TypeScript `PostureFrameSchema` value.
/// Validation is provided by [`validate_posture_frame_schema`].
pub struct PostureFrameSchema;

fn deserialize_required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn deserialize_optional_non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

// ============================================================================
// Validation result and error formatting
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaValidationError {
    pub issues: Vec<SchemaIssue>,
}

impl fmt::Display for SchemaValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = self
            .issues
            .iter()
            .map(|issue| format!("{}: {}", issue.path, issue.message))
            .collect::<Vec<_>>()
            .join(", ");
        formatter.write_str(&message)
    }
}

impl std::error::Error for SchemaValidationError {}

pub enum SchemaValidationResult<T> {
    Success { data: T },
    Failure { error: String },
}

impl<T> SchemaValidationResult<T> {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

/// The Rust equivalent of `validateWithSchema`: run a schema validator and
/// retain the parsed value on success, or flatten issues into one message.
pub fn validate_with_schema<T>(
    schema: impl FnOnce(&T) -> Result<(), SchemaValidationError>,
    data: T,
) -> SchemaValidationResult<T> {
    match schema(&data) {
        Ok(()) => SchemaValidationResult::Success { data },
        Err(error) => SchemaValidationResult::Failure {
            error: error.to_string(),
        },
    }
}

fn issue(path: impl Into<String>, message: impl Into<String>) -> SchemaIssue {
    SchemaIssue {
        path: path.into(),
        message: message.into(),
    }
}

fn finish(issues: Vec<SchemaIssue>) -> Result<(), SchemaValidationError> {
    if issues.is_empty() {
        Ok(())
    } else {
        Err(SchemaValidationError { issues })
    }
}

fn finite(value: f64, path: &str, issues: &mut Vec<SchemaIssue>) {
    if !value.is_finite() {
        issues.push(issue(path, "must be a finite number"));
    }
}

fn finite_vec(values: &[f64], path: &str, issues: &mut Vec<SchemaIssue>) {
    for (index, value) in values.iter().enumerate() {
        finite(*value, &format!("{path}.{index}"), issues);
    }
}

fn finite_matrix(values: &[Vec<f64>], path: &str, issues: &mut Vec<SchemaIssue>) {
    for (index, row) in values.iter().enumerate() {
        finite_vec(row, &format!("{path}.{index}"), issues);
    }
}

fn positive_integer(value: usize, path: &str, issues: &mut Vec<SchemaIssue>) {
    if value == 0 {
        issues.push(issue(path, "must be a positive integer"));
    }
}

fn positive_number(value: f64, path: &str, issues: &mut Vec<SchemaIssue>) {
    finite(value, path, issues);
    if value <= 0.0 {
        issues.push(issue(path, "must be positive"));
    }
}

// ============================================================================
// Schema validators
// ============================================================================

fn validate_dimensionality_reduction_config(
    config: &DimensionalityReductionConfigSchema,
    path: &str,
    issues: &mut Vec<SchemaIssue>,
) {
    positive_integer(config.components, &format!("{path}.components"), issues);
}

fn validate_mlp(model: &SerializedMlpSchema, path: &str, issues: &mut Vec<SchemaIssue>) {
    finite_matrix(
        &model.layer_weights,
        &format!("{path}.layerWeights"),
        issues,
    );
    finite_matrix(&model.layer_biases, &format!("{path}.layerBiases"), issues);
    finite_vec(&model.layer_shapes, &format!("{path}.layerShapes"), issues);
    finite(model.hidden_layers, &format!("{path}.hiddenLayers"), issues);
    finite(model.hidden_size, &format!("{path}.hiddenSize"), issues);
    finite_vec(
        &model.class_weights,
        &format!("{path}.classWeights"),
        issues,
    );
}

fn validate_classifier_state(
    state: &SerializedClassifierStateSchema,
    path: &str,
    issues: &mut Vec<SchemaIssue>,
) {
    match state {
        SerializedClassifierStateSchema::Mlp(model) => validate_mlp(model, path, issues),
        SerializedClassifierStateSchema::Knn(model) => {
            finite_matrix(
                &model.training_data,
                &format!("{path}.trainingData"),
                issues,
            );
            finite_vec(
                &model.training_labels,
                &format!("{path}.trainingLabels"),
                issues,
            );
            finite(model.k, &format!("{path}.k"), issues);
            if let Some(gamma) = model.gamma {
                finite(gamma, &format!("{path}.gamma"), issues);
            }
        }
        SerializedClassifierStateSchema::Svm(model) => {
            finite_vec(&model.weights, &format!("{path}.weights"), issues);
            finite(model.bias, &format!("{path}.bias"), issues);
            finite_vec(
                &model.class_weights,
                &format!("{path}.classWeights"),
                issues,
            );
        }
        SerializedClassifierStateSchema::KMeansPrototype(model) => {
            for (index, cluster) in model.clusters.iter().enumerate() {
                finite_vec(
                    &cluster.centroid,
                    &format!("{path}.clusters.{index}.centroid"),
                    issues,
                );
                if let Some(values) = &cluster.prototype_good {
                    finite_vec(
                        values,
                        &format!("{path}.clusters.{index}.prototypeGood"),
                        issues,
                    );
                }
                if let Some(values) = &cluster.prototype_bad {
                    finite_vec(
                        values,
                        &format!("{path}.clusters.{index}.prototypeBad"),
                        issues,
                    );
                }
            }
            finite_vec(
                &model.global_prototype_good,
                &format!("{path}.globalPrototypeGood"),
                issues,
            );
            finite_vec(
                &model.global_prototype_bad,
                &format!("{path}.globalPrototypeBad"),
                issues,
            );
            finite(model.temperature, &format!("{path}.temperature"), issues);
        }
        SerializedClassifierStateSchema::GaussianNb(model) => {
            for (index, values) in model.class_means.iter().enumerate() {
                finite_vec(values, &format!("{path}.classMeans.{index}"), issues);
            }
            for (index, values) in model.class_variances.iter().enumerate() {
                finite_vec(values, &format!("{path}.classVariances.{index}"), issues);
            }
            finite_vec(&model.class_priors, &format!("{path}.classPriors"), issues);
            finite(model.epsilon, &format!("{path}.epsilon"), issues);
        }
        SerializedClassifierStateSchema::KMeansLogistic(model) => {
            finite_matrix(&model.centroids, &format!("{path}.centroids"), issues);
            for (index, cluster) in model.cluster_models.iter().enumerate() {
                if let Some(cluster) = cluster {
                    validate_mlp(cluster, &format!("{path}.clusterModels.{index}"), issues);
                }
            }
            validate_mlp(&model.global_model, &format!("{path}.globalModel"), issues);
            finite(model.temperature, &format!("{path}.temperature"), issues);
        }
    }
}

fn validate_transformer(
    transformer: &DimReductionTransformerSchema,
    path: &str,
    issues: &mut Vec<SchemaIssue>,
) {
    match transformer {
        DimReductionTransformerSchema::RandomProjection(state) => {
            finite_matrix(
                &state.projection_matrix,
                &format!("{path}.data.projectionMatrix"),
                issues,
            );
            finite(
                state.n_components,
                &format!("{path}.data.nComponents"),
                issues,
            );
            finite(state.n_features, &format!("{path}.data.nFeatures"), issues);
            finite(state.seed, &format!("{path}.data.seed"), issues);
        }
        DimReductionTransformerSchema::Pca(state) => {
            finite_matrix(
                &state.components,
                &format!("{path}.data.components"),
                issues,
            );
            finite_vec(&state.mean, &format!("{path}.data.mean"), issues);
            finite(
                state.n_components,
                &format!("{path}.data.nComponents"),
                issues,
            );
            finite(state.n_features, &format!("{path}.data.nFeatures"), issues);
            if let Some(values) = &state.explained_variance {
                finite_vec(values, &format!("{path}.data.explainedVariance"), issues);
            }
        }
    }
}

/// Validates the serialized classifier-state union.
pub use self::validate_serialized_classifier_state_schema as validate_serialized_classifier_state;

pub fn validate_serialized_classifier_state_schema(
    state: &SerializedClassifierStateSchema,
) -> Result<(), SchemaValidationError> {
    let mut issues = Vec::new();
    validate_classifier_state(state, "state", &mut issues);
    finish(issues)
}

/// Validates a serialized feature extractor using the source schema's bounds.
pub fn validate_serialized_feature_extractor_schema(
    extractor: &SerializedFeatureExtractorSchema,
) -> Result<(), SchemaValidationError> {
    let mut issues = Vec::new();
    if extractor.feature_types.is_empty() {
        issues.push(issue(
            "featureTypes",
            "must contain at least one feature type",
        ));
    }
    positive_integer(
        extractor.concatenated_dimensions,
        "concatenatedDimensions",
        &mut issues,
    );
    validate_dimensionality_reduction_config(
        &extractor.dim_reduction_config,
        "dimReductionConfig",
        &mut issues,
    );
    if let Some(values) = &extractor.normalization_mean {
        finite_vec(values, "normalizationMean", &mut issues);
    }
    if let Some(values) = &extractor.normalization_std {
        finite_vec(values, "normalizationStd", &mut issues);
    }
    if let Some(transformer) = &extractor.dim_reduction_transformer {
        validate_transformer(transformer, "dimReductionTransformer", &mut issues);
    }
    finish(issues)
}

/// Validates the complete serialized model envelope.
pub fn validate_serialized_model_schema(
    model: &SerializedModelSchema,
) -> Result<(), SchemaValidationError> {
    let mut issues = Vec::new();
    if let Err(error) = validate_serialized_feature_extractor_schema(&model.feature_extractor) {
        issues.extend(error.issues.into_iter().map(|mut item| {
            item.path = format!("featureExtractor.{}", item.path);
            item
        }));
    }
    validate_classifier_state(&model.classifier.state, "classifier.state", &mut issues);
    finite(model.trained_at, "trainedAt", &mut issues);
    finite(model.version, "version", &mut issues);
    finish(issues)
}

/// Validates the fields represented by `PostureFrameSchema`.
///
/// The TypeScript schema deliberately accepts any `Float32Array` length and
/// does not inspect Blob contents.  This function therefore avoids the stricter
/// feature-dimension and thumbnail MIME checks used by other domain validators.
pub fn validate_posture_frame_schema(frame: &PostureFrame) -> Result<(), SchemaValidationError> {
    let mut issues = Vec::new();
    if frame.id.is_empty() {
        issues.push(issue("id", "must contain at least one character"));
    }
    positive_number(frame.timestamp, "timestamp", &mut issues);
    if frame.keypoints.len() != COCO_KEYPOINT_COUNT {
        issues.push(issue(
            "keypoints",
            format!("must contain exactly {COCO_KEYPOINT_COUNT} keypoints"),
        ));
    }
    for (index, keypoint) in frame.keypoints.iter().enumerate() {
        finite(keypoint.x, &format!("keypoints.{index}.x"), &mut issues);
        finite(keypoint.y, &format!("keypoints.{index}.y"), &mut issues);
        finite(
            keypoint.score,
            &format!("keypoints.{index}.score"),
            &mut issues,
        );
    }
    for (field, value) in [
        ("x1", frame.bbox.x1),
        ("y1", frame.bbox.y1),
        ("x2", frame.bbox.x2),
        ("y2", frame.bbox.y2),
        ("score", frame.bbox.score),
        ("width", frame.bbox.width),
        ("height", frame.bbox.height),
    ] {
        finite(value, &format!("bbox.{field}"), &mut issues);
    }
    if !(0.0..=1.0).contains(&frame.bbox.score) {
        issues.push(issue("bbox.score", "must be between 0 and 1"));
    }
    if frame.bbox.width < 0.0 {
        issues.push(issue("bbox.width", "must be non-negative"));
    }
    if frame.bbox.height < 0.0 {
        issues.push(issue("bbox.height", "must be non-negative"));
    }
    finish(issues)
}

impl PostureFrameSchema {
    pub fn validate(frame: &PostureFrame) -> Result<(), SchemaValidationError> {
        validate_posture_frame_schema(frame)
    }
}
