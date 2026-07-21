//! Serialization and configuration types for the machine-learning pipeline.
//!
//! This is a mechanical Rust port of `src/services/ml/types.ts`. The source
//! uses JavaScript `number` values for serialized model data, so floating-point
//! values remain `f64`; integer-shaped settings use Rust integer types.

use serde::{Deserialize, Serialize};

use super::pca::SerializedPca;
use super::random_projection::RandomProjectionState;

// ============================================================================
// Normalization and dimensionality-reduction types
// ============================================================================

/// Normalization mode used for feature preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    None,
    Layer,
    ZScore,
    Calibrated,
}

/// Dimensionality-reduction method stored in a serialized feature extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimensionalityReductionMethod {
    #[serde(rename = "random_projection")]
    RandomProjection,
    #[serde(rename = "pls-da")]
    PlsDa,
    #[serde(rename = "linear_nca")]
    LinearNca,
    #[serde(rename = "pca")]
    Pca,
    #[serde(rename = "none")]
    None,
}

/// Dimensionality-reduction configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionalityReductionConfig {
    pub method: DimensionalityReductionMethod,
    pub components: usize,
}

/// A serialized fitted dimensionality-reduction transformer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum DimReductionTransformer {
    #[serde(rename = "random_projection")]
    RandomProjection(RandomProjectionState),
    #[serde(rename = "pca")]
    Pca(SerializedPca),
}

/// Raw dimensionality-reduction state used by the source-facing type guards.
/// Unlike [`DimReductionTransformer`], this representation has no `{type,data}`
/// wrapper on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RawDimReductionTransformerState {
    RandomProjection(RandomProjectionState),
    Pca(SerializedPca),
}

/// Optional raw transformer state, matching the TypeScript union's `null`.
pub type DimReductionTransformerState = Option<RawDimReductionTransformerState>;

/// Optional tagged transformer state stored by a serialized feature extractor.
pub type DimReductionTransformerWrapper = Option<DimReductionTransformer>;

// ============================================================================
// Serialized classifier states
// ============================================================================

/// Serialized state for an MLP classifier.
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

/// Serialized state for a K-nearest-neighbors classifier.
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

/// Kernel used by the K-nearest-neighbors classifier at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnKernel {
    Cosine,
    Rbf,
}

/// Serialized state for a support-vector machine classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedSvm {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub class_weights: [f64; 2],
}

/// One cluster and its optional class prototypes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KMeansPrototypeCluster {
    pub centroid: Vec<f64>,
    pub prototype_good: Option<Vec<f64>>,
    pub prototype_bad: Option<Vec<f64>>,
}

/// Serialized state for the K-means prototype classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansPrototype {
    pub clusters: Vec<KMeansPrototypeCluster>,
    pub global_prototype_good: Vec<f64>,
    pub global_prototype_bad: Vec<f64>,
    pub temperature: f64,
}

/// Serialized state for Gaussian Naive Bayes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedGaussianNb {
    pub class_means: [Vec<f64>; 2],
    pub class_variances: [Vec<f64>; 2],
    pub class_priors: [f64; 2],
    pub epsilon: f64,
}

/// Serialized state for K-means logistic classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedKMeansLogistic {
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<Option<SerializedMlp>>,
    pub global_model: SerializedMlp,
    pub temperature: f64,
}

/// Union of all serialized classifier states.
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

/// Classifier ID and its serialized state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedClassifier {
    pub classifier_id: String,
    pub state: SerializedClassifierState,
}

// ============================================================================
// Classifier parameters
// ============================================================================

/// Parameters for the MLP classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlpParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: MlpClassifierId,
    #[serde(rename = "weightDecay")]
    pub weight_decay: f64,
    #[serde(rename = "maxIterations")]
    pub max_iterations: usize,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
    pub use_class_weights: Option<bool>,
    #[serde(rename = "labelSmoothing", skip_serializing_if = "Option::is_none")]
    pub label_smoothing: Option<f64>,
    #[serde(rename = "hiddenLayers", skip_serializing_if = "Option::is_none")]
    pub hidden_layers: Option<usize>,
    #[serde(rename = "hiddenSize", skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,
}

/// The literal classifier ID used by MLP parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlpClassifierId {
    #[serde(rename = "mlp")]
    Mlp,
}

/// Parameters for the K-nearest-neighbors classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnnParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: KnnClassifierId,
    pub k: usize,
    pub distance: KnnDistance,
}

/// The literal classifier ID used by KNN parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnnClassifierId {
    #[serde(rename = "knn")]
    Knn,
}

/// Distance metric declared by the KNN parameter contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnDistance {
    Euclidean,
    Manhattan,
}

/// Parameters for the support-vector-machine classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SvmParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: SvmClassifierId,
    #[serde(rename = "C")]
    pub c: f64,
    #[serde(rename = "maxIterations")]
    pub max_iterations: usize,
    #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
    pub use_class_weights: Option<bool>,
}

/// The literal classifier ID used by SVM parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvmClassifierId {
    #[serde(rename = "svm")]
    Svm,
}

/// Parameters for the K-means prototype classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KMeansPrototypeParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: KMeansPrototypeClassifierId,
    pub temperature: f64,
}

/// The literal classifier ID used by K-means prototype parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KMeansPrototypeClassifierId {
    #[serde(rename = "kmeans_prototype")]
    KMeansPrototype,
}

/// Parameters for Gaussian Naive Bayes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianNbParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: GaussianNbClassifierId,
    #[serde(rename = "varianceSmoothing")]
    pub variance_smoothing: f64,
}

/// The literal classifier ID used by Gaussian-NB parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaussianNbClassifierId {
    #[serde(rename = "gaussian_nb")]
    GaussianNb,
}

/// Parameters for K-means logistic classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KMeansLogisticParams {
    #[serde(rename = "classifierId")]
    pub classifier_id: KMeansLogisticClassifierId,
    pub temperature: f64,
    #[serde(rename = "weightDecay")]
    pub weight_decay: f64,
    #[serde(rename = "maxIterations")]
    pub max_iterations: usize,
    #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
    pub use_class_weights: Option<bool>,
}

/// The literal classifier ID used by K-means logistic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KMeansLogisticClassifierId {
    #[serde(rename = "kmeans_logistic")]
    KMeansLogistic,
}

/// Discriminated union for all classifier parameter types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "classifierId")]
pub enum ClassifierParams {
    #[serde(rename = "mlp")]
    Mlp {
        #[serde(rename = "weightDecay")]
        weight_decay: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "learningRate")]
        learning_rate: f64,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
        #[serde(rename = "hiddenLayers", skip_serializing_if = "Option::is_none")]
        hidden_layers: Option<usize>,
        #[serde(rename = "hiddenSize", skip_serializing_if = "Option::is_none")]
        hidden_size: Option<usize>,
    },
    #[serde(rename = "knn")]
    Knn { k: usize, distance: KnnDistance },
    #[serde(rename = "svm")]
    Svm {
        #[serde(rename = "C")]
        c: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
    },
    #[serde(rename = "kmeans_prototype")]
    KMeansPrototype { temperature: f64 },
    #[serde(rename = "gaussian_nb")]
    GaussianNb {
        #[serde(rename = "varianceSmoothing")]
        variance_smoothing: f64,
    },
    #[serde(rename = "kmeans_logistic")]
    KMeansLogistic {
        temperature: f64,
        #[serde(rename = "weightDecay")]
        weight_decay: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
    },
}

// ============================================================================
// Serialized feature extractor and model
// ============================================================================

/// Serialized state for a fitted feature extractor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedFeatureExtractor {
    pub feature_types: Vec<String>,
    pub normalization_mode: NormalizationMode,
    pub dim_reduction_config: DimensionalityReductionConfig,
    pub concatenated_dimensions: usize,
    pub normalization_mean: Option<Vec<f64>>,
    pub normalization_std: Option<Vec<f64>>,
    pub dim_reduction_transformer: DimReductionTransformerWrapper,
}

/// Serialized state for a complete feature-extractor/classifier model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedModel {
    pub feature_extractor: SerializedFeatureExtractor,
    pub classifier: SerializedClassifier,
    pub trained_at: f64,
    pub version: f64,
}

// ============================================================================
// Type guards
// ============================================================================

pub fn is_mlp_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::Mlp { .. })
}

pub fn is_knn_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::Knn { .. })
}

pub fn is_svm_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::Svm { .. })
}

pub fn is_kmeans_prototype_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::KMeansPrototype { .. })
}

pub fn is_gaussian_nb_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::GaussianNb { .. })
}

pub fn is_kmeans_logistic_params(params: &ClassifierParams) -> bool {
    matches!(params, ClassifierParams::KMeansLogistic { .. })
}

pub fn is_serialized_mlp(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::Mlp(_))
}

pub fn is_serialized_knn(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::Knn(_))
}

pub fn is_serialized_svm(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::Svm(_))
}

pub fn is_serialized_kmeans_prototype(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::KMeansPrototype(_))
}

pub fn is_serialized_gaussian_nb(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::GaussianNb(_))
}

pub fn is_serialized_kmeans_logistic(classifier: &SerializedClassifierState) -> bool {
    matches!(classifier, SerializedClassifierState::KMeansLogistic(_))
}

pub fn is_serialized_random_projection(transformer: &DimReductionTransformerState) -> bool {
    matches!(
        transformer,
        Some(RawDimReductionTransformerState::RandomProjection(_))
    )
}

pub fn is_serialized_pca(transformer: &DimReductionTransformerState) -> bool {
    matches!(transformer, Some(RawDimReductionTransformerState::Pca(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn raw_and_wrapped_transformer_states_keep_distinct_wire_shapes() {
        let raw_random: DimReductionTransformerState = serde_json::from_value(json!({
            "projectionMatrix": [[1.0]],
            "nComponents": 1,
            "nFeatures": 1,
            "seed": 42.0
        }))
        .unwrap();
        assert!(is_serialized_random_projection(&raw_random));
        assert!(!is_serialized_pca(&raw_random));
        assert!(serde_json::to_value(&raw_random)
            .unwrap()
            .get("type")
            .is_none());

        let raw_pca: DimReductionTransformerState = serde_json::from_value(json!({
            "components": [[1.0]],
            "mean": [0.0],
            "nComponents": 1,
            "nFeatures": 1
        }))
        .unwrap();
        assert!(is_serialized_pca(&raw_pca));

        let wrapped: DimReductionTransformerWrapper = serde_json::from_value(json!({
            "type": "pca",
            "data": {
                "components": [[1.0]],
                "mean": [0.0],
                "nComponents": 1,
                "nFeatures": 1
            }
        }))
        .unwrap();
        let wrapped_json = serde_json::to_value(wrapped).unwrap();
        assert_eq!(wrapped_json["type"], "pca");
        assert!(wrapped_json.get("data").is_some());
    }
}
