//! Shared serialization helpers for classifier model state.
//!
//! This is the Rust port of
//! `src/services/ml/serialization/serializationHelpers.ts`.  The native model
//! envelope keeps classifier state, reduction state, and normalization state
//! in typed values; tensors and feature vectors remain Rust arrays and never
//! cross the native boundary as JSON arrays.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::ported::base_classifier::BaseClassifier;
use crate::ported::kmeans_logistic_classifier::KMeansLogisticClassifier;
use crate::ported::kmeans_prototype_classifier::KMeansPrototypeClassifier;
use crate::ported::knn_classifier::KnnClassifier;
use crate::ported::mlp_classifier::MlpClassifier;
use crate::ported::naive_bayes_classifier::GaussianNbClassifier;
use crate::ported::random_projection::{RandomProjectionError, RandomProjectionTransformer};
use crate::ported::svm_classifier::SvmClassifier;
use crate::ported::types::{
    ClassifierParams, DimReductionTransformer as TransformerState, DimReductionTransformerWrapper,
    NormalizationMode, SerializedClassifier, SerializedClassifierState, SerializedModel,
};

/// Errors raised while reconstructing or validating serialized model state.
#[derive(Debug, Clone, PartialEq)]
pub enum SerializationError {
    MissingTransformer,
    TransformerNotFitted,
    UnsupportedTransformer,
    Transformer(RandomProjectionError),
    MissingNormalizationParameters,
    InvalidNormalizationParameters(String),
    InvalidModel(String),
    MissingWeights { field: String },
    MissingBias,
    MissingClassWeights,
    InvalidKnnData,
    MissingClassifierParameters,
    ClassifierParameterMismatch { classifier_id: String },
    InvalidSerializedFields(String),
}

impl fmt::Display for SerializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTransformer => {
                formatter.write_str("Transformer reconstruction returned null")
            }
            Self::TransformerNotFitted => {
                formatter.write_str("Reconstructed transformer is not fitted")
            }
            Self::UnsupportedTransformer => {
                formatter.write_str("unsupported dimensionality reduction transformer")
            }
            Self::Transformer(error) => error.fmt(formatter),
            Self::MissingNormalizationParameters => formatter.write_str(
                "Cannot load model: Z-score normalization was used during training, but normalization parameters are missing. The model may be corrupted. Please retrain your model.",
            ),
            Self::InvalidNormalizationParameters(message) => {
                write!(formatter, "invalid normalization parameters: {message}")
            }
            Self::InvalidModel(message) => write!(formatter, "invalid serialized model: {message}"),
            Self::MissingWeights { field } => write!(
                formatter,
                "Cannot deserialize model: {field} array is empty or missing"
            ),
            Self::MissingBias => formatter.write_str(
                "Cannot load model: This model was trained with an older version that used \"intercept\" instead of \"bias\". Please retrain your model with the current version for proper L2 regularization support.",
            ),
            Self::MissingClassWeights => formatter.write_str(
                "Cannot load model: This model is missing class weights. Please retrain your model with the current version to include automatic class weight balancing.",
            ),
            Self::InvalidKnnData => formatter.write_str(
                "Cannot load model: Invalid KNN model format (missing trainingData or trainingLabels). The model may be corrupted. Please retrain your model.",
            ),
            Self::MissingClassifierParameters => formatter.write_str(
                "Cannot load model: Missing classifier parameters. This model was trained with an older version. Please retrain your model.",
            ),
            Self::ClassifierParameterMismatch { classifier_id } => write!(
                formatter,
                "classifier parameters do not match classifier ID {classifier_id}"
            ),
            Self::InvalidSerializedFields(message) => write!(
                formatter,
                "invalid serialized classifier fields: {message}"
            ),
        }
    }
}

impl Error for SerializationError {}

impl From<RandomProjectionError> for SerializationError {
    fn from(error: RandomProjectionError) -> Self {
        Self::Transformer(error)
    }
}

/// A transformer that can participate in the model envelope.
pub trait DimReductionTransformer {
    fn serialized_state(&self) -> Result<Option<TransformerState>, SerializationError>;
    fn is_fitted(&self) -> bool;
    fn n_components(&self) -> usize;
}

impl DimReductionTransformer for RandomProjectionTransformer {
    fn serialized_state(&self) -> Result<Option<TransformerState>, SerializationError> {
        Ok(Some(TransformerState::RandomProjection(self.to_json()?)))
    }

    fn is_fitted(&self) -> bool {
        RandomProjectionTransformer::is_fitted(self)
    }

    fn n_components(&self) -> usize {
        RandomProjectionTransformer::n_components(self)
    }
}

/// Serializes the supported dimensionality-reduction transformer.
pub fn serialize_dim_reduction_transformer(
    transformer: Option<&dyn DimReductionTransformer>,
) -> Result<DimReductionTransformerWrapper, SerializationError> {
    let Some(transformer) = transformer else {
        return Ok(None);
    };

    transformer.serialized_state()
}

/// Reconstructs the random-projection transformer from its typed state.
pub fn deserialize_dim_reduction_transformer(
    wrapper: DimReductionTransformerWrapper,
) -> Result<Option<RandomProjectionTransformer>, SerializationError> {
    let Some(wrapper) = wrapper else {
        return Ok(None);
    };

    let result = match wrapper {
        TransformerState::RandomProjection(state) => Some(
            RandomProjectionTransformer::from_json(state)
                .map_err(SerializationError::from)
                .map_err(reconstruction_error)?,
        ),
        TransformerState::Pca(_) => {
            return Err(reconstruction_error(SerializationError::MissingTransformer));
        }
    };

    let Some(transformer) = result else {
        return Err(reconstruction_error(SerializationError::MissingTransformer));
    };
    if !transformer.is_fitted() {
        return Err(reconstruction_error(
            SerializationError::TransformerNotFitted,
        ));
    }
    Ok(Some(transformer))
}

fn reconstruction_error(error: SerializationError) -> SerializationError {
    SerializationError::InvalidModel(format!(
        "Failed to reconstruct dimensionality reduction transformer: {error}. The model cannot be loaded. Please retrain your model with the current codebase version."
    ))
}

/// Alias retaining the source terminology for callers handling model records.
pub type DimensionalityTransformerWrapper = DimReductionTransformerWrapper;

/// Extracts the selected feature IDs in their stored order.
pub fn extract_feature_types_from_model(model: &SerializedModel) -> Vec<String> {
    model.feature_extractor.feature_types.clone()
}

/// Extracts the stored normalization mode from the model envelope.
pub fn extract_normalization_mode(model: &SerializedModel) -> NormalizationMode {
    model.feature_extractor.normalization_mode
}

/// Normalization and reduction metadata restored during model loading.
#[derive(Debug, Clone, PartialEq)]
pub struct RestoredModelMetadata {
    pub n_features: usize,
    pub dim_reduction_transformer: Option<RandomProjectionTransformer>,
    pub normalization_mean: Option<Vec<f64>>,
    pub normalization_std: Option<Vec<f64>>,
}

/// Restores and validates metadata from the current serialized model envelope.
pub fn restore_model_metadata(
    model: &SerializedModel,
) -> Result<RestoredModelMetadata, SerializationError> {
    let extractor = &model.feature_extractor;
    let n_features = extractor.concatenated_dimensions;

    let (normalization_mean, normalization_std) = match extractor.normalization_mode {
        NormalizationMode::ZScore | NormalizationMode::Calibrated => {
            let (Some(mean), Some(std)) = (
                extractor.normalization_mean.clone(),
                extractor.normalization_std.clone(),
            ) else {
                return Err(SerializationError::MissingNormalizationParameters);
            };
            validate_normalization_parameters(&mean, &std, n_features)?;
            (Some(mean), Some(std))
        }
        NormalizationMode::None | NormalizationMode::Layer => {
            if extractor.normalization_mean.is_some() || extractor.normalization_std.is_some() {
                return Err(SerializationError::InvalidNormalizationParameters(
                    "none/layer normalization must not contain z-score parameters".into(),
                ));
            }
            (None, None)
        }
    };

    let transformer =
        deserialize_dim_reduction_transformer(extractor.dim_reduction_transformer.clone())?;

    if let Some(transformer) = &transformer {
        if transformer.n_features != n_features {
            return Err(SerializationError::InvalidModel(format!(
                "random projection expects {} input features, got {n_features}",
                transformer.n_features
            )));
        }
        if transformer.n_components() != extractor.dim_reduction_config.components {
            return Err(SerializationError::InvalidModel(format!(
                "random projection has {} components, expected {}",
                transformer.n_components(),
                extractor.dim_reduction_config.components
            )));
        }
    }

    Ok(RestoredModelMetadata {
        n_features,
        dim_reduction_transformer: transformer,
        normalization_mean,
        normalization_std,
    })
}

fn validate_normalization_parameters(
    mean: &[f64],
    std: &[f64],
    n_features: usize,
) -> Result<(), SerializationError> {
    if mean.len() != n_features || std.len() != n_features {
        return Err(SerializationError::InvalidNormalizationParameters(
            "mean and std dimensions do not match the feature dimension".into(),
        ));
    }
    if mean.iter().any(|value| !value.is_finite()) {
        return Err(SerializationError::InvalidNormalizationParameters(
            "means must be finite".into(),
        ));
    }
    if std.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(SerializationError::InvalidNormalizationParameters(
            "standard deviations must be finite and positive".into(),
        ));
    }
    Ok(())
}

/// Creates the classifier-only serialized state used by classifier loaders.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedClassifierModel {
    pub classifier: SerializedClassifierState,
    pub dim_reduction_transformer: DimReductionTransformerWrapper,
    pub params: ClassifierParams,
    pub n_features: usize,
}

pub fn create_serialized_model(
    classifier: SerializedClassifierState,
    dim_reduction_transformer: DimReductionTransformerWrapper,
    params: ClassifierParams,
    n_features: usize,
) -> SerializedClassifierModel {
    SerializedClassifierModel {
        classifier,
        dim_reduction_transformer,
        params,
        n_features,
    }
}

/// Validates that a weight vector is present and non-empty.
pub fn validate_weights_array<T>(
    weights: Option<&[T]>,
    field_name: &str,
) -> Result<(), SerializationError> {
    if weights.is_none_or(|values| values.is_empty()) {
        return Err(SerializationError::MissingWeights {
            field: field_name.to_owned(),
        });
    }
    Ok(())
}

/// Validates the current bias field and rejects the removed `intercept` form.
pub fn validate_bias_field(bias: Option<f64>) -> Result<(), SerializationError> {
    match bias {
        Some(value) if value.is_finite() => Ok(()),
        _ => Err(SerializationError::MissingBias),
    }
}

/// Validates balanced weights for the two-class classifiers.
pub fn validate_class_weights(class_weights: Option<&[f64]>) -> Result<(), SerializationError> {
    let Some(class_weights) = class_weights else {
        return Err(SerializationError::MissingClassWeights);
    };
    if class_weights.len() != 2
        || class_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(SerializationError::MissingClassWeights);
    }
    Ok(())
}

/// Validates the required KNN training arrays.
pub fn validate_knn_data(
    training_data: Option<&[Vec<f64>]>,
    training_labels: Option<&[f64]>,
) -> Result<(), SerializationError> {
    let (Some(training_data), Some(training_labels)) = (training_data, training_labels) else {
        return Err(SerializationError::InvalidKnnData);
    };
    if training_data.is_empty()
        || training_data.len() != training_labels.len()
        || training_data
            .iter()
            .any(|row| row.is_empty() || row.iter().any(|value| !value.is_finite()))
        || training_labels
            .iter()
            .any(|label| !label.is_finite() || (*label != 0.0 && *label != 1.0))
    {
        return Err(SerializationError::InvalidKnnData);
    }
    let dimensions = training_data[0].len();
    if training_data.iter().any(|row| row.len() != dimensions) {
        return Err(SerializationError::InvalidKnnData);
    }
    Ok(())
}

/// Ensures the classifier parameters identify the requested classifier.
pub fn extract_and_validate_params<'a>(
    params: Option<&'a ClassifierParams>,
    classifier_id: &str,
) -> Result<&'a ClassifierParams, SerializationError> {
    let Some(params) = params else {
        return Err(SerializationError::MissingClassifierParameters);
    };

    let matches = matches!(
        (classifier_id, params),
        ("mlp" | "MLPClassifier", ClassifierParams::Mlp { .. })
            | ("knn" | "KNNClassifier", ClassifierParams::Knn { .. })
            | ("svm" | "SVMClassifier", ClassifierParams::Svm { .. })
            | (
                "kmeans_prototype"
                    | "prototypical"
                    | "PrototypicalClassifier"
                    | "KMeansPrototypeClassifier",
                ClassifierParams::KMeansPrototype { .. },
            )
            | (
                "gaussian_nb" | "GaussianNBClassifier",
                ClassifierParams::GaussianNb { .. }
            )
            | (
                "kmeans_logistic" | "KMeansLogisticClassifier",
                ClassifierParams::KMeansLogistic { .. },
            )
    );

    if matches {
        Ok(params)
    } else {
        Err(SerializationError::ClassifierParameterMismatch {
            classifier_id: classifier_id.to_owned(),
        })
    }
}

/// Reconstructs and validates a complete classifier envelope. Returning the
/// canonical state keeps the source helper's concrete reconstruction boundary
/// while preserving the public serialized-envelope API used by native callers.
pub fn deserialize_classifier(
    model: &SerializedClassifierModel,
    classifier_id: &str,
) -> Result<SerializedClassifier, SerializationError> {
    extract_and_validate_params(Some(&model.params), classifier_id)?;
    let transformer =
        deserialize_dim_reduction_transformer(model.dim_reduction_transformer.clone())?;
    if model.n_features == 0 {
        return Err(SerializationError::InvalidModel(
            "nFeatures must be positive".into(),
        ));
    }
    if !state_matches_classifier(classifier_id, &model.classifier) {
        return Err(SerializationError::InvalidSerializedFields(format!(
            "classifier ID {classifier_id} does not match serialized state variant"
        )));
    }
    validate_serialized_classifier_state(&model.classifier)?;

    let expected_dimensions = transformer
        .as_ref()
        .map_or(model.n_features, RandomProjectionTransformer::n_components);
    let actual_dimensions = classifier_input_dimensions(&model.classifier)?;
    if actual_dimensions != expected_dimensions {
        return Err(SerializationError::InvalidSerializedFields(format!(
            "classifier input dimension {actual_dimensions} does not match expected dimension {expected_dimensions}"
        )));
    }

    let state = reconstruct_classifier_state(&model.classifier)?;
    Ok(SerializedClassifier {
        classifier_id: normalized_classifier_id(classifier_id).to_owned(),
        state,
    })
}

fn normalized_classifier_id(classifier_id: &str) -> &str {
    match classifier_id {
        "MLPClassifier" => "mlp",
        "KNNClassifier" => "knn",
        "SVMClassifier" => "svm",
        "prototypical" | "PrototypicalClassifier" | "KMeansPrototypeClassifier" => {
            "kmeans_prototype"
        }
        "GaussianNBClassifier" => "gaussian_nb",
        "KMeansLogisticClassifier" => "kmeans_logistic",
        other => other,
    }
}

fn state_matches_classifier(classifier_id: &str, state: &SerializedClassifierState) -> bool {
    matches!(
        (normalized_classifier_id(classifier_id), state),
        ("mlp", SerializedClassifierState::Mlp(_))
            | ("knn", SerializedClassifierState::Knn(_))
            | ("svm", SerializedClassifierState::Svm(_))
            | (
                "kmeans_prototype",
                SerializedClassifierState::KMeansPrototype(_)
            )
            | ("gaussian_nb", SerializedClassifierState::GaussianNb(_))
            | (
                "kmeans_logistic",
                SerializedClassifierState::KMeansLogistic(_)
            )
    )
}

fn classifier_input_dimensions(
    state: &SerializedClassifierState,
) -> Result<usize, SerializationError> {
    let dimensions = match state {
        SerializedClassifierState::Mlp(model) => model.layer_shapes.first().copied(),
        SerializedClassifierState::Knn(model) => model.training_data.first().map(Vec::len),
        SerializedClassifierState::Svm(model) => Some(model.weights.len()),
        SerializedClassifierState::KMeansPrototype(model) => {
            Some(model.global_prototype_good.len())
        }
        SerializedClassifierState::GaussianNb(model) => Some(model.class_means[0].len()),
        SerializedClassifierState::KMeansLogistic(model) => model.centroids.first().map(Vec::len),
    };
    dimensions.filter(|value| *value > 0).ok_or_else(|| {
        SerializationError::InvalidSerializedFields(
            "classifier input dimension must be positive".into(),
        )
    })
}

fn reconstruct_classifier_state(
    state: &SerializedClassifierState,
) -> Result<SerializedClassifierState, SerializationError> {
    let result = match state.clone() {
        SerializedClassifierState::Mlp(model) => MlpClassifier::from_state(model)
            .map_err(classifier_error)?
            .to_json(),
        SerializedClassifierState::Knn(model) => KnnClassifier::from_state(model)
            .map_err(classifier_error)?
            .to_json(),
        SerializedClassifierState::Svm(model) => SvmClassifier::from_state(model)
            .map_err(classifier_error)?
            .to_json(),
        SerializedClassifierState::KMeansPrototype(model) => {
            KMeansPrototypeClassifier::from_state(model)
                .map_err(classifier_error)?
                .to_json()
        }
        SerializedClassifierState::GaussianNb(model) => GaussianNbClassifier::from_state(model)
            .map_err(classifier_error)?
            .to_json(),
        SerializedClassifierState::KMeansLogistic(model) => {
            KMeansLogisticClassifier::from_state(model)
                .map_err(classifier_error)?
                .to_json()
        }
    };
    result.map_err(classifier_error)
}

fn classifier_error(error: impl fmt::Display) -> SerializationError {
    SerializationError::InvalidSerializedFields(error.to_string())
}

fn validate_serialized_classifier_state(
    state: &SerializedClassifierState,
) -> Result<(), SerializationError> {
    match state {
        SerializedClassifierState::Mlp(model) => {
            if model.layer_weights.is_empty()
                || model.layer_biases.is_empty()
                || model.layer_weights.len() != model.layer_biases.len()
                || model
                    .class_weights
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(SerializationError::InvalidSerializedFields(
                    "invalid MLP weights or class weights".into(),
                ));
            }
        }
        SerializedClassifierState::Knn(model) => {
            validate_knn_data(Some(&model.training_data), Some(&model.training_labels))?;
            if model.k == 0 || model.k > model.training_data.len() {
                return Err(SerializationError::InvalidSerializedFields(
                    "KNN k must be within the training-data size".into(),
                ));
            }
        }
        SerializedClassifierState::Svm(model) => {
            validate_weights_array(Some(&model.weights), "weights")?;
            validate_bias_field(Some(model.bias))?;
            validate_class_weights(Some(&model.class_weights))?;
        }
        SerializedClassifierState::KMeansPrototype(model) => {
            if model.clusters.is_empty()
                || model.global_prototype_good.is_empty()
                || model.global_prototype_good.len() != model.global_prototype_bad.len()
                || !model.temperature.is_finite()
                || model.temperature <= 0.0
            {
                return Err(SerializationError::InvalidSerializedFields(
                    "invalid K-Means Prototype state".into(),
                ));
            }
        }
        SerializedClassifierState::GaussianNb(model) => {
            if model.class_means.iter().any(|values| values.is_empty())
                || model.class_means[0].len() != model.class_means[1].len()
                || model.class_variances[0].len() != model.class_means[0].len()
                || model.class_variances[1].len() != model.class_means[0].len()
                || model
                    .class_priors
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                || !model.epsilon.is_finite()
                || model.epsilon <= 0.0
            {
                return Err(SerializationError::InvalidSerializedFields(
                    "invalid Gaussian Naive Bayes state".into(),
                ));
            }
        }
        SerializedClassifierState::KMeansLogistic(model) => {
            if model.centroids.is_empty()
                || model.cluster_models.len() != model.centroids.len()
                || !model.temperature.is_finite()
                || model.temperature <= 0.0
            {
                return Err(SerializationError::InvalidSerializedFields(
                    "invalid K-Means Logistic state".into(),
                ));
            }
        }
    }
    Ok(())
}
