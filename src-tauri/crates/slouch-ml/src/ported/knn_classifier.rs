//! K-nearest-neighbors classifier.
//!
//! This is a mechanical Rust port of `src/services/ml/knnClassifier.ts`.
//! Features are stored as `f32`, while serialized values and probability
//! calculations use `f64` at the native boundary.

use std::cmp::Ordering;

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::layer_norm::l2_normalize_single;
use super::types::{KnnKernel, SerializedClassifierState, SerializedKnn};

const DEFAULT_K: usize = 5;
const DEFAULT_GAMMA: f64 = 1.0;
const WEIGHT_EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnnConfig {
    pub k: usize,
    pub kernel: KnnKernel,
    pub gamma: f64,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            kernel: KnnKernel::Cosine,
            gamma: DEFAULT_GAMMA,
        }
    }
}

#[derive(Debug, Clone)]
struct TrainedModel {
    training_data: Vec<Vec<f32>>,
    training_labels: Vec<i32>,
    k: usize,
    kernel: KnnKernel,
    gamma: f64,
}

#[derive(Debug, Clone, Default)]
pub struct KnnClassifier {
    model: Option<TrainedModel>,
    params: KnnConfig,
}

impl KnnClassifier {
    pub fn new(k: usize, kernel: KnnKernel, gamma: f64) -> Result<Self, ClassifierError> {
        Self::with_config(KnnConfig { k, kernel, gamma })
    }

    pub fn with_config(config: KnnConfig) -> Result<Self, ClassifierError> {
        validate_config(config)?;
        Ok(Self {
            model: None,
            params: config,
        })
    }

    pub fn from_json(data: SerializedKnn, params: KnnConfig) -> Result<Self, ClassifierError> {
        let classifier = Self::with_config(params)?;
        let training_data = normalize_serialized_data(&data)?;
        let training_labels = validate_training_state(&training_data, &data.training_labels)?;

        let k = if data.k == 0 { params.k } else { data.k };
        let kernel = data.kernel.unwrap_or(params.kernel);
        let gamma = data.gamma.unwrap_or(params.gamma);
        validate_config(KnnConfig { k, kernel, gamma })?;

        Ok(Self {
            model: Some(TrainedModel {
                training_data,
                training_labels,
                k,
                kernel,
                gamma,
            }),
            params: classifier.params,
        })
    }

    pub fn from_state(data: SerializedKnn) -> Result<Self, ClassifierError> {
        let params = KnnConfig {
            // The registry/factory loader supplies k=3 for falsy legacy state;
            // keep the standalone constructor default at 5.
            k: if data.k == 0 { 3 } else { data.k },
            kernel: data.kernel.unwrap_or(KnnKernel::Cosine),
            gamma: data.gamma.unwrap_or(DEFAULT_GAMMA),
        };
        Self::from_json(data, params)
    }

    fn validate_features(features: &[Vec<f32>]) -> Result<usize, ClassifierError> {
        let Some(first) = features.first() else {
            return Err(ClassifierError::EmptyDataset);
        };
        if first.is_empty() {
            return Err(ClassifierError::EmptyFeatureVector);
        }

        let dimensions = first.len();
        for (row, feature) in features.iter().enumerate() {
            if feature.len() != dimensions {
                return Err(ClassifierError::RaggedFeatures {
                    row,
                    expected: dimensions,
                    actual: feature.len(),
                });
            }
            if let Some(column) = feature.iter().position(|value| !value.is_finite()) {
                return Err(ClassifierError::NonFiniteFeature { row, column });
            }
        }
        Ok(dimensions)
    }

    fn similarity(query: &[f32], training: &[f32], kernel: KnnKernel, gamma: f64) -> f64 {
        match kernel {
            KnnKernel::Cosine => {
                let dot = query
                    .iter()
                    .zip(training)
                    .fold(0.0_f32, |sum, (left, right)| sum + left * right);
                f64::from(dot)
            }
            KnnKernel::Rbf => {
                let squared_distance =
                    query
                        .iter()
                        .zip(training)
                        .fold(0.0_f32, |sum, (left, right)| {
                            let difference = left - right;
                            sum + difference * difference
                        });
                let gamma_f32 = gamma as f32;
                f64::from((-gamma_f32 * squared_distance).exp())
            }
        }
    }
}

impl BaseClassifier for KnnClassifier {
    fn classifier_id(&self) -> &'static str {
        "knn"
    }

    fn train(&mut self, features: &[Vec<f32>], labels: &[i32]) -> Result<(), ClassifierError> {
        if features.is_empty() {
            return Err(ClassifierError::EmptyDataset);
        }
        if features.len() != labels.len() {
            return Err(ClassifierError::LengthMismatch {
                features: features.len(),
                labels: labels.len(),
            });
        }
        Self::validate_features(features)?;

        let training_data = features
            .iter()
            .map(|feature| {
                l2_normalize_single(feature)
                    .map_err(|error| ClassifierError::InvalidState(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let training_labels = labels.to_vec();

        self.model = Some(TrainedModel {
            training_data,
            training_labels,
            k: self.params.k,
            kernel: self.params.kernel,
            gamma: self.params.gamma,
        });
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        let Some(first) = model.training_data.first() else {
            return Err(ClassifierError::InvalidState(
                "trained model contains no samples".into(),
            ));
        };
        if features.len() != first.len() {
            return Err(ClassifierError::PredictionDimension {
                expected: first.len(),
                actual: features.len(),
            });
        }
        if let Some(column) = features.iter().position(|value| !value.is_finite()) {
            return Err(ClassifierError::NonFiniteFeature { row: 0, column });
        }

        let normalized_features = l2_normalize_single(features)
            .map_err(|error| ClassifierError::InvalidState(error.to_string()))?;
        let mut similarities = model
            .training_data
            .iter()
            .enumerate()
            .map(|(index, training)| {
                (
                    index,
                    Self::similarity(
                        &normalized_features,
                        training,
                        self.params.kernel,
                        self.params.gamma,
                    ),
                )
            })
            .collect::<Vec<_>>();
        similarities.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

        // The source recreates the tensor from serialized data but continues
        // to use constructor parameters for prediction.
        let effective_k = self.params.k.min(similarities.len());
        if effective_k == 0 {
            return Err(ClassifierError::InvalidState(
                "k must be positive for a trained model".into(),
            ));
        }

        let mut weighted_good = 0.0_f64;
        let mut sum_weights = 0.0_f64;
        for (index, similarity) in similarities.into_iter().take(effective_k) {
            let weight = similarity.max(0.0) + WEIGHT_EPSILON;
            sum_weights += weight;
            if model.training_labels[index] == 0 {
                weighted_good += weight;
            }
        }

        Ok(weighted_good / sum_weights)
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        Ok(SerializedClassifierState::Knn(SerializedKnn {
            training_data: model
                .training_data
                .iter()
                .map(|feature| feature.iter().map(|value| f64::from(*value)).collect())
                .collect(),
            training_labels: model
                .training_labels
                .iter()
                .map(|label| f64::from(*label))
                .collect(),
            k: model.k,
            kernel: Some(model.kernel),
            gamma: Some(model.gamma),
        }))
    }

    fn dispose(&mut self) {
        self.model = None;
    }
}

fn validate_config(config: KnnConfig) -> Result<(), ClassifierError> {
    if config.k == 0 {
        return Err(ClassifierError::InvalidState("k must be positive".into()));
    }
    if !config.gamma.is_finite() {
        return Err(ClassifierError::InvalidState("gamma must be finite".into()));
    }
    if matches!(config.kernel, KnnKernel::Rbf) && config.gamma <= 0.0 {
        return Err(ClassifierError::InvalidState(
            "RBF gamma must be positive".into(),
        ));
    }
    Ok(())
}

fn normalize_serialized_data(data: &SerializedKnn) -> Result<Vec<Vec<f32>>, ClassifierError> {
    if data.training_data.is_empty() {
        return Err(ClassifierError::EmptyDataset);
    }

    let mut converted = Vec::with_capacity(data.training_data.len());
    for (row, feature) in data.training_data.iter().enumerate() {
        if feature.is_empty() {
            return Err(ClassifierError::EmptyFeatureVector);
        }
        let mut converted_feature = Vec::with_capacity(feature.len());
        for (column, value) in feature.iter().enumerate() {
            if !value.is_finite() {
                return Err(ClassifierError::NonFiniteFeature { row, column });
            }
            let narrowed = *value as f32;
            if !narrowed.is_finite() {
                return Err(ClassifierError::NonFiniteFeature { row, column });
            }
            converted_feature.push(narrowed);
        }
        converted.push(converted_feature);
    }

    let dimensions = converted[0].len();
    for (row, feature) in converted.iter().enumerate() {
        if feature.len() != dimensions {
            return Err(ClassifierError::RaggedFeatures {
                row,
                expected: dimensions,
                actual: feature.len(),
            });
        }
    }

    converted
        .iter()
        .map(|feature| {
            l2_normalize_single(feature)
                .map_err(|error| ClassifierError::InvalidState(error.to_string()))
        })
        .collect()
}

fn validate_training_state(
    training_data: &[Vec<f32>],
    training_labels: &[f64],
) -> Result<Vec<i32>, ClassifierError> {
    if training_data.len() != training_labels.len() {
        return Err(ClassifierError::LengthMismatch {
            features: training_data.len(),
            labels: training_labels.len(),
        });
    }
    let mut labels = Vec::with_capacity(training_labels.len());
    for (index, label) in training_labels.iter().copied().enumerate() {
        if label != 0.0 && label != 1.0 {
            return Err(ClassifierError::InvalidState(format!(
                "training label at index {index} must be 0 or 1"
            )));
        }
        labels.push(label as i32);
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_model_predicts_with_constructor_configuration() {
        let state = SerializedKnn {
            training_data: vec![vec![1.0, 0.0], vec![0.8, 0.6]],
            training_labels: vec![0.0, 1.0],
            k: 1,
            kernel: Some(KnnKernel::Rbf),
            gamma: Some(100.0),
        };
        let classifier = KnnClassifier::from_json(
            state,
            KnnConfig {
                k: 2,
                kernel: KnnKernel::Cosine,
                gamma: 1.0,
            },
        )
        .unwrap();

        let probability = classifier.predict_proba(&[1.0, 0.0]).unwrap();
        assert!(probability > 0.5 && probability < 0.7);
        let SerializedClassifierState::Knn(serialized) = classifier.to_json().unwrap() else {
            panic!("expected KNN state");
        };
        assert_eq!(serialized.k, 1);
        assert_eq!(serialized.kernel, Some(KnnKernel::Rbf));
    }

    #[test]
    fn serialized_f32_overflow_and_non_binary_labels_are_rejected() {
        let config = KnnConfig::default();
        let overflow = SerializedKnn {
            training_data: vec![vec![f64::MAX]],
            training_labels: vec![0.0],
            k: 1,
            kernel: None,
            gamma: None,
        };
        assert!(KnnClassifier::from_json(overflow, config).is_err());

        let invalid_label = SerializedKnn {
            training_data: vec![vec![1.0]],
            training_labels: vec![2.0],
            k: 1,
            kernel: None,
            gamma: None,
        };
        assert!(KnnClassifier::from_json(invalid_label, config).is_err());
    }
}
