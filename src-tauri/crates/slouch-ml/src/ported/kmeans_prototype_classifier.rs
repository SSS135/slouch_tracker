//! K-means prototype classifier.
//!
//! This is a mechanical Rust port of
//! `src/services/ml/kmeansPrototypeClassifier.ts`. Features and prototypes
//! use `f32`; distances, softmax values, probabilities, and serialized values
//! use `f64`.

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::kmeans::{kmeans_default, silhouette_score, KMeansError};
use super::types::{KMeansPrototypeCluster, SerializedClassifierState, SerializedKMeansPrototype};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansPrototypeConfig {
    pub temperature: f64,
    pub n_clusters: usize,
}

impl Default for KMeansPrototypeConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            n_clusters: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ClusterData {
    centroid: Vec<f32>,
    prototype_good: Option<Vec<f32>>,
    prototype_bad: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct TrainedModel {
    clusters: Vec<ClusterData>,
    global_prototype_good: Vec<f32>,
    global_prototype_bad: Vec<f32>,
    temperature: f64,
}

#[derive(Debug, Clone)]
pub struct KMeansPrototypeClassifier {
    model: Option<TrainedModel>,
    params: KMeansPrototypeConfig,
}

impl KMeansPrototypeClassifier {
    pub fn new(temperature: f64, n_clusters: usize) -> Result<Self, ClassifierError> {
        let config = KMeansPrototypeConfig {
            temperature,
            n_clusters,
        };
        Self::with_config(config)
    }

    pub fn with_config(config: KMeansPrototypeConfig) -> Result<Self, ClassifierError> {
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(ClassifierError::InvalidState(
                "temperature must be finite and positive".into(),
            ));
        }
        Ok(Self {
            model: None,
            params: config,
        })
    }

    pub fn from_json(
        data: SerializedKMeansPrototype,
        temperature: f64,
    ) -> Result<Self, ClassifierError> {
        Self::from_json_with_clusters(data, temperature, 0)
    }

    pub fn from_json_with_clusters(
        data: SerializedKMeansPrototype,
        temperature: f64,
        n_clusters: usize,
    ) -> Result<Self, ClassifierError> {
        let classifier = Self::with_config(KMeansPrototypeConfig {
            temperature,
            n_clusters,
        })?;
        let model = Self::model_from_state(data)?;
        Ok(Self {
            model: Some(model),
            params: classifier.params,
        })
    }

    pub fn from_state(data: SerializedKMeansPrototype) -> Result<Self, ClassifierError> {
        let temperature = data.temperature;
        Self::from_json(data, temperature)
    }

    fn model_from_state(data: SerializedKMeansPrototype) -> Result<TrainedModel, ClassifierError> {
        if !data.temperature.is_finite() || data.temperature <= 0.0 {
            return Err(ClassifierError::InvalidState(
                "temperature must be finite and positive".into(),
            ));
        }
        let dimensions = data.global_prototype_good.len();
        if dimensions == 0 || data.global_prototype_bad.len() != dimensions {
            return Err(ClassifierError::InvalidState(
                "global prototypes must have equal nonzero dimensions".into(),
            ));
        }
        if !data
            .global_prototype_good
            .iter()
            .chain(data.global_prototype_bad.iter())
            .all(|value| value.is_finite())
        {
            return Err(ClassifierError::InvalidState(
                "global prototypes must contain finite values".into(),
            ));
        }

        let mut clusters = Vec::with_capacity(data.clusters.len());
        for cluster in data.clusters {
            if cluster.centroid.len() != dimensions {
                return Err(ClassifierError::InvalidState(
                    "cluster centroids must match prototype dimensions".into(),
                ));
            }
            if !cluster.centroid.iter().all(|value| value.is_finite()) {
                return Err(ClassifierError::InvalidState(
                    "cluster centroids must contain finite values".into(),
                ));
            }
            if cluster.prototype_good.as_ref().is_some_and(|prototype| {
                prototype.len() != dimensions || !prototype.iter().all(|value| value.is_finite())
            }) || cluster.prototype_bad.as_ref().is_some_and(|prototype| {
                prototype.len() != dimensions || !prototype.iter().all(|value| value.is_finite())
            }) {
                return Err(ClassifierError::InvalidState(
                    "cluster prototypes must match centroid dimensions and be finite".into(),
                ));
            }
            clusters.push(ClusterData {
                centroid: Self::narrow_vector(cluster.centroid, "cluster centroid")?,
                prototype_good: cluster
                    .prototype_good
                    .map(|prototype| Self::narrow_vector(prototype, "good cluster prototype"))
                    .transpose()?,
                prototype_bad: cluster
                    .prototype_bad
                    .map(|prototype| Self::narrow_vector(prototype, "bad cluster prototype"))
                    .transpose()?,
            });
        }
        if clusters.is_empty() {
            return Err(ClassifierError::InvalidState(
                "serialized model must contain at least one cluster".into(),
            ));
        }

        Ok(TrainedModel {
            clusters,
            global_prototype_good: Self::narrow_vector(
                data.global_prototype_good,
                "global good prototype",
            )?,
            global_prototype_bad: Self::narrow_vector(
                data.global_prototype_bad,
                "global bad prototype",
            )?,
            temperature: data.temperature,
        })
    }

    fn narrow_vector(values: Vec<f64>, field: &'static str) -> Result<Vec<f32>, ClassifierError> {
        values
            .into_iter()
            .map(|value| {
                let narrowed = value as f32;
                if narrowed.is_finite() {
                    Ok(narrowed)
                } else {
                    Err(ClassifierError::InvalidState(format!(
                        "{field} contains a value outside the finite f32 range"
                    )))
                }
            })
            .collect()
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

    fn map_kmeans_error(error: KMeansError) -> ClassifierError {
        ClassifierError::InvalidState(error.to_string())
    }

    fn compute_mean_for_class(
        features: &[Vec<f32>],
        labels: &[i32],
        target_label: i32,
        dimensions: usize,
    ) -> Option<Vec<f32>> {
        let mut sum = vec![0.0_f32; dimensions];
        let mut count = 0_usize;
        for (feature, label) in features.iter().zip(labels) {
            if *label == target_label {
                for (total, value) in sum.iter_mut().zip(feature) {
                    *total += *value;
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        for value in &mut sum {
            *value /= count as f32;
        }
        Some(sum)
    }

    fn compute_mean_for_subset(
        features: &[Vec<f32>],
        labels: &[i32],
        target_label: i32,
        dimensions: usize,
    ) -> Option<Vec<f32>> {
        Self::compute_mean_for_class(features, labels, target_label, dimensions)
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum::<f64>()
            .sqrt()
    }

    fn distance_softmax(distances: &[f64], temperature: f64) -> Result<Vec<f64>, ClassifierError> {
        if distances.is_empty() || distances.iter().any(|distance| !distance.is_finite()) {
            return Err(ClassifierError::InvalidState(
                "prototype distances must be finite and nonempty".into(),
            ));
        }
        let minimum = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let exponentials = distances
            .iter()
            .map(|distance| (-((*distance - minimum) / temperature)).exp())
            .collect::<Vec<_>>();
        let total = exponentials.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(ClassifierError::InvalidState(
                "prototype softmax produced a non-finite normalization".into(),
            ));
        }
        let probabilities = exponentials
            .into_iter()
            .map(|value| value / total)
            .collect::<Vec<_>>();
        if probabilities.iter().any(|value| !value.is_finite()) {
            return Err(ClassifierError::InvalidState(
                "prototype softmax produced a non-finite probability".into(),
            ));
        }
        Ok(probabilities)
    }
}

impl BaseClassifier for KMeansPrototypeClassifier {
    fn classifier_id(&self) -> &'static str {
        "kmeans_prototype"
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
        let dimensions = Self::validate_features(features)?;

        let (best_k, best_result) = if self.params.n_clusters > 0 {
            let best_k = self.params.n_clusters.min(features.len());
            let result = kmeans_default(features, best_k).map_err(Self::map_kmeans_error)?;
            (best_k, result)
        } else {
            let candidate_ks = [1_usize, 2, 3, 5, 7]
                .into_iter()
                .filter(|candidate| *candidate <= features.len())
                .collect::<Vec<_>>();
            let mut best_k = 1_usize;
            let mut best_score = f64::NEG_INFINITY;
            let mut temporary_result = None;

            for candidate in candidate_ks {
                let result = kmeans_default(features, candidate).map_err(Self::map_kmeans_error)?;
                if candidate == 1 {
                    if temporary_result.is_none() {
                        temporary_result = Some(result);
                        best_k = candidate;
                    }
                    continue;
                }
                let score = silhouette_score(features, &result.assignments, &result.centroids)
                    .map_err(Self::map_kmeans_error)?;
                if score > best_score {
                    best_score = score;
                    best_k = candidate;
                    temporary_result = Some(result);
                }
            }
            let result = temporary_result.ok_or_else(|| {
                ClassifierError::InvalidState("K-means failed to converge".into())
            })?;
            (best_k, result)
        };

        let global_prototype_good = Self::compute_mean_for_class(features, labels, 0, dimensions);
        let global_prototype_bad = Self::compute_mean_for_class(features, labels, 1, dimensions);
        let (Some(global_prototype_good), Some(global_prototype_bad)) =
            (global_prototype_good, global_prototype_bad)
        else {
            return Err(ClassifierError::MissingClass);
        };

        let mut clusters = Vec::with_capacity(best_k);
        for cluster_index in 0..best_k {
            let mut cluster_features = Vec::new();
            let mut cluster_labels = Vec::new();
            for (index, assignment) in best_result.assignments.iter().enumerate() {
                if *assignment == cluster_index {
                    cluster_features.push(features[index].clone());
                    cluster_labels.push(labels[index]);
                }
            }
            let centroid = best_result.centroids[cluster_index].clone();
            let prototype_good =
                Self::compute_mean_for_subset(&cluster_features, &cluster_labels, 0, dimensions);
            let prototype_bad =
                Self::compute_mean_for_subset(&cluster_features, &cluster_labels, 1, dimensions);
            clusters.push(ClusterData {
                centroid,
                prototype_good,
                prototype_bad,
            });
        }

        self.model = Some(TrainedModel {
            clusters,
            global_prototype_good,
            global_prototype_bad,
            temperature: self.params.temperature,
        });
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        let expected = model.global_prototype_good.len();
        if features.len() != expected {
            return Err(ClassifierError::PredictionDimension {
                expected,
                actual: features.len(),
            });
        }
        if let Some(column) = features.iter().position(|value| !value.is_finite()) {
            return Err(ClassifierError::NonFiniteFeature { row: 0, column });
        }

        let centroid_distances = model
            .clusters
            .iter()
            .map(|cluster| Self::euclidean_distance(features, &cluster.centroid))
            .collect::<Vec<_>>();
        let cluster_weights = Self::distance_softmax(&centroid_distances, model.temperature)?;

        let mut final_probability_good = 0.0_f64;
        for (index, cluster) in model.clusters.iter().enumerate() {
            let prototype_good = cluster
                .prototype_good
                .as_deref()
                .unwrap_or(&model.global_prototype_good);
            let prototype_bad = cluster
                .prototype_bad
                .as_deref()
                .unwrap_or(&model.global_prototype_bad);
            let distance_good = Self::euclidean_distance(features, prototype_good);
            let distance_bad = Self::euclidean_distance(features, prototype_bad);
            let probabilities =
                Self::distance_softmax(&[distance_good, distance_bad], model.temperature)?;
            final_probability_good += cluster_weights[index] * probabilities[0];
        }
        if !final_probability_good.is_finite() {
            return Err(ClassifierError::InvalidState(
                "prototype classifier produced a non-finite probability".into(),
            ));
        }
        Ok(final_probability_good)
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        Ok(SerializedClassifierState::KMeansPrototype(
            SerializedKMeansPrototype {
                clusters: model
                    .clusters
                    .iter()
                    .map(|cluster| KMeansPrototypeCluster {
                        centroid: cluster
                            .centroid
                            .iter()
                            .map(|value| f64::from(*value))
                            .collect(),
                        prototype_good: cluster.prototype_good.as_ref().map(|prototype| {
                            prototype.iter().map(|value| f64::from(*value)).collect()
                        }),
                        prototype_bad: cluster.prototype_bad.as_ref().map(|prototype| {
                            prototype.iter().map(|value| f64::from(*value)).collect()
                        }),
                    })
                    .collect(),
                global_prototype_good: model
                    .global_prototype_good
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                global_prototype_bad: model
                    .global_prototype_bad
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                temperature: model.temperature,
            },
        ))
    }

    fn dispose(&mut self) {
        self.model = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(value: f64, temperature: f64) -> SerializedKMeansPrototype {
        SerializedKMeansPrototype {
            clusters: vec![KMeansPrototypeCluster {
                centroid: vec![value],
                prototype_good: Some(vec![0.0]),
                prototype_bad: Some(vec![1.0]),
            }],
            global_prototype_good: vec![0.0],
            global_prototype_bad: vec![1.0],
            temperature,
        }
    }

    #[test]
    fn empty_features_take_precedence_over_length_mismatch() {
        let mut classifier = KMeansPrototypeClassifier::new(1.0, 1).unwrap();
        assert!(matches!(
            classifier.train(&[], &[0]),
            Err(ClassifierError::EmptyDataset)
        ));
    }

    #[test]
    fn serialized_values_outside_f32_range_are_rejected() {
        assert!(matches!(
            KMeansPrototypeClassifier::from_state(state(f64::MAX, 1.0)),
            Err(ClassifierError::InvalidState(_))
        ));
    }

    #[test]
    fn subnormal_temperature_never_returns_nan() {
        let classifier = KMeansPrototypeClassifier::from_state(state(0.0, f64::MIN_POSITIVE))
            .expect("finite model");
        let probability = classifier
            .predict_proba(&[0.5])
            .expect("finite probability");
        assert!(probability.is_finite());
    }
}
