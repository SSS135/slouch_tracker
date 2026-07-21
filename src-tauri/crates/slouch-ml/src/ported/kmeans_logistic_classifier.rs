//! K-means logistic classifier.
//!
//! Mechanical Rust port of `src/services/ml/kmeansLogisticClassifier.ts`.
//! Feature vectors and centroids use `f32`; distances, temperatures, routing
//! weights, probabilities, and serialized model values use `f64`.

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::kmeans::{kmeans_default, silhouette_score, KMeansError};
use super::mlp_classifier::{MlpClassifier, MlpConfig};
use super::types::{SerializedClassifierState, SerializedKMeansLogistic, SerializedMlp};

const DEFAULT_TEMPERATURE: f64 = 1.0;
const DEFAULT_WEIGHT_DECAY: f64 = 1.0;
const DEFAULT_MAX_ITERATIONS: usize = 100;

type ClusterSelection = (usize, Vec<Vec<f32>>, Vec<usize>);

/// Configuration for k-means logistic classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KMeansLogisticConfig {
    pub temperature: f64,
    pub weight_decay: f64,
    pub max_iterations: usize,
    pub use_class_weights: bool,
    pub n_clusters: usize,
}

impl Default for KMeansLogisticConfig {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            use_class_weights: false,
            n_clusters: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ClusterModel {
    centroid: Vec<f32>,
    model: Option<MlpClassifier>,
}

#[derive(Debug, Clone)]
struct TrainedModel {
    clusters: Vec<ClusterModel>,
    global_model: MlpClassifier,
    temperature: f64,
}

/// Combines k-means clustering with one logistic model per mixed-class
/// cluster and uses temperature-scaled soft routing at inference time.
#[derive(Debug, Clone)]
pub struct KMeansLogisticClassifier {
    model: Option<TrainedModel>,
    params: KMeansLogisticConfig,
}

impl KMeansLogisticClassifier {
    /// Creates an untrained classifier from explicit source-style parameters.
    pub fn new(
        temperature: f64,
        weight_decay: f64,
        max_iterations: usize,
        use_class_weights: bool,
        n_clusters: usize,
    ) -> Result<Self, ClassifierError> {
        Self::with_config(KMeansLogisticConfig {
            temperature,
            weight_decay,
            max_iterations,
            use_class_weights,
            n_clusters,
        })
    }

    /// Creates an untrained classifier from a complete configuration.
    pub fn with_config(config: KMeansLogisticConfig) -> Result<Self, ClassifierError> {
        validate_config(&config)?;
        Ok(Self {
            model: None,
            params: config,
        })
    }

    /// Creates an untrained classifier using the source defaults.
    pub fn default_classifier() -> Result<Self, ClassifierError> {
        Self::with_config(KMeansLogisticConfig::default())
    }

    /// Restores a serialized classifier with the supplied training settings.
    pub fn from_json(
        data: SerializedKMeansLogistic,
        config: KMeansLogisticConfig,
    ) -> Result<Self, ClassifierError> {
        let classifier = Self::with_config(config)?;
        let model = Self::model_from_state(data, &config)?;
        Ok(Self {
            model: Some(model),
            params: classifier.params,
        })
    }

    /// Restores a classifier from its serialized state using source defaults.
    pub fn from_state(data: SerializedKMeansLogistic) -> Result<Self, ClassifierError> {
        let temperature = data.temperature;
        Self::from_json(
            data,
            KMeansLogisticConfig {
                temperature,
                ..KMeansLogisticConfig::default()
            },
        )
    }

    fn mlp_config(config: &KMeansLogisticConfig) -> MlpConfig {
        MlpConfig {
            weight_decay: config.weight_decay,
            max_iterations: config.max_iterations,
            use_class_weights: config.use_class_weights,
            hidden_layers: 0,
            ..MlpConfig::default()
        }
    }

    fn map_kmeans_error(error: KMeansError) -> ClassifierError {
        ClassifierError::InvalidState(error.to_string())
    }

    fn model_from_state(
        data: SerializedKMeansLogistic,
        config: &KMeansLogisticConfig,
    ) -> Result<TrainedModel, ClassifierError> {
        if !data.temperature.is_finite() || data.temperature <= 0.0 {
            return Err(ClassifierError::InvalidState(
                "temperature must be finite and positive".into(),
            ));
        }
        if data.centroids.is_empty() || data.cluster_models.len() != data.centroids.len() {
            return Err(ClassifierError::InvalidState(
                "serialized model must contain one cluster model entry per centroid".into(),
            ));
        }
        if data.centroids.iter().any(|centroid| {
            centroid.is_empty()
                || centroid.iter().any(|value| {
                    let narrowed = *value as f32;
                    !value.is_finite() || !narrowed.is_finite()
                })
        }) {
            return Err(ClassifierError::InvalidState(
                "cluster centroids must be nonempty and remain finite in f32".into(),
            ));
        }
        let dimensions = data.centroids[0].len();
        if data
            .centroids
            .iter()
            .any(|centroid| centroid.len() != dimensions)
        {
            return Err(ClassifierError::InvalidState(
                "cluster centroids must have equal dimensions".into(),
            ));
        }

        validate_mlp_input_dimensions(&data.global_model, dimensions, "global model")?;
        for (index, model) in data.cluster_models.iter().enumerate() {
            if let Some(model) = model {
                validate_mlp_input_dimensions(
                    model,
                    dimensions,
                    &format!("cluster model {index}"),
                )?;
            }
        }

        let mlp_config = Self::mlp_config(config);
        let global_model = MlpClassifier::from_json(data.global_model, mlp_config)?;
        let mut clusters = Vec::with_capacity(data.centroids.len());
        for (centroid, model_data) in data.centroids.into_iter().zip(data.cluster_models) {
            let model = model_data
                .map(|state: SerializedMlp| MlpClassifier::from_json(state, mlp_config))
                .transpose()?;
            clusters.push(ClusterModel {
                centroid: centroid.into_iter().map(|value| value as f32).collect(),
                model,
            });
        }

        Ok(TrainedModel {
            clusters,
            global_model,
            temperature: data.temperature,
        })
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

    fn select_clusters(
        features: &[Vec<f32>],
        n_clusters: usize,
    ) -> Result<ClusterSelection, ClassifierError> {
        if n_clusters > 0 {
            let best_k = n_clusters.min(features.len());
            let result = kmeans_default(features, best_k).map_err(Self::map_kmeans_error)?;
            return Ok((best_k, result.centroids, result.assignments));
        }

        let k_min = (features.len() as f64 / 15.0).ceil() as usize;
        let k_max = features.len() / 5;
        if k_max < k_min || k_max < 1 {
            let result = kmeans_default(features, 1).map_err(Self::map_kmeans_error)?;
            return Ok((1, result.centroids, result.assignments));
        }

        let mut best_k = 1_usize;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_centroids = Vec::new();
        let mut best_assignments = Vec::new();

        for candidate in k_min..=k_max {
            let result = kmeans_default(features, candidate).map_err(Self::map_kmeans_error)?;
            if candidate == 1 {
                if best_centroids.is_empty() {
                    best_k = 1;
                    best_centroids = result.centroids;
                    best_assignments = result.assignments;
                }
                continue;
            }

            let score = silhouette_score(features, &result.assignments, &result.centroids)
                .map_err(Self::map_kmeans_error)?;
            if score > best_score {
                best_score = score;
                best_k = candidate;
                best_centroids = result.centroids;
                best_assignments = result.assignments;
            }
        }

        if best_centroids.is_empty() {
            return Err(ClassifierError::InvalidState(
                "K-means failed to produce clusters".into(),
            ));
        }
        Ok((best_k, best_centroids, best_assignments))
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

    fn softmax(values: &[f64]) -> Vec<f64> {
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = values
            .iter()
            .map(|value| (*value - maximum).exp())
            .collect::<Vec<_>>();
        let total = exponentials.iter().sum::<f64>();
        exponentials
            .into_iter()
            .map(|value| value / total)
            .collect()
    }
}

impl BaseClassifier for KMeansLogisticClassifier {
    fn classifier_id(&self) -> &'static str {
        "kmeans_logistic"
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
        self.dispose();
        let dimensions = Self::validate_features(features)?;
        let (best_k, centroids, assignments) =
            Self::select_clusters(features, self.params.n_clusters)?;
        if centroids.len() != best_k || assignments.len() != features.len() {
            return Err(ClassifierError::InvalidState(
                "K-means returned inconsistent cluster data".into(),
            ));
        }

        let mlp_config = Self::mlp_config(&self.params);
        let mut global_model = MlpClassifier::new(mlp_config)?;
        global_model.train(features, labels)?;

        let mut clusters = Vec::with_capacity(best_k);
        for cluster_index in 0..best_k {
            let centroid = centroids
                .get(cluster_index)
                .ok_or_else(|| ClassifierError::InvalidState("missing cluster centroid".into()))?;
            if centroid.len() != dimensions {
                return Err(ClassifierError::InvalidState(
                    "cluster centroid dimension does not match features".into(),
                ));
            }

            let mut cluster_features = Vec::new();
            let mut cluster_labels = Vec::new();
            for (index, assignment) in assignments.iter().enumerate() {
                if *assignment == cluster_index {
                    cluster_features.push(features[index].clone());
                    cluster_labels.push(labels[index]);
                }
            }

            let has_good = cluster_labels.contains(&0);
            let has_bad = cluster_labels.contains(&1);
            let model = if has_good && has_bad {
                let mut cluster_model = MlpClassifier::new(mlp_config)?;
                cluster_model.train(&cluster_features, &cluster_labels)?;
                Some(cluster_model)
            } else {
                None
            };
            clusters.push(ClusterModel {
                centroid: centroid.clone(),
                model,
            });
        }

        self.model = Some(TrainedModel {
            clusters,
            global_model,
            temperature: self.params.temperature,
        });
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        let expected = model
            .clusters
            .first()
            .map(|cluster| cluster.centroid.len())
            .ok_or_else(|| ClassifierError::InvalidState("model has no clusters".into()))?;
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
        let cluster_weights = Self::softmax(
            &centroid_distances
                .iter()
                .map(|distance| -distance / model.temperature)
                .collect::<Vec<_>>(),
        );

        let mut final_probability_good = 0.0_f64;
        for (index, cluster) in model.clusters.iter().enumerate() {
            let probability_good = match &cluster.model {
                Some(cluster_model) => cluster_model.predict_proba(features)?,
                None => model.global_model.predict_proba(features)?,
            };
            final_probability_good += cluster_weights[index] * probability_good;
        }
        Ok(final_probability_good)
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        let Some(model) = &self.model else {
            return Err(ClassifierError::Untrained);
        };
        let cluster_models = model
            .clusters
            .iter()
            .map(|cluster| {
                cluster
                    .model
                    .as_ref()
                    .map(MlpClassifier::to_mlp_json)
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SerializedClassifierState::KMeansLogistic(
            SerializedKMeansLogistic {
                centroids: model
                    .clusters
                    .iter()
                    .map(|cluster| {
                        cluster
                            .centroid
                            .iter()
                            .map(|value| f64::from(*value))
                            .collect()
                    })
                    .collect(),
                cluster_models,
                global_model: model.global_model.to_mlp_json()?,
                temperature: model.temperature,
            },
        ))
    }

    fn dispose(&mut self) {
        self.model = None;
    }
}

fn validate_mlp_input_dimensions(
    model: &SerializedMlp,
    expected: usize,
    name: &str,
) -> Result<(), ClassifierError> {
    if model.layer_shapes.first().copied() != Some(expected) {
        return Err(ClassifierError::InvalidState(format!(
            "{name} input dimension must match centroid dimension {expected}"
        )));
    }
    Ok(())
}

fn validate_config(config: &KMeansLogisticConfig) -> Result<(), ClassifierError> {
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(ClassifierError::InvalidState(
            "temperature must be finite and positive".into(),
        ));
    }
    if !config.weight_decay.is_finite() {
        return Err(ClassifierError::InvalidState(
            "weight decay must be finite".into(),
        ));
    }
    Ok(())
}
