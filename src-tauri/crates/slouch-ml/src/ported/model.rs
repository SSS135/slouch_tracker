//! Composition of the feature-extraction pipeline and a posture classifier.
//!
//! This is the Rust port of `src/services/ml/model.ts`.  Feature containers are
//! represented by `slouch_domain::FeatureSource`; the concrete source may be an
//! inference result or a stored posture frame.

use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};
use slouch_domain::FeatureSource;

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::classifier_factory::deserialize_classifier;
use super::config::TRAINING_CONFIG;
use super::feature_extractor::{FeatureExtractor, FeatureExtractorConfig};
use super::types::{SerializedClassifier, SerializedModel};

/// The part emitted by the source `Model.toJSON()` method.
///
/// The TypeScript method deliberately leaves `trainedAt` and `version` to the
/// caller.  `to_serialized_model` is provided for callers that need the full
/// persisted envelope accepted by `from_json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedModelPayload {
    pub feature_extractor: super::types::SerializedFeatureExtractor,
    pub classifier: SerializedClassifier,
}

/// Errors raised by model composition and dataset validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Dataset(String),
    FeatureExtractor(String),
    Classifier(ClassifierError),
    ClassifierFactory(String),
    Serialization(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(message)
            | Self::FeatureExtractor(message)
            | Self::ClassifierFactory(message)
            | Self::Serialization(message) => formatter.write_str(message),
            Self::Classifier(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<ClassifierError> for ModelError {
    fn from(error: ClassifierError) -> Self {
        Self::Classifier(error)
    }
}

/// A fitted feature extractor and classifier used as one prediction model.
pub struct Model<S = slouch_domain::InferenceResult>
where
    S: FeatureSource,
{
    feature_extractor: FeatureExtractor<S>,
    classifier: Box<dyn BaseClassifier>,
}

impl<S> Model<S>
where
    S: FeatureSource,
{
    /// Creates a model from an already-created extractor and classifier.
    pub fn new(
        feature_extractor: FeatureExtractor<S>,
        classifier: Box<dyn BaseClassifier>,
    ) -> Self {
        Self {
            feature_extractor,
            classifier,
        }
    }

    /// Creates a model from extractor configuration and a classifier.
    pub fn from_config(
        extractor_config: FeatureExtractorConfig<S>,
        classifier: Box<dyn BaseClassifier>,
    ) -> Self {
        Self::new(FeatureExtractor::new(extractor_config), classifier)
    }

    /// Fits preprocessing on the supplied feature containers, then trains the
    /// classifier on the transformed vectors.
    pub fn fit(&mut self, features: &[S], labels: &[i32]) -> Result<(), ModelError> {
        if features.is_empty() {
            return Err(ModelError::Dataset(
                "Cannot train on empty dataset".to_owned(),
            ));
        }

        if features.len() != labels.len() {
            return Err(ModelError::Dataset(format!(
                "Length mismatch: {} features, {} labels",
                features.len(),
                labels.len()
            )));
        }

        self.validate_dataset(labels)?;

        self.feature_extractor
            .fit(features, labels)
            .map_err(|error| ModelError::FeatureExtractor(error.to_string()))?;
        let processed = self
            .feature_extractor
            .transform_batch(features)
            .map_err(|error| ModelError::FeatureExtractor(error.to_string()))?;

        self.classifier.train(&processed, labels)?;
        Ok(())
    }

    /// Transforms one feature container and returns the classifier's good-posture
    /// probability.
    pub fn predict(&self, features: &S) -> Result<f64, ModelError> {
        let processed = self
            .feature_extractor
            .transform(features)
            .map_err(|error| ModelError::FeatureExtractor(error.to_string()))?;
        Ok(self.classifier.predict_proba(&processed)?)
    }

    /// Serializes the same two fields returned by the TypeScript `toJSON()`.
    pub fn to_json(&self) -> Result<SerializedModelPayload, ModelError> {
        let feature_extractor = self
            .feature_extractor
            .to_json()
            .map_err(|error| ModelError::Serialization(error.to_string()))?;
        let classifier = SerializedClassifier {
            classifier_id: self.classifier.classifier_id().to_owned(),
            state: self.classifier.to_json()?,
        };
        Ok(SerializedModelPayload {
            feature_extractor,
            classifier,
        })
    }

    /// Serializes a complete persisted model envelope.
    pub fn to_serialized_model(
        &self,
        trained_at: f64,
        version: f64,
    ) -> Result<SerializedModel, ModelError> {
        let payload = self.to_json()?;
        Ok(SerializedModel {
            feature_extractor: payload.feature_extractor,
            classifier: payload.classifier,
            trained_at,
            version,
        })
    }

    /// Restores a model from its validated serialized envelope.
    pub fn from_json(data: SerializedModel) -> Result<Self, ModelError> {
        if !data.trained_at.is_finite() || !data.version.is_finite() {
            return Err(ModelError::Serialization(
                "serialized model metadata must be finite".to_owned(),
            ));
        }

        let feature_extractor = FeatureExtractor::from_json(data.feature_extractor)
            .map_err(|error| ModelError::Serialization(error.to_string()))?;
        let classifier_id = data.classifier.classifier_id.clone();
        let classifier =
            deserialize_classifier(&classifier_id, data.classifier.state, &BTreeMap::new())
                .map_err(|error| ModelError::ClassifierFactory(error.to_string()))?;

        Ok(Self::new(feature_extractor, classifier))
    }

    /// Releases transformer and classifier resources.
    pub fn dispose(&mut self) {
        self.feature_extractor.dispose();
        self.classifier.dispose();
    }

    fn validate_dataset(&self, labels: &[i32]) -> Result<(), ModelError> {
        let mut class_counts = [0usize; 2];
        for &label in labels.iter() {
            if label == 0 || label == 1 {
                class_counts[label as usize] += 1;
            }
        }

        let class0_count = class_counts[0];
        let class1_count = class_counts[1];

        if class0_count == 0 && class1_count == 0 {
            return Err(ModelError::Dataset(
                "Dataset has no labeled frames".to_owned(),
            ));
        }

        if class0_count < TRAINING_CONFIG.min_frames_per_class {
            return Err(ModelError::Dataset(format!(
                "Not enough class 0 frames: {class0_count} < {}",
                TRAINING_CONFIG.min_frames_per_class
            )));
        }

        if class1_count < TRAINING_CONFIG.min_frames_per_class {
            return Err(ModelError::Dataset(format!(
                "Not enough class 1 frames: {class1_count} < {}",
                TRAINING_CONFIG.min_frames_per_class
            )));
        }

        Ok(())
    }
}
