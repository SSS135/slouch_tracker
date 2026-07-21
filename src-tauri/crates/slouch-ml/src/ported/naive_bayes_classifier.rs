//! Gaussian Naive Bayes classifier.
//!
//! Mechanical port of `src/services/ml/naiveBayesClassifier.ts`. Feature
//! statistics are kept as `f32` to preserve the source `Float32Array`
//! accumulation and serialization values are represented as `f64`.

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::types::{SerializedClassifierState, SerializedGaussianNb};

const DEFAULT_VARIANCE_SMOOTHING: f64 = 1e-6;

type StateParts<'a> = (&'a [Vec<f32>; 2], &'a [Vec<f32>; 2], &'a [f64; 2]);
type ValidatedState = ([Vec<f32>; 2], [Vec<f32>; 2], [f64; 2], f64);

#[derive(Debug, Clone)]
pub struct GaussianNbClassifier {
    class_means: Option<[Vec<f32>; 2]>,
    class_variances: Option<[Vec<f32>; 2]>,
    class_priors: Option<[f64; 2]>,
    epsilon: f64,
}

impl Default for GaussianNbClassifier {
    fn default() -> Self {
        Self {
            class_means: None,
            class_variances: None,
            class_priors: None,
            epsilon: DEFAULT_VARIANCE_SMOOTHING,
        }
    }
}

impl GaussianNbClassifier {
    pub fn new(variance_smoothing: f64) -> Result<Self, ClassifierError> {
        if !valid_epsilon(variance_smoothing) {
            return Err(ClassifierError::InvalidEpsilon);
        }
        Ok(Self {
            epsilon: variance_smoothing,
            ..Self::default()
        })
    }

    pub fn from_json(
        data: SerializedGaussianNb,
        variance_smoothing: f64,
    ) -> Result<Self, ClassifierError> {
        if !valid_epsilon(variance_smoothing) {
            return Err(ClassifierError::InvalidEpsilon);
        }
        let (means, variances, priors, epsilon) = Self::validated_state(&data)?;
        Ok(Self {
            class_means: Some(means),
            class_variances: Some(variances),
            class_priors: Some(priors),
            epsilon,
        })
    }

    pub fn from_state(data: SerializedGaussianNb) -> Result<Self, ClassifierError> {
        let (means, variances, priors, epsilon) = Self::validated_state(&data)?;
        Ok(Self {
            class_means: Some(means),
            class_variances: Some(variances),
            class_priors: Some(priors),
            epsilon,
        })
    }

    fn validated_state(data: &SerializedGaussianNb) -> Result<ValidatedState, ClassifierError> {
        let dimensions = data.class_means[0].len();
        if dimensions == 0
            || data.class_means[1].len() != dimensions
            || data.class_variances[0].len() != dimensions
            || data.class_variances[1].len() != dimensions
        {
            return Err(ClassifierError::InvalidState(
                "class means and variances must have equal nonzero dimensions".into(),
            ));
        }
        let epsilon_f32 = data.epsilon as f32;
        if !data.epsilon.is_finite()
            || data.epsilon <= 0.0
            || !epsilon_f32.is_finite()
            || epsilon_f32 <= 0.0
        {
            return Err(ClassifierError::InvalidState(
                "epsilon must remain finite and positive in f32".into(),
            ));
        }
        if data
            .class_priors
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
            || (data.class_priors.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(ClassifierError::InvalidState(
                "state contains invalid numeric values".into(),
            ));
        }

        let means = [
            narrow_finite_values(&data.class_means[0], "class means")?,
            narrow_finite_values(&data.class_means[1], "class means")?,
        ];
        let variances = [
            narrow_positive_values(&data.class_variances[0], "class variances")?,
            narrow_positive_values(&data.class_variances[1], "class variances")?,
        ];

        Ok((means, variances, data.class_priors, data.epsilon))
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

    fn compute_mean(samples: &[&[f32]], dimensions: usize) -> Vec<f32> {
        let mut mean = vec![0.0_f32; dimensions];
        for sample in samples {
            for (sum, value) in mean.iter_mut().zip(sample.iter()) {
                *sum += *value;
            }
        }
        for value in &mut mean {
            *value = (f64::from(*value) / samples.len() as f64) as f32;
        }
        mean
    }

    fn compute_variance(&self, samples: &[&[f32]], mean: &[f32]) -> Vec<f32> {
        if samples.len() == 1 {
            return vec![self.epsilon as f32; mean.len()];
        }

        let mut variance = vec![0.0_f32; mean.len()];
        for sample in samples {
            for ((sum, value), average) in variance.iter_mut().zip(sample.iter()).zip(mean.iter()) {
                let difference = f64::from(*value) - f64::from(*average);
                *sum = (f64::from(*sum) + difference * difference) as f32;
            }
        }
        for value in &mut variance {
            *value = (f64::from(*value) / samples.len() as f64 + self.epsilon) as f32;
        }
        variance
    }

    fn state_parts(&self) -> Result<StateParts<'_>, ClassifierError> {
        match (&self.class_means, &self.class_variances, &self.class_priors) {
            (Some(means), Some(variances), Some(priors)) => Ok((means, variances, priors)),
            _ => Err(ClassifierError::Untrained),
        }
    }

    fn compute_log_probability(
        features: &[f32],
        mean: &[f32],
        variance: &[f32],
        prior: f64,
    ) -> f64 {
        let mut log_probability = prior.ln();
        for ((feature, mean), variance) in features.iter().zip(mean).zip(variance) {
            let difference = f64::from(*feature) - f64::from(*mean);
            let variance = f64::from(*variance);
            let log_pdf = -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
                - difference * difference / (2.0 * variance);
            log_probability += log_pdf;
        }
        log_probability
    }

    pub fn to_state(&self) -> Result<SerializedGaussianNb, ClassifierError> {
        let (means, variances, priors) = self.state_parts()?;
        Ok(SerializedGaussianNb {
            class_means: [
                means[0].iter().map(|value| f64::from(*value)).collect(),
                means[1].iter().map(|value| f64::from(*value)).collect(),
            ],
            class_variances: [
                variances[0].iter().map(|value| f64::from(*value)).collect(),
                variances[1].iter().map(|value| f64::from(*value)).collect(),
            ],
            class_priors: *priors,
            epsilon: self.epsilon,
        })
    }
}

fn valid_epsilon(value: f64) -> bool {
    let narrowed = value as f32;
    value.is_finite() && value > 0.0 && narrowed.is_finite() && narrowed > 0.0
}

fn narrow_finite_values(values: &[f64], field: &str) -> Result<Vec<f32>, ClassifierError> {
    values
        .iter()
        .map(|value| {
            let narrowed = *value as f32;
            if !value.is_finite() || !narrowed.is_finite() {
                return Err(ClassifierError::InvalidState(format!(
                    "{field} must remain finite in f32"
                )));
            }
            Ok(narrowed)
        })
        .collect()
}

fn narrow_positive_values(values: &[f64], field: &str) -> Result<Vec<f32>, ClassifierError> {
    values
        .iter()
        .map(|value| {
            let narrowed = *value as f32;
            if !value.is_finite() || !narrowed.is_finite() || narrowed <= 0.0 {
                return Err(ClassifierError::InvalidState(format!(
                    "{field} must remain finite and positive in f32"
                )));
            }
            Ok(narrowed)
        })
        .collect()
}

fn validate_trained_state(
    means: &[Vec<f32>; 2],
    variances: &[Vec<f32>; 2],
    priors: &[f64; 2],
) -> Result<(), ClassifierError> {
    if means.iter().flatten().any(|value| !value.is_finite())
        || variances
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || priors
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
    {
        return Err(ClassifierError::InvalidState(
            "training produced invalid Gaussian Naive Bayes state".into(),
        ));
    }
    Ok(())
}

impl BaseClassifier for GaussianNbClassifier {
    fn classifier_id(&self) -> &'static str {
        "gaussian_nb"
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
        let mut class0 = Vec::new();
        let mut class1 = Vec::new();
        for (sample, label) in features.iter().zip(labels) {
            if *label == 0 {
                class0.push(sample.as_slice());
            } else {
                class1.push(sample.as_slice());
            }
        }
        if class0.is_empty() || class1.is_empty() {
            return Err(ClassifierError::MissingClass);
        }

        let means = [
            Self::compute_mean(&class0, dimensions),
            Self::compute_mean(&class1, dimensions),
        ];
        let variances = [
            self.compute_variance(&class0, &means[0]),
            self.compute_variance(&class1, &means[1]),
        ];
        let sample_count = features.len() as f64;
        let priors = [
            class0.len() as f64 / sample_count,
            class1.len() as f64 / sample_count,
        ];
        validate_trained_state(&means, &variances, &priors)?;

        self.class_means = Some(means);
        self.class_variances = Some(variances);
        self.class_priors = Some(priors);
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        let (means, variances, priors) = self.state_parts()?;
        let expected = means[0].len();
        if features.len() != expected {
            return Err(ClassifierError::PredictionDimension {
                expected,
                actual: features.len(),
            });
        }
        if let Some(column) = features.iter().position(|value| !value.is_finite()) {
            return Err(ClassifierError::NonFiniteFeature { row: 0, column });
        }

        let scale = (features.len() as f64).sqrt();
        let log0 =
            Self::compute_log_probability(features, &means[0], &variances[0], priors[0]) / scale;
        let log1 =
            Self::compute_log_probability(features, &means[1], &variances[1], priors[1]) / scale;
        let maximum = log0.max(log1);
        let exp0 = (log0 - maximum).exp();
        let exp1 = (log1 - maximum).exp();
        let probability_bad = exp1 / (exp0 + exp1);
        Ok(1.0 - probability_bad)
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        Ok(SerializedClassifierState::GaussianNb(self.to_state()?))
    }

    fn dispose(&mut self) {
        self.class_means = None;
        self.class_variances = None;
        self.class_priors = None;
    }
}
