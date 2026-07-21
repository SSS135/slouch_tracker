//! Feature extraction, normalization, and dimensionality reduction for ML models.
//!
//! Mechanical Rust port of `src/services/ml/featureExtractor.ts`.
//! Feature containers remain generic over the shared `FeatureSource` contract so
//! the same extractor can be fitted from stored frames or inference results.

use std::{fmt, str::FromStr};

use slouch_domain::{FeatureId, FeatureSource};

use super::feature_extraction::{
    concatenate_features, get_expected_concatenated_dimensions, FeatureExtractionError,
};
use super::layer_norm::{apply_layer_norm_batch, layer_norm_tf};
use super::pca::{PcaError, PcaTransformer};
use super::random_projection::{RandomProjectionError, RandomProjectionTransformer};
use super::types::{
    DimReductionTransformer, DimensionalityReductionConfig, DimensionalityReductionMethod,
    NormalizationMode, SerializedFeatureExtractor,
};
use crate::ported::constants::EPSILON_STABLE;

type ZScoreNormalization = (Vec<Vec<f32>>, Vec<f64>, Vec<f64>);

/// Configuration for a feature extractor.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureExtractorConfig<S = slouch_domain::InferenceResult> {
    pub feature_types: Vec<FeatureId>,
    pub normalization_mode: NormalizationMode,
    pub dim_reduction_config: DimensionalityReductionConfig,
    pub unlabeled_samples: Vec<S>,
}

/// Errors raised while fitting, transforming, or restoring an extractor.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureExtractorError {
    EmptyDataset,
    LengthMismatch {
        features: usize,
        labels: usize,
    },
    FeatureExtraction(FeatureExtractionError),
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    RaggedFeatures {
        row: usize,
        expected: usize,
        actual: usize,
    },
    EmptyFeatureArray,
    InvalidConfiguration(String),
    RandomProjection(RandomProjectionError),
    Pca(PcaError),
}

impl fmt::Display for FeatureExtractorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => formatter.write_str("Cannot fit FeatureExtractor on empty dataset"),
            Self::LengthMismatch { features, labels } => {
                write!(formatter, "Length mismatch: {features} features, {labels} labels")
            }
            Self::FeatureExtraction(error) => error.fmt(formatter),
            Self::DimensionMismatch { expected, actual } => write!(
                formatter,
                "Feature dimension mismatch: Expected {expected} features after concatenation, got {actual}"
            ),
            Self::RaggedFeatures { row, expected, actual } => write!(
                formatter,
                "Feature row {row} has length {actual}, expected {expected}"
            ),
            Self::EmptyFeatureArray => formatter.write_str("Cannot normalize empty feature array"),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::RandomProjection(error) => error.fmt(formatter),
            Self::Pca(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FeatureExtractorError {}

impl From<FeatureExtractionError> for FeatureExtractorError {
    fn from(error: FeatureExtractionError) -> Self {
        Self::FeatureExtraction(error)
    }
}

impl From<RandomProjectionError> for FeatureExtractorError {
    fn from(error: RandomProjectionError) -> Self {
        Self::RandomProjection(error)
    }
}

impl From<PcaError> for FeatureExtractorError {
    fn from(error: PcaError) -> Self {
        Self::Pca(error)
    }
}

/// Stateful feature preprocessing pipeline.
#[derive(Debug, Clone)]
pub struct FeatureExtractor<S = slouch_domain::InferenceResult> {
    feature_types: Vec<FeatureId>,
    normalization_mode: NormalizationMode,
    dim_reduction_config: DimensionalityReductionConfig,
    unlabeled_samples: Vec<S>,
    concatenated_dimensions: usize,
    normalization_mean: Option<Vec<f64>>,
    normalization_std: Option<Vec<f64>>,
    dim_reduction_transformer: Option<DimensionalityReducer>,
}

#[derive(Debug, Clone)]
enum DimensionalityReducer {
    RandomProjection(RandomProjectionTransformer),
    Pca(PcaTransformer),
}

impl<S> FeatureExtractor<S>
where
    S: FeatureSource,
{
    pub fn new(config: FeatureExtractorConfig<S>) -> Self {
        Self {
            feature_types: config.feature_types,
            normalization_mode: config.normalization_mode,
            dim_reduction_config: config.dim_reduction_config,
            unlabeled_samples: config.unlabeled_samples,
            concatenated_dimensions: 0,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        }
    }

    /// Fits normalization and dimensionality-reduction state on labeled data.
    pub fn fit(&mut self, features: &[S], labels: &[i32]) -> Result<(), FeatureExtractorError> {
        if features.is_empty() {
            return Err(FeatureExtractorError::EmptyDataset);
        }
        if features.len() != labels.len() {
            return Err(FeatureExtractorError::LengthMismatch {
                features: features.len(),
                labels: labels.len(),
            });
        }

        log_info(&format!(
            "[FEATURE_EXTRACTOR] Fitting on {} labeled samples",
            features.len()
        ));

        let labeled_concatenated = features
            .iter()
            .map(|container| concatenate_features(container, &self.feature_types))
            .collect::<Result<Vec<_>, _>>()?;
        validate_feature_values(&labeled_concatenated)?;
        self.concatenated_dimensions = labeled_concatenated
            .first()
            .ok_or(FeatureExtractorError::EmptyFeatureArray)?
            .len();
        if self.concatenated_dimensions == 0 {
            return Err(FeatureExtractorError::EmptyFeatureArray);
        }
        validate_ragged(&labeled_concatenated)?;
        log_info(&format!(
            "[FEATURE_EXTRACTOR] Concatenated dimensions: {}",
            self.concatenated_dimensions
        ));

        let use_combined_data = !self.unlabeled_samples.is_empty()
            && (matches!(
                self.dim_reduction_config.method,
                DimensionalityReductionMethod::Pca
            ) || matches!(self.normalization_mode, NormalizationMode::ZScore));

        // The source intentionally ignores unlabeled samples for modes which
        // cannot use them. Do not validate semantically unused input.
        let unlabeled_concatenated = if use_combined_data {
            let values = self
                .unlabeled_samples
                .iter()
                .map(|container| concatenate_features(container, &self.feature_types))
                .collect::<Result<Vec<_>, _>>()?;
            validate_ragged(&values)?;
            validate_feature_values(&values)?;
            for (row, sample) in values.iter().enumerate() {
                if sample.len() != self.concatenated_dimensions {
                    return Err(FeatureExtractorError::RaggedFeatures {
                        row,
                        expected: self.concatenated_dimensions,
                        actual: sample.len(),
                    });
                }
            }
            values
        } else {
            Vec::new()
        };

        let mut combined_concatenated = labeled_concatenated;
        if use_combined_data {
            combined_concatenated.extend(unlabeled_concatenated);
            log_info(&format!(
                "[FEATURE_EXTRACTOR] Combined {} labeled + {} unlabeled samples for fitting",
                features.len(),
                self.unlabeled_samples.len()
            ));
        }

        let normalized_combined = match self.normalization_mode {
            NormalizationMode::None => {
                log_info("[FEATURE_EXTRACTOR] No normalization applied");
                combined_concatenated
            }
            NormalizationMode::Layer => {
                log_info("[FEATURE_EXTRACTOR] Applying layer normalization...");
                apply_layer_norm_batch(&combined_concatenated).map_err(|error| {
                    FeatureExtractorError::InvalidConfiguration(error.to_string())
                })?
            }
            NormalizationMode::ZScore => {
                log_info("[FEATURE_EXTRACTOR] Applying z-score normalization...");
                let (normalized, mean, std) = normalize_z_score(&combined_concatenated)?;
                self.normalization_mean = Some(mean);
                self.normalization_std = Some(std);
                log_info(&format!(
                    "[FEATURE_EXTRACTOR] Z-score normalization params saved (computed on {} samples)",
                    combined_concatenated.len()
                ));
                normalized
            }
            NormalizationMode::Calibrated => {
                log_info(
                    "[FEATURE_EXTRACTOR] Applying calibrated (reference-class) normalization...",
                );
                // Fit mean/std on the reference class (label 0 = good/present) only so the
                // baseline centers near zero and deviations read as slouch. The unlabeled
                // reservoir is intentionally excluded (Calibrated is absent from
                // `use_combined_data`), so `combined_concatenated` is aligned with `labels`.
                let reference: Vec<Vec<f32>> = combined_concatenated
                    .iter()
                    .zip(labels)
                    .filter(|(_, label)| **label == 0)
                    .map(|(row, _)| row.clone())
                    .collect();
                let stats_source = if reference.is_empty() {
                    &combined_concatenated
                } else {
                    &reference
                };
                let (mean, std) = z_score_stats(stats_source)?;
                let normalized = apply_z_score_stats(&combined_concatenated, &mean, &std);
                self.normalization_mean = Some(mean);
                self.normalization_std = Some(std);
                log_info(&format!(
                    "[FEATURE_EXTRACTOR] Calibrated normalization params saved (fit on {} reference-class samples)",
                    stats_source.len()
                ));
                normalized
            }
        };

        let normalized_labeled = if use_combined_data {
            normalized_combined[..features.len()].to_vec()
        } else {
            normalized_combined.clone()
        };

        self.dim_reduction_transformer = match self.dim_reduction_config.method {
            DimensionalityReductionMethod::RandomProjection => {
                log_info(&format!(
                    "[FEATURE_EXTRACTOR] Fitting Random Projection ({} components)",
                    self.dim_reduction_config.components
                ));
                let mut transformer = RandomProjectionTransformer::with_default_seed(
                    self.dim_reduction_config.components,
                )?;
                transformer.fit(&normalized_labeled)?;
                log_info("[FEATURE_EXTRACTOR] Random Projection fitted");
                Some(DimensionalityReducer::RandomProjection(transformer))
            }
            DimensionalityReductionMethod::Pca => {
                log_info(&format!(
                    "[FEATURE_EXTRACTOR] Fitting PCA ({} components) on {} samples",
                    self.dim_reduction_config.components,
                    normalized_combined.len()
                ));
                let data = normalized_combined
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| f64::from(*value))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut transformer = PcaTransformer::new();
                transformer.fit(&data, self.dim_reduction_config.components as isize)?;
                // PCA yields at most min(fit_rows - 1, n_features) components. Record the
                // EFFECTIVE count as the config's component count so the serialized config,
                // the fitted PCA state (`n_components`), `get_output_dimensions`, and the
                // model-container validator all agree. A requested count larger than the data
                // can support then degrades to the effective rank instead of writing a state
                // whose `n_components` disagrees with the config the loader re-validates.
                let effective_components = transformer.get_output_dimension();
                if effective_components < self.dim_reduction_config.components {
                    log_info(&format!(
                        "[FEATURE_EXTRACTOR] PCA components reduced {} -> {} (limited by fitted data)",
                        self.dim_reduction_config.components, effective_components
                    ));
                }
                self.dim_reduction_config.components = effective_components;
                log_info("[FEATURE_EXTRACTOR] PCA fitted");
                Some(DimensionalityReducer::Pca(transformer))
            }
            DimensionalityReductionMethod::None => {
                log_info("[FEATURE_EXTRACTOR] No dimensionality reduction");
                None
            }
            DimensionalityReductionMethod::PlsDa | DimensionalityReductionMethod::LinearNca => {
                return Err(FeatureExtractorError::InvalidConfiguration(
                    "Unsupported dimensionality reduction method".to_owned(),
                ));
            }
        };

        log_info(&format!(
            "[FEATURE_EXTRACTOR] Fit complete. Output dimensions: {}",
            self.get_output_dimensions()
        ));
        Ok(())
    }

    /// Transforms one feature container using fitted preprocessing state.
    pub fn transform(&self, features: &S) -> Result<Vec<f32>, FeatureExtractorError> {
        let concatenated = concatenate_features(features, &self.feature_types)?;
        if concatenated.len() != self.concatenated_dimensions {
            return Err(FeatureExtractorError::DimensionMismatch {
                expected: self.concatenated_dimensions,
                actual: concatenated.len(),
            });
        }

        let normalized = match self.normalization_mode {
            NormalizationMode::None => concatenated,
            NormalizationMode::Layer => layer_norm_tf(&concatenated)
                .map_err(|error| FeatureExtractorError::InvalidConfiguration(error.to_string()))?,
            NormalizationMode::ZScore | NormalizationMode::Calibrated => {
                let mean = self.normalization_mean.as_ref().ok_or_else(|| {
                    FeatureExtractorError::InvalidConfiguration(
                        "Z-score normalization was used during training, but normalization parameters are missing. The model may be corrupted. Please retrain your model.".to_owned(),
                    )
                })?;
                let std = self.normalization_std.as_ref().ok_or_else(|| {
                    FeatureExtractorError::InvalidConfiguration(
                        "Z-score normalization was used during training, but normalization parameters are missing. The model may be corrupted. Please retrain your model.".to_owned(),
                    )
                })?;
                apply_z_score_norm(&concatenated, mean, std)?
            }
        };

        match &self.dim_reduction_transformer {
            Some(DimensionalityReducer::Pca(transformer)) => {
                let data = vec![normalized
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>()];
                Ok(transformer
                    .transform(&data)?
                    .into_iter()
                    .next()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|value| value as f32)
                    .collect())
            }
            Some(DimensionalityReducer::RandomProjection(transformer)) => {
                Ok(transformer.transform(&normalized)?)
            }
            None => Ok(normalized),
        }
    }

    pub fn transform_batch(&self, features: &[S]) -> Result<Vec<Vec<f32>>, FeatureExtractorError> {
        features
            .iter()
            .map(|container| self.transform(container))
            .collect()
    }

    /// Serializes the fitted extractor state.
    pub fn to_json(&self) -> Result<SerializedFeatureExtractor, FeatureExtractorError> {
        let dim_reduction_transformer = match &self.dim_reduction_transformer {
            Some(DimensionalityReducer::RandomProjection(transformer)) => Some(
                DimReductionTransformer::RandomProjection(transformer.to_json()?),
            ),
            Some(DimensionalityReducer::Pca(transformer)) => {
                Some(DimReductionTransformer::Pca(transformer.to_json()?))
            }
            None => None,
        };

        Ok(SerializedFeatureExtractor {
            feature_types: self
                .feature_types
                .iter()
                .map(|feature_type| feature_type.as_str().to_owned())
                .collect(),
            normalization_mode: self.normalization_mode,
            dim_reduction_config: self.dim_reduction_config,
            concatenated_dimensions: self.concatenated_dimensions,
            normalization_mean: self.normalization_mean.clone(),
            normalization_std: self.normalization_std.clone(),
            dim_reduction_transformer,
        })
    }

    /// Restores an extractor from a serialized, validated model state.
    pub fn from_json(data: SerializedFeatureExtractor) -> Result<Self, FeatureExtractorError> {
        if data.feature_types.is_empty() {
            return Err(FeatureExtractorError::InvalidConfiguration(
                "featureTypes must contain at least one feature type".to_owned(),
            ));
        }
        let SerializedFeatureExtractor {
            feature_types: serialized_feature_types,
            normalization_mode,
            dim_reduction_config,
            concatenated_dimensions,
            normalization_mean,
            normalization_std,
            dim_reduction_transformer: serialized_transformer,
        } = data;

        let feature_types = serialized_feature_types
            .iter()
            .map(|value| {
                FeatureId::from_str(value).map_err(|_| {
                    FeatureExtractorError::InvalidConfiguration(format!(
                        "Unknown feature type: {value}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if concatenated_dimensions == 0 || dim_reduction_config.components == 0 {
            return Err(FeatureExtractorError::InvalidConfiguration(
                "Serialized feature extractor dimensions must be positive".to_owned(),
            ));
        }
        let expected_dimensions = get_expected_concatenated_dimensions(&feature_types)?;
        if concatenated_dimensions != expected_dimensions {
            return Err(FeatureExtractorError::InvalidConfiguration(format!(
                "concatenatedDimensions is {concatenated_dimensions}, expected {expected_dimensions} for the configured feature types"
            )));
        }

        match normalization_mode {
            NormalizationMode::ZScore | NormalizationMode::Calibrated => {
                let mean = normalization_mean.as_ref().ok_or_else(|| {
                    FeatureExtractorError::InvalidConfiguration(
                        "z_score normalization requires normalizationMean".to_owned(),
                    )
                })?;
                let std = normalization_std.as_ref().ok_or_else(|| {
                    FeatureExtractorError::InvalidConfiguration(
                        "z_score normalization requires normalizationStd".to_owned(),
                    )
                })?;
                if mean.len() != concatenated_dimensions || std.len() != concatenated_dimensions {
                    return Err(FeatureExtractorError::InvalidConfiguration(
                        "z_score normalization vectors must match concatenatedDimensions"
                            .to_owned(),
                    ));
                }
                if mean.iter().any(|value| !value.is_finite())
                    || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(FeatureExtractorError::InvalidConfiguration(
                        "z_score normalization statistics must be finite and standard deviations must be positive".to_owned(),
                    ));
                }
            }
            NormalizationMode::None | NormalizationMode::Layer => {
                if normalization_mean.is_some() || normalization_std.is_some() {
                    return Err(FeatureExtractorError::InvalidConfiguration(
                        "normalization statistics are only valid for z_score normalization"
                            .to_owned(),
                    ));
                }
            }
        }

        let dim_reduction_transformer = match (dim_reduction_config.method, serialized_transformer)
        {
            (
                DimensionalityReductionMethod::RandomProjection,
                Some(DimReductionTransformer::RandomProjection(state)),
            ) => {
                if state.n_features != concatenated_dimensions
                    || state.n_components != dim_reduction_config.components
                {
                    return Err(FeatureExtractorError::InvalidConfiguration(
                        "random projection state dimensions do not match extractor configuration"
                            .to_owned(),
                    ));
                }
                Some(DimensionalityReducer::RandomProjection(
                    RandomProjectionTransformer::from_json(state)?,
                ))
            }
            (DimensionalityReductionMethod::Pca, Some(DimReductionTransformer::Pca(state))) => {
                if state.n_features != concatenated_dimensions
                    || state.n_components != dim_reduction_config.components
                {
                    return Err(FeatureExtractorError::InvalidConfiguration(
                        "PCA state dimensions do not match extractor configuration".to_owned(),
                    ));
                }
                Some(DimensionalityReducer::Pca(PcaTransformer::from_json(
                    state,
                )?))
            }
            (DimensionalityReductionMethod::None, None) => None,
            (DimensionalityReductionMethod::PlsDa, _)
            | (DimensionalityReductionMethod::LinearNca, _) => {
                return Err(FeatureExtractorError::InvalidConfiguration(
                    "Unsupported dimensionality reduction method".to_owned(),
                ));
            }
            _ => {
                return Err(FeatureExtractorError::InvalidConfiguration(
                    "dimensionality reduction state does not match its configured method"
                        .to_owned(),
                ));
            }
        };

        Ok(Self {
            feature_types,
            normalization_mode,
            dim_reduction_config,
            unlabeled_samples: Vec::new(),
            concatenated_dimensions,
            normalization_mean,
            normalization_std,
            dim_reduction_transformer,
        })
    }

    /// Releases cached numeric buffers while retaining serialized metadata.
    pub fn dispose(&mut self) {
        if let Some(transformer) = &mut self.dim_reduction_transformer {
            match transformer {
                DimensionalityReducer::RandomProjection(value) => value.dispose(),
                DimensionalityReducer::Pca(value) => value.dispose(),
            }
        }
    }

    pub fn get_output_dimensions(&self) -> usize {
        match &self.dim_reduction_transformer {
            Some(DimensionalityReducer::RandomProjection(value)) => value.output_dimension(),
            Some(DimensionalityReducer::Pca(value)) => value.get_output_dimension(),
            None => self.concatenated_dimensions,
        }
    }

    pub fn is_fitted(&self) -> bool {
        self.concatenated_dimensions > 0
    }

    pub fn feature_types(&self) -> &[FeatureId] {
        &self.feature_types
    }

    pub fn normalization_mode(&self) -> NormalizationMode {
        self.normalization_mode
    }
}

fn validate_feature_values(features: &[Vec<f32>]) -> Result<(), FeatureExtractorError> {
    for (row, values) in features.iter().enumerate() {
        if let Some(column) = values.iter().position(|value| !value.is_finite()) {
            return Err(FeatureExtractorError::InvalidConfiguration(format!(
                "Feature value at row {row}, column {column} must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_ragged(features: &[Vec<f32>]) -> Result<(), FeatureExtractorError> {
    let Some(first) = features.first() else {
        return Err(FeatureExtractorError::EmptyFeatureArray);
    };
    let expected = first.len();
    for (row, values) in features.iter().enumerate() {
        if values.len() != expected {
            return Err(FeatureExtractorError::RaggedFeatures {
                row,
                expected,
                actual: values.len(),
            });
        }
    }
    Ok(())
}

/// Computes per-dimension mean and standard deviation, applying the stability
/// guard that collapses a near-zero deviation to 1.0. The statistics source may
/// be a subset of the rows that are ultimately transformed (calibrated mode).
fn z_score_stats(features: &[Vec<f32>]) -> Result<(Vec<f64>, Vec<f64>), FeatureExtractorError> {
    let first = features
        .first()
        .ok_or(FeatureExtractorError::EmptyFeatureArray)?;
    let n_features = first.len();
    if n_features == 0 {
        return Err(FeatureExtractorError::EmptyFeatureArray);
    }
    validate_ragged(features)?;
    validate_feature_values(features)?;

    let n_samples = features.len() as f64;
    let mut mean = vec![0.0; n_features];
    for row in features {
        for (column, value) in row.iter().enumerate() {
            mean[column] += f64::from(*value);
        }
    }
    for value in &mut mean {
        *value /= n_samples;
    }

    let mut variance = vec![0.0; n_features];
    for row in features {
        for (column, value) in row.iter().enumerate() {
            let difference = f64::from(*value) - mean[column];
            variance[column] += difference * difference;
        }
    }

    let mut std = Vec::with_capacity(n_features);
    for value in variance {
        let standard_deviation = (value / n_samples).sqrt();
        std.push(if standard_deviation < f64::from(EPSILON_STABLE) {
            1.0
        } else {
            standard_deviation
        });
    }

    Ok((mean, std))
}

fn apply_z_score_stats(features: &[Vec<f32>], mean: &[f64], std: &[f64]) -> Vec<Vec<f32>> {
    features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, value)| ((f64::from(*value) - mean[column]) / std[column]) as f32)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn normalize_z_score(features: &[Vec<f32>]) -> Result<ZScoreNormalization, FeatureExtractorError> {
    let (mean, std) = z_score_stats(features)?;
    let normalized = apply_z_score_stats(features, &mean, &std);
    Ok((normalized, mean, std))
}

fn apply_z_score_norm(
    features: &[f32],
    mean: &[f64],
    std: &[f64],
) -> Result<Vec<f32>, FeatureExtractorError> {
    if features.len() != mean.len() || features.len() != std.len() {
        return Err(FeatureExtractorError::DimensionMismatch {
            expected: mean.len(),
            actual: features.len(),
        });
    }
    if features.iter().any(|value| !value.is_finite())
        || mean.iter().any(|value| !value.is_finite())
        || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(FeatureExtractorError::InvalidConfiguration(
            "z_score normalization inputs and statistics must be finite with positive standard deviations"
                .to_owned(),
        ));
    }
    let normalized = features
        .iter()
        .enumerate()
        .map(|(index, value)| ((f64::from(*value) - mean[index]) / std[index]) as f32)
        .collect::<Vec<_>>();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(FeatureExtractorError::InvalidConfiguration(
            "z_score normalization produced a non-finite value".to_owned(),
        ));
    }
    Ok(normalized)
}

fn log_info(message: &str) {
    eprintln!("[training] {message}");
}
