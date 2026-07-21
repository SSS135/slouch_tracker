//! Common classifier contract for the machine-learning pipeline.
//!
//! This is a mechanical Rust port of `src/services/ml/baseClassifier.ts`.
//! Feature vectors use `f32`, while probabilities and serialized model values
//! use the Rust equivalents defined by the porting contract.

use std::fmt;

use super::types::SerializedClassifierState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierError {
    EmptyDataset,
    LengthMismatch {
        features: usize,
        labels: usize,
    },
    MissingClass,
    InvalidEpsilon,
    EmptyFeatureVector,
    RaggedFeatures {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteFeature {
        row: usize,
        column: usize,
    },
    PredictionDimension {
        expected: usize,
        actual: usize,
    },
    Untrained,
    InvalidState(String),
}

impl fmt::Display for ClassifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => formatter.write_str("cannot train on empty dataset"),
            Self::LengthMismatch { features, labels } => {
                write!(
                    formatter,
                    "length mismatch: {features} features, {labels} labels"
                )
            }
            Self::MissingClass => formatter.write_str("both classes must have at least one sample"),
            Self::InvalidEpsilon => {
                formatter.write_str("variance smoothing must be finite and positive")
            }
            Self::EmptyFeatureVector => formatter.write_str("feature vectors must not be empty"),
            Self::RaggedFeatures {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "feature row {row} has length {actual}, expected {expected}"
            ),
            Self::NonFiniteFeature { row, column } => {
                write!(
                    formatter,
                    "feature at row {row}, column {column} is not finite"
                )
            }
            Self::PredictionDimension { expected, actual } => write!(
                formatter,
                "prediction has {actual} features, expected {expected}"
            ),
            Self::Untrained => formatter.write_str("model not trained"),
            Self::InvalidState(message) => write!(formatter, "invalid serialized state: {message}"),
        }
    }
}

impl std::error::Error for ClassifierError {}

/// Shared interface implemented by every posture classifier.
pub trait BaseClassifier: Send {
    fn classifier_id(&self) -> &'static str;
    fn train(&mut self, features: &[Vec<f32>], labels: &[i32]) -> Result<(), ClassifierError>;
    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError>;
    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError>;
    fn dispose(&mut self);
}
