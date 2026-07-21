//! Feature validation before tensor operations.
//!
//! This preserves the source utility's validation boundary: the feature batch
//! must contain at least one row. Rust slices cannot contain an `undefined` or
//! `null` row, so a non-empty `&[Vec<f32>]` already satisfies the source's
//! first-row-defined check.

use std::fmt;

/// Errors returned when a feature batch cannot be used for the requested
/// operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureValidationError {
    /// The feature batch contains no rows.
    EmptyFeatures { context: String },
}

impl fmt::Display for FeatureValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFeatures { context } => {
                write!(formatter, "Cannot {context}: features array is empty")
            }
        }
    }
}

impl std::error::Error for FeatureValidationError {}

/// Validates a feature batch before tensor operations.
///
/// Empty rows are intentionally accepted because the TypeScript source only
/// checks that the batch exists and that its first row is defined.
pub fn validate_features(
    features: &[Vec<f32>],
    context: &str,
) -> Result<(), FeatureValidationError> {
    if features.is_empty() {
        return Err(FeatureValidationError::EmptyFeatures {
            context: context.to_owned(),
        });
    }

    Ok(())
}
