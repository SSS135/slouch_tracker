//! Layer normalization and L2 normalization for feature vectors.
//!
//! This is the Rust equivalent of the TensorFlow.js implementation in
//! `src/services/ml/layerNorm.ts`. Native reductions use `f64` accumulation
//! before narrowing the output to reproduce the frozen TensorFlow.js oracle
//! results and avoid order-sensitive `f32` drift.

use std::fmt;

use crate::ported::constants::EPSILON_STABLE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNormError {
    EmptyTensor,
    EmptySample {
        index: usize,
    },
    RaggedBatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidEpsilon,
    NonFiniteInput {
        sample: usize,
        index: usize,
    },
}

impl fmt::Display for LayerNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTensor => formatter.write_str("tensor must contain at least one value"),
            Self::EmptySample { index } => {
                write!(
                    formatter,
                    "sample at index {index} must contain at least one value"
                )
            }
            Self::RaggedBatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "sample at index {index} has {actual} values, expected {expected}"
            ),
            Self::InvalidEpsilon => formatter.write_str("epsilon must be finite and nonnegative"),
            Self::NonFiniteInput { sample, index } => {
                write!(
                    formatter,
                    "sample {sample} value at index {index} is not finite"
                )
            }
        }
    }
}

impl std::error::Error for LayerNormError {}

/// Normalizes one tensor flattened in its source iteration order.
///
/// TensorFlow.js computes one global mean and variance for `layerNormTF`, so a
/// flattened slice preserves its observable arithmetic for any tensor shape.
pub fn layer_norm_tf(values: &[f32]) -> Result<Vec<f32>, LayerNormError> {
    layer_norm_tf_with_epsilon(values, EPSILON_STABLE)
}

/// Normalizes one tensor with an explicit numerical-stability epsilon.
pub fn layer_norm_tf_with_epsilon(
    values: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, LayerNormError> {
    validate_epsilon(epsilon)?;
    validate_tensor(values)?;
    Ok(normalize_f32(values, epsilon))
}

/// Applies layer normalization independently to every feature vector.
pub fn apply_layer_norm_batch(samples: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, LayerNormError> {
    apply_layer_norm_batch_with_epsilon(samples, EPSILON_STABLE)
}

/// Applies layer normalization independently to every feature vector with an
/// explicit numerical-stability epsilon.
pub fn apply_layer_norm_batch_with_epsilon(
    samples: &[Vec<f32>],
    epsilon: f32,
) -> Result<Vec<Vec<f32>>, LayerNormError> {
    validate_epsilon(epsilon)?;
    validate_batch(samples)?;
    Ok(samples
        .iter()
        .map(|sample| normalize_f32(sample, epsilon))
        .collect())
}

/// Applies row-wise L2 normalization to a batch of feature vectors.
///
/// A zero or near-zero row is returned unchanged, matching the TensorFlow.js
/// `tf.where(norm < EPSILON_STABLE, 1, norm)` guard.
pub fn apply_l2_norm_batch(samples: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, LayerNormError> {
    validate_batch(samples)?;
    Ok(samples
        .iter()
        .map(|sample| l2_normalize_f32(sample))
        .collect())
}

/// L2-normalizes one feature vector, returning near-zero vectors unchanged.
pub fn l2_normalize_single(values: &[f32]) -> Result<Vec<f32>, LayerNormError> {
    validate_tensor(values)?;
    Ok(l2_normalize_f64(values))
}

fn normalize_f32(values: &[f32], epsilon: f32) -> Vec<f32> {
    // TensorFlow.js' backend reduction order is captured by the committed
    // compatibility oracle. f64 accumulation followed by one narrowing step
    // matches those source-generated values; sequential Rust f32 summation does not.
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let difference = f64::from(*value) - mean;
            difference * difference
        })
        .sum::<f64>()
        / values.len() as f64;
    let scale = (variance + f64::from(epsilon)).sqrt();

    values
        .iter()
        .map(|value| ((f64::from(*value) - mean) / scale) as f32)
        .collect()
}

fn l2_normalize_f32(values: &[f32]) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|value| *value * *value)
        .sum::<f32>()
        .sqrt();
    if norm < EPSILON_STABLE {
        return values.to_vec();
    }

    values.iter().map(|value| *value / norm).collect()
}

fn l2_normalize_f64(values: &[f32]) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm < f64::from(EPSILON_STABLE) {
        return values.to_vec();
    }

    values
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect()
}

fn validate_epsilon(epsilon: f32) -> Result<(), LayerNormError> {
    if epsilon.is_finite() && epsilon >= 0.0 {
        Ok(())
    } else {
        Err(LayerNormError::InvalidEpsilon)
    }
}

fn validate_tensor(values: &[f32]) -> Result<(), LayerNormError> {
    if values.is_empty() {
        return Err(LayerNormError::EmptyTensor);
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(LayerNormError::NonFiniteInput { sample: 0, index });
    }
    Ok(())
}

fn validate_batch(samples: &[Vec<f32>]) -> Result<(), LayerNormError> {
    if samples.is_empty() {
        return Ok(());
    }

    for (index, sample) in samples.iter().enumerate() {
        if sample.is_empty() {
            return Err(LayerNormError::EmptySample { index });
        }
    }

    let expected = samples[0].len();
    for (index, sample) in samples.iter().enumerate() {
        if sample.len() != expected {
            return Err(LayerNormError::RaggedBatch {
                index,
                expected,
                actual: sample.len(),
            });
        }
        if let Some(value_index) = sample.iter().position(|value| !value.is_finite()) {
            return Err(LayerNormError::NonFiniteInput {
                sample: index,
                index: value_index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_and_batch_empty_errors_are_distinct() {
        assert_eq!(layer_norm_tf(&[]), Err(LayerNormError::EmptyTensor));
        assert_eq!(
            apply_layer_norm_batch(&[vec![]]),
            Err(LayerNormError::EmptySample { index: 0 }),
        );
        assert_eq!(
            apply_l2_norm_batch(&[vec![1.0], vec![]]),
            Err(LayerNormError::EmptySample { index: 1 }),
        );
    }

    #[test]
    fn layer_norm_and_single_l2_match_source_generated_extreme_values() {
        let cancellation = [1.0e20, -1.0e20, 1.0, -1.0];
        let normalized = layer_norm_tf(&cancellation).unwrap();
        assert!((normalized[0] - std::f32::consts::SQRT_2).abs() < 1e-6);
        assert!((normalized[1] + std::f32::consts::SQRT_2).abs() < 1e-6);
        assert!(normalized[2].is_sign_positive() && normalized[2] > 0.0);
        assert!(normalized[3].is_sign_negative() && normalized[3] < 0.0);

        let extremes = [f32::MAX, -f32::MAX];
        assert_eq!(
            apply_l2_norm_batch(&[extremes.to_vec()]).unwrap()[0],
            vec![0.0, -0.0]
        );
        let single = l2_normalize_single(&extremes).unwrap();
        assert!(single[0].is_finite() && single[0] > 0.0);
        assert!(single[1].is_finite() && single[1] < 0.0);
    }
}
