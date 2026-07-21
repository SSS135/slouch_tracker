//! Generic spatial pooling over flattened row-major tensors: average, spatial
//! maximum, and population standard-deviation reductions over an arbitrary set of
//! axes. This is model-agnostic feature-extraction math — RTMDet extracted features
//! pool their `cls_p5`/`reg_p5` maps through it, and pooled backbone embedding
//! features reuse the same reductions.

use std::fmt;

use super::constants::EPSILON_STABLE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingKind {
    Mean,
    Max,
    PopulationStd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolingError {
    ZeroDimension {
        tensor: &'static str,
        dimension: &'static str,
    },
    DimensionOverflow {
        tensor: &'static str,
    },
    Shape {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidShape {
        tensor: &'static str,
    },
    AxisOutOfBounds {
        tensor: &'static str,
        axis: usize,
        rank: usize,
    },
    DuplicateAxis {
        tensor: &'static str,
        axis: usize,
    },
    NonFiniteInput {
        tensor: &'static str,
        index: usize,
    },
}

impl fmt::Display for PoolingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { tensor, dimension } => {
                write!(
                    formatter,
                    "{tensor} tensor {dimension} dimension must be positive"
                )
            }
            Self::DimensionOverflow { tensor } => {
                write!(formatter, "{tensor} tensor dimensions overflow usize")
            }
            Self::Shape {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "{tensor} tensor has {actual} values, expected {expected}"
            ),
            Self::InvalidShape { tensor } => {
                write!(formatter, "{tensor} tensor shape is invalid")
            }
            Self::AxisOutOfBounds { tensor, axis, rank } => write!(
                formatter,
                "{tensor} pooling axis {axis} is out of bounds for rank {rank}"
            ),
            Self::DuplicateAxis { tensor, axis } => {
                write!(formatter, "{tensor} pooling axis {axis} is duplicated")
            }
            Self::NonFiniteInput { tensor, index } => {
                write!(
                    formatter,
                    "{tensor} tensor value at index {index} is not finite"
                )
            }
        }
    }
}

impl std::error::Error for PoolingError {}

/// Generic mean pooling equivalent to TensorFlow.js `tf.mean` followed by flattening.
pub fn pool_features_mean(
    raw: &[f32],
    shape: &[usize],
    pooling_axes: &[usize],
) -> Result<Vec<f32>, PoolingError> {
    pool_features(raw, shape, pooling_axes, PoolingKind::Mean)
}

/// Generic maximum pooling equivalent to TensorFlow.js `tf.max` followed by flattening.
pub fn pool_features_max(
    raw: &[f32],
    shape: &[usize],
    pooling_axes: &[usize],
) -> Result<Vec<f32>, PoolingError> {
    pool_features(raw, shape, pooling_axes, PoolingKind::Max)
}

/// Generic population standard-deviation pooling equivalent to TensorFlow.js `tf.moments`.
pub fn pool_features_std(
    raw: &[f32],
    shape: &[usize],
    pooling_axes: &[usize],
) -> Result<Vec<f32>, PoolingError> {
    pool_features(raw, shape, pooling_axes, PoolingKind::PopulationStd)
}

/// Pools a flattened row-major tensor over the requested axes.
pub fn pool_features(
    raw: &[f32],
    shape: &[usize],
    pooling_axes: &[usize],
    kind: PoolingKind,
) -> Result<Vec<f32>, PoolingError> {
    let (expected, reduced_axes, kept_axes) = validate_generic_shape(shape, pooling_axes)?;
    validate_input("features", raw, expected)?;

    let reduced_len = product_of_axes(shape, &reduced_axes, "features")?;
    let output_len = product_of_axes(shape, &kept_axes, "features")?;
    let mut output = Vec::with_capacity(output_len);
    let mut coordinates = vec![0; shape.len()];

    for output_index in 0..output_len {
        decode_index(output_index, &kept_axes, shape, &mut coordinates);
        let mut lane = Vec::with_capacity(reduced_len);
        for reduced_index in 0..reduced_len {
            decode_index(reduced_index, &reduced_axes, shape, &mut coordinates);
            lane.push(raw[flat_index(&coordinates, shape)]);
        }
        output.push(pool_contiguous(&lane, kind));
    }

    Ok(output)
}

fn pool_contiguous(values: &[f32], kind: PoolingKind) -> f32 {
    match kind {
        PoolingKind::Mean => mean(values),
        PoolingKind::Max => values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |maximum, value| maximum.max(value)),
        PoolingKind::PopulationStd => {
            let average =
                values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| {
                    let difference = f64::from(*value) - average;
                    difference * difference
                })
                .sum::<f64>()
                / values.len() as f64;
            (variance + f64::from(EPSILON_STABLE)).sqrt() as f32
        }
    }
}

fn mean(values: &[f32]) -> f32 {
    (values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64) as f32
}

fn validate_generic_shape(
    shape: &[usize],
    pooling_axes: &[usize],
) -> Result<(usize, Vec<usize>, Vec<usize>), PoolingError> {
    if shape.is_empty() || pooling_axes.is_empty() {
        return Err(PoolingError::InvalidShape { tensor: "features" });
    }

    for (index, dimension) in shape.iter().enumerate() {
        if *dimension == 0 {
            return Err(PoolingError::ZeroDimension {
                tensor: "features",
                dimension: if index == 0 { "first" } else { "shape" },
            });
        }
    }

    let mut reduced_axes = Vec::with_capacity(pooling_axes.len());
    for axis in pooling_axes {
        if *axis >= shape.len() {
            return Err(PoolingError::AxisOutOfBounds {
                tensor: "features",
                axis: *axis,
                rank: shape.len(),
            });
        }
        if reduced_axes.contains(axis) {
            return Err(PoolingError::DuplicateAxis {
                tensor: "features",
                axis: *axis,
            });
        }
        reduced_axes.push(*axis);
    }

    let expected = shape.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(PoolingError::DimensionOverflow { tensor: "features" })
    })?;
    let kept_axes = (0..shape.len())
        .filter(|axis| !reduced_axes.contains(axis))
        .collect();
    Ok((expected, reduced_axes, kept_axes))
}

fn product_of_axes(
    shape: &[usize],
    axes: &[usize],
    tensor: &'static str,
) -> Result<usize, PoolingError> {
    axes.iter().try_fold(1usize, |product, axis| {
        product
            .checked_mul(shape[*axis])
            .ok_or(PoolingError::DimensionOverflow { tensor })
    })
}

fn decode_index(index: usize, axes: &[usize], shape: &[usize], coordinates: &mut [usize]) {
    let mut remainder = index;
    for axis in axes.iter().rev() {
        coordinates[*axis] = remainder % shape[*axis];
        remainder /= shape[*axis];
    }
}

fn flat_index(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |index, (coordinate, dimension)| {
            index * dimension + coordinate
        })
}

fn validate_input(
    tensor: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), PoolingError> {
    if values.len() != expected {
        return Err(PoolingError::Shape {
            tensor,
            expected,
            actual: values.len(),
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(PoolingError::NonFiniteInput { tensor, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs()
    }

    #[test]
    fn mean_pools_each_row_of_a_two_dimensional_tensor() {
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = pool_features_mean(&raw, &[2, 3], &[1]).unwrap();
        assert_eq!(pooled.len(), 2);
        assert!(close(pooled[0], 2.0));
        assert!(close(pooled[1], 5.0));
    }

    #[test]
    fn max_pools_each_row_and_preserves_exact_bits() {
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = pool_features_max(&raw, &[2, 3], &[1]).unwrap();
        assert_eq!(pooled[0].to_bits(), 3.0_f32.to_bits());
        assert_eq!(pooled[1].to_bits(), 6.0_f32.to_bits());
    }

    #[test]
    fn population_std_matches_the_manual_reduction() {
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pooled = pool_features_std(&raw, &[2, 3], &[1]).unwrap();
        // Population variance of {1,2,3} is 2/3; std = sqrt(2/3 + eps).
        let expected = ((2.0_f64 / 3.0) + f64::from(EPSILON_STABLE)).sqrt() as f32;
        assert!(close(pooled[0], expected));
        assert!(close(pooled[1], expected));
    }

    #[test]
    fn constant_input_std_collapses_to_sqrt_epsilon() {
        let raw = vec![3.7_f32; 12];
        let pooled = pool_features_std(&raw, &[3, 4], &[1]).unwrap();
        let expected = EPSILON_STABLE.sqrt();
        assert!(pooled.iter().all(|value| (*value - expected).abs() < 1e-6));
    }

    #[test]
    fn nchw_channel_pooling_reduces_spatial_axes() {
        // Shape [batch=1, channels=2, h=2, w=2]; pool the spatial axes [2, 3].
        let raw = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mean = pool_features_mean(&raw, &[1, 2, 2, 2], &[2, 3]).unwrap();
        assert_eq!(mean.len(), 2);
        assert!(close(mean[0], 2.5));
        assert!(close(mean[1], 25.0));
        let max = pool_features_max(&raw, &[1, 2, 2, 2], &[2, 3]).unwrap();
        assert_eq!(max[0].to_bits(), 4.0_f32.to_bits());
        assert_eq!(max[1].to_bits(), 40.0_f32.to_bits());
    }

    #[test]
    fn rejects_length_mismatch() {
        assert_eq!(
            pool_features_mean(&[1.0, 2.0], &[2, 3], &[1]),
            Err(PoolingError::Shape {
                tensor: "features",
                expected: 6,
                actual: 2,
            })
        );
    }

    #[test]
    fn rejects_non_finite_values_at_their_index() {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let raw = [1.0, non_finite, 3.0, 4.0, 5.0, 6.0];
            assert_eq!(
                pool_features_mean(&raw, &[2, 3], &[1]),
                Err(PoolingError::NonFiniteInput {
                    tensor: "features",
                    index: 1,
                })
            );
        }
    }

    #[test]
    fn rejects_zero_dimension() {
        assert_eq!(
            pool_features_mean(&[], &[2, 0], &[1]),
            Err(PoolingError::ZeroDimension {
                tensor: "features",
                dimension: "shape",
            })
        );
    }

    #[test]
    fn rejects_out_of_bounds_and_duplicate_axes() {
        assert_eq!(
            pool_features_mean(&[1.0, 2.0], &[2], &[1]),
            Err(PoolingError::AxisOutOfBounds {
                tensor: "features",
                axis: 1,
                rank: 1,
            })
        );
        assert_eq!(
            pool_features_mean(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[1, 1]),
            Err(PoolingError::DuplicateAxis {
                tensor: "features",
                axis: 1,
            })
        );
    }

    #[test]
    fn rejects_empty_shape_or_axes() {
        assert_eq!(
            pool_features_mean(&[], &[], &[0]),
            Err(PoolingError::InvalidShape { tensor: "features" })
        );
        assert_eq!(
            pool_features_mean(&[1.0], &[1], &[]),
            Err(PoolingError::InvalidShape { tensor: "features" })
        );
    }
}
