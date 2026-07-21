use std::fmt;

use crate::ported::constants::{
    EPSILON_STABLE, RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_BACKBONE_RAW_DIMS,
    RTMPOSE_BACKBONE_SHAPE, RTMPOSE_GAU_POOLED_DIMS, RTMPOSE_GAU_RAW_DIMS, RTMPOSE_GAU_SHAPE,
};

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

/// Extracts the flattened backbone tensor without changing its source order.
pub fn extract_backbone_raw(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    validate_input("backbone", values, RTMPOSE_BACKBONE_RAW_DIMS)?;
    Ok(values.to_vec())
}

/// Extracts the flattened GAU tensor without changing its source order.
pub fn extract_gau_raw(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    validate_input("gau", values, RTMPOSE_GAU_RAW_DIMS)?;
    Ok(values.to_vec())
}

/// Pools backbone features over height and width.
pub fn pool_backbone_features(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_backbone(
        values,
        RTMPOSE_BACKBONE_SHAPE[1],
        RTMPOSE_BACKBONE_SHAPE[2],
        RTMPOSE_BACKBONE_SHAPE[3],
        PoolingKind::Mean,
    )
}

/// Pools backbone features with a spatial maximum.
pub fn pool_backbone_features_max(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_backbone(
        values,
        RTMPOSE_BACKBONE_SHAPE[1],
        RTMPOSE_BACKBONE_SHAPE[2],
        RTMPOSE_BACKBONE_SHAPE[3],
        PoolingKind::Max,
    )
}

/// Pools backbone features with a spatial population standard deviation.
pub fn pool_backbone_features_std(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_backbone(
        values,
        RTMPOSE_BACKBONE_SHAPE[1],
        RTMPOSE_BACKBONE_SHAPE[2],
        RTMPOSE_BACKBONE_SHAPE[3],
        PoolingKind::PopulationStd,
    )
}

/// Pools GAU features over the keypoint dimension.
pub fn pool_gau_features(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_gau(
        values,
        RTMPOSE_GAU_SHAPE[1],
        RTMPOSE_GAU_SHAPE[2],
        PoolingKind::Mean,
    )
}

/// Pools GAU features with a keypoint maximum.
pub fn pool_gau_features_max(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_gau(
        values,
        RTMPOSE_GAU_SHAPE[1],
        RTMPOSE_GAU_SHAPE[2],
        PoolingKind::Max,
    )
}

/// Pools GAU features with a keypoint population standard deviation.
pub fn pool_gau_features_std(values: &[f32]) -> Result<Vec<f32>, PoolingError> {
    pool_gau(
        values,
        RTMPOSE_GAU_SHAPE[1],
        RTMPOSE_GAU_SHAPE[2],
        PoolingKind::PopulationStd,
    )
}

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

/// Pools an NCHW-style backbone tensor with configurable dimensions.
pub fn pool_backbone(
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    kind: PoolingKind,
) -> Result<Vec<f32>, PoolingError> {
    validate_dimension("backbone", "channels", channels)?;
    validate_dimension("backbone", "height", height)?;
    validate_dimension("backbone", "width", width)?;

    let spatial = height
        .checked_mul(width)
        .ok_or(PoolingError::DimensionOverflow { tensor: "backbone" })?;
    let expected = channels
        .checked_mul(spatial)
        .ok_or(PoolingError::DimensionOverflow { tensor: "backbone" })?;
    validate_input("backbone", values, expected)?;

    let mut output = Vec::with_capacity(channels);
    for channel in values.chunks_exact(spatial) {
        output.push(pool_contiguous(channel, kind));
    }
    Ok(output)
}

/// Pools a keypoint-major GAU tensor with configurable dimensions.
pub fn pool_gau(
    values: &[f32],
    keypoints: usize,
    features: usize,
    kind: PoolingKind,
) -> Result<Vec<f32>, PoolingError> {
    validate_dimension("gau", "keypoints", keypoints)?;
    validate_dimension("gau", "features", features)?;

    let expected = keypoints
        .checked_mul(features)
        .ok_or(PoolingError::DimensionOverflow { tensor: "gau" })?;
    validate_input("gau", values, expected)?;

    let mut output = Vec::with_capacity(features);
    for feature in 0..features {
        let mut lane = Vec::with_capacity(keypoints);
        for keypoint in 0..keypoints {
            lane.push(values[keypoint * features + feature]);
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

fn validate_dimension(
    tensor: &'static str,
    dimension: &'static str,
    value: usize,
) -> Result<(), PoolingError> {
    if value == 0 {
        Err(PoolingError::ZeroDimension { tensor, dimension })
    } else {
        Ok(())
    }
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

const _: () = {
    assert!(RTMPOSE_BACKBONE_POOLED_DIMS == RTMPOSE_BACKBONE_SHAPE[1]);
    assert!(RTMPOSE_GAU_POOLED_DIMS == RTMPOSE_GAU_SHAPE[2]);
};
