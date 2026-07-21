use std::fmt;

use super::constants::{RTMDET_EXTRACTED_DIMS, RTMDET_SHAPE};
use super::rtmpose_features::{
    pool_features_max, pool_features_mean, pool_features_std, PoolingError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtmDetFeaturesError {
    Pooling(PoolingError),
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for RtmDetFeaturesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pooling(error) => error.fmt(formatter),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} extracted dimensions, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for RtmDetFeaturesError {}

impl From<PoolingError> for RtmDetFeaturesError {
    fn from(error: PoolingError) -> Self {
        Self::Pooling(error)
    }
}

/// Pools and concatenates RTMDet cls_p5 and reg_p5 features.
///
/// The output order matches the TypeScript TensorFlow.js implementation:
/// cls average, cls population standard deviation, cls maximum, followed by
/// the corresponding three reg vectors. No normalization is applied here.
pub fn extract_rtm_det_features(
    raw_cls_p5: &[f32],
    raw_reg_p5: &[f32],
) -> Result<Vec<f32>, RtmDetFeaturesError> {
    let shape = [
        RTMDET_SHAPE.batch,
        RTMDET_SHAPE.channels,
        RTMDET_SHAPE.height,
        RTMDET_SHAPE.width,
    ];
    let pooling_axes = [2, 3];

    let cls_avg = pool_features_mean(raw_cls_p5, &shape, &pooling_axes)?;
    let cls_std = pool_features_std(raw_cls_p5, &shape, &pooling_axes)?;
    let cls_max = pool_features_max(raw_cls_p5, &shape, &pooling_axes)?;
    let reg_avg = pool_features_mean(raw_reg_p5, &shape, &pooling_axes)?;
    let reg_std = pool_features_std(raw_reg_p5, &shape, &pooling_axes)?;
    let reg_max = pool_features_max(raw_reg_p5, &shape, &pooling_axes)?;

    let mut extracted = Vec::with_capacity(RTMDET_EXTRACTED_DIMS);
    extracted.extend_from_slice(&cls_avg);
    extracted.extend_from_slice(&cls_std);
    extracted.extend_from_slice(&cls_max);
    extracted.extend_from_slice(&reg_avg);
    extracted.extend_from_slice(&reg_std);
    extracted.extend_from_slice(&reg_max);

    if extracted.len() != RTMDET_EXTRACTED_DIMS {
        return Err(RtmDetFeaturesError::DimensionMismatch {
            expected: RTMDET_EXTRACTED_DIMS,
            actual: extracted.len(),
        });
    }

    Ok(extracted)
}

const _: () = {
    assert!(RTMDET_EXTRACTED_DIMS == 6 * RTMDET_SHAPE.channels);
};
