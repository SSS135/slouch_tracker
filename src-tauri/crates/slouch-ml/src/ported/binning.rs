//! Soft binning with fixed bin edges using Gaussian RBF kernels.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinningError {
    TooFewEdges,
    NonFiniteValue,
    InvalidConfidence,
    NonFiniteEdge { index: usize },
    InvalidEdgeSpacing { index: usize },
}

impl fmt::Display for BinningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewEdges => write!(formatter, "edges must have at least 2 elements"),
            Self::NonFiniteValue => write!(formatter, "value must be a finite number"),
            Self::InvalidConfidence => write!(
                formatter,
                "confidence must be a finite number between 0 and 1"
            ),
            Self::NonFiniteEdge { index } => {
                write!(formatter, "edge at index {index} must be a finite number")
            }
            Self::InvalidEdgeSpacing { index } => write!(
                formatter,
                "edge spacing at index {index} must produce a finite, nonzero sigma"
            ),
        }
    }
}

impl std::error::Error for BinningError {}

/// Computes a soft histogram using each edge as a Gaussian kernel center.
///
/// The local kernel sigma is half the distance to the nearest neighboring
/// edge, allowing non-uniform edge spacing. The returned probabilities are
/// stored as `f32`, matching the source `Float32Array` output.
pub fn soft_bin_with_fixed_edges(
    value: f64,
    confidence: f64,
    edges: &[f64],
) -> Result<Vec<f32>, BinningError> {
    if edges.len() < 2 {
        return Err(BinningError::TooFewEdges);
    }
    if !value.is_finite() {
        return Err(BinningError::NonFiniteValue);
    }
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(BinningError::InvalidConfidence);
    }
    let num_bins = edges.len();
    let mut probabilities = vec![0.0_f32; num_bins];
    let mut sum = 0.0_f64;

    for index in 0..num_bins {
        let left_distance = if index > 0 {
            edges[index] - edges[index - 1]
        } else {
            f64::INFINITY
        };
        let right_distance = if index < num_bins - 1 {
            edges[index + 1] - edges[index]
        } else {
            f64::INFINITY
        };
        let local_sigma = left_distance.min(right_distance) * 0.5;

        let difference = value - edges[index];
        let weight = (-0.5 * (difference / local_sigma).powi(2)).exp() as f32;
        probabilities[index] = weight;
        sum += f64::from(weight);
    }

    if sum > 0.0 {
        for probability in &mut probabilities {
            *probability = (f64::from(*probability) / sum) as f32;
        }
    } else {
        let uniform = (1.0_f64 / num_bins as f64) as f32;
        probabilities.fill(uniform);
    }

    if confidence < 1.0 {
        let uniform = 1.0_f64 / num_bins as f64;
        for probability in &mut probabilities {
            *probability =
                (confidence * f64::from(*probability) + (1.0 - confidence) * uniform) as f32;
        }
    }

    Ok(probabilities)
}
