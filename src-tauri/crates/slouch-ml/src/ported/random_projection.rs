//! Gaussian random projection for feature vectors.
//!
//! This is a mechanical Rust port of `src/services/ml/randomProjection.ts`.
//! The serialized matrix keeps the source's `number` precision, while the
//! matrix used for projection is stored as `f32`, matching TensorFlow.js
//! tensor conversion before matrix multiplication.

use std::fmt;

use serde::{Deserialize, Serialize, Serializer};

use super::async_utils::{batch_process_async, AsyncUtilsError, BatchProcessOptions};
use super::config::TRAINING_CONFIG;

/// Serializes an integral seed (for example `42.0`) as an integer so downstream
/// consumers that model the seed as an unsigned integer accept it verbatim,
/// while non-integral or out-of-range seeds keep their floating-point form. The
/// seed itself stays an `f64` to preserve the source's JavaScript `number`
/// semantics; only the on-the-wire encoding changes.
fn serialize_integral_seed<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_finite() && value.fract() == 0.0 && *value >= 0.0 && *value <= u64::MAX as f64 {
        serializer.serialize_u64(*value as u64)
    } else {
        serializer.serialize_f64(*value)
    }
}

/// Serialized random-projection state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RandomProjectionState {
    pub projection_matrix: Vec<Vec<f64>>,
    pub n_components: usize,
    pub n_features: usize,
    #[serde(serialize_with = "serialize_integral_seed")]
    pub seed: f64,
}

/// Errors produced by random-projection fitting, transformation, or loading.
#[derive(Debug, Clone, PartialEq)]
pub enum RandomProjectionError {
    InvalidComponents,
    InvalidSeed,
    EmptyDataset,
    EmptyFeatureVector {
        row: usize,
    },
    RaggedDataset {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        row: usize,
        column: usize,
    },
    NotFitted,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    UnfittedSerialization,
    MatrixRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    MatrixRowLengthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteMatrix {
        row: usize,
        column: usize,
    },
    MatrixValueOutOfRange {
        row: usize,
        column: usize,
    },
    Async(AsyncUtilsError),
}

impl fmt::Display for RandomProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidComponents => formatter.write_str("nComponents must be positive"),
            Self::InvalidSeed => formatter.write_str("seed must be finite"),
            Self::EmptyDataset => {
                formatter.write_str("Cannot fit random projection on empty dataset")
            }
            Self::EmptyFeatureVector { row } => {
                write!(formatter, "feature vector at row {row} must not be empty")
            }
            Self::RaggedDataset {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "feature vector at row {row} has {actual} values, expected {expected}"
            ),
            Self::NonFiniteInput { row, column } => {
                write!(
                    formatter,
                    "feature value at row {row}, column {column} is not finite"
                )
            }
            Self::NotFitted => {
                formatter.write_str("Random projection must be fitted before transform")
            }
            Self::DimensionMismatch { expected, actual } => write!(
                formatter,
                "Input dimension ({actual}) does not match fitted dimension ({expected})"
            ),
            Self::UnfittedSerialization => {
                formatter.write_str("Cannot serialize unfitted random projection")
            }
            Self::MatrixRowCountMismatch { expected, actual } => write!(
                formatter,
                "Projection matrix dimension mismatch: expected {expected} rows, got {actual}"
            ),
            Self::MatrixRowLengthMismatch {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "Projection matrix row {row} has invalid length: expected {expected}, got {actual}"
            ),
            Self::NonFiniteMatrix { row, column } => write!(
                formatter,
                "projection matrix value at row {row}, column {column} is not finite"
            ),
            Self::MatrixValueOutOfRange { row, column } => write!(
                formatter,
                "projection matrix value at row {row}, column {column} cannot be represented as f32"
            ),
            Self::Async(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RandomProjectionError {}

impl From<AsyncUtilsError> for RandomProjectionError {
    fn from(error: AsyncUtilsError) -> Self {
        Self::Async(error)
    }
}

/// Gaussian random projection transformer.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomProjectionTransformer {
    projection_matrix: Option<Vec<Vec<f64>>>,
    projection_matrix_f32: Option<Vec<Vec<f32>>>,
    n_components: usize,
    pub n_features: usize,
    seed: f64,
}

impl RandomProjectionTransformer {
    /// Creates an unfitted transformer.
    pub fn new(n_components: usize, seed: f64) -> Result<Self, RandomProjectionError> {
        if n_components == 0 {
            return Err(RandomProjectionError::InvalidComponents);
        }
        if !seed.is_finite() {
            return Err(RandomProjectionError::InvalidSeed);
        }

        Ok(Self {
            projection_matrix: None,
            projection_matrix_f32: None,
            n_components,
            n_features: 0,
            seed,
        })
    }

    /// Creates an unfitted transformer using the source training seed.
    pub fn with_default_seed(n_components: usize) -> Result<Self, RandomProjectionError> {
        Self::new(n_components, TRAINING_CONFIG.random_seed as f64)
    }

    /// Fits the projection matrix to the input feature dimension.
    pub fn fit(&mut self, samples: &[Vec<f32>]) -> Result<(), RandomProjectionError> {
        validate_samples(samples)?;
        let n_features = samples[0].len();
        let matrix = generate_gaussian_random_matrix(n_features, self.n_components, self.seed)?;
        let matrix_f32 = matrix
            .iter()
            .enumerate()
            .map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .map(|(column, value)| {
                        let converted = *value as f32;
                        if converted.is_finite() {
                            Ok(converted)
                        } else {
                            Err(RandomProjectionError::MatrixValueOutOfRange { row, column })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.n_features = n_features;
        self.projection_matrix = Some(matrix);
        self.projection_matrix_f32 = Some(matrix_f32);
        Ok(())
    }

    /// Projects one feature vector.
    pub fn transform(&self, sample: &[f32]) -> Result<Vec<f32>, RandomProjectionError> {
        let Some(matrix) = self.projection_matrix_f32.as_ref() else {
            return Err(RandomProjectionError::NotFitted);
        };
        if sample.len() != self.n_features {
            return Err(RandomProjectionError::DimensionMismatch {
                expected: self.n_features,
                actual: sample.len(),
            });
        }
        if let Some(column) = sample.iter().position(|value| !value.is_finite()) {
            return Err(RandomProjectionError::NonFiniteInput { row: 0, column });
        }

        let mut result = Vec::with_capacity(self.n_components);
        for row in matrix {
            let value = row
                .iter()
                .zip(sample)
                .map(|(weight, feature)| f64::from(*weight) * f64::from(*feature))
                .sum::<f64>() as f32;
            result.push(value);
        }
        Ok(result)
    }

    /// Fits the transformer and projects all samples.
    pub fn fit_transform(
        &mut self,
        samples: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RandomProjectionError> {
        self.fit(samples)?;
        samples
            .iter()
            .map(|sample| self.transform(sample))
            .collect()
    }

    /// Fits the transformer and projects samples in batches, yielding between
    /// batches through the shared asynchronous ML helper.
    pub async fn fit_transform_async(
        &mut self,
        samples: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RandomProjectionError> {
        self.fit(samples)?;
        let transformed = batch_process_async(
            samples.to_vec(),
            |sample| self.transform(&sample),
            BatchProcessOptions::default(),
        )
        .await
        .map_err(RandomProjectionError::Async)?;
        transformed.into_iter().collect()
    }

    /// Projects multiple samples in input order.
    pub fn transform_batch(
        &self,
        samples: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RandomProjectionError> {
        samples
            .iter()
            .map(|sample| self.transform(sample))
            .collect()
    }

    /// Returns whether a projection matrix has been fitted or loaded.
    pub fn is_fitted(&self) -> bool {
        self.projection_matrix.is_some()
    }

    /// Returns the configured number of output components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Returns the output dimension.
    pub fn output_dimension(&self) -> usize {
        self.n_components
    }

    /// Returns the configured random seed.
    pub fn seed(&self) -> f64 {
        self.seed
    }

    /// Serializes the fitted projection state.
    pub fn to_state(&self) -> Result<RandomProjectionState, RandomProjectionError> {
        let Some(projection_matrix) = self.projection_matrix.as_ref() else {
            return Err(RandomProjectionError::UnfittedSerialization);
        };
        Ok(RandomProjectionState {
            projection_matrix: projection_matrix.clone(),
            n_components: self.n_components,
            n_features: self.n_features,
            seed: self.seed,
        })
    }

    /// Alias matching the source class's JSON terminology.
    pub fn to_json(&self) -> Result<RandomProjectionState, RandomProjectionError> {
        self.to_state()
    }

    /// Restores a transformer from serialized state.
    pub fn from_state(state: RandomProjectionState) -> Result<Self, RandomProjectionError> {
        let mut transformer = Self::new(state.n_components, state.seed)?;
        if state.projection_matrix.len() != state.n_components {
            return Err(RandomProjectionError::MatrixRowCountMismatch {
                expected: state.n_components,
                actual: state.projection_matrix.len(),
            });
        }

        let mut matrix_f32 = Vec::with_capacity(state.projection_matrix.len());
        for (row, values) in state.projection_matrix.iter().enumerate() {
            if values.len() != state.n_features {
                return Err(RandomProjectionError::MatrixRowLengthMismatch {
                    row,
                    expected: state.n_features,
                    actual: values.len(),
                });
            }
            let mut converted_row = Vec::with_capacity(values.len());
            for (column, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(RandomProjectionError::NonFiniteMatrix { row, column });
                }
                let converted = *value as f32;
                if !converted.is_finite() {
                    return Err(RandomProjectionError::MatrixValueOutOfRange { row, column });
                }
                converted_row.push(converted);
            }
            matrix_f32.push(converted_row);
        }

        transformer.n_features = state.n_features;
        transformer.projection_matrix = Some(state.projection_matrix);
        transformer.projection_matrix_f32 = Some(matrix_f32);
        Ok(transformer)
    }

    /// Alias matching the source class's JSON terminology.
    pub fn from_json(state: RandomProjectionState) -> Result<Self, RandomProjectionError> {
        Self::from_state(state)
    }

    /// Releases the cached tensor-equivalent matrix.
    ///
    /// The serialized matrix intentionally remains available, matching the
    /// source's observable behavior where `isFitted()` stays true after
    /// `dispose()` while `transform()` rejects use of the disposed cache.
    pub fn dispose(&mut self) {
        self.projection_matrix_f32 = None;
    }
}

fn validate_samples(samples: &[Vec<f32>]) -> Result<(), RandomProjectionError> {
    if samples.is_empty() {
        return Err(RandomProjectionError::EmptyDataset);
    }

    let expected = samples[0].len();
    if expected == 0 {
        return Err(RandomProjectionError::EmptyFeatureVector { row: 0 });
    }
    for (row, sample) in samples.iter().enumerate() {
        if sample.len() != expected {
            return Err(RandomProjectionError::RaggedDataset {
                row,
                expected,
                actual: sample.len(),
            });
        }
        if let Some(column) = sample.iter().position(|value| !value.is_finite()) {
            return Err(RandomProjectionError::NonFiniteInput { row, column });
        }
    }
    Ok(())
}

/// Returns raw source-compatible seedrandom values for fixture verification.
pub fn compatibility_seedrandom_trace(
    seed: f64,
    count: usize,
) -> Result<(String, Vec<f64>), RandomProjectionError> {
    let seed_string = super::js_number::to_string(seed);
    let mut rng = SeedRandom::from_number(seed)?;
    Ok((seed_string, (0..count).map(|_| rng.next_f64()).collect()))
}

fn generate_gaussian_random_matrix(
    n_features: usize,
    n_components: usize,
    seed: f64,
) -> Result<Vec<Vec<f64>>, RandomProjectionError> {
    let mut rng = SeedRandom::from_number(seed)?;
    let scale = 1.0 / (n_components as f64).sqrt();
    let mut matrix = Vec::with_capacity(n_components);

    for _ in 0..n_components {
        let mut row = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            row.push(z * scale);
        }
        matrix.push(row);
    }

    Ok(matrix)
}

/// The ARC4 generator used by seedrandom 3.0.5's default algorithm.
struct SeedRandom {
    i: usize,
    j: usize,
    state: [u8; 256],
}

impl SeedRandom {
    fn from_number(seed: f64) -> Result<Self, RandomProjectionError> {
        if !seed.is_finite() {
            return Err(RandomProjectionError::InvalidSeed);
        }

        let seed_string = super::js_number::to_string(seed);
        let mut key = Vec::with_capacity(seed_string.len());
        let mut smear = 0_i32;
        for (index, code_unit) in seed_string.encode_utf16().enumerate() {
            let key_index = index & 255;
            let previous = key.get(key_index).copied().unwrap_or(0);
            smear ^= i32::from(previous) * 19;
            if key_index == key.len() {
                key.push(((smear + i32::from(code_unit)) & 255) as u8);
            } else {
                key[key_index] = ((smear + i32::from(code_unit)) & 255) as u8;
            }
        }
        if key.is_empty() {
            key.push(0);
        }

        let mut state = [0_u8; 256];
        for (index, value) in state.iter_mut().enumerate() {
            *value = index as u8;
        }

        let mut j = 0_usize;
        for index in 0..256 {
            let value = state[index];
            j = (j + usize::from(value) + usize::from(key[index % key.len()])) & 255;
            state.swap(index, j);
        }

        let mut generator = Self { i: 0, j: 0, state };
        generator.generate(256);
        Ok(generator)
    }

    fn generate(&mut self, count: usize) -> u64 {
        let mut result = 0_u64;
        for _ in 0..count {
            self.i = (self.i + 1) & 255;
            let value = self.state[self.i];
            self.j = (self.j + usize::from(value)) & 255;
            self.state[self.i] = self.state[self.j];
            self.state[self.j] = value;
            let output_index =
                (usize::from(self.state[self.i]) + usize::from(self.state[self.j])) & 255;
            result = result
                .wrapping_mul(256)
                .wrapping_add(u64::from(self.state[output_index]));
        }
        result
    }

    fn next_f64(&mut self) -> f64 {
        const WIDTH: f64 = 256.0;
        const SIGNIFICANCE: f64 = 4_503_599_627_370_496.0;
        const OVERFLOW: f64 = 9_007_199_254_740_992.0;

        let mut numerator = self.generate(6) as f64;
        let mut denominator = WIDTH.powi(6);
        let mut extra = 0_u64;

        while numerator < SIGNIFICANCE {
            numerator = (numerator + extra as f64) * WIDTH;
            denominator *= WIDTH;
            extra = self.generate(1);
        }
        while numerator >= OVERFLOW {
            numerator /= 2.0;
            denominator /= 2.0;
            extra >>= 1;
        }

        (numerator + extra as f64) / denominator
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::RandomProjectionState;

    fn state(seed: f64) -> RandomProjectionState {
        RandomProjectionState {
            projection_matrix: vec![vec![0.1, 0.2]],
            n_components: 1,
            n_features: 2,
            seed,
        }
    }

    #[test]
    fn integral_seed_serializes_as_an_integer() {
        // Guards the model bridge: the wire schema types the seed as `u64`, so an
        // integral seed must serialize as an integer rather than `42.0`.
        let value = serde_json::to_value(state(42.0)).expect("serialize state");
        assert!(
            value["seed"].is_u64(),
            "integral seed must serialize as an integer, got {}",
            value["seed"]
        );
        assert_eq!(value["seed"], serde_json::json!(42));
    }

    #[test]
    fn integral_seed_round_trips_back_to_f64() {
        let serialized = serde_json::to_value(state(42.0)).expect("serialize state");
        let restored: RandomProjectionState =
            serde_json::from_value(serialized).expect("deserialize state");
        assert_eq!(restored.seed, 42.0);
    }
}
