//! K-means clustering with silhouette-score selection.
//!
//! This is a mechanical Rust port of `src/services/ml/kmeans.ts`. Feature
//! vectors and centroids remain `f32`; distances, scores, and seedrandom's
//! generated values use `f64`, matching the source's JavaScript number math.

use std::fmt;

use super::config::TRAINING_CONFIG;

#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub k: usize,
    pub silhouette_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KMeansInitializationStep {
    pub total_distance_squared: f64,
    pub threshold: f64,
    pub selected_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KMeansRunTrace {
    pub seed: f64,
    pub requested_k: usize,
    pub initial_centroid_indices: Vec<usize>,
    pub initialization_steps: Vec<KMeansInitializationStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectBestKRunTrace {
    pub seed: f64,
    pub requested_k: usize,
    pub initial_centroid_indices: Vec<usize>,
    pub initialization_steps: Vec<KMeansInitializationStep>,
    pub candidate_k: usize,
    pub run: usize,
    pub silhouette_score: f64,
    pub selected_for_candidate: bool,
    pub selected_overall: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KMeansError {
    EmptyFeatureVector {
        row: usize,
    },
    RaggedFeatures {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteFeature {
        row: usize,
        column: usize,
    },
    InvalidSeed,
    AssignmentLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidAssignment {
        index: usize,
        cluster: usize,
        clusters: usize,
    },
    CentroidDimensionMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for KMeansError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFeatureVector { row } => {
                write!(formatter, "feature vector at row {row} must not be empty")
            }
            Self::RaggedFeatures {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "feature vector at row {row} has {actual} values, expected {expected}"
            ),
            Self::NonFiniteFeature { row, column } => write!(
                formatter,
                "feature value at row {row}, column {column} is not finite"
            ),
            Self::InvalidSeed => formatter.write_str("seed must be finite"),
            Self::AssignmentLengthMismatch { expected, actual } => write!(
                formatter,
                "assignment count {actual} does not match feature count {expected}"
            ),
            Self::InvalidAssignment {
                index,
                cluster,
                clusters,
            } => write!(
                formatter,
                "assignment at index {index} selects cluster {cluster}, but there are {clusters} clusters"
            ),
            Self::CentroidDimensionMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "centroid at index {index} has {actual} values, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for KMeansError {}

/// Computes squared Euclidean distance between two vectors.
fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for (left, right) in a.iter().zip(b) {
        let difference = f64::from(*left) - f64::from(*right);
        sum += difference * difference;
    }
    sum
}

/// Computes Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    squared_distance(a, b).sqrt()
}

/// The ARC4 generator used by seedrandom 3.0.5's default algorithm.
///
/// Keeping this small implementation local avoids adding a runtime dependency
/// while preserving the seeded sequences used by the TypeScript source.
struct SeedRandom {
    i: usize,
    j: usize,
    state: [u8; 256],
}

impl SeedRandom {
    fn from_number(seed: f64) -> Result<Self, KMeansError> {
        if !seed.is_finite() {
            return Err(KMeansError::InvalidSeed);
        }

        let seed_string = super::js_number::to_string(seed);
        let mut key = Vec::with_capacity(seed_string.len());
        let mut smear = 0_i32;
        for (index, code_unit) in seed_string.encode_utf16().enumerate() {
            let key_index = index & 255;
            let previous = key.get(key_index).copied().unwrap_or(0);
            smear ^= i32::from(previous) * 19;
            key_index_for_push(
                &mut key,
                key_index,
                ((smear + i32::from(code_unit)) & 255) as u8,
            );
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
            let t = state[index];
            j = (j + usize::from(t) + usize::from(key[index % key.len()])) & 255;
            state.swap(index, j);
        }

        let mut generator = Self { i: 0, j: 0, state };
        generator.generate(256);
        Ok(generator)
    }

    fn generate(&mut self, count: usize) -> u64 {
        let mut result = 0_u64;
        for _ in 0..count {
            let i = (self.i + 1) & 255;
            let t = self.state[i];
            self.i = i;
            let j = (self.j + usize::from(t)) & 255;
            self.j = j;
            self.state[i] = self.state[j];
            self.state[j] = t;
            let output_index = (usize::from(self.state[i]) + usize::from(self.state[j])) & 255;
            result = result
                .wrapping_mul(256)
                .wrapping_add(u64::from(self.state[output_index]));
        }
        result
    }

    fn next_f64(&mut self) -> f64 {
        const WIDTH: f64 = 256.0;
        const SIGNIFICANCE: f64 = 4_503_599_627_370_496.0; // 2^52
        const OVERFLOW: f64 = 9_007_199_254_740_992.0; // 2^53

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

fn key_index_for_push(key: &mut Vec<u8>, index: usize, value: u8) {
    if index == key.len() {
        key.push(value);
    } else {
        key[index] = value;
    }
}

/// Returns the source-compatible seed string and raw seedrandom values used by
/// compatibility oracles. Keeping this seam beside the production generator
/// makes RNG fixture fields mutation-sensitive without duplicating ARC4.
pub fn compatibility_seedrandom_trace(
    seed: f64,
    count: usize,
) -> Result<(String, Vec<f64>), KMeansError> {
    let seed_string = super::js_number::to_string(seed);
    let mut rng = SeedRandom::from_number(seed)?;
    let values = (0..count).map(|_| rng.next_f64()).collect();
    Ok((seed_string, values))
}

fn k_means_plus_plus_init(
    features: &[Vec<f32>],
    k: usize,
    rng: &mut SeedRandom,
) -> (Vec<Vec<f32>>, Vec<usize>, Vec<KMeansInitializationStep>) {
    let n = features.len();
    let first_index = (rng.next_f64() * n as f64).floor() as usize;
    let mut centroids = vec![features[first_index].clone()];
    let mut initial_centroid_indices = vec![first_index];
    let mut initialization_steps = Vec::with_capacity(k.saturating_sub(1));
    let mut minimum_distance_squared = vec![f32::INFINITY; n];

    for centroid_index in 1..k {
        let last_centroid = &centroids[centroid_index - 1];
        for (index, feature) in features.iter().enumerate() {
            // Assignment to the source Float32Array rounds before comparison
            // on subsequent centroid-selection passes.
            let distance_squared = squared_distance(feature, last_centroid) as f32;
            if distance_squared < minimum_distance_squared[index] {
                minimum_distance_squared[index] = distance_squared;
            }
        }

        let total_distance_squared = minimum_distance_squared
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        let threshold = rng.next_f64() * total_distance_squared;
        let mut cumulative = 0.0_f64;
        let mut selected_index = 0_usize;
        for (index, distance_squared) in minimum_distance_squared.iter().enumerate() {
            cumulative += f64::from(*distance_squared);
            if cumulative >= threshold {
                selected_index = index;
                break;
            }
        }
        initial_centroid_indices.push(selected_index);
        initialization_steps.push(KMeansInitializationStep {
            total_distance_squared,
            threshold,
            selected_index,
        });
        centroids.push(features[selected_index].clone());
    }

    (centroids, initial_centroid_indices, initialization_steps)
}

fn assign_clusters(features: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    let mut assignments = Vec::with_capacity(features.len());
    for feature in features {
        let mut minimum_distance = f64::INFINITY;
        let mut minimum_index = 0_usize;
        for (centroid_index, centroid) in centroids.iter().enumerate() {
            let distance = squared_distance(feature, centroid);
            if distance < minimum_distance {
                minimum_distance = distance;
                minimum_index = centroid_index;
            }
        }
        assignments.push(minimum_index);
    }
    assignments
}

fn update_centroids(
    features: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    dimension: usize,
    rng: &mut SeedRandom,
) -> Vec<Vec<f32>> {
    let mut counts = vec![0_usize; k];
    let mut sums = vec![vec![0.0_f32; dimension]; k];

    for (feature, cluster) in features.iter().zip(assignments) {
        counts[*cluster] += 1;
        for (sum, value) in sums[*cluster].iter_mut().zip(feature) {
            *sum += *value;
        }
    }

    let mut centroids = Vec::with_capacity(k);
    for cluster in 0..k {
        if counts[cluster] == 0 {
            let random_index = (rng.next_f64() * features.len() as f64).floor() as usize;
            centroids.push(features[random_index].clone());
        } else {
            let mut centroid = vec![0.0_f32; dimension];
            for (value, sum) in centroid.iter_mut().zip(&sums[cluster]) {
                *value = (f64::from(*sum) / counts[cluster] as f64) as f32;
            }
            centroids.push(centroid);
        }
    }
    centroids
}

fn validate_features(features: &[Vec<f32>]) -> Result<(), KMeansError> {
    let Some(first) = features.first() else {
        return Ok(());
    };
    let dimension = first.len();
    if dimension == 0 {
        return Err(KMeansError::EmptyFeatureVector { row: 0 });
    }
    for (row, feature) in features.iter().enumerate() {
        if feature.len() != dimension {
            return Err(KMeansError::RaggedFeatures {
                row,
                expected: dimension,
                actual: feature.len(),
            });
        }
        if let Some(column) = feature.iter().position(|value| !value.is_finite()) {
            return Err(KMeansError::NonFiniteFeature { row, column });
        }
    }
    Ok(())
}

/// Runs k-means++ clustering and returns the mean silhouette score.
pub fn kmeans(
    features: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: f64,
) -> Result<KMeansResult, KMeansError> {
    validate_features(features)?;
    let n = features.len();
    if n == 0 || k == 0 {
        return Ok(KMeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
            k: 0,
            silhouette_score: 0.0,
        });
    }

    let effective_k = k.min(n);
    let dimension = features[0].len();
    if effective_k == 1 {
        let mut centroid = vec![0.0_f32; dimension];
        for feature in features {
            for (sum, value) in centroid.iter_mut().zip(feature) {
                *sum += *value;
            }
        }
        for value in &mut centroid {
            *value = (f64::from(*value) / n as f64) as f32;
        }
        return Ok(KMeansResult {
            centroids: vec![centroid],
            assignments: vec![0; n],
            k: 1,
            silhouette_score: 0.0,
        });
    }

    let mut rng = SeedRandom::from_number(seed)?;
    let (mut centroids, _, _) = k_means_plus_plus_init(features, effective_k, &mut rng);
    let mut assignments = Vec::new();
    let mut previous_assignments = Vec::new();

    for _ in 0..max_iter {
        assignments = assign_clusters(features, &centroids);
        if !previous_assignments.is_empty() && assignments == previous_assignments {
            break;
        }
        previous_assignments = assignments.clone();
        centroids = update_centroids(features, &assignments, effective_k, dimension, &mut rng);
    }

    let score = silhouette_score(features, &assignments, &centroids)?;
    Ok(KMeansResult {
        centroids,
        assignments,
        k: effective_k,
        silhouette_score: score,
    })
}

/// Runs k-means and records the exact k-means++ selection evidence.
pub fn kmeans_with_trace(
    features: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: f64,
) -> Result<(KMeansResult, KMeansRunTrace), KMeansError> {
    validate_features(features)?;
    let effective_k = k.min(features.len());
    let mut trace = KMeansRunTrace {
        seed,
        requested_k: k,
        initial_centroid_indices: Vec::new(),
        initialization_steps: Vec::new(),
    };
    if effective_k > 1 {
        let mut rng = SeedRandom::from_number(seed)?;
        let (_, indices, steps) = k_means_plus_plus_init(features, effective_k, &mut rng);
        trace.initial_centroid_indices = indices;
        trace.initialization_steps = steps;
    }
    Ok((kmeans(features, k, max_iter, seed)?, trace))
}

/// Runs k-means with the source function's default iteration count and seed.
pub fn kmeans_default(features: &[Vec<f32>], k: usize) -> Result<KMeansResult, KMeansError> {
    kmeans(features, k, 100, TRAINING_CONFIG.random_seed as f64)
}

/// Computes the mean silhouette score for a clustering.
pub fn silhouette_score(
    features: &[Vec<f32>],
    assignments: &[usize],
    centroids: &[Vec<f32>],
) -> Result<f64, KMeansError> {
    validate_features(features)?;
    if assignments.len() != features.len() {
        return Err(KMeansError::AssignmentLengthMismatch {
            expected: features.len(),
            actual: assignments.len(),
        });
    }
    let dimension = features.first().map_or(0, Vec::len);
    for (index, centroid) in centroids.iter().enumerate() {
        if centroid.len() != dimension {
            return Err(KMeansError::CentroidDimensionMismatch {
                index,
                expected: dimension,
                actual: centroid.len(),
            });
        }
        if let Some(column) = centroid.iter().position(|value| !value.is_finite()) {
            return Err(KMeansError::NonFiniteFeature { row: index, column });
        }
    }
    let n = features.len();
    let k = centroids.len();
    for (index, cluster) in assignments.iter().enumerate() {
        if *cluster >= k {
            return Err(KMeansError::InvalidAssignment {
                index,
                cluster: *cluster,
                clusters: k,
            });
        }
    }
    if k <= 1 || n <= 1 {
        return Ok(0.0);
    }

    let mut clusters = vec![Vec::new(); k];
    for (index, cluster) in assignments.iter().enumerate() {
        clusters[*cluster].push(index);
    }

    let mut total_silhouette = 0.0_f64;
    let mut valid_points = 0_usize;
    for index in 0..n {
        let my_cluster = assignments[index];
        let members = &clusters[my_cluster];
        let mut a = 0.0_f64;
        if members.len() > 1 {
            for member in members {
                if *member != index {
                    a += euclidean_distance(&features[index], &features[*member]);
                }
            }
            a /= (members.len() - 1) as f64;
        }

        let mut b = f64::INFINITY;
        for (cluster, other_members) in clusters.iter().enumerate() {
            if cluster == my_cluster || other_members.is_empty() {
                continue;
            }
            let mut average_distance = 0.0_f64;
            for member in other_members {
                average_distance += euclidean_distance(&features[index], &features[*member]);
            }
            average_distance /= other_members.len() as f64;
            if average_distance < b {
                b = average_distance;
            }
        }

        if b == f64::INFINITY {
            continue;
        }
        let maximum = a.max(b);
        if maximum > 0.0 {
            total_silhouette += (b - a) / maximum;
            valid_points += 1;
        }
    }

    Ok(if valid_points > 0 {
        total_silhouette / valid_points as f64
    } else {
        0.0
    })
}

/// Selects the candidate k with the highest silhouette score.
pub fn select_best_k(
    features: &[Vec<f32>],
    k_values: &[usize],
    seed: f64,
) -> Result<KMeansResult, KMeansError> {
    validate_features(features)?;
    let n = features.len();
    if n == 0 {
        return Ok(KMeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
            k: 0,
            silhouette_score: 0.0,
        });
    }

    let valid_values = k_values
        .iter()
        .copied()
        .filter(|candidate| *candidate > 0 && *candidate <= n)
        .collect::<Vec<_>>();
    if valid_values.is_empty() {
        return kmeans(features, 1, 100, seed);
    }

    let mut best_result = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut seed_offset = 0.0_f64;
    for candidate in valid_values {
        let runs = if candidate == 1 { 1 } else { 3 };
        let mut best_run_result = None;
        let mut best_run_score = f64::NEG_INFINITY;
        for _ in 0..runs {
            let result = kmeans(features, candidate, 100, seed + seed_offset)?;
            seed_offset += 1.0;
            if result.silhouette_score > best_run_score {
                best_run_score = result.silhouette_score;
                best_run_result = Some(result);
            }
        }
        if best_run_score > best_score {
            best_score = best_run_score;
            best_result = best_run_result;
        }
    }

    best_result
        .map(Ok)
        .unwrap_or_else(|| kmeans(features, 1, 100, seed))
}

/// Selects the best k and records every candidate/run seed and initializer.
pub fn select_best_k_with_trace(
    features: &[Vec<f32>],
    k_values: &[usize],
    seed: f64,
) -> Result<(KMeansResult, Vec<SelectBestKRunTrace>), KMeansError> {
    validate_features(features)?;
    if features.is_empty() {
        return Ok((kmeans(features, 0, 100, seed)?, Vec::new()));
    }
    let valid_values = k_values
        .iter()
        .copied()
        .filter(|candidate| *candidate > 0 && *candidate <= features.len())
        .collect::<Vec<_>>();
    if valid_values.is_empty() {
        return Ok((kmeans(features, 1, 100, seed)?, Vec::new()));
    }

    let mut traces = Vec::new();
    let mut results = Vec::new();
    let mut best_overall = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut seed_offset = 0.0;
    for candidate_k in valid_values {
        let mut best_candidate = None;
        let mut best_candidate_score = f64::NEG_INFINITY;
        let runs = if candidate_k == 1 { 1 } else { 3 };
        for run in 0..runs {
            let (result, trace) =
                kmeans_with_trace(features, candidate_k, 100, seed + seed_offset)?;
            seed_offset += 1.0;
            if result.silhouette_score > best_candidate_score {
                best_candidate_score = result.silhouette_score;
                best_candidate = Some(traces.len());
            }
            traces.push(SelectBestKRunTrace {
                seed: trace.seed,
                requested_k: trace.requested_k,
                initial_centroid_indices: trace.initial_centroid_indices.clone(),
                initialization_steps: trace.initialization_steps.clone(),
                candidate_k,
                run,
                silhouette_score: result.silhouette_score,
                selected_for_candidate: false,
                selected_overall: false,
            });
            results.push(result);
        }
        if let Some(index) = best_candidate {
            traces[index].selected_for_candidate = true;
            if best_candidate_score > best_score {
                best_score = best_candidate_score;
                best_overall = Some(index);
            }
        }
    }
    let selected = best_overall.expect("valid candidates always produce a run");
    traces[selected].selected_overall = true;
    Ok((results[selected].clone(), traces))
}

/// Selects the best k using `[1, 2, 3, 5, 7]`, matching the source default.
pub fn select_best_k_default(features: &[Vec<f32>]) -> Result<KMeansResult, KMeansError> {
    select_best_k(
        features,
        &[1, 2, 3, 5, 7],
        TRAINING_CONFIG.random_seed as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_trace_exposes_source_fields_directly() {
        let features = vec![vec![0.0_f32], vec![10.0_f32]];
        let (_, runs) = select_best_k_with_trace(&features, &[2], 42.0).unwrap();
        let run = &runs[0];

        assert_eq!(run.seed, 42.0);
        assert_eq!(run.requested_k, 2);
        assert!(!run.initial_centroid_indices.is_empty());
        assert!(!run.initialization_steps.is_empty());
    }
}
