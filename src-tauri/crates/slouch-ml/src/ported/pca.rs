use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SerializedPca {
    pub components: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    pub n_components: usize,
    pub n_features: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explained_variance: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcaError {
    EmptyData,
    EmptyFeatureVector,
    RaggedData {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        row: usize,
        column: usize,
    },
    NonFiniteState {
        field: &'static str,
        row: usize,
        column: usize,
    },
    InvalidComponentCount,
    InvalidState(String),
    NotFitted,
}

impl std::fmt::Display for PcaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => formatter.write_str("Cannot fit PCA on empty data"),
            Self::EmptyFeatureVector => formatter.write_str("Cannot fit PCA on empty data"),
            Self::RaggedData {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "PCA data row {row} has {actual} features; expected {expected}"
            ),
            Self::NonFiniteValue { row, column } => {
                write!(
                    formatter,
                    "PCA data contains a non-finite value at [{row}, {column}]"
                )
            }
            Self::NonFiniteState { field, row, column } => write!(
                formatter,
                "PCA state field {field} contains a non-finite value at [{row}, {column}]"
            ),
            Self::InvalidComponentCount => formatter.write_str("nComponents must be positive"),
            Self::InvalidState(message) => formatter.write_str(message),
            Self::NotFitted => formatter.write_str("PCA must be fitted before transform"),
        }
    }
}

impl std::error::Error for PcaError {}

#[derive(Debug, Clone, Default)]
pub struct PcaTransformer {
    components: Option<Vec<Vec<f64>>>,
    mean: Option<Vec<f64>>,
    n_components: usize,
    n_features: usize,
    explained_variance: Option<Vec<f64>>,
}

impl PcaTransformer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, data: &[Vec<f64>], requested_components: isize) -> Result<(), PcaError> {
        let n_samples = data.len();
        let n_features = data.first().map_or(0, Vec::len);
        if n_samples == 0 || n_features == 0 {
            return Err(PcaError::EmptyData);
        }
        validate_rows(data, n_features)?;

        if requested_components <= 0 {
            return Err(PcaError::InvalidComponentCount);
        }

        // Centering removes one degree of freedom, matching ml-pca's maximum.
        let max_components = (n_samples - 1).min(n_features);
        let n_components = (requested_components as usize).min(max_components);
        if n_components == 0 {
            return Err(PcaError::InvalidComponentCount);
        }

        let mean = column_means(data, n_features);
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(PcaError::InvalidState(
                "PCA column means are not finite".to_owned(),
            ));
        }
        let centered = centered_data(data, &mean);
        if centered.iter().flatten().any(|value| !value.is_finite()) {
            return Err(PcaError::InvalidState(
                "PCA centered data is not finite".to_owned(),
            ));
        }
        // With few samples the exact full decomposition costs O(min(n,d)^2 * max(n,d))
        // because both SVDs materialise every singular vector over all d feature
        // dimensions. On deep features (d up to 1024) inflated by the unlabeled
        // reservoir (n up to ~1000) that is billions of operations and stalls
        // training for many minutes. The truncated randomized path returns the same
        // retained top-k components in O(n * d * k). The exact path is kept for small
        // d (all frozen ml-pca oracle cases have d <= 6) where it is already fast and
        // its component signs / null-space basis stay pinned to ml-matrix.
        let (components, explained_variance) = if n_features > FAST_PATH_MIN_FEATURES {
            compute_reduction_randomized(&centered, n_samples, n_features, n_components)?
        } else {
            compute_reduction_full(&centered, n_samples, n_features, n_components)?
        };

        self.components = Some(components);
        self.mean = Some(mean);
        self.n_components = n_components;
        self.n_features = n_features;
        self.explained_variance = Some(explained_variance);
        Ok(())
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PcaError> {
        let (components, mean) = match (&self.components, &self.mean) {
            (Some(components), Some(mean)) => (components, mean),
            _ => return Err(PcaError::NotFitted),
        };
        validate_rows(data, self.n_features)?;

        let mut projected = Vec::with_capacity(data.len());
        for row in data {
            let mut output = Vec::with_capacity(self.n_components);
            for component in components {
                let mut dot = 0.0;
                for (value, (average, loading)) in row.iter().zip(mean.iter().zip(component)) {
                    let centered = value - average;
                    let term = centered * loading;
                    if !centered.is_finite() || !term.is_finite() {
                        return Err(PcaError::InvalidState(
                            "PCA projection produced a non-finite intermediate".to_owned(),
                        ));
                    }
                    dot += term;
                    if !dot.is_finite() {
                        return Err(PcaError::InvalidState(
                            "PCA projection produced a non-finite result".to_owned(),
                        ));
                    }
                }
                output.push(dot);
            }
            projected.push(output);
        }
        Ok(projected)
    }

    pub fn get_output_dimension(&self) -> usize {
        self.n_components
    }

    pub fn to_json(&self) -> Result<SerializedPca, PcaError> {
        let (components, mean) = match (&self.components, &self.mean) {
            (Some(components), Some(mean)) => (components, mean),
            _ => {
                return Err(PcaError::InvalidState(
                    "Cannot serialize unfitted PCA".to_owned(),
                ))
            }
        };

        Ok(SerializedPca {
            components: components.clone(),
            mean: mean.clone(),
            n_components: self.n_components,
            n_features: self.n_features,
            explained_variance: self.explained_variance.clone(),
        })
    }

    pub fn from_json(data: SerializedPca) -> Result<Self, PcaError> {
        validate_serialized_state(&data)?;
        Ok(Self {
            components: Some(data.components),
            mean: Some(data.mean),
            n_components: data.n_components,
            n_features: data.n_features,
            // The source loader does not restore its `PCA` instance. Its next
            // `toJSON` therefore omits training-only explained variance.
            explained_variance: None,
        })
    }

    pub fn dispose(&mut self) {
        self.components = None;
        self.mean = None;
        self.explained_variance = None;
    }
}

fn validate_rows(data: &[Vec<f64>], n_features: usize) -> Result<(), PcaError> {
    for (row, values) in data.iter().enumerate() {
        if values.len() != n_features {
            return Err(PcaError::RaggedData {
                row,
                expected: n_features,
                actual: values.len(),
            });
        }
        if let Some(column) = values.iter().position(|value| !value.is_finite()) {
            return Err(PcaError::NonFiniteValue { row, column });
        }
    }
    Ok(())
}

fn validate_serialized_state(data: &SerializedPca) -> Result<(), PcaError> {
    if data.n_components == 0 {
        return Err(PcaError::InvalidState(
            "Serialized PCA must contain at least one component".to_owned(),
        ));
    }
    if data.n_features == 0 {
        return Err(PcaError::InvalidState(
            "Serialized PCA must contain at least one feature".to_owned(),
        ));
    }
    if data.components.len() != data.n_components {
        return Err(PcaError::InvalidState(
            "Serialized PCA component count does not match nComponents".to_owned(),
        ));
    }
    if data.mean.len() != data.n_features {
        return Err(PcaError::InvalidState(
            "Serialized PCA mean length does not match nFeatures".to_owned(),
        ));
    }
    for (row, component) in data.components.iter().enumerate() {
        if component.len() != data.n_features {
            return Err(PcaError::InvalidState(
                "Serialized PCA component width does not match nFeatures".to_owned(),
            ));
        }
        if let Some(column) = component.iter().position(|value| !value.is_finite()) {
            return Err(PcaError::NonFiniteState {
                field: "components",
                row,
                column,
            });
        }
    }
    if let Some(column) = data.mean.iter().position(|value| !value.is_finite()) {
        return Err(PcaError::NonFiniteState {
            field: "mean",
            row: 0,
            column,
        });
    }
    if let Some(explained_variance) = &data.explained_variance {
        if explained_variance.len() != data.n_components {
            return Err(PcaError::InvalidState(
                "Serialized PCA explained variance length does not match nComponents".to_owned(),
            ));
        }
        if let Some(column) = explained_variance
            .iter()
            .position(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PcaError::NonFiniteState {
                field: "explainedVariance",
                row: 0,
                column,
            });
        }
    }
    Ok(())
}

fn column_means(data: &[Vec<f64>], n_features: usize) -> Vec<f64> {
    let mut mean = vec![0.0; n_features];
    for row in data {
        for (index, value) in row.iter().enumerate() {
            mean[index] += value;
        }
    }
    let sample_count = data.len() as f64;
    for value in &mut mean {
        *value /= sample_count;
    }
    mean
}

fn centered_data(data: &[Vec<f64>], mean: &[f64]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|row| {
            row.iter()
                .zip(mean)
                .map(|(value, average)| value - average)
                .collect()
        })
        .collect()
}

/// Feature-dimension threshold above which the truncated randomized path is used.
/// Every frozen ml-pca oracle case has `d <= 6`, so they always take the exact
/// path; production deep features (>= 256 dims) always take the fast path.
const FAST_PATH_MIN_FEATURES: usize = 64;

/// Extra range-finder columns beyond the requested rank (Halko et al. oversampling).
const RANDOMIZED_OVERSAMPLES: usize = 10;

/// Subspace power iterations sharpening the sampled range against the top spectrum.
const RANDOMIZED_POWER_ITERATIONS: usize = 4;

/// Exact decomposition preserved for small `d`. This is the original ml-pca-parity
/// path: it runs both the ml-matrix Golub-Reinsch port (for sign / null-space
/// reference) and nalgebra's SVD (for stable components), so the frozen oracles keep
/// matching bit-for-bit. Returns `(components, explained_variance)`.
fn compute_reduction_full(
    centered: &[Vec<f64>],
    n_samples: usize,
    n_features: usize,
    n_components: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), PcaError> {
    // `ml-pca` 4.1.1 delegates to `ml-matrix`'s Golub-Reinsch SVD with
    // autoTranspose. The local port preserves its operation order so
    // component signs and null-space bases are not independently chosen.
    let (ml_singular_values, ml_right_vectors) =
        super::ml_matrix_svd::right_singular_vectors(centered);
    let matrix =
        nalgebra::DMatrix::from_fn(n_samples, n_features, |row, column| centered[row][column]);
    let decomposition = matrix.svd(false, true);
    let right_transposed = decomposition.v_t.ok_or_else(|| {
        PcaError::InvalidState("PCA SVD did not return right singular vectors".to_owned())
    })?;
    let largest = decomposition.singular_values.get(0).copied().unwrap_or(0.0);
    if !largest.is_finite()
        || decomposition
            .singular_values
            .iter()
            .any(|value| !value.is_finite())
        || ml_singular_values.iter().any(|value| !value.is_finite())
    {
        return Err(PcaError::InvalidState(
            "PCA singular values are not finite".to_owned(),
        ));
    }
    let components = (0..n_components)
        .map(|component| {
            let ml_component = (0..n_features)
                .map(|feature| ml_right_vectors[feature][component])
                .collect::<Vec<_>>();
            if decomposition.singular_values[component] <= largest * 1e-12 {
                // A null-space basis is not unique. Preserve ml-matrix's
                // Householder completion rather than selecting a new one.
                return ml_component;
            }
            let mut stable_component = (0..n_features)
                .map(|feature| right_transposed[(component, feature)])
                .collect::<Vec<_>>();
            let alignment = stable_component
                .iter()
                .zip(&ml_component)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            if alignment < 0.0 {
                for value in &mut stable_component {
                    *value = -*value;
                }
            }
            stable_component
        })
        .collect::<Vec<_>>();
    if components.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PcaError::InvalidState(
            "PCA components are not finite".to_owned(),
        ));
    }
    let denominator = (n_samples - 1) as f64;
    let eigenvalues = decomposition
        .singular_values
        .iter()
        .map(|value| value * value / denominator)
        .collect::<Vec<_>>();
    debug_assert_eq!(ml_singular_values.len(), eigenvalues.len());
    if eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PcaError::InvalidState(
            "PCA eigenvalues are not finite".to_owned(),
        ));
    }
    let total_variance = eigenvalues.iter().sum::<f64>();
    if !total_variance.is_finite() || total_variance <= 0.0 {
        return Err(PcaError::InvalidState(
            "PCA total variance must be finite and positive".to_owned(),
        ));
    }
    let explained_variance = eigenvalues
        .iter()
        .take(n_components)
        .map(|value| value / total_variance)
        .collect::<Vec<_>>();
    if explained_variance
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PcaError::InvalidState(
            "PCA explained variance is not finite".to_owned(),
        ));
    }
    Ok((components, explained_variance))
}

/// Truncated PCA via randomized SVD (Halko, Martinsson & Tropp, 2011) — the same
/// approach as scikit-learn's `svd_solver="randomized"`. It builds a small
/// orthonormal basis for the sampled range of the centered data, projects onto it,
/// and reads the top-`k` singular triplets from an `l x l` eigenproblem, costing
/// `O(n * d * l)` instead of a full `O(min(n,d)^2 * max(n,d))` SVD. When
/// `l >= rank(X)` (few samples) the result is exact; for the reservoir-inflated
/// near-square case only the retained top-k directions are computed, which is what
/// makes it finish in milliseconds where the exact path took tens of minutes.
///
/// Output matches the exact path's contract: `n_components` unit loading vectors of
/// length `n_features`, ordered by descending explained variance, plus the
/// explained-variance *ratio* (each retained eigenvalue over the full covariance
/// trace, identical semantics to the exact path). Component signs are fixed
/// deterministically (largest-magnitude entry positive), so refits are reproducible.
fn compute_reduction_randomized(
    centered: &[Vec<f64>],
    n_samples: usize,
    n_features: usize,
    n_components: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), PcaError> {
    use nalgebra::DMatrix;

    let x = DMatrix::from_fn(n_samples, n_features, |row, column| centered[row][column]);

    // Full covariance trace = ||X||_F^2 / (n-1); the (n-1) cancels in the ratio, so
    // the raw sum of squares is the denominator for explained variance. Computing it
    // directly avoids materialising every eigenvalue.
    let total_sumsq: f64 = centered.iter().flatten().map(|value| value * value).sum();
    if !total_sumsq.is_finite() || total_sumsq <= 0.0 {
        return Err(PcaError::InvalidState(
            "PCA total variance must be finite and positive".to_owned(),
        ));
    }

    let target_rank = (n_components + RANDOMIZED_OVERSAMPLES)
        .min(n_samples)
        .min(n_features)
        .max(n_components);

    let xt = x.transpose();
    let omega = gaussian_matrix(n_features, target_rank);
    let mut basis = orthonormalize(&x * &omega)?;
    for _ in 0..RANDOMIZED_POWER_ITERATIONS {
        let sampled = &x * (&xt * &basis);
        basis = orthonormalize(sampled)?;
    }

    // B = Q^T X (l x d); right singular vectors of B are those of X within range(Q).
    let b = basis.transpose() * &x;
    let b_transposed = b.transpose();
    // Left singular vectors and squared singular values from the small l x l Gram.
    let gram = &b * &b_transposed;
    let eigen = gram.symmetric_eigen();

    // `symmetric_eigen` leaves eigenpairs unordered; keep the largest n_components.
    let mut order: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
    order.sort_by(|&left, &right| {
        eigen.eigenvalues[right]
            .partial_cmp(&eigen.eigenvalues[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut components = Vec::with_capacity(n_components);
    let mut explained_variance = Vec::with_capacity(n_components);
    for &index in order.iter().take(n_components) {
        let eigenvalue = eigen.eigenvalues[index].max(0.0);
        // v = B^T w normalized; ||B^T w|| = sigma, so this is the unit right vector.
        let left_vector = eigen.eigenvectors.column(index);
        let projected = &b_transposed * left_vector;
        let norm = projected.norm();
        let mut component: Vec<f64> = if norm > f64::EPSILON {
            projected.iter().map(|value| value / norm).collect()
        } else {
            // Degenerate null-space direction: fall back to a canonical unit axis so
            // the serialized state stays finite. Full-rank feature data never reaches
            // this for the retained top-k.
            let mut axis = vec![0.0; n_features];
            axis[index.min(n_features - 1)] = 1.0;
            axis
        };
        flip_component_sign(&mut component);
        if component.iter().any(|value| !value.is_finite()) {
            return Err(PcaError::InvalidState(
                "PCA components are not finite".to_owned(),
            ));
        }
        let ratio = eigenvalue / total_sumsq;
        if !ratio.is_finite() || ratio < 0.0 {
            return Err(PcaError::InvalidState(
                "PCA explained variance is not finite".to_owned(),
            ));
        }
        components.push(component);
        explained_variance.push(ratio);
    }
    Ok((components, explained_variance))
}

/// Deterministic sign convention: negate so the largest-magnitude entry is positive.
fn flip_component_sign(component: &mut [f64]) {
    let mut max_abs = 0.0;
    let mut negative = false;
    for &value in component.iter() {
        if value.abs() > max_abs {
            max_abs = value.abs();
            negative = value < 0.0;
        }
    }
    if negative {
        for value in component.iter_mut() {
            *value = -*value;
        }
    }
}

/// Thin QR orthonormalisation of the range sample, guarding against non-finite bases.
fn orthonormalize(matrix: nalgebra::DMatrix<f64>) -> Result<nalgebra::DMatrix<f64>, PcaError> {
    let basis = matrix.qr().q();
    if basis.iter().any(|value| !value.is_finite()) {
        return Err(PcaError::InvalidState(
            "PCA randomized range finder produced a non-finite basis".to_owned(),
        ));
    }
    Ok(basis)
}

/// A fixed-seed Gaussian matrix so randomized fits are reproducible across runs.
fn gaussian_matrix(rows: usize, columns: usize) -> nalgebra::DMatrix<f64> {
    let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15);
    nalgebra::DMatrix::from_fn(rows, columns, |_, _| rng.next_gaussian())
}

/// Minimal deterministic PRNG (SplitMix64) with Box-Muller Gaussians. Local so the
/// crate takes no new RNG dependency for a purely internal, seeded sampling matrix.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_uniform().max(f64::MIN_POSITIVE);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod fast_path_tests {
    use super::*;

    fn seeded_matrix(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / ((1u64 << 53) as f64)
        };
        (0..n)
            .map(|_| (0..d).map(|_| next()).collect::<Vec<f64>>())
            .collect()
    }

    fn align_sign(component: &mut [f64], reference: &[f64]) {
        let dot: f64 = component
            .iter()
            .zip(reference)
            .map(|(left, right)| left * right)
            .sum();
        if dot < 0.0 {
            for value in component.iter_mut() {
                *value = -*value;
            }
        }
    }

    #[test]
    fn randomized_matches_exact_on_small_full_rank_matrix() {
        // n=12, d=8: wide enough that the sampled range (l >= rank) is exact, so the
        // randomized path must reproduce the exact ml-pca path up to component sign.
        let (n, d, k) = (12usize, 8usize, 6usize);
        let data = seeded_matrix(n, d);
        let mean = column_means(&data, d);
        let centered = centered_data(&data, &mean);

        let (full_components, full_variance) =
            compute_reduction_full(&centered, n, d, k).unwrap();
        let (fast_components, fast_variance) =
            compute_reduction_randomized(&centered, n, d, k).unwrap();

        assert_eq!(full_components.len(), fast_components.len());
        for (index, (full, fast)) in full_components.iter().zip(&fast_components).enumerate() {
            let mut fast = fast.clone();
            align_sign(&mut fast, full);
            for (expected, actual) in full.iter().zip(&fast) {
                assert!(
                    (expected - actual).abs() < 1e-5,
                    "component {index} differs: {expected} vs {actual}"
                );
            }
        }
        for (expected, actual) in full_variance.iter().zip(&fast_variance) {
            assert!(
                (expected - actual).abs() < 1e-6,
                "explained variance differs: {expected} vs {actual}"
            );
        }
    }

    #[test]
    fn public_fit_completes_quickly_on_wide_deep_features() {
        // 40 x 1024 -> 30. The exact path took hundreds of seconds at this width; the
        // test finishing in normal test time is the regression guard, and the loose
        // debug wall-clock bound documents the intent.
        let (n, d) = (40usize, 1024usize);
        let data = seeded_matrix(n, d);
        let start = std::time::Instant::now();
        let mut transformer = PcaTransformer::new();
        transformer.fit(&data, 30).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "randomized PCA fit took too long: {elapsed:?}"
        );

        let state = transformer.to_json().unwrap();
        assert_eq!(state.n_components, 30);
        assert_eq!(state.n_features, d);
        assert_eq!(state.components.len(), 30);
        assert!(state.components.iter().all(|component| component.len() == d));
        let variance = state.explained_variance.as_ref().unwrap();
        assert_eq!(variance.len(), 30);
        assert!(variance.iter().all(|value| value.is_finite() && *value >= 0.0));
        for pair in variance.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12, "explained variance not descending");
        }
        assert!(variance.iter().sum::<f64>() <= 1.0 + 1e-9);

        let projected = transformer.transform(&data).unwrap();
        assert_eq!(projected.len(), n);
        assert!(projected.iter().all(|row| row.len() == 30));
    }

    #[test]
    fn randomized_components_are_orthonormal_and_reproducible() {
        // Truncated (l < rank) path: retained loadings stay unit-norm and mutually
        // orthogonal, and the fixed seed makes refits bit-identical.
        let (n, d, k) = (60usize, 256usize, 10usize);
        let data = seeded_matrix(n, d);
        let mean = column_means(&data, d);
        let centered = centered_data(&data, &mean);
        let (components, _) = compute_reduction_randomized(&centered, n, d, k).unwrap();

        for component in &components {
            let norm = component.iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "component not unit norm: {norm}");
        }
        for left in 0..components.len() {
            for right in (left + 1)..components.len() {
                let dot: f64 = components[left]
                    .iter()
                    .zip(&components[right])
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(dot.abs() < 1e-5, "components {left},{right} not orthogonal: {dot}");
            }
        }

        let (again, _) = compute_reduction_randomized(&centered, n, d, k).unwrap();
        assert_eq!(components, again, "seeded randomized fit must be reproducible");
    }
}
