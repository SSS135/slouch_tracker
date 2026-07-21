/// Training configuration for the L2-regularized logistic regression classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingConfig {
    /// L2 regularization strength (Ridge penalty).
    pub l2_regularization: f64,
    /// Maximum iterations for gradient descent.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub convergence_tol: f64,
    /// Number of folds for cross-validation (always 3-fold stratified CV).
    pub cv_folds: usize,
    /// Minimum frames required per class to train.
    pub min_frames_per_class: usize,
    /// Maximum acceptable class imbalance ratio (warning threshold).
    pub max_imbalance_ratio: f64,
    /// Random seed for reproducibility.
    pub random_seed: u64,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Batch size for mini-batch gradient descent (`None` means full batch).
    pub batch_size: Option<usize>,
}

pub const TRAINING_CONFIG: TrainingConfig = TrainingConfig {
    l2_regularization: 0.01,
    max_iterations: 1000,
    convergence_tol: 1e-6,
    cv_folds: 3,
    min_frames_per_class: 1,
    max_imbalance_ratio: 0.35,
    random_seed: 42,
    learning_rate: 0.01,
    batch_size: None,
};
