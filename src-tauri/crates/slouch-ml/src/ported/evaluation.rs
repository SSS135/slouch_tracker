//! Cross-validation evaluation for trained posture classifiers.
//!
//! This is a mechanical Rust port of `src/services/ml/evaluation.ts`.
//! Feature containers are borrowed through the domain `FeatureSource` contract;
//! the classifier and model implementations own preprocessing and inference.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use slouch_domain::{FeatureMap, FeatureSource, Keypoint};

use super::base_classifier::BaseClassifier;
use super::config::TRAINING_CONFIG;
use super::cross_validation::{
    create_stratified_k_fold, create_temporal_block_k_fold, create_time_ordered_folds, CvFold,
};
use super::feature_extractor::FeatureExtractor;
use super::model::Model;

/// Fold-construction strategy selected by [`CrossValidationOptions`].
///
/// `TimeOrdered` builds forward-chaining, capture-time-ordered folds and is the
/// default: it guarantees every test frame is later than its training frames and
/// purges near-duplicates around the split, so reported metrics reflect real
/// behaviour rather than temporal leakage. `TemporalBlock` retains the legacy
/// round-robin temporal-block folds with a shuffled-stratified fallback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CvStrategy {
    #[default]
    TimeOrdered,
    TemporalBlock,
}

/// Options accepted by [`cross_validate`].
///
/// `None` values mirror the TypeScript function's `Partial<CrossValidationOptions>`
/// and receive the same defaults at call time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CrossValidationOptions {
    pub cv_folds: Option<usize>,
    pub random_seed: Option<u64>,
    pub timestamps: Option<Vec<f64>>,
    pub gap_threshold_ms: Option<f64>,
    pub strategy: Option<CvStrategy>,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy)]
struct BorrowedFeature<'a, S>(&'a S);

impl<S: FeatureSource> FeatureSource for BorrowedFeature<'_, S> {
    type Bbox = S::Bbox;

    fn features(&self) -> &FeatureMap {
        self.0.features()
    }

    fn keypoints(&self) -> &[Keypoint] {
        self.0.keypoints()
    }

    fn bbox(&self) -> &Self::Bbox {
        self.0.bbox()
    }
}

/// The fold strategy used to produce a set of evaluation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvType {
    TemporalBlock,
    ShuffledStratified,
}

/// Aggregate metrics returned by cross-validation.
#[derive(Debug, Clone, PartialEq)]
pub struct CvMetrics {
    pub cv_accuracy: f64,
    pub cv_std: f64,
    pub mcc: f64,
    pub f1_score: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub fold_accuracies: Vec<f64>,
    pub balanced_accuracy: f64,
    pub accuracy_ci_low: f64,
    pub accuracy_ci_high: f64,
    pub worst_fold_accuracy: f64,
    pub cv_type: CvType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    MissingTimestamps,
    EmptyDataset,
    LengthMismatch { features: usize, labels: usize },
    NoValidFolds,
    Cancelled,
    Model(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTimestamps => {
                formatter.write_str("timestamps are required and must match labels length")
            }
            Self::EmptyDataset => {
                formatter.write_str("Cannot perform cross-validation on empty dataset")
            }
            Self::LengthMismatch { features, labels } => {
                write!(
                    formatter,
                    "Length mismatch: {features} features, {labels} labels"
                )
            }
            Self::NoValidFolds => formatter.write_str("Cross-validation produced no valid folds"),
            Self::Cancelled => formatter.write_str("Cross-validation cancelled"),
            Self::Model(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EvaluationError {}

#[derive(Debug, Clone, Copy)]
struct FoldMetrics {
    accuracy: f64,
    tp: usize,
    fp: usize,
    fn_: usize,
    tn: usize,
}

/// Runs temporal-block cross-validation with shuffled stratified fallback.
///
/// The classifier factory is invoked once per fold so no fitted state can leak
/// between folds. The source's prediction polarity is intentionally retained:
/// probabilities above `0.5` map to label `0`, otherwise to label `1`.
pub fn cross_validate<S, Factory>(
    extractor_config: &super::feature_extractor::FeatureExtractorConfig<S>,
    classifier_factory: Factory,
    features: &[S],
    labels: &[i32],
    options: CrossValidationOptions,
) -> Result<Option<CvMetrics>, EvaluationError>
where
    S: FeatureSource,
    Factory: Fn() -> Box<dyn BaseClassifier>,
{
    cross_validate_fallible_with_cancellation(
        extractor_config,
        || Ok(classifier_factory()),
        features,
        labels,
        options,
        None,
    )
}

pub fn cross_validate_cancellable<S, Factory>(
    extractor_config: &super::feature_extractor::FeatureExtractorConfig<S>,
    classifier_factory: Factory,
    features: &[S],
    labels: &[i32],
    options: CrossValidationOptions,
    cancellation: &CancellationToken,
) -> Result<Option<CvMetrics>, EvaluationError>
where
    S: FeatureSource,
    Factory: Fn() -> Box<dyn BaseClassifier>,
{
    cross_validate_fallible_with_cancellation(
        extractor_config,
        || Ok(classifier_factory()),
        features,
        labels,
        options,
        Some(cancellation),
    )
}

pub fn cross_validate_fallible<S, Factory>(
    extractor_config: &super::feature_extractor::FeatureExtractorConfig<S>,
    classifier_factory: Factory,
    features: &[S],
    labels: &[i32],
    options: CrossValidationOptions,
) -> Result<Option<CvMetrics>, EvaluationError>
where
    S: FeatureSource,
    Factory: Fn() -> Result<Box<dyn BaseClassifier>, String>,
{
    cross_validate_fallible_with_cancellation(
        extractor_config,
        classifier_factory,
        features,
        labels,
        options,
        None,
    )
}

fn cross_validate_fallible_with_cancellation<S, Factory>(
    extractor_config: &super::feature_extractor::FeatureExtractorConfig<S>,
    classifier_factory: Factory,
    features: &[S],
    labels: &[i32],
    options: CrossValidationOptions,
    cancellation: Option<&CancellationToken>,
) -> Result<Option<CvMetrics>, EvaluationError>
where
    S: FeatureSource,
    Factory: Fn() -> Result<Box<dyn BaseClassifier>, String>,
{
    check_cancellation(cancellation)?;
    let cv_folds = options.cv_folds.unwrap_or(5);
    let random_seed = options.random_seed.unwrap_or(TRAINING_CONFIG.random_seed) as f64;
    let gap_threshold_ms = options.gap_threshold_ms.unwrap_or(15_000.0);

    // Match the TypeScript early return: disabling CV does not require timestamps
    // or a dataset because no evaluation work is performed.
    if cv_folds <= 1 {
        eprintln!("[CV] Skipping CV (cvFolds={cv_folds})");
        return Ok(None);
    }

    let Some(timestamps) = options.timestamps.as_deref() else {
        return Err(EvaluationError::MissingTimestamps);
    };
    if timestamps.len() != labels.len() {
        return Err(EvaluationError::MissingTimestamps);
    }

    if features.is_empty() || labels.is_empty() {
        return Err(EvaluationError::EmptyDataset);
    }
    if features.len() != labels.len() {
        return Err(EvaluationError::LengthMismatch {
            features: features.len(),
            labels: labels.len(),
        });
    }

    eprintln!(
        "[CV] Starting {cv_folds}-fold cross-validation on {} samples",
        features.len()
    );

    let actual_folds = cv_folds.min(labels.len());
    if actual_folds < cv_folds {
        eprintln!(
            "Fold count limited from {cv_folds} to {actual_folds} (dataset size: {})",
            labels.len()
        );
    }

    let strategy = options.strategy.unwrap_or_default();

    // A single fold containing identical train/test indices is each helper's
    // sentinel for "no real cross-validation".
    let (folds, cv_type) = match strategy {
        CvStrategy::TimeOrdered => {
            let folds =
                create_time_ordered_folds(timestamps, labels, actual_folds, gap_threshold_ms);
            if is_single_identity_fold(&folds) || !has_multiclass_train_fold(&folds, labels) {
                // Two degeneracies force the shuffled-stratified fallback here.
                // (1) Small or time-clustered datasets cannot form a purged
                // time-ordered holdout, so `create_time_ordered_folds` returns
                // the all-data identity sentinel. (2) When the classes were
                // captured in disjoint time spans (e.g. every GOOD frame before
                // every BAD one, or all PRESENT before AWAY), the expanding
                // time-ordered windows can be empty or contain only the
                // earliest-captured class - no fold offers a two-class training
                // split, and fitting a single-class split aborts the whole run
                // in `Model::fit`. Falling back to shuffled stratified k-fold
                // caps the fold count to the smallest class size, so every fold
                // keeps both classes in its training split whenever that class
                // holds at least two samples.
                eprintln!(
                    "[CV] No time-ordered fold has a two-class training split - falling back to shuffled stratified k-fold"
                );
                let stratified = create_stratified_k_fold(labels, actual_folds, random_seed);
                if stratified.is_empty() || is_single_identity_fold(&stratified) {
                    eprintln!(
                        "[CV] Smallest class has fewer than 2 samples - cannot stratify; returning null metrics"
                    );
                    return Ok(None);
                }
                eprintln!(
                    "[CV] Created {} shuffled stratified folds",
                    stratified.len()
                );
                (stratified, CvType::ShuffledStratified)
            } else {
                eprintln!(
                    "[CV] Created {} time-ordered folds (gap threshold: {gap_threshold_ms}ms)",
                    folds.len()
                );
                (folds, CvType::TemporalBlock)
            }
        }
        CvStrategy::TemporalBlock => {
            let mut folds = create_temporal_block_k_fold(
                timestamps,
                labels,
                actual_folds,
                gap_threshold_ms,
                random_seed,
            );
            let mut using_temporal_blocks = true;

            if is_single_identity_fold(&folds) {
                eprintln!(
                    "[CV] Insufficient temporal blocks - falling back to regular shuffled k-fold CV"
                );
                folds = create_stratified_k_fold(labels, actual_folds, random_seed);
                using_temporal_blocks = false;

                if is_single_identity_fold(&folds) {
                    eprintln!(
                        "[CV] Insufficient data for cross-validation - returning null metrics"
                    );
                    return Ok(None);
                }
            }

            if using_temporal_blocks {
                eprintln!(
                    "[CV] Created {} temporal block folds (gap threshold: {gap_threshold_ms}ms)",
                    folds.len()
                );
                (folds, CvType::TemporalBlock)
            } else {
                eprintln!("[CV] Created {} shuffled stratified folds", folds.len());
                (folds, CvType::ShuffledStratified)
            }
        }
    };

    let mut fold_metrics = Vec::new();

    for (fold_index, fold) in folds.iter().enumerate() {
        eprintln!("[CV] Processing fold {}/{}", fold_index + 1, folds.len());

        check_cancellation(cancellation)?;
        let x_train = fold
            .train_indices
            .iter()
            .map(|&index| BorrowedFeature(&features[index]))
            .collect::<Vec<_>>();
        let y_train: Vec<i32> = fold
            .train_indices
            .iter()
            .map(|&index| labels[index])
            .collect();
        let x_test = fold
            .test_indices
            .iter()
            .map(|&index| BorrowedFeature(&features[index]))
            .collect::<Vec<_>>();
        let y_test: Vec<i32> = fold
            .test_indices
            .iter()
            .map(|&index| labels[index])
            .collect();

        if x_train.is_empty() {
            eprintln!(
                "[CV] Fold {} has empty training set, skipping",
                fold_index + 1
            );
            continue;
        }
        if x_test.is_empty() {
            eprintln!("[CV] Fold {} has empty test set, skipping", fold_index + 1);
            continue;
        }
        // A training split confined to one class cannot fit a binary model:
        // `Model::fit` requires at least one sample per class and would otherwise
        // abort the entire cross-validation (and thus the whole training run).
        // Expanding time-ordered windows can legitimately land inside a single
        // class, so skip such a fold exactly like an empty split and let the
        // class-balanced folds carry the metrics.
        if y_train.iter().all(|&label| label == y_train[0]) {
            eprintln!(
                "[CV] Fold {} training set has a single class, skipping",
                fold_index + 1
            );
            continue;
        }

        let borrowed_extractor_config = super::feature_extractor::FeatureExtractorConfig {
            feature_types: extractor_config.feature_types.clone(),
            normalization_mode: extractor_config.normalization_mode,
            dim_reduction_config: extractor_config.dim_reduction_config,
            unlabeled_samples: extractor_config
                .unlabeled_samples
                .iter()
                .map(BorrowedFeature)
                .collect(),
        };
        let extractor = FeatureExtractor::new(borrowed_extractor_config);
        let classifier = classifier_factory().map_err(EvaluationError::Model)?;
        let mut fold_model = Model::new(extractor, classifier);

        let result = (|| {
            check_cancellation(cancellation)?;
            fold_model
                .fit(&x_train, &y_train)
                .map_err(|error| EvaluationError::Model(error.to_string()))?;
            check_cancellation(cancellation)?;

            let predictions: Result<Vec<i32>, EvaluationError> = x_test
                .iter()
                .map(|sample| {
                    check_cancellation(cancellation)?;
                    let probability = fold_model
                        .predict(sample)
                        .map_err(|error| EvaluationError::Model(error.to_string()))?;
                    Ok(if probability > 0.5 { 0 } else { 1 })
                })
                .collect();
            let metrics = calculate_fold_metrics(&predictions?, &y_test);
            Ok(metrics)
        })();

        fold_model.dispose();
        let metrics = result?;
        eprintln!(
            "[CV] Fold {} accuracy: {:.1}%",
            fold_index + 1,
            metrics.accuracy * 100.0
        );
        fold_metrics.push(metrics);
    }

    if fold_metrics.is_empty() {
        return Err(EvaluationError::NoValidFolds);
    }

    let aggregated = aggregate_metrics(&fold_metrics, cv_type);
    eprintln!(
        "[CV] Cross-validation complete: {:.1}% ± {:.1}%",
        aggregated.cv_accuracy * 100.0,
        aggregated.cv_std * 100.0
    );

    Ok(Some(aggregated))
}

fn check_cancellation(cancellation: Option<&CancellationToken>) -> Result<(), EvaluationError> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        Err(EvaluationError::Cancelled)
    } else {
        Ok(())
    }
}

fn is_single_identity_fold(folds: &[CvFold]) -> bool {
    if folds.len() != 1 {
        return false;
    }

    folds[0].train_indices == folds[0].test_indices
}

/// True when at least one fold can actually train a binary classifier, i.e. its
/// training split holds two distinct labels. Time-ordered folds split purely by
/// capture time, so a dataset whose classes occupy disjoint time spans can yield
/// folds whose only non-empty training windows are single-class; those cannot be
/// fitted, so the caller falls back to a class-balanced split when this returns
/// false.
fn has_multiclass_train_fold(folds: &[CvFold], labels: &[i32]) -> bool {
    folds.iter().any(|fold| {
        let mut train_labels = fold.train_indices.iter().map(|&index| labels[index]);
        match train_labels.next() {
            Some(first) => train_labels.any(|label| label != first),
            None => false,
        }
    })
}

fn calculate_fold_metrics(predictions: &[i32], ground_truth: &[i32]) -> FoldMetrics {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    let mut tn = 0;

    for (index, &prediction) in predictions.iter().enumerate() {
        let actual = ground_truth[index];
        match (prediction, actual) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => tn += 1,
        }
    }

    FoldMetrics {
        accuracy: (tp + tn) as f64 / predictions.len() as f64,
        tp,
        fp,
        fn_,
        tn,
    }
}

fn aggregate_metrics(fold_metrics: &[FoldMetrics], cv_type: CvType) -> CvMetrics {
    let fold_accuracies: Vec<f64> = fold_metrics
        .iter()
        .map(|metrics| metrics.accuracy)
        .collect();
    let cv_accuracy = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;
    let variance = fold_accuracies
        .iter()
        .map(|accuracy| (accuracy - cv_accuracy).powi(2))
        .sum::<f64>()
        / fold_accuracies.len() as f64;
    let cv_std = variance.sqrt();

    let total_tp = fold_metrics.iter().map(|metrics| metrics.tp).sum::<usize>();
    let total_fp = fold_metrics.iter().map(|metrics| metrics.fp).sum::<usize>();
    let total_fn = fold_metrics
        .iter()
        .map(|metrics| metrics.fn_)
        .sum::<usize>();
    let total_tn = fold_metrics.iter().map(|metrics| metrics.tn).sum::<usize>();

    let numerator = total_tp as f64 * total_tn as f64 - total_fp as f64 * total_fn as f64;
    let denominator = (total_tp + total_fp) as f64
        * (total_tp + total_fn) as f64
        * (total_tn + total_fp) as f64
        * (total_tn + total_fn) as f64;
    let mcc = if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator.sqrt()
    };

    let precision = if total_tp + total_fp > 0 {
        total_tp as f64 / (total_tp + total_fp) as f64
    } else {
        0.0
    };
    let recall = if total_tp + total_fn > 0 {
        total_tp as f64 / (total_tp + total_fn) as f64
    } else {
        0.0
    };
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    let total_correct = total_tp + total_tn;
    let total_samples = total_tp + total_fp + total_fn + total_tn;
    let (accuracy_ci_low, accuracy_ci_high) = wilson_interval(total_correct, total_samples);
    let balanced_accuracy =
        balanced_accuracy_from_confusion(total_tn, total_fp, total_fn, total_tp);
    let worst_fold_accuracy = fold_accuracies
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let worst_fold_accuracy = if worst_fold_accuracy.is_finite() {
        worst_fold_accuracy
    } else {
        0.0
    };

    CvMetrics {
        cv_accuracy,
        cv_std,
        mcc,
        f1_score,
        confusion_matrix: vec![vec![total_tn, total_fp], vec![total_fn, total_tp]],
        fold_accuracies,
        balanced_accuracy,
        accuracy_ci_low,
        accuracy_ci_high,
        worst_fold_accuracy,
        cv_type,
    }
}

/// Wilson score 95% confidence interval (z = 1.959964) for a binomial proportion
/// `successes / total`, clamped to `[0, 1]`. An empty sample yields a degenerate
/// `[0, 0]` interval.
fn wilson_interval(successes: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    const Z: f64 = 1.959964;
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = Z * Z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    (
        (center - half).clamp(0.0, 1.0),
        (center + half).clamp(0.0, 1.0),
    )
}

/// Mean of per-class recall over the classes that actually appear in the pooled
/// binary confusion matrix. Absent classes are skipped so a dataset without one
/// class (e.g. no `away` frames) still yields an honest score.
fn balanced_accuracy_from_confusion(tn: usize, fp: usize, fn_: usize, tp: usize) -> f64 {
    let mut recall_sum = 0.0;
    let mut classes = 0;
    let actual_negative = tn + fp;
    if actual_negative > 0 {
        recall_sum += tn as f64 / actual_negative as f64;
        classes += 1;
    }
    let actual_positive = tp + fn_;
    if actual_positive > 0 {
        recall_sum += tp as f64 / actual_positive as f64;
        classes += 1;
    }
    if classes == 0 {
        0.0
    } else {
        recall_sum / classes as f64
    }
}

/// Returns zeroed metrics for callers that need a stable non-null shape.
pub fn get_empty_metrics() -> CvMetrics {
    CvMetrics {
        cv_accuracy: 0.0,
        cv_std: 0.0,
        mcc: 0.0,
        f1_score: 0.0,
        confusion_matrix: vec![vec![0, 0], vec![0, 0]],
        fold_accuracies: Vec::new(),
        balanced_accuracy: 0.0,
        accuracy_ci_low: 0.0,
        accuracy_ci_high: 0.0,
        worst_fold_accuracy: 0.0,
        cv_type: CvType::ShuffledStratified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_matches_reference_bounds() {
        let (low, high) = wilson_interval(8, 10);
        assert!((low - 0.490_1).abs() < 1e-3, "low={low}");
        assert!((high - 0.943_3).abs() < 1e-3, "high={high}");
        assert_eq!(wilson_interval(0, 0), (0.0, 0.0));
        // A perfect proportion keeps the upper bound at (clamped to) 1.
        let high_perfect = wilson_interval(4, 4).1;
        assert!(
            high_perfect <= 1.0 && (1.0 - high_perfect) < 1e-5,
            "high={high_perfect}"
        );
    }

    #[test]
    fn aggregate_metrics_reports_balanced_accuracy_ci_and_mcc() {
        // Hand-built pooled confusion matrix: tp=3, fp=2, fn=1, tn=4 over 10 samples.
        let fold = FoldMetrics {
            accuracy: 0.7,
            tp: 3,
            fp: 2,
            fn_: 1,
            tn: 4,
        };
        let metrics = aggregate_metrics(&[fold], CvType::TemporalBlock);

        // Balanced accuracy = mean(recall_pos=3/4, recall_neg=4/6).
        assert!((metrics.balanced_accuracy - 0.708_333).abs() < 1e-4);
        // MCC = (3*4 - 2*1) / sqrt(5*4*6*5) = 10 / sqrt(600).
        assert!((metrics.mcc - 0.408_248).abs() < 1e-4);

        let (low, high) = wilson_interval(7, 10);
        assert!((metrics.accuracy_ci_low - low).abs() < 1e-12);
        assert!((metrics.accuracy_ci_high - high).abs() < 1e-12);
        assert!((metrics.worst_fold_accuracy - 0.7).abs() < 1e-12);
    }

    #[test]
    fn aggregate_metrics_tracks_worst_fold_accuracy() {
        let folds = [
            FoldMetrics {
                accuracy: 0.8,
                tp: 4,
                fp: 1,
                fn_: 0,
                tn: 5,
            },
            FoldMetrics {
                accuracy: 0.6,
                tp: 3,
                fp: 2,
                fn_: 2,
                tn: 3,
            },
        ];
        let metrics = aggregate_metrics(&folds, CvType::TemporalBlock);
        assert!((metrics.worst_fold_accuracy - 0.6).abs() < 1e-12);
    }

    #[test]
    fn balanced_accuracy_uses_only_present_classes() {
        // No positive samples: balanced accuracy equals the single present class recall.
        assert!((balanced_accuracy_from_confusion(3, 1, 0, 0) - 0.75).abs() < 1e-12);
        assert_eq!(balanced_accuracy_from_confusion(0, 0, 0, 0), 0.0);
    }
}
