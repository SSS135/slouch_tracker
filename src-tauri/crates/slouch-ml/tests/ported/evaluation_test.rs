use std::collections::BTreeMap;

use slouch_domain::{BoundingBox, ExpandedBbox, FeatureId, InferenceResult};
use slouch_ml::ported::{
    base_classifier::BaseClassifier,
    constants::{RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_GAU_POOLED_DIMS},
    evaluation::{
        cross_validate, cross_validate_cancellable, get_empty_metrics, CancellationToken,
        CrossValidationOptions, CvStrategy, CvType, EvaluationError,
    },
    feature_extractor::FeatureExtractorConfig,
    knn_classifier::KnnClassifier,
    mlp_classifier::{MlpClassifier, MlpConfig},
    svm_classifier::{SvmClassifier, SvmConfig},
    types::{DimensionalityReductionConfig, DimensionalityReductionMethod, NormalizationMode},
};
const FEATURE_BACKBONE_AVG: FeatureId = FeatureId::BackboneFeatures;
const FEATURE_GAU_AVG: FeatureId = FeatureId::GauFeatures;

fn bbox() -> ExpandedBbox {
    let box_ = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
        score: 1.0,
        width: 1.0,
        height: 1.0,
    };
    ExpandedBbox {
        original: box_,
        expanded: box_,
    }
}

fn create_mock_features(label: bool, seed: usize) -> InferenceResult {
    let base_value = if label { 0.8_f64 } else { 0.2_f64 };
    let gau_features = (0..RTMPOSE_GAU_POOLED_DIMS)
        .map(|index| (base_value + (seed as f64 + index as f64).sin() * 0.1) as f32)
        .collect();
    let backbone_features = (0..RTMPOSE_BACKBONE_POOLED_DIMS)
        .map(|index| (base_value + (seed as f64 + index as f64).cos() * 0.1) as f32)
        .collect();

    InferenceResult {
        features: BTreeMap::from([
            (FEATURE_GAU_AVG, gau_features),
            (FEATURE_BACKBONE_AVG, backbone_features),
        ]),
        keypoints: Vec::new(),
        bbox: bbox(),
        classification: None,
    }
}

fn create_mock_dataset(
    good_count: usize,
    bad_count: usize,
) -> (Vec<InferenceResult>, Vec<i32>, Vec<f64>) {
    let mut raw_features = Vec::with_capacity(good_count + bad_count);
    let mut labels = Vec::with_capacity(good_count + bad_count);
    let mut timestamps = Vec::with_capacity(good_count + bad_count);
    let block_gap = 20_000.0;

    for index in 0..good_count {
        raw_features.push(create_mock_features(true, index));
        labels.push(0);
        timestamps.push(index as f64 * block_gap);
    }

    let bad_start_time = (good_count + 1) as f64 * block_gap;
    for index in 0..bad_count {
        raw_features.push(create_mock_features(false, index + good_count));
        labels.push(1);
        timestamps.push(bad_start_time + index as f64 * block_gap);
    }

    (raw_features, labels, timestamps)
}

#[test]
fn mock_feature_generation_matches_float32array_narrowing() {
    // Frozen golden f32 bit patterns for `base + sin(seed + index) * 0.1` narrowed to f32,
    // captured independently from the TS oracle formula (evaluation.test.ts:19-22) rather than
    // re-derived from create_mock_features. Comparing against fixed constants (not a re-run of
    // the generator's own expression) ensures a regression in the formula or the f64->f32
    // narrowing is actually caught instead of passing vacuously.
    const LAST: usize = RTMPOSE_GAU_POOLED_DIMS - 1;
    type BitCase = (bool, usize, [(usize, u32); 4]);
    let cases: [BitCase; 3] = [
        (
            true,
            0,
            [
                (0, 0x3f4c_cccd),
                (1, 0x3f62_5777),
                (17, 0x3f34_302f),
                (LAST, 0x3f3f_d61d),
            ],
        ),
        (
            false,
            7,
            [
                (0, 0x3e88_09a8),
                (1, 0x3e99_0e1e),
                (17, 0x3de0_2337),
                (LAST, 0x3dd7_63ce),
            ],
        ),
        (
            true,
            9_999,
            [
                (0, 0x3f5d_1576),
                (1, 0x3f44_f9ed),
                (17, 0x3f5b_4f67),
                (LAST, 0x3f48_c2e8),
            ],
        ),
    ];
    for (label, seed, expected) in cases {
        let generated = create_mock_features(label, seed);
        for (index, expected_bits) in expected {
            assert_eq!(
                generated.features[&FEATURE_GAU_AVG][index].to_bits(),
                expected_bits,
                "GAU feature bit mismatch at label={label} seed={seed} index={index}"
            );
        }
    }
}

fn config(
    feature_type: FeatureId,
    normalization_mode: NormalizationMode,
    method: DimensionalityReductionMethod,
    components: usize,
) -> FeatureExtractorConfig {
    FeatureExtractorConfig {
        feature_types: vec![feature_type],
        normalization_mode,
        dim_reduction_config: DimensionalityReductionConfig { method, components },
        unlabeled_samples: Vec::new(),
    }
}

// These cases exercise the legacy temporal-block path with singleton-block
// datasets, so they pin the strategy rather than using the time-ordered default.
fn options(cv_folds: usize, timestamps: Vec<f64>) -> CrossValidationOptions {
    CrossValidationOptions {
        cv_folds: Some(cv_folds),
        random_seed: None,
        timestamps: Some(timestamps),
        gap_threshold_ms: None,
        strategy: Some(CvStrategy::TemporalBlock),
    }
}

fn evaluate<F>(
    extractor_config: FeatureExtractorConfig,
    classifier_factory: F,
    features: &[InferenceResult],
    labels: &[i32],
    options: CrossValidationOptions,
) -> slouch_ml::ported::evaluation::CvMetrics
where
    F: Fn() -> Box<dyn BaseClassifier>,
{
    cross_validate(
        &extractor_config,
        classifier_factory,
        features,
        labels,
        options,
    )
    .unwrap()
    .expect("cross-validation should produce metrics")
}

#[test]
fn performs_two_fold_cv_with_knn_classifier() {
    let (features, labels, timestamps) = create_mock_dataset(5, 5);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(3, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    assert!(metrics.cv_accuracy > 0.0);
    assert!(metrics.cv_accuracy <= 1.0);
    assert!(metrics.cv_std >= 0.0);
    assert_eq!(metrics.fold_accuracies.len(), 2);
    assert_eq!(metrics.confusion_matrix.len(), 2);
    assert_eq!(metrics.confusion_matrix[0].len(), 2);
    assert!((-1.0..=1.0).contains(&metrics.mcc));
    assert!(metrics.f1_score >= 0.0);
}

#[test]
fn performs_cv_with_mlp_classifier() {
    let (features, labels, timestamps) = create_mock_dataset(4, 4);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::ZScore,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                MlpClassifier::new(MlpConfig {
                    hidden_layers: 0,
                    hidden_size: 64,
                    max_iterations: 2,
                    learning_rate: 0.01,
                    ..MlpConfig::default()
                })
                .unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    assert!(metrics.cv_accuracy > 0.0);
    assert_eq!(metrics.fold_accuracies.len(), 2);
}

#[test]
fn performs_cv_with_svm_classifier() {
    let (features, labels, timestamps) = create_mock_dataset(4, 4);
    let metrics = evaluate(
        config(
            FEATURE_BACKBONE_AVG,
            NormalizationMode::ZScore,
            DimensionalityReductionMethod::RandomProjection,
            8,
        ),
        || {
            Box::new(
                SvmClassifier::new(SvmConfig {
                    c: 1.0,
                    max_iterations: 3,
                    ..SvmConfig::default()
                })
                .unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    assert!(metrics.cv_accuracy > 0.0);
    assert_eq!(metrics.fold_accuracies.len(), 2);
}

#[test]
fn supports_all_normalization_modes() {
    let (features, labels, timestamps) = create_mock_dataset(3, 3);
    for normalization_mode in [
        NormalizationMode::None,
        NormalizationMode::Layer,
        NormalizationMode::ZScore,
    ] {
        let metrics = evaluate(
            config(
                FEATURE_GAU_AVG,
                normalization_mode,
                DimensionalityReductionMethod::None,
                1,
            ),
            || {
                Box::new(
                    KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0)
                        .unwrap(),
                )
            },
            &features,
            &labels,
            options(2, timestamps.clone()),
        );
        assert!(metrics.cv_accuracy > 0.0);
        assert_eq!(metrics.fold_accuracies.len(), 2);
    }
}

#[test]
fn supports_random_projection_dimensionality_reduction() {
    let (features, labels, timestamps) = create_mock_dataset(3, 3);
    let metrics = evaluate(
        config(
            FEATURE_BACKBONE_AVG,
            NormalizationMode::ZScore,
            DimensionalityReductionMethod::RandomProjection,
            4,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    assert!(metrics.cv_accuracy > 0.0);
    assert_eq!(metrics.fold_accuracies.len(), 2);
}

#[test]
fn rejects_empty_dataset() {
    let result = cross_validate(
        &config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(3, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &[],
        &[],
        options(5, Vec::new()),
    );

    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Cannot perform cross-validation on empty dataset"));
}

#[test]
fn rejects_length_mismatch_before_training() {
    let (features, _, timestamps) = create_mock_dataset(4, 4);
    let result = cross_validate(
        &config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(3, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &[0, 1],
        options(5, timestamps),
    );

    assert!(result
        .unwrap_err()
        .to_string()
        .contains("timestamps are required and must match labels length"));
}

#[test]
fn limits_fold_count_to_dataset_size() {
    let (features, labels, timestamps) = create_mock_dataset(3, 3);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(1, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(5, timestamps),
    );

    assert!(metrics.fold_accuracies.len() <= 3);
}

#[test]
fn default_strategy_runs_stratified_cv_on_small_time_clustered_dataset() {
    // Regression: a small, balanced dataset captured inside the purge window
    // (all timestamps equal, so no time-ordered holdout is possible) must still
    // produce cross-validation metrics under the default TimeOrdered strategy by
    // falling back to shuffled stratified k-fold. Previously CV was skipped, which
    // left the model with empty fold accuracies and broke persistence.
    let (features, labels, _) = create_mock_dataset(6, 6);
    let clustered_timestamps = vec![1_000.0; features.len()];
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        CrossValidationOptions {
            cv_folds: Some(5),
            random_seed: Some(42),
            timestamps: Some(clustered_timestamps),
            gap_threshold_ms: None,
            // None => default TimeOrdered strategy, which must fall back here.
            strategy: None,
        },
    );

    assert!(
        !metrics.fold_accuracies.is_empty(),
        "CV must run and report fold accuracies for a tiny balanced dataset"
    );
    // Fold count is capped to the smallest class (6 samples) and the requested 5.
    assert!(metrics.fold_accuracies.len() <= 5);
    assert!(metrics.fold_accuracies.len() >= 2);
    assert_eq!(metrics.cv_type, CvType::ShuffledStratified);
    assert!(metrics.cv_accuracy >= 0.0 && metrics.cv_accuracy <= 1.0);
}

#[test]
fn creates_fresh_classifier_instances_for_each_fold() {
    let (features, labels, timestamps) = create_mock_dataset(4, 4);
    let created = std::cell::Cell::new(0_usize);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::ZScore,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            created.set(created.get() + 1);
            Box::new(
                KnnClassifier::new(3, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    assert_eq!(created.get(), 2);
    assert_eq!(metrics.fold_accuracies.len(), 2);
}

#[test]
fn custom_random_seed_is_reproducible() {
    let (features, labels, timestamps) = create_mock_dataset(3, 3);
    let mut first_options = options(2, timestamps.clone());
    first_options.random_seed = Some(42);
    let mut second_options = options(2, timestamps);
    second_options.random_seed = Some(42);

    let first = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        first_options,
    );
    let second = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        second_options,
    );

    assert!((first.cv_accuracy - second.cv_accuracy).abs() <= 1e-5);
    assert_eq!(first.fold_accuracies, second.fold_accuracies);
}

#[test]
fn computes_confusion_matrix_and_mcc() {
    let (features, labels, timestamps) = create_mock_dataset(4, 4);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(3, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
    );

    let tn = metrics.confusion_matrix[0][0];
    let fp = metrics.confusion_matrix[0][1];
    let fn_ = metrics.confusion_matrix[1][0];
    let tp = metrics.confusion_matrix[1][1];
    assert_eq!(tp + fp + fn_ + tn, 8);
    assert!([tp, fp, fn_, tn].iter().all(|count| *count <= labels.len()));

    let numerator = (tp * tn) as f64 - (fp * fn_) as f64;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f64;
    let calculated_mcc = if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator.sqrt()
    };
    assert!((metrics.mcc - calculated_mcc).abs() <= 1e-5);
}

#[test]
fn cancellation_is_observed_before_fold_work() {
    let (features, labels, timestamps) = create_mock_dataset(3, 3);
    let token = CancellationToken::default();
    token.cancel();
    let result = cross_validate_cancellable(
        &config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        options(2, timestamps),
        &token,
    );

    assert_eq!(result.unwrap_err(), EvaluationError::Cancelled);
}

#[test]
fn returns_empty_metrics_structure() {
    let metrics = get_empty_metrics();

    assert_eq!(metrics.cv_accuracy, 0.0);
    assert_eq!(metrics.cv_std, 0.0);
    assert_eq!(metrics.mcc, 0.0);
    assert_eq!(metrics.f1_score, 0.0);
    assert_eq!(metrics.confusion_matrix, vec![vec![0, 0], vec![0, 0]]);
    assert!(metrics.fold_accuracies.is_empty());
}

#[test]
fn time_ordered_cv_skips_single_class_windows_when_classes_are_time_separated() {
    // Regression: all GOOD frames were captured before all BAD frames, >15s apart,
    // so the default time-ordered folds place several early expanding windows
    // entirely inside the GOOD span. Fitting such a single-class fold used to abort
    // the whole run with "Not enough class 1 frames: 0 < 1". CV must now skip those
    // folds and still report metrics from the later two-class folds.
    let (features, labels, timestamps) = create_mock_dataset(6, 5);
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        CrossValidationOptions {
            cv_folds: Some(5),
            random_seed: Some(42),
            timestamps: Some(timestamps),
            gap_threshold_ms: None,
            // None => default TimeOrdered strategy.
            strategy: None,
        },
    );

    assert!(
        !metrics.fold_accuracies.is_empty(),
        "CV must report metrics despite time-separated classes"
    );
    assert!(metrics.cv_accuracy >= 0.0 && metrics.cv_accuracy <= 1.0);
}

#[test]
fn time_ordered_cv_falls_back_when_no_fold_has_a_two_class_train_window() {
    // Exact reproduction of the reported posture failure: GOOD frames bracket the
    // BAD frames in time (good early, bad in the middle, good again), so the only
    // non-empty purged time-ordered training window is a single early GOOD frame.
    // No fold offers a two-class training split, so the default strategy must fall
    // back to shuffled stratified k-fold rather than aborting the run.
    let labels = vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0];
    let timestamps_ms = vec![
        0.0, 1_577.0, 2_641.0, 4_293.0, 5_328.0, 6_406.0, 10_024.0, 11_563.0, 15_104.0, 15_937.0,
        19_310.0,
    ];
    let features: Vec<_> = labels
        .iter()
        .enumerate()
        .map(|(index, &label)| create_mock_features(label == 1, index))
        .collect();
    let metrics = evaluate(
        config(
            FEATURE_GAU_AVG,
            NormalizationMode::None,
            DimensionalityReductionMethod::None,
            1,
        ),
        || {
            Box::new(
                KnnClassifier::new(2, slouch_ml::ported::types::KnnKernel::Cosine, 1.0).unwrap(),
            )
        },
        &features,
        &labels,
        CrossValidationOptions {
            cv_folds: Some(5),
            random_seed: Some(42),
            timestamps: Some(timestamps_ms),
            gap_threshold_ms: None,
            strategy: None,
        },
    );

    assert!(
        !metrics.fold_accuracies.is_empty(),
        "fallback CV must report fold accuracies"
    );
    assert_eq!(metrics.cv_type, CvType::ShuffledStratified);
}
