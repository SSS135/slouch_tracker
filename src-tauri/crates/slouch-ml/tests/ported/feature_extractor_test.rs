use std::collections::BTreeMap;

use slouch_domain::{BoundingBox, FeatureId, FeatureMap, FeatureSource, Keypoint};
use slouch_ml::ported::{
    constants::{RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_GAU_POOLED_DIMS},
    feature_extractor::{FeatureExtractor, FeatureExtractorConfig},
    types::{
        DimensionalityReductionConfig, DimensionalityReductionMethod, NormalizationMode,
        SerializedFeatureExtractor,
    },
};

const GAU: FeatureId = FeatureId::GauFeatures;
const BACKBONE: FeatureId = FeatureId::BackboneFeatures;

#[derive(Clone)]
struct MockFeatures {
    features: FeatureMap,
    keypoints: Vec<Keypoint>,
    bbox: BoundingBox,
}

impl FeatureSource for MockFeatures {
    type Bbox = BoundingBox;

    fn features(&self) -> &FeatureMap {
        &self.features
    }

    fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    fn bbox(&self) -> &Self::Bbox {
        &self.bbox
    }
}

fn create_mock_features(seed: usize) -> MockFeatures {
    let gau_features = (0..RTMPOSE_GAU_POOLED_DIMS)
        .map(|index| ((seed as f64 + index as f64 * 0.1).sin() + 0.5) as f32)
        .collect();
    let backbone_features = (0..RTMPOSE_BACKBONE_POOLED_DIMS)
        .map(|index| ((seed as f64 + index as f64 * 0.1).cos() + 0.5) as f32)
        .collect();

    MockFeatures {
        features: BTreeMap::from([(GAU, gau_features), (BACKBONE, backbone_features)]),
        keypoints: Vec::new(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

fn create_mock_dataset(count: usize) -> (Vec<MockFeatures>, Vec<i32>) {
    (
        (0..count).map(create_mock_features).collect(),
        (0..count).map(|index| (index % 2) as i32).collect(),
    )
}

fn config(
    feature_types: Vec<FeatureId>,
    normalization_mode: NormalizationMode,
    method: DimensionalityReductionMethod,
    components: usize,
) -> FeatureExtractorConfig<MockFeatures> {
    FeatureExtractorConfig {
        feature_types,
        normalization_mode,
        dim_reduction_config: DimensionalityReductionConfig { method, components },
        unlabeled_samples: Vec::new(),
    }
}

fn none_config(feature_types: Vec<FeatureId>) -> FeatureExtractorConfig<MockFeatures> {
    config(
        feature_types,
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    )
}

fn assert_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual} to be within {tolerance} of {expected}"
    );
}

fn assert_error_contains<T, E: std::fmt::Display>(result: Result<T, E>, text: &str) {
    assert!(
        result.is_err_and(|error| error.to_string().contains(text)),
        "expected error containing {text:?}"
    );
}

#[test]
fn constructor_creates_unfitted_extractor_with_default_config() {
    let extractor = FeatureExtractor::new(none_config(vec![GAU]));

    assert!(!extractor.is_fitted());
}

#[test]
fn constructor_accepts_z_score_normalization() {
    let extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));

    assert!(!extractor.is_fitted());
}

#[test]
fn constructor_accepts_random_projection() {
    let extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));

    assert!(!extractor.is_fitted());
}

#[test]
fn fits_on_dataset_with_sufficient_data() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);

    extractor.fit(&features, &labels).unwrap();

    assert!(extractor.is_fitted());
    assert_eq!(extractor.get_output_dimensions(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn fit_with_z_score_saves_normalization_parameters() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);

    extractor.fit(&features, &labels).unwrap();
    let serialized = extractor.to_json().unwrap();

    assert!(serialized.normalization_mean.is_some());
    assert!(serialized.normalization_std.is_some());
    assert_eq!(
        serialized.normalization_mean.unwrap().len(),
        RTMPOSE_GAU_POOLED_DIMS
    );
    assert_eq!(
        serialized.normalization_std.unwrap().len(),
        RTMPOSE_GAU_POOLED_DIMS
    );
}

#[test]
fn fit_with_layer_normalization_does_not_save_batch_parameters() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::Layer,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);

    extractor.fit(&features, &labels).unwrap();
    let serialized = extractor.to_json().unwrap();

    assert!(serialized.normalization_mean.is_none());
    assert!(serialized.normalization_std.is_none());
}

#[test]
fn fit_with_random_projection_reduces_dimensions() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);

    extractor.fit(&features, &labels).unwrap();

    assert!(extractor.is_fitted());
    assert_eq!(extractor.get_output_dimensions(), 50);
}

#[test]
fn fit_concatenates_multiple_feature_types() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU, BACKBONE]));
    let (features, labels) = create_mock_dataset(10);

    extractor.fit(&features, &labels).unwrap();

    assert!(extractor.is_fitted());
    assert_eq!(
        extractor.get_output_dimensions(),
        RTMPOSE_GAU_POOLED_DIMS + RTMPOSE_BACKBONE_POOLED_DIMS
    );
}

#[test]
fn fit_rejects_empty_dataset() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));

    assert_error_contains(
        extractor.fit(&[], &[]),
        "Cannot fit FeatureExtractor on empty dataset",
    );
}

#[test]
fn fit_rejects_mismatched_features_and_labels() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, _) = create_mock_dataset(10);

    assert_error_contains(extractor.fit(&features, &[0, 1]), "Length mismatch");
}

#[test]
fn transforms_single_sample_after_fitting() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let transformed = extractor.transform(&create_mock_features(999)).unwrap();

    assert_eq!(transformed.len(), RTMPOSE_GAU_POOLED_DIMS);
    assert_finite(&transformed);
}

#[test]
fn transforms_with_z_score_normalization_parameters() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let transformed = extractor.transform(&create_mock_features(999)).unwrap();

    assert_finite(&transformed);
}

#[test]
fn transforms_with_layer_normalization() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::Layer,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let transformed = extractor.transform(&create_mock_features(999)).unwrap();

    assert_finite(&transformed);
}

#[test]
fn transforms_with_random_projection() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let transformed = extractor.transform(&create_mock_features(999)).unwrap();

    assert_eq!(transformed.len(), 50);
}

#[test]
fn transform_before_fit_reports_dimension_mismatch() {
    let extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));

    let error = extractor.transform(&create_mock_features(999)).unwrap_err();
    assert!(error
        .to_string()
        .contains("Expected 0 features after concatenation"));
}

#[test]
fn transform_rejects_missing_feature_dependency() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let invalid = MockFeatures {
        features: BTreeMap::new(),
        keypoints: Vec::new(),
        bbox: create_mock_features(0).bbox,
    };

    assert_error_contains(
        extractor.transform(&invalid),
        "not available in this container",
    );
}

#[test]
fn transforms_a_batch_of_samples() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let test_features = vec![create_mock_features(998), create_mock_features(999)];
    let transformed = extractor.transform_batch(&test_features).unwrap();

    assert_eq!(transformed.len(), 2);
    assert_eq!(transformed[0].len(), RTMPOSE_GAU_POOLED_DIMS);
    assert_eq!(transformed[1].len(), RTMPOSE_GAU_POOLED_DIMS);
    assert_finite(&transformed[0]);
    assert_finite(&transformed[1]);
}

#[test]
fn transforms_an_empty_batch() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let empty: Vec<MockFeatures> = Vec::new();
    assert!(extractor.transform_batch(&empty).unwrap().is_empty());
}

#[test]
fn serializes_and_deserializes_without_normalization() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let serialized = extractor.to_json().unwrap();
    assert_eq!(serialized.feature_types, vec![GAU.as_str().to_owned()]);
    assert_eq!(serialized.normalization_mode, NormalizationMode::None);
    assert_eq!(
        serialized.dim_reduction_config.method,
        DimensionalityReductionMethod::None
    );
    assert_eq!(serialized.concatenated_dimensions, RTMPOSE_GAU_POOLED_DIMS);
    assert!(serialized.normalization_mean.is_none());
    assert!(serialized.normalization_std.is_none());
    assert!(serialized.dim_reduction_transformer.is_none());

    let restored = FeatureExtractor::<MockFeatures>::from_json(serialized).unwrap();
    assert!(restored.is_fitted());
    assert_eq!(restored.get_output_dimensions(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn serializes_and_deserializes_with_z_score_normalization() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let serialized = extractor.to_json().unwrap();
    assert_eq!(serialized.normalization_mode, NormalizationMode::ZScore);
    assert!(serialized.normalization_mean.is_some());
    assert!(serialized.normalization_std.is_some());

    let restored = FeatureExtractor::<MockFeatures>::from_json(serialized).unwrap();
    assert!(restored.is_fitted());

    let test_features = create_mock_features(999);
    let original = extractor.transform(&test_features).unwrap();
    let round_trip = restored.transform(&test_features).unwrap();
    assert_eq!(original.len(), round_trip.len());
    for (actual, expected) in round_trip.iter().zip(original.iter()) {
        assert_close(*actual, *expected, 4.999e-6);
    }
}

#[test]
fn serializes_and_deserializes_with_random_projection() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let serialized = extractor.to_json().unwrap();
    let json = serde_json::to_value(&serialized).unwrap();
    assert_eq!(json["dimReductionTransformer"]["type"], "random_projection");

    let restored = FeatureExtractor::<MockFeatures>::from_json(serialized).unwrap();
    assert!(restored.is_fitted());
    assert_eq!(restored.get_output_dimensions(), 50);

    let test_features = create_mock_features(999);
    let original = extractor.transform(&test_features).unwrap();
    let round_trip = restored.transform(&test_features).unwrap();
    assert_eq!(original.len(), round_trip.len());
    for (actual, expected) in round_trip.iter().zip(original.iter()) {
        assert_close(*actual, *expected, 4.999e-6);
    }
}

#[test]
fn source_five_decimal_tolerance_accepts_four_micro_and_rejects_six_micro() {
    assert_close(4e-6, 0.0, 4.999e-6);
    assert!(std::panic::catch_unwind(|| assert_close(6e-6, 0.0, 4.999e-6)).is_err());
}

#[test]
fn fit_rejects_non_finite_labeled_and_unlabeled_values() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (mut features, labels) = create_mock_dataset(2);
        features[0].features.get_mut(&GAU).unwrap()[0] = invalid;
        let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
        assert_error_contains(extractor.fit(&features, &labels), "must be finite");

        let (features, labels) = create_mock_dataset(2);
        let mut unlabeled = create_mock_features(99);
        unlabeled.features.get_mut(&GAU).unwrap()[0] = invalid;
        let mut extractor = FeatureExtractor::new(FeatureExtractorConfig {
            feature_types: vec![GAU],
            normalization_mode: NormalizationMode::ZScore,
            dim_reduction_config: DimensionalityReductionConfig {
                method: DimensionalityReductionMethod::None,
                components: 1,
            },
            unlabeled_samples: vec![unlabeled],
        });
        assert_error_contains(extractor.fit(&features, &labels), "must be finite");
    }
}

#[test]
fn restoration_rejects_inconsistent_dimensions_and_normalization_state() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(4);
    extractor.fit(&features, &labels).unwrap();
    let valid = extractor.to_json().unwrap();

    let mut wrong_dimensions = valid.clone();
    wrong_dimensions.concatenated_dimensions += 1;
    assert_error_contains(
        FeatureExtractor::<MockFeatures>::from_json(wrong_dimensions),
        "concatenatedDimensions",
    );

    let mut wrong_mean = valid.clone();
    wrong_mean.normalization_mean = Some(vec![0.0]);
    assert_error_contains(
        FeatureExtractor::<MockFeatures>::from_json(wrong_mean),
        "normalization vectors",
    );

    let mut invalid_std = valid;
    invalid_std.normalization_std.as_mut().unwrap()[0] = 0.0;
    assert_error_contains(
        FeatureExtractor::<MockFeatures>::from_json(invalid_std),
        "standard deviations",
    );
}

#[test]
fn restoration_rejects_unsupported_reduction_methods() {
    for method in [
        DimensionalityReductionMethod::PlsDa,
        DimensionalityReductionMethod::LinearNca,
    ] {
        let state = SerializedFeatureExtractor {
            feature_types: vec![GAU.as_str().to_owned()],
            normalization_mode: NormalizationMode::None,
            dim_reduction_config: DimensionalityReductionConfig {
                method,
                components: 1,
            },
            concatenated_dimensions: RTMPOSE_GAU_POOLED_DIMS,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        };
        assert_error_contains(
            FeatureExtractor::<MockFeatures>::from_json(state),
            "Unsupported dimensionality reduction method",
        );
    }
}

#[test]
fn rejects_corrupt_serialized_data_at_the_deserialization_boundary() {
    let invalid = serde_json::json!({
        "featureTypes": ["invalid_feature"],
        "normalizationMode": "invalid",
        "dimReductionConfig": { "method": "invalid", "components": -1 },
        "concatenatedDimensions": "not a number"
    });

    assert!(serde_json::from_value::<SerializedFeatureExtractor>(invalid).is_err());
}

#[test]
fn dispose_does_not_throw() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    extractor.dispose();
}

#[test]
fn dispose_releases_random_projection_transformer() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();
    assert!(extractor.transform(&create_mock_features(999)).is_ok());

    extractor.dispose();

    assert!(extractor.transform(&create_mock_features(999)).is_err());
}

#[test]
fn returns_concatenated_dimensions_without_reduction() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU, BACKBONE]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    assert_eq!(
        extractor.get_output_dimensions(),
        RTMPOSE_GAU_POOLED_DIMS + RTMPOSE_BACKBONE_POOLED_DIMS
    );
}

#[test]
fn returns_reduced_dimensions_with_random_projection() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::RandomProjection,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    assert_eq!(extractor.get_output_dimensions(), 50);
}

#[test]
fn none_normalization_preserves_input_values() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let test_features = create_mock_features(999);
    let original = test_features.features.get(&GAU).unwrap();
    let transformed = extractor.transform(&test_features).unwrap();

    for (actual, expected) in transformed.iter().zip(original.iter()) {
        assert_close(*actual, *expected, 1e-6);
    }
}

#[test]
fn layer_normalization_has_zero_mean_and_unit_std_without_l2_normalization() {
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::Layer,
        DimensionalityReductionMethod::None,
        1,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    let transformed = extractor.transform(&create_mock_features(999)).unwrap();
    let mean = transformed.iter().sum::<f32>() / transformed.len() as f32;
    let variance = transformed
        .iter()
        .map(|value| {
            let difference = *value - mean;
            difference * difference
        })
        .sum::<f32>()
        / transformed.len() as f32;
    let std = variance.sqrt();

    assert_close(mean, 0.0, 1e-3);
    assert_close(std, 1.0, 1e-2);
}

#[test]
fn is_fitted_is_false_before_fit() {
    let extractor = FeatureExtractor::new(none_config(vec![GAU]));

    assert!(!extractor.is_fitted());
}

#[test]
fn is_fitted_is_true_after_fit() {
    let mut extractor = FeatureExtractor::new(none_config(vec![GAU]));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    assert!(extractor.is_fitted());
}

fn gau_constant(value: f32) -> MockFeatures {
    MockFeatures {
        features: BTreeMap::from([(GAU, vec![value; RTMPOSE_GAU_POOLED_DIMS])]),
        keypoints: Vec::new(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

#[test]
fn calibrated_normalization_centers_reference_class_unlike_z_score() {
    // Good rows (label 0) form the reference cluster; the bad row (label 1) sits away.
    let features = vec![gau_constant(0.2), gau_constant(0.3), gau_constant(1.0)];
    let labels = vec![0, 0, 1];

    let mut calibrated = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::Calibrated,
        DimensionalityReductionMethod::None,
        1,
    ));
    calibrated.fit(&features, &labels).unwrap();

    let good_a = calibrated.transform(&features[0]).unwrap();
    let good_b = calibrated.transform(&features[1]).unwrap();
    let bad = calibrated.transform(&features[2]).unwrap();

    // The mean of the transformed good rows is ~0 per dimension.
    for index in 0..RTMPOSE_GAU_POOLED_DIMS {
        assert_close((good_a[index] + good_b[index]) / 2.0, 0.0, 1e-4);
    }
    // The bad row is a consistent, large deviation from the good baseline.
    assert!(bad.iter().all(|value| value.abs() > 5.0));

    // z-score centers on the overall mean, so the good rows are NOT centered.
    let mut z_score = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    z_score.fit(&features, &labels).unwrap();
    let z_good_a = z_score.transform(&features[0]).unwrap();
    let z_good_b = z_score.transform(&features[1]).unwrap();
    let z_good_mean = (z_good_a[0] + z_good_b[0]) / 2.0;
    assert!(
        z_good_mean.abs() > 0.3,
        "z-score good-class mean should be far from zero, got {z_good_mean}"
    );
}

#[test]
fn calibrated_falls_back_to_z_score_when_reference_class_is_empty() {
    // No label-0 samples: calibrated must not panic and must compute mean/std over
    // all rows, exactly like z-score.
    let features = vec![gau_constant(0.2), gau_constant(0.3), gau_constant(1.0)];
    let labels = vec![1, 1, 1];

    let mut calibrated = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::Calibrated,
        DimensionalityReductionMethod::None,
        1,
    ));
    calibrated.fit(&features, &labels).unwrap();
    let calibrated_state = calibrated.to_json().unwrap();

    let mut z_score = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    ));
    z_score.fit(&features, &labels).unwrap();
    let z_state = z_score.to_json().unwrap();

    let calibrated_mean = calibrated_state.normalization_mean.unwrap();
    let calibrated_std = calibrated_state.normalization_std.unwrap();
    let z_mean = z_state.normalization_mean.unwrap();
    let z_std = z_state.normalization_std.unwrap();
    assert_eq!(calibrated_mean.len(), z_mean.len());
    for (calibrated_value, z_value) in calibrated_mean.iter().zip(z_mean.iter()) {
        assert!((calibrated_value - z_value).abs() < 1e-9);
    }
    for (calibrated_value, z_value) in calibrated_std.iter().zip(z_std.iter()) {
        assert!((calibrated_value - z_value).abs() < 1e-9);
    }
}

/// An upright COCO-17 pose whose head/trunk geometry shifts with `seed`, so a batch
/// of these carries real variance for torso_invariant (a 7-dim computed feature).
fn create_torso_mock(seed: usize) -> MockFeatures {
    let t = seed as f64;
    let points: [(f64, f64); 17] = [
        (0.50, 0.20 + t * 0.006), // nose — head flexion varies with seed
        (0.47, 0.18),             // left eye
        (0.53, 0.18),             // right eye
        (0.45, 0.20),             // left ear
        (0.55, 0.20),             // right ear
        (0.40 - t * 0.001, 0.35), // left shoulder
        (0.60 + t * 0.001, 0.35), // right shoulder
        (0.38, 0.50),             // left elbow
        (0.62, 0.50),             // right elbow
        (0.37, 0.62),             // left wrist
        (0.63, 0.62),             // right wrist
        (0.44, 0.65),             // left hip
        (0.56, 0.65),             // right hip
        (0.44, 0.80),             // left knee
        (0.56, 0.80),             // right knee
        (0.44, 0.95),             // left ankle
        (0.56, 0.95),             // right ankle
    ];
    MockFeatures {
        features: BTreeMap::new(),
        keypoints: points
            .iter()
            .map(|(x, y)| Keypoint::new(*x, *y, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

#[test]
fn pca_clamped_to_feature_dimensions_records_effective_count_and_round_trips() {
    // torso_invariant is 7-dimensional, so PCA with 30 requested components cannot
    // exceed rank 7 no matter how many rows are fitted — the user's failing
    // configuration. The extractor must degrade to 7 and stay internally consistent
    // through serialize/deserialize instead of the loader rejecting it.
    let mut extractor = FeatureExtractor::new(config(
        vec![FeatureId::TorsoInvariant],
        NormalizationMode::None,
        DimensionalityReductionMethod::Pca,
        30,
    ));
    let features = (0..30).map(create_torso_mock).collect::<Vec<_>>();
    let labels = (0..30usize).map(|index| (index % 2) as i32).collect::<Vec<_>>();
    extractor.fit(&features, &labels).unwrap();

    assert_eq!(extractor.get_output_dimensions(), 7);
    let serialized = extractor.to_json().unwrap();
    assert_eq!(serialized.concatenated_dimensions, 7);
    assert_eq!(serialized.dim_reduction_config.components, 7);

    // from_json is the exact validator (reached via Model::from_json) that previously
    // rejected the model with "PCA state dimensions do not match extractor
    // configuration"; it must now accept the clamped state.
    let restored = FeatureExtractor::<MockFeatures>::from_json(serialized).unwrap();
    assert_eq!(restored.get_output_dimensions(), 7);
    let transformed = restored.transform(&create_torso_mock(999)).unwrap();
    assert_eq!(transformed.len(), 7);
}

#[test]
fn pca_clamped_by_sample_count_records_effective_count_and_round_trips() {
    // GAU is 256-dimensional but only 10 rows are fitted, so PCA caps at rows - 1 = 9
    // and the serialized config must record 9, agreeing with the fitted state.
    let mut extractor = FeatureExtractor::new(config(
        vec![GAU],
        NormalizationMode::None,
        DimensionalityReductionMethod::Pca,
        50,
    ));
    let (features, labels) = create_mock_dataset(10);
    extractor.fit(&features, &labels).unwrap();

    assert_eq!(extractor.get_output_dimensions(), 9);
    let serialized = extractor.to_json().unwrap();
    assert_eq!(serialized.dim_reduction_config.components, 9);
    assert_eq!(serialized.concatenated_dimensions, RTMPOSE_GAU_POOLED_DIMS);

    let restored = FeatureExtractor::<MockFeatures>::from_json(serialized).unwrap();
    assert_eq!(restored.get_output_dimensions(), 9);
    let transformed = restored.transform(&create_mock_features(999)).unwrap();
    assert_eq!(transformed.len(), 9);
}
