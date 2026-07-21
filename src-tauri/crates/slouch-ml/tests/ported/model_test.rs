use slouch_domain::{BoundingBox, ExpandedBbox, FeatureId, FeatureMap, InferenceResult};
use slouch_ml::ported::base_classifier::BaseClassifier;
use slouch_ml::ported::feature_extractor::{FeatureExtractor, FeatureExtractorConfig};
use slouch_ml::ported::knn_classifier::{KnnClassifier, KnnConfig};
use slouch_ml::ported::mlp_classifier::{MlpClassifier, MlpConfig};
use slouch_ml::ported::model::Model;
use slouch_ml::ported::types::{
    DimensionalityReductionConfig, DimensionalityReductionMethod, NormalizationMode,
    SerializedModel,
};

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
    let gau_features = (0..slouch_ml::ported::constants::RTMPOSE_GAU_POOLED_DIMS)
        .map(|index| (base_value + (seed as f64 + index as f64).sin() * 0.1) as f32)
        .collect();
    let backbone_features = (0..slouch_ml::ported::constants::RTMPOSE_BACKBONE_POOLED_DIMS)
        .map(|index| (base_value + (seed as f64 + index as f64).cos() * 0.1) as f32)
        .collect();

    let mut features = FeatureMap::new();
    features.insert(FeatureId::GauFeatures, gau_features);
    features.insert(FeatureId::BackboneFeatures, backbone_features);

    InferenceResult {
        features,
        keypoints: Vec::new(),
        bbox: bbox(),
        classification: None,
    }
}

fn create_mock_dataset(good_count: usize, bad_count: usize) -> (Vec<InferenceResult>, Vec<i32>) {
    let mut raw_features = Vec::with_capacity(good_count + bad_count);
    let mut labels = Vec::with_capacity(good_count + bad_count);

    for index in 0..good_count {
        raw_features.push(create_mock_features(true, index));
        labels.push(0);
    }
    for index in 0..bad_count {
        raw_features.push(create_mock_features(false, index + good_count));
        labels.push(1);
    }

    (raw_features, labels)
}

fn extractor_config(
    feature_types: Vec<FeatureId>,
    normalization_mode: NormalizationMode,
    method: DimensionalityReductionMethod,
    components: usize,
) -> FeatureExtractorConfig {
    FeatureExtractorConfig {
        feature_types,
        normalization_mode,
        dim_reduction_config: DimensionalityReductionConfig { method, components },
        unlabeled_samples: Vec::new(),
    }
}

fn model(config: FeatureExtractorConfig, classifier: Box<dyn BaseClassifier>) -> Model {
    Model::new(FeatureExtractor::new(config), classifier)
}

fn knn_model(config: FeatureExtractorConfig) -> Model {
    let classifier = KnnClassifier::with_config(KnnConfig {
        k: 3,
        ..KnnConfig::default()
    })
    .unwrap();
    model(config, Box::new(classifier))
}

#[test]
fn constructor_creates_model_with_feature_extractor_and_classifier() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let _model = knn_model(config);
}

#[test]
fn fit_trains_model_successfully_with_sufficient_data() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(4, 4);

    assert!(model.fit(&features, &labels).is_ok());
}

#[test]
fn fit_rejects_empty_dataset() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);

    let error = model.fit(&[], &[]).unwrap_err();
    assert!(error.to_string().contains("Cannot train on empty dataset"));
}

#[test]
fn fit_rejects_length_mismatch() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, _) = create_mock_dataset(4, 4);

    let error = model.fit(&features, &[0, 1]).unwrap_err();
    assert!(error.to_string().contains("Length mismatch"));
}

#[test]
fn fit_rejects_insufficient_data_per_class() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(0, 10);

    let error = model.fit(&features, &labels).unwrap_err();
    assert!(error.to_string().contains("Not enough class 0 frames"));
}

#[test]
fn fit_supports_all_normalization_modes() {
    let (features, labels) = create_mock_dataset(4, 4);

    for normalization_mode in [
        NormalizationMode::None,
        NormalizationMode::Layer,
        NormalizationMode::ZScore,
    ] {
        let config = extractor_config(
            vec![FeatureId::GauFeatures],
            normalization_mode,
            DimensionalityReductionMethod::None,
            1,
        );
        let mut model = knn_model(config);
        assert!(model.fit(&features, &labels).is_ok());
    }
}

#[test]
fn fit_supports_random_projection_with_mlp_classifier() {
    let config = extractor_config(
        vec![FeatureId::BackboneFeatures],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::RandomProjection,
        16,
    );
    let classifier = MlpClassifier::new(MlpConfig {
        hidden_layers: 0,
        hidden_size: 64,
        max_iterations: 2,
        learning_rate: 0.01,
        ..MlpConfig::default()
    })
    .unwrap();
    let mut model = model(config, Box::new(classifier));
    let (features, labels) = create_mock_dataset(3, 3);

    assert!(model.fit(&features, &labels).is_ok());
}

#[test]
fn predict_returns_a_probability_after_training() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(4, 4);
    model.fit(&features, &labels).unwrap();

    let prediction = model.predict(&create_mock_features(true, 9_999)).unwrap();
    assert!((0.0..=1.0).contains(&prediction));
}

#[test]
fn predict_is_consistent_for_the_same_input() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(4, 4);
    model.fit(&features, &labels).unwrap();

    let test_features = create_mock_features(true, 9_999);
    let prediction_one = model.predict(&test_features).unwrap();
    let prediction_two = model.predict(&test_features).unwrap();
    assert!((prediction_one - prediction_two).abs() < 5e-6);
}

#[test]
fn serialization_round_trip_restores_model_predictions() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(5, 5);
    model.fit(&features, &labels).unwrap();

    let payload = model.to_json().unwrap();
    let payload_json = serde_json::to_value(&payload).unwrap();
    assert!(payload_json.get("trainedAt").is_none());
    assert!(payload_json.get("version").is_none());
    assert_eq!(payload.classifier.classifier_id, "knn");
    assert_eq!(
        payload.feature_extractor.feature_types,
        vec![FeatureId::GauFeatures.to_string()]
    );
    let serialized = SerializedModel {
        feature_extractor: payload.feature_extractor,
        classifier: payload.classifier,
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    };

    let test_features = create_mock_features(true, 9_999);
    let original_prediction = model.predict(&test_features).unwrap();
    let restored = Model::from_json(serialized).unwrap();
    let restored_prediction = restored.predict(&test_features).unwrap();

    assert!((restored_prediction - original_prediction).abs() < 5e-6);
}

#[test]
fn serialization_preserves_z_score_normalization_parameters() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::None,
        1,
    );
    let classifier = MlpClassifier::new(MlpConfig {
        hidden_layers: 0,
        hidden_size: 64,
        max_iterations: 2,
        learning_rate: 0.01,
        ..MlpConfig::default()
    })
    .unwrap();
    let mut model = model(config, Box::new(classifier));
    let (features, labels) = create_mock_dataset(3, 3);
    model.fit(&features, &labels).unwrap();

    let serialized = model.to_json().unwrap();
    assert_eq!(
        serialized.feature_extractor.normalization_mode,
        NormalizationMode::ZScore
    );
    assert!(serialized.feature_extractor.normalization_mean.is_some());
    assert!(serialized.feature_extractor.normalization_std.is_some());
}

#[test]
fn serialization_round_trip_restores_random_projection_model() {
    let config = extractor_config(
        vec![FeatureId::BackboneFeatures],
        NormalizationMode::ZScore,
        DimensionalityReductionMethod::RandomProjection,
        8,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(5, 5);
    model.fit(&features, &labels).unwrap();

    let payload = model.to_json().unwrap();
    let serialized = SerializedModel {
        feature_extractor: payload.feature_extractor,
        classifier: payload.classifier,
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    };
    let test_features = create_mock_features(true, 9_999);
    let original_prediction = model.predict(&test_features).unwrap();
    let restored = Model::from_json(serialized).unwrap();
    let restored_prediction = restored.predict(&test_features).unwrap();

    assert!((restored_prediction - original_prediction).abs() < 5e-6);
}

#[test]
fn fit_ignores_non_binary_labels_matching_oracle() {
    // Oracle (model.ts validateDataset) only reads classCounts[0]/[1]; labels >= 2
    // are silently ignored, so a dataset with an extra class trains successfully.
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, mut labels) = create_mock_dataset(4, 4);
    labels[0] = 2;
    assert!(model.fit(&features, &labels).is_ok());
}

#[test]
fn fit_reports_no_labeled_frames_when_all_labels_are_non_binary() {
    // When no label is 0 or 1 (class0Count == 0 && class1Count == 0), the oracle
    // returns the 'Dataset has no labeled frames' error before the per-class checks.
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(4, 4);
    let labels: Vec<i32> = labels.iter().map(|_| 2).collect();
    let error = model.fit(&features, &labels).unwrap_err();
    assert!(error.to_string().contains("Dataset has no labeled frames"));
}

#[test]
fn five_decimal_comparison_rejects_six_micro_difference() {
    let within = (4.0_f64 / 1_000_000.0).abs();
    let outside = (6.0_f64 / 1_000_000.0).abs();
    let tolerance = 5.0_f64 / 1_000_000.0;
    assert!(within < tolerance);
    assert!(outside >= tolerance);
}

#[test]
fn dispose_releases_model_resources_without_error() {
    let config = extractor_config(
        vec![FeatureId::GauFeatures],
        NormalizationMode::None,
        DimensionalityReductionMethod::None,
        1,
    );
    let mut model = knn_model(config);
    let (features, labels) = create_mock_dataset(4, 4);
    model.fit(&features, &labels).unwrap();
    model.dispose();
}
