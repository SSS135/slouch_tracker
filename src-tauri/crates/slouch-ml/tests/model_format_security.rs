//! Adversarial regression tests for slouch-ml's classifier (de)serialization
//! trust boundary: the layer that turns serialized classifier state back into
//! runnable models. The binary container envelope (SLMD/SLCF) is owned and
//! tested by slouch-store; everything here targets validation that slouch-ml
//! itself performs on decoded state.

use std::collections::BTreeMap;

use serde_json::json;
use slouch_domain::{
    BoundingBox, ExpandedBbox, FeatureId, FeatureMap, InferenceResult, ParameterValue,
};
use slouch_ml::ported::base_classifier::{BaseClassifier, ClassifierError};
use slouch_ml::ported::classifier_factory::deserialize_classifier as factory_deserialize_classifier;
use slouch_ml::ported::feature_extractor::{FeatureExtractor, FeatureExtractorConfig};
use slouch_ml::ported::knn_classifier::{KnnClassifier, KnnConfig};
use slouch_ml::ported::mlp_classifier::{MlpClassifier, MlpConfig};
use slouch_ml::ported::model::{Model, ModelError};
use slouch_ml::ported::naive_bayes_classifier::GaussianNbClassifier;
use slouch_ml::ported::random_projection::{RandomProjectionState, RandomProjectionTransformer};
use slouch_ml::ported::serialization::serialization_helpers::{
    create_serialized_model, deserialize_classifier as envelope_deserialize_classifier,
    deserialize_dim_reduction_transformer, restore_model_metadata, SerializationError,
};
use slouch_ml::ported::svm_classifier::{SvmClassifier, SvmConfig};
use slouch_ml::ported::types::{
    ClassifierParams, DimReductionTransformer, DimensionalityReductionConfig,
    DimensionalityReductionMethod, KnnDistance, NormalizationMode, SerializedClassifier,
    SerializedClassifierState, SerializedFeatureExtractor, SerializedGaussianNb,
    SerializedKMeansLogistic, SerializedKnn, SerializedMlp, SerializedModel, SerializedSvm,
};

const TRAINED_DIMENSIONS: usize = 3;

fn training_features() -> Vec<Vec<f32>> {
    vec![
        vec![0.9, 0.1, 0.2],
        vec![1.0, 0.2, 0.1],
        vec![0.8, 0.0, 0.3],
        vec![0.1, 0.9, 0.8],
        vec![0.0, 1.0, 0.9],
        vec![0.2, 0.8, 1.0],
    ]
}

fn training_labels() -> Vec<i32> {
    vec![0, 0, 0, 1, 1, 1]
}

fn no_params() -> BTreeMap<String, ParameterValue> {
    BTreeMap::new()
}

fn svm_params() -> ClassifierParams {
    ClassifierParams::Svm {
        c: 1.0,
        max_iterations: 100,
        use_class_weights: None,
    }
}

fn knn_params() -> ClassifierParams {
    ClassifierParams::Knn {
        k: 3,
        distance: KnnDistance::Euclidean,
    }
}

fn trained_svm_state() -> SerializedSvm {
    let mut classifier = SvmClassifier::new(SvmConfig {
        c: 1.0,
        max_iterations: 50,
        use_class_weights: false,
    })
    .unwrap();
    classifier
        .train(&training_features(), &training_labels())
        .unwrap();
    match classifier.to_json().unwrap() {
        SerializedClassifierState::Svm(state) => state,
        _ => panic!("SVM classifier must serialize SVM state"),
    }
}

fn trained_mlp_state() -> SerializedMlp {
    let mut classifier = MlpClassifier::new(MlpConfig {
        hidden_layers: 0,
        max_iterations: 5,
        ..MlpConfig::default()
    })
    .unwrap();
    classifier
        .train(&training_features(), &training_labels())
        .unwrap();
    classifier.to_mlp_json().unwrap()
}

fn trained_gaussian_nb_state() -> SerializedGaussianNb {
    let mut classifier = GaussianNbClassifier::new(1e-6).unwrap();
    classifier
        .train(&training_features(), &training_labels())
        .unwrap();
    classifier.to_state().unwrap()
}

fn trained_knn_state() -> SerializedKnn {
    let mut classifier = KnnClassifier::with_config(KnnConfig {
        k: 3,
        ..KnnConfig::default()
    })
    .unwrap();
    classifier
        .train(&training_features(), &training_labels())
        .unwrap();
    match classifier.to_json().unwrap() {
        SerializedClassifierState::Knn(state) => state,
        _ => panic!("KNN classifier must serialize KNN state"),
    }
}

fn trained_kmeans_logistic_state() -> SerializedKMeansLogistic {
    SerializedKMeansLogistic {
        centroids: vec![vec![0.5, 0.5, 0.5]],
        cluster_models: vec![None],
        global_model: trained_mlp_state(),
        temperature: 1.0,
    }
}

fn factory_error(classifier_id: &str, state: SerializedClassifierState) -> ClassifierError {
    match factory_deserialize_classifier(classifier_id, state, &no_params()) {
        Err(error) => error,
        Ok(_) => panic!("adversarial state for {classifier_id} unexpectedly restored"),
    }
}

fn gau_dimensions() -> usize {
    FeatureId::GauFeatures.metadata().dimensions
}

fn mock_bbox() -> ExpandedBbox {
    let bounding_box = BoundingBox {
        x1: 0.0,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
        score: 1.0,
        width: 1.0,
        height: 1.0,
    };
    ExpandedBbox {
        original: bounding_box,
        expanded: bounding_box,
    }
}

fn mock_inference(good: bool, seed: usize) -> InferenceResult {
    let base_value = if good { 0.8_f64 } else { 0.2_f64 };
    let gau_features = (0..gau_dimensions())
        .map(|index| (base_value + (seed as f64 + index as f64).sin() * 0.1) as f32)
        .collect();
    let mut features = FeatureMap::new();
    features.insert(FeatureId::GauFeatures, gau_features);
    InferenceResult {
        features,
        keypoints: Vec::new(),
        bbox: mock_bbox(),
        classification: None,
    }
}

/// Trains a real KNN posture model end to end and returns its persisted
/// envelope; the adversarial tests mutate this genuine baseline.
fn trained_model_envelope() -> SerializedModel {
    let dimensions = gau_dimensions();
    let mut samples = Vec::new();
    let mut labels = Vec::new();
    for seed in 0..4 {
        samples.push(mock_inference(true, seed));
        labels.push(0);
    }
    for seed in 4..8 {
        samples.push(mock_inference(false, seed));
        labels.push(1);
    }

    let extractor = FeatureExtractor::new(FeatureExtractorConfig {
        feature_types: vec![FeatureId::GauFeatures],
        normalization_mode: NormalizationMode::None,
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: dimensions,
        },
        unlabeled_samples: Vec::new(),
    });
    let classifier = KnnClassifier::with_config(KnnConfig {
        k: 3,
        ..KnnConfig::default()
    })
    .unwrap();
    let mut model = Model::new(extractor, Box::new(classifier));
    model.fit(&samples, &labels).unwrap();
    model.to_serialized_model(1_700_000_000_000.0, 1.0).unwrap()
}

fn metadata_envelope(feature_extractor: SerializedFeatureExtractor) -> SerializedModel {
    SerializedModel {
        feature_extractor,
        classifier: SerializedClassifier {
            classifier_id: "svm".to_owned(),
            state: SerializedClassifierState::Svm(trained_svm_state()),
        },
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    }
}

fn z_score_extractor(mean: Option<Vec<f64>>, std: Option<Vec<f64>>) -> SerializedFeatureExtractor {
    SerializedFeatureExtractor {
        feature_types: vec![FeatureId::GauFeatures.as_str().to_owned()],
        normalization_mode: NormalizationMode::ZScore,
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: TRAINED_DIMENSIONS,
        },
        concatenated_dimensions: TRAINED_DIMENSIONS,
        normalization_mean: mean,
        normalization_std: std,
        dim_reduction_transformer: None,
    }
}

fn projection_extractor(
    dimensions: usize,
    components: usize,
    state: RandomProjectionState,
) -> SerializedFeatureExtractor {
    SerializedFeatureExtractor {
        feature_types: vec![FeatureId::GauFeatures.as_str().to_owned()],
        normalization_mode: NormalizationMode::None,
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::RandomProjection,
            components,
        },
        concatenated_dimensions: dimensions,
        normalization_mean: None,
        normalization_std: None,
        dim_reduction_transformer: Some(DimReductionTransformer::RandomProjection(state)),
    }
}

fn fitted_projection(n_features: usize, n_components: usize) -> RandomProjectionTransformer {
    let samples = vec![
        vec![0.5_f32; n_features],
        vec![0.25; n_features],
        vec![0.75; n_features],
    ];
    let mut transformer = RandomProjectionTransformer::with_default_seed(n_components).unwrap();
    transformer.fit(&samples).unwrap();
    transformer
}

// ============================================================================
// Baseline: a genuinely trained model must round-trip; the adversarial cases
// below mutate exactly this envelope, so their failures are caused by the
// mutation and not by an unrepresentative fixture.
// ============================================================================

#[test]
fn trained_envelope_round_trips_through_json_and_restores_predictions() {
    let envelope = trained_model_envelope();
    let encoded = serde_json::to_value(&envelope).unwrap();
    let decoded: SerializedModel = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, envelope);

    let restored = Model::from_json(decoded).unwrap();
    let prediction = restored.predict(&mock_inference(true, 9_999)).unwrap();
    assert!((0.0..=1.0).contains(&prediction));
}

// ============================================================================
// Scenario 1: corrupted serialized state built by JSON surgery.
// ============================================================================

#[test]
fn removing_required_fields_from_a_trained_envelope_fails_json_decoding() {
    let base = serde_json::to_value(trained_model_envelope()).unwrap();

    for field in ["trainingData", "trainingLabels", "k"] {
        let mut mutated = base.clone();
        mutated["classifier"]["state"]
            .as_object_mut()
            .unwrap()
            .remove(field)
            .expect("field must exist in the serialized KNN state");
        assert!(
            serde_json::from_value::<SerializedModel>(mutated).is_err(),
            "decoding must fail without classifier state field {field}"
        );
    }

    for field in ["featureExtractor", "classifier"] {
        let mut mutated = base.clone();
        mutated
            .as_object_mut()
            .unwrap()
            .remove(field)
            .expect("field must exist in the serialized envelope");
        assert!(
            serde_json::from_value::<SerializedModel>(mutated).is_err(),
            "decoding must fail without envelope field {field}"
        );
    }
}

#[test]
fn renamed_and_wrong_typed_state_fields_fail_json_decoding() {
    let base = serde_json::to_value(trained_model_envelope()).unwrap();

    let mut renamed = base.clone();
    let state = renamed["classifier"]["state"].as_object_mut().unwrap();
    let moved = state
        .remove("trainingData")
        .expect("trainingData must exist in the serialized KNN state");
    state.insert("training_data".to_owned(), moved);
    assert!(
        serde_json::from_value::<SerializedModel>(renamed).is_err(),
        "the wire contract is camelCase; a snake_case alias must not decode"
    );

    let mut string_k = base.clone();
    string_k["classifier"]["state"]["k"] = json!("three");
    assert!(serde_json::from_value::<SerializedModel>(string_k).is_err());

    let mut scalar_matrix = base;
    scalar_matrix["classifier"]["state"]["trainingData"] = json!(42);
    assert!(serde_json::from_value::<SerializedModel>(scalar_matrix).is_err());
}

// ============================================================================
// Scenario 2: truncated state. Serde cannot detect shortened tensors, so
// these pin the explicit shape validation in the restore paths.
// ============================================================================

#[test]
fn truncated_mlp_weights_decode_but_are_rejected_by_explicit_shape_validation() {
    let mut tampered = serde_json::to_value(trained_mlp_state()).unwrap();
    tampered["layerWeights"][0]
        .as_array_mut()
        .unwrap()
        .pop()
        .expect("trained MLP layer must have weights to remove");

    let truncated: SerializedMlp =
        serde_json::from_value(tampered).expect("truncation is invisible to serde decoding");
    let error = MlpClassifier::from_state(truncated).unwrap_err();
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("invalid tensor lengths")
    ));
}

#[test]
fn envelope_rejects_svm_weight_vectors_shorter_than_declared_dimension() {
    let intact = create_serialized_model(
        SerializedClassifierState::Svm(trained_svm_state()),
        None,
        svm_params(),
        TRAINED_DIMENSIONS,
    );
    let restored = envelope_deserialize_classifier(&intact, "svm").unwrap();
    assert_eq!(restored.classifier_id, "svm");

    let mut short = trained_svm_state();
    short.weights.truncate(TRAINED_DIMENSIONS - 1);
    let truncated = create_serialized_model(
        SerializedClassifierState::Svm(short),
        None,
        svm_params(),
        TRAINED_DIMENSIONS,
    );
    let error = envelope_deserialize_classifier(&truncated, "svm").unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidSerializedFields(message)
            if message.contains("does not match expected dimension 3")
    ));

    let mut empty = trained_svm_state();
    empty.weights.clear();
    let missing = create_serialized_model(
        SerializedClassifierState::Svm(empty),
        None,
        svm_params(),
        TRAINED_DIMENSIONS,
    );
    assert_eq!(
        envelope_deserialize_classifier(&missing, "svm").unwrap_err(),
        SerializationError::MissingWeights {
            field: "weights".into()
        }
    );
}

#[test]
fn truncated_knn_rows_and_gaussian_nb_variances_are_rejected() {
    let mut ragged = trained_knn_state();
    ragged.training_data[1]
        .pop()
        .expect("trained KNN rows must be nonempty");
    assert_eq!(
        factory_error("knn", SerializedClassifierState::Knn(ragged)),
        ClassifierError::RaggedFeatures {
            row: 1,
            expected: TRAINED_DIMENSIONS,
            actual: TRAINED_DIMENSIONS - 1,
        }
    );

    let mut truncated = trained_gaussian_nb_state();
    truncated.class_variances[1]
        .pop()
        .expect("trained variances must be nonempty");
    let error = GaussianNbClassifier::from_state(truncated).unwrap_err();
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("equal nonzero dimensions")
    ));
}

#[test]
fn decodable_but_inconsistent_knn_k_is_rejected_by_envelope_validation() {
    let mut state = trained_knn_state();
    state.k = state.training_data.len() + 1;
    let model = create_serialized_model(
        SerializedClassifierState::Knn(state),
        None,
        knn_params(),
        TRAINED_DIMENSIONS,
    );
    let error = envelope_deserialize_classifier(&model, "knn").unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidSerializedFields(message)
            if message.contains("within the training-data size")
    ));
}

// ============================================================================
// Scenario 3: classifier ID and state-variant routing.
// ============================================================================

#[test]
fn factory_rejects_state_belonging_to_a_different_classifier() {
    let cases = [
        ("svm", SerializedClassifierState::Knn(trained_knn_state())),
        ("knn", SerializedClassifierState::Mlp(trained_mlp_state())),
        (
            "gaussian_nb",
            SerializedClassifierState::Svm(trained_svm_state()),
        ),
        (
            "kmeans_logistic",
            SerializedClassifierState::GaussianNb(trained_gaussian_nb_state()),
        ),
        (
            "MLPClassifier",
            SerializedClassifierState::Svm(trained_svm_state()),
        ),
    ];
    for (classifier_id, state) in cases {
        let error = factory_error(classifier_id, state);
        assert!(
            matches!(
                &error,
                ClassifierError::InvalidState(message)
                    if message.contains("serialized state does not match classifier")
            ),
            "unexpected routing error for {classifier_id}: {error}"
        );
    }
}

#[test]
fn factory_rejects_unknown_and_deprecated_classifier_ids() {
    assert_eq!(
        factory_error(
            "transformer_xl",
            SerializedClassifierState::Svm(trained_svm_state())
        ),
        ClassifierError::InvalidState("Unknown classifier ID: transformer_xl".to_owned())
    );

    for deprecated in ["logistic_regression", "LogisticRegressionClassifier"] {
        let error = factory_error(
            deprecated,
            SerializedClassifierState::Mlp(trained_mlp_state()),
        );
        assert!(matches!(
            &error,
            ClassifierError::InvalidState(message) if message.contains("no longer supported")
        ));
    }
}

#[test]
fn envelope_rejects_parameter_and_state_variant_mismatches() {
    let wrong_params = create_serialized_model(
        SerializedClassifierState::Svm(trained_svm_state()),
        None,
        knn_params(),
        TRAINED_DIMENSIONS,
    );
    assert_eq!(
        envelope_deserialize_classifier(&wrong_params, "svm").unwrap_err(),
        SerializationError::ClassifierParameterMismatch {
            classifier_id: "svm".into()
        }
    );

    let wrong_state = create_serialized_model(
        SerializedClassifierState::Knn(trained_knn_state()),
        None,
        svm_params(),
        TRAINED_DIMENSIONS,
    );
    let error = envelope_deserialize_classifier(&wrong_state, "svm").unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidSerializedFields(message)
            if message.contains("does not match serialized state variant")
    ));

    let zero_features = create_serialized_model(
        SerializedClassifierState::Svm(trained_svm_state()),
        None,
        svm_params(),
        0,
    );
    assert_eq!(
        envelope_deserialize_classifier(&zero_features, "svm").unwrap_err(),
        SerializationError::InvalidModel("nFeatures must be positive".into())
    );
}

// ============================================================================
// Scenario 4: NaN/Infinity smuggling. Typed state can carry non-finite f64
// values, so the restore paths must validate finiteness; the JSON wire format
// cannot represent them at all, which the wire test pins explicitly.
// ============================================================================

#[test]
fn non_finite_numbers_in_restored_state_are_rejected() {
    assert!(factory_deserialize_classifier(
        "kmeans_logistic",
        SerializedClassifierState::KMeansLogistic(trained_kmeans_logistic_state()),
        &no_params(),
    )
    .is_ok());

    let mut nan_weight = trained_svm_state();
    nan_weight.weights[0] = f64::NAN;
    let error = factory_error("svm", SerializedClassifierState::Svm(nan_weight));
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite f32 range")
    ));

    let mut infinite_bias = trained_svm_state();
    infinite_bias.bias = f64::INFINITY;
    let error = factory_error("svm", SerializedClassifierState::Svm(infinite_bias));
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite f32 range")
    ));

    let mut nan_mlp = trained_mlp_state();
    nan_mlp.layer_weights[0][0] = f64::NAN;
    let error = factory_error("mlp", SerializedClassifierState::Mlp(nan_mlp));
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite f32 range")
    ));

    let mut nan_mean = trained_gaussian_nb_state();
    nan_mean.class_means[0][0] = f64::NAN;
    let error = factory_error(
        "gaussian_nb",
        SerializedClassifierState::GaussianNb(nan_mean),
    );
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("class means must remain finite")
    ));

    let mut infinite_variance = trained_gaussian_nb_state();
    infinite_variance.class_variances[0][0] = f64::NEG_INFINITY;
    let error = factory_error(
        "gaussian_nb",
        SerializedClassifierState::GaussianNb(infinite_variance),
    );
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite and positive")
    ));

    let mut nan_prior = trained_gaussian_nb_state();
    nan_prior.class_priors[0] = f64::NAN;
    let error = factory_error(
        "gaussian_nb",
        SerializedClassifierState::GaussianNb(nan_prior),
    );
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("invalid numeric values")
    ));

    let mut infinite_knn = trained_knn_state();
    infinite_knn.training_data[0][1] = f64::INFINITY;
    assert_eq!(
        factory_error("knn", SerializedClassifierState::Knn(infinite_knn)),
        ClassifierError::NonFiniteFeature { row: 0, column: 1 }
    );

    let mut nan_centroid = trained_kmeans_logistic_state();
    nan_centroid.centroids[0][0] = f64::NAN;
    let error = factory_error(
        "kmeans_logistic",
        SerializedClassifierState::KMeansLogistic(nan_centroid),
    );
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite in f32")
    ));

    let mut nan_temperature = trained_kmeans_logistic_state();
    nan_temperature.temperature = f64::NAN;
    let error = factory_error(
        "kmeans_logistic",
        SerializedClassifierState::KMeansLogistic(nan_temperature),
    );
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("temperature")
    ));
}

#[test]
fn finite_f64_values_that_overflow_f32_are_rejected_not_saturated() {
    let mut overflow_weight = trained_svm_state();
    overflow_weight.weights[0] = 1e300;
    let error = factory_error("svm", SerializedClassifierState::Svm(overflow_weight));
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite f32 range")
    ));

    let mut overflow_mlp = trained_mlp_state();
    overflow_mlp.layer_weights[0][0] = -1e40;
    let error = factory_error("mlp", SerializedClassifierState::Mlp(overflow_mlp));
    assert!(matches!(
        &error,
        ClassifierError::InvalidState(message) if message.contains("finite f32 range")
    ));

    let mut overflow_knn = trained_knn_state();
    overflow_knn.training_data[0][0] = f64::MAX;
    assert_eq!(
        factory_error("knn", SerializedClassifierState::Knn(overflow_knn)),
        ClassifierError::NonFiniteFeature { row: 0, column: 0 }
    );
}

#[test]
fn json_wire_format_cannot_smuggle_non_finite_floats() {
    let mut state = trained_svm_state();
    state.weights[0] = f64::NAN;
    state.bias = f64::INFINITY;
    let encoded = serde_json::to_value(&state).unwrap();

    // serde_json has no NaN/Infinity representation: non-finite floats become
    // null on encode, and the strongly typed decode then refuses null where a
    // number is required. A switch to a float encoding that silently
    // round-trips non-finite values would break both assertions.
    assert!(encoded["weights"][0].is_null());
    assert!(encoded["bias"].is_null());
    assert!(serde_json::from_value::<SerializedSvm>(encoded).is_err());
}

#[test]
fn restored_classifiers_reject_non_finite_prediction_inputs() {
    let svm = SvmClassifier::from_state(trained_svm_state()).unwrap();
    assert_eq!(
        svm.predict_proba(&[f32::NAN, 0.0, 0.0]),
        Err(ClassifierError::NonFiniteFeature { row: 0, column: 0 })
    );

    let mlp = MlpClassifier::from_state(trained_mlp_state()).unwrap();
    assert_eq!(
        mlp.predict_proba(&[0.0, f32::INFINITY, 0.0]),
        Err(ClassifierError::NonFiniteFeature { row: 0, column: 1 })
    );

    let gaussian_nb = GaussianNbClassifier::from_state(trained_gaussian_nb_state()).unwrap();
    assert_eq!(
        gaussian_nb.predict_proba(&[0.0, 0.0, f32::NEG_INFINITY]),
        Err(ClassifierError::NonFiniteFeature { row: 0, column: 2 })
    );
}

#[test]
fn z_score_normalization_parameters_are_validated_on_restore() {
    let valid = metadata_envelope(z_score_extractor(
        Some(vec![0.0; TRAINED_DIMENSIONS]),
        Some(vec![1.0; TRAINED_DIMENSIONS]),
    ));
    let metadata = restore_model_metadata(&valid).unwrap();
    assert_eq!(metadata.n_features, TRAINED_DIMENSIONS);
    assert_eq!(
        metadata.normalization_mean,
        Some(vec![0.0; TRAINED_DIMENSIONS])
    );

    let missing_std =
        metadata_envelope(z_score_extractor(Some(vec![0.0; TRAINED_DIMENSIONS]), None));
    assert_eq!(
        restore_model_metadata(&missing_std).unwrap_err(),
        SerializationError::MissingNormalizationParameters
    );

    let nan_mean = metadata_envelope(z_score_extractor(
        Some(vec![0.0, f64::NAN, 0.0]),
        Some(vec![1.0; TRAINED_DIMENSIONS]),
    ));
    let error = restore_model_metadata(&nan_mean).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidNormalizationParameters(message)
            if message.contains("finite")
    ));

    let zero_std = metadata_envelope(z_score_extractor(
        Some(vec![0.0; TRAINED_DIMENSIONS]),
        Some(vec![1.0, 0.0, 1.0]),
    ));
    let error = restore_model_metadata(&zero_std).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidNormalizationParameters(message)
            if message.contains("positive")
    ));

    let short_std = metadata_envelope(z_score_extractor(
        Some(vec![0.0; TRAINED_DIMENSIONS]),
        Some(vec![1.0; TRAINED_DIMENSIONS - 1]),
    ));
    let error = restore_model_metadata(&short_std).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidNormalizationParameters(message)
            if message.contains("do not match the feature dimension")
    ));

    let mut stray = z_score_extractor(
        Some(vec![0.0; TRAINED_DIMENSIONS]),
        Some(vec![1.0; TRAINED_DIMENSIONS]),
    );
    stray.normalization_mode = NormalizationMode::None;
    let error = restore_model_metadata(&metadata_envelope(stray)).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidNormalizationParameters(message)
            if message.contains("must not contain z-score parameters")
    ));
}

// ============================================================================
// Scenario 5: dimension mismatches at restore time and predict time.
// ============================================================================

#[test]
fn transformer_and_classifier_dimension_agreement_is_enforced() {
    let transformer_state = fitted_projection(6, TRAINED_DIMENSIONS).to_state().unwrap();
    let wrapper = Some(DimReductionTransformer::RandomProjection(
        transformer_state.clone(),
    ));

    // The classifier operates in the 3-dimensional projected space while the
    // envelope declares 6 raw input features.
    let reduced = create_serialized_model(
        SerializedClassifierState::Svm(trained_svm_state()),
        wrapper.clone(),
        svm_params(),
        6,
    );
    let restored = envelope_deserialize_classifier(&reduced, "svm").unwrap();
    assert_eq!(restored.classifier_id, "svm");

    let mut unreduced_state = trained_svm_state();
    let doubled = unreduced_state.weights.clone();
    unreduced_state.weights.extend(doubled);
    let unreduced = create_serialized_model(
        SerializedClassifierState::Svm(unreduced_state),
        wrapper,
        svm_params(),
        6,
    );
    let error = envelope_deserialize_classifier(&unreduced, "svm").unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidSerializedFields(message)
            if message.contains("does not match expected dimension 3")
    ));

    let no_transformer = create_serialized_model(
        SerializedClassifierState::Svm(trained_svm_state()),
        None,
        svm_params(),
        6,
    );
    let error = envelope_deserialize_classifier(&no_transformer, "svm").unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidSerializedFields(message)
            if message.contains("does not match expected dimension 6")
    ));
}

#[test]
fn restore_metadata_rejects_transformer_shape_disagreements_and_truncation() {
    let state = fitted_projection(6, 3).to_state().unwrap();

    let valid = metadata_envelope(projection_extractor(6, 3, state.clone()));
    let metadata = restore_model_metadata(&valid).unwrap();
    assert_eq!(
        metadata
            .dim_reduction_transformer
            .map(|transformer| transformer.n_components()),
        Some(3)
    );

    let feature_lie = metadata_envelope(projection_extractor(4, 3, state.clone()));
    let error = restore_model_metadata(&feature_lie).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidModel(message) if message.contains("expects 6 input features")
    ));

    let component_lie = metadata_envelope(projection_extractor(6, 2, state.clone()));
    let error = restore_model_metadata(&component_lie).unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidModel(message)
            if message.contains("has 3 components, expected 2")
    ));

    let mut truncated = state;
    truncated.projection_matrix[0]
        .pop()
        .expect("fitted projection rows must be nonempty");
    let error = deserialize_dim_reduction_transformer(Some(
        DimReductionTransformer::RandomProjection(truncated),
    ))
    .unwrap_err();
    assert!(matches!(
        &error,
        SerializationError::InvalidModel(message) if message.contains("Failed to reconstruct")
    ));
}

#[test]
fn model_restore_rejects_declared_dimension_and_feature_type_lies() {
    let mut dimension_lie = trained_model_envelope();
    dimension_lie.feature_extractor.concatenated_dimensions = 999;
    let Err(error) = Model::<InferenceResult>::from_json(dimension_lie) else {
        panic!("dimension lie must not restore");
    };
    assert!(error.to_string().contains("concatenatedDimensions is 999"));

    let mut unknown_feature = trained_model_envelope();
    unknown_feature.feature_extractor.feature_types = vec!["nonexistent_feature".into()];
    let Err(error) = Model::<InferenceResult>::from_json(unknown_feature) else {
        panic!("unknown feature type must not restore");
    };
    assert!(error
        .to_string()
        .contains("Unknown feature type: nonexistent_feature"));
}

#[test]
fn predict_after_restore_reports_dimension_mismatch_errors_instead_of_panicking() {
    let svm = SvmClassifier::from_state(trained_svm_state()).unwrap();
    assert_eq!(
        svm.predict_proba(&[0.5, 0.5]),
        Err(ClassifierError::PredictionDimension {
            expected: TRAINED_DIMENSIONS,
            actual: 2,
        })
    );
    assert_eq!(
        svm.predict_proba(&[]),
        Err(ClassifierError::PredictionDimension {
            expected: TRAINED_DIMENSIONS,
            actual: 0,
        })
    );

    let mlp = MlpClassifier::from_state(trained_mlp_state()).unwrap();
    assert_eq!(
        mlp.predict_proba(&[0.5; 4]),
        Err(ClassifierError::PredictionDimension {
            expected: TRAINED_DIMENSIONS,
            actual: 4,
        })
    );

    let gaussian_nb = GaussianNbClassifier::from_state(trained_gaussian_nb_state()).unwrap();
    assert_eq!(
        gaussian_nb.predict_proba(&[0.5; 5]),
        Err(ClassifierError::PredictionDimension {
            expected: TRAINED_DIMENSIONS,
            actual: 5,
        })
    );

    let knn = KnnClassifier::from_state(trained_knn_state()).unwrap();
    assert_eq!(
        knn.predict_proba(&[0.5; 4]),
        Err(ClassifierError::PredictionDimension {
            expected: TRAINED_DIMENSIONS,
            actual: 4,
        })
    );
}

// ============================================================================
// Scenario 6: staleness metadata and the determinism the pair-level
// training-config fingerprint depends on. The SHA-256 fingerprint itself is
// computed and compared in slouch-store; slouch-ml owns the trainedAt/version
// metadata gate and the reproducibility of serialized classifier state.
// ============================================================================

#[test]
fn staleness_metadata_must_be_finite_to_restore() {
    let mut nan_trained_at = trained_model_envelope();
    nan_trained_at.trained_at = f64::NAN;
    let Err(error) = Model::<InferenceResult>::from_json(nan_trained_at) else {
        panic!("NaN trainedAt must not restore");
    };
    assert_eq!(
        error,
        ModelError::Serialization("serialized model metadata must be finite".to_owned())
    );

    let mut infinite_version = trained_model_envelope();
    infinite_version.version = f64::INFINITY;
    let Err(error) = Model::<InferenceResult>::from_json(infinite_version) else {
        panic!("infinite version must not restore");
    };
    assert_eq!(
        error,
        ModelError::Serialization("serialized model metadata must be finite".to_owned())
    );
}

fn deterministic_mlp_state(features: &[Vec<f32>]) -> SerializedMlp {
    let mut classifier = MlpClassifier::new(MlpConfig {
        hidden_layers: 1,
        hidden_size: 4,
        max_iterations: 5,
        ..MlpConfig::default()
    })
    .unwrap();
    classifier.train(features, &training_labels()).unwrap();
    classifier.to_mlp_json().unwrap()
}

#[test]
fn identical_training_yields_identical_state_and_input_changes_are_visible() {
    // Model comparison across generations is only meaningful when the seeded
    // training pipeline is reproducible bit for bit, and when a real change in
    // the training inputs actually shows up in the serialized weights.
    let baseline = deterministic_mlp_state(&training_features());
    assert_eq!(baseline, deterministic_mlp_state(&training_features()));

    let mut perturbed = training_features();
    perturbed[0][0] += 0.5;
    assert_ne!(deterministic_mlp_state(&perturbed), baseline);
}

#[test]
fn restore_then_reserialize_reproduces_the_exact_persisted_state() {
    // Payload hashes over serialized state stay valid only if a load/save
    // cycle is an exact fixed point; any narrowing or re-canonicalization
    // drift on these paths would silently invalidate stored digests.
    let svm_state = trained_svm_state();
    let restored = SvmClassifier::from_state(svm_state.clone()).unwrap();
    assert_eq!(
        restored.to_json().unwrap(),
        SerializedClassifierState::Svm(svm_state)
    );

    let nb_state = trained_gaussian_nb_state();
    let restored = GaussianNbClassifier::from_state(nb_state.clone()).unwrap();
    assert_eq!(
        restored.to_json().unwrap(),
        SerializedClassifierState::GaussianNb(nb_state)
    );

    let mlp_state = trained_mlp_state();
    let restored = MlpClassifier::from_state(mlp_state.clone()).unwrap();
    assert_eq!(
        restored.to_json().unwrap(),
        SerializedClassifierState::Mlp(mlp_state)
    );
}
