use slouch_ml::ported::base_classifier::{BaseClassifier, ClassifierError};
use slouch_ml::ported::kmeans_logistic_classifier::{
    KMeansLogisticClassifier, KMeansLogisticConfig,
};
use slouch_ml::ported::types::{SerializedClassifierState, SerializedKMeansLogistic};

fn classifier(config: KMeansLogisticConfig) -> KMeansLogisticClassifier {
    KMeansLogisticClassifier::with_config(config).unwrap()
}

fn two_cluster_features() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ]
}

fn serialized_state(classifier: &KMeansLogisticClassifier) -> SerializedKMeansLogistic {
    match classifier.to_json().unwrap() {
        SerializedClassifierState::KMeansLogistic(state) => state,
        _ => panic!("classifier returned the wrong serialized state"),
    }
}

fn with_max_iterations(max_iterations: usize) -> KMeansLogisticConfig {
    KMeansLogisticConfig {
        max_iterations,
        ..KMeansLogisticConfig::default()
    }
}

#[test]
fn rejects_empty_dataset() {
    let mut classifier = classifier(KMeansLogisticConfig::default());
    assert_eq!(
        classifier.train(&[], &[]),
        Err(ClassifierError::EmptyDataset)
    );
}

#[test]
fn rejects_feature_label_length_mismatch() {
    let mut classifier = classifier(KMeansLogisticConfig::default());
    assert_eq!(
        classifier.train(&[vec![1.0, 2.0]], &[0, 1]),
        Err(ClassifierError::LengthMismatch {
            features: 1,
            labels: 2,
        })
    );
}

#[test]
fn trains_successfully_with_valid_data() {
    let mut classifier = classifier(with_max_iterations(50));
    assert!(classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .is_ok());
}

#[test]
fn handles_class_weight_balancing() {
    let mut classifier = classifier(KMeansLogisticConfig {
        use_class_weights: true,
        max_iterations: 50,
        ..KMeansLogisticConfig::default()
    });
    let features = vec![
        vec![0.0, 0.0],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.2],
    ];
    assert!(classifier.train(&features, &[0, 1, 1, 1]).is_ok());
}

#[test]
fn handles_small_dataset_where_k_max_is_below_k_min() {
    let mut classifier = classifier(with_max_iterations(50));
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn rejects_prediction_with_untrained_model() {
    let classifier = classifier(KMeansLogisticConfig::default());
    assert_eq!(
        classifier.predict_proba(&[1.0, 2.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn returns_probabilities_in_the_unit_interval() {
    let mut classifier = classifier(with_max_iterations(50));
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    for probability in [
        classifier.predict_proba(&[0.0, 0.0]).unwrap(),
        classifier.predict_proba(&[10.0, 10.0]).unwrap(),
    ] {
        assert!((0.0..=1.0).contains(&probability));
    }
}

#[test]
fn predicts_correctly_on_linearly_separable_data() {
    let mut classifier = classifier(KMeansLogisticConfig {
        max_iterations: 200,
        ..KMeansLogisticConfig::default()
    });
    let features = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![0.2, 0.0],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
        vec![10.2, 10.0],
    ];
    classifier.train(&features, &[0, 0, 0, 1, 1, 1]).unwrap();

    let probability_good = classifier.predict_proba(&[0.0, 0.0]).unwrap();
    let probability_bad = classifier.predict_proba(&[10.0, 10.0]).unwrap();
    assert!(probability_good > 0.5);
    assert!(probability_bad < 0.5);
}

#[test]
fn uses_default_temperature_when_not_specified() {
    let mut classifier = classifier(KMeansLogisticConfig::default());
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn respects_custom_temperature() {
    let mut classifier = classifier(KMeansLogisticConfig {
        temperature: 0.5,
        max_iterations: 50,
        ..KMeansLogisticConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!((0.0..=1.0).contains(&probability));
}

#[test]
fn rejects_serialization_of_an_untrained_model() {
    let classifier = classifier(KMeansLogisticConfig::default());
    assert_eq!(classifier.to_json(), Err(ClassifierError::Untrained));
}

#[test]
fn serialized_state_has_the_expected_structure() {
    let mut classifier = classifier(with_max_iterations(50));
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let state = serialized_state(&classifier);
    assert!(!state.centroids.is_empty());
    assert_eq!(state.cluster_models.len(), state.centroids.len());
    assert!(!state.global_model.layer_weights.is_empty());
    assert!(state.temperature.is_finite());
}

#[test]
fn restores_model_with_from_json() {
    let mut classifier = classifier(with_max_iterations(50));
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let state = serialized_state(&classifier);
    let restored = KMeansLogisticClassifier::from_json(state, with_max_iterations(50)).unwrap();
    assert!(restored.predict_proba(&[5.0, 5.0]).is_ok());
}

#[test]
fn preserves_predictions_through_round_trip_serialization() {
    let mut classifier = classifier(KMeansLogisticConfig {
        max_iterations: 100,
        ..KMeansLogisticConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let test_input = [5.0, 5.0];
    let original_probability = classifier.predict_proba(&test_input).unwrap();
    let state = serialized_state(&classifier);
    let restored = KMeansLogisticClassifier::from_json(
        state,
        KMeansLogisticConfig {
            max_iterations: 100,
            ..KMeansLogisticConfig::default()
        },
    )
    .unwrap();
    let restored_probability = restored.predict_proba(&test_input).unwrap();

    assert!((restored_probability - original_probability).abs() < 5e-7);
}

#[test]
fn handles_edge_case_inputs_without_nan() {
    let mut classifier = classifier(with_max_iterations(50));
    let features = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    classifier.train(&features, &[0, 1]).unwrap();

    let probability = classifier.predict_proba(&[0.5, 0.5]).unwrap();
    assert!(probability.is_finite());
}

#[test]
fn handles_very_small_feature_values() {
    let mut classifier = classifier(with_max_iterations(50));
    let features = vec![vec![1e-10, 1e-10], vec![1e-9, 1e-9]];
    classifier.train(&features, &[0, 1]).unwrap();

    let probability = classifier.predict_proba(&[5e-10, 5e-10]).unwrap();
    assert!(probability.is_finite());
}

#[test]
fn dispose_restores_untrained_state() {
    let mut classifier = classifier(with_max_iterations(50));
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    classifier.train(&features, &[0, 1]).unwrap();
    classifier.dispose();

    assert_eq!(
        classifier.predict_proba(&[5.0, 5.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn dispose_is_idempotent() {
    let mut classifier = classifier(with_max_iterations(50));
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    classifier.train(&features, &[0, 1]).unwrap();

    classifier.dispose();
    classifier.dispose();
    classifier.dispose();
}

#[test]
fn dispose_is_safe_on_an_untrained_model() {
    let mut classifier = classifier(KMeansLogisticConfig::default());
    classifier.dispose();
}

#[test]
fn uses_default_hyperparameters_when_not_specified() {
    let mut classifier = classifier(KMeansLogisticConfig::default());
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn respects_custom_weight_decay() {
    let mut classifier = classifier(KMeansLogisticConfig {
        weight_decay: 0.1,
        max_iterations: 50,
        ..KMeansLogisticConfig::default()
    });
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn handles_data_with_clear_cluster_structure() {
    let mut classifier = classifier(KMeansLogisticConfig {
        max_iterations: 100,
        ..KMeansLogisticConfig::default()
    });
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for index in 0..20 {
        let value = index as f32 * 0.1;
        features.push(vec![value, value]);
        labels.push(0);
    }
    for index in 0..20 {
        let value = 10.0 + index as f32 * 0.1;
        features.push(vec![value, value]);
        labels.push(1);
    }

    classifier.train(&features, &labels).unwrap();

    let probability_good = classifier.predict_proba(&[1.0, 1.0]).unwrap();
    let probability_bad = classifier.predict_proba(&[11.0, 11.0]).unwrap();
    assert!(probability_good > 0.5);
    assert!(probability_bad < 0.5);
}

#[test]
fn handles_mixed_clusters_with_both_classes() {
    let mut classifier = classifier(KMeansLogisticConfig {
        max_iterations: 100,
        ..KMeansLogisticConfig::default()
    });
    let features = vec![
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![10.0, 10.0],
        vec![10.5, 10.5],
        vec![11.0, 10.0],
        vec![10.0, 11.0],
    ];
    let labels = [0, 1, 0, 1, 0, 1, 0, 1];

    classifier.train(&features, &labels).unwrap();
    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!((0.0..=1.0).contains(&probability));
}

#[test]
fn failed_retraining_discards_the_previous_model() {
    let mut classifier = classifier(with_max_iterations(10));
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    assert!(matches!(
        classifier.train(&[vec![0.0, 0.0], vec![1.0]], &[0, 1]),
        Err(ClassifierError::RaggedFeatures { .. })
    ));
    assert_eq!(
        classifier.predict_proba(&[0.0, 0.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn rejects_centroids_that_overflow_when_narrowed_to_f32() {
    let mut classifier = classifier(with_max_iterations(10));
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();
    let mut state = serialized_state(&classifier);
    state.centroids[0][0] = 1e300;

    assert!(matches!(
        KMeansLogisticClassifier::from_json(state, with_max_iterations(10)),
        Err(ClassifierError::InvalidState(message))
            if message == "cluster centroids must be nonempty and remain finite in f32"
    ));
}

#[test]
fn rejects_global_and_cluster_mlp_dimension_mismatches_without_panicking() {
    let config = KMeansLogisticConfig {
        n_clusters: 1,
        max_iterations: 10,
        ..KMeansLogisticConfig::default()
    };
    let mut classifier = classifier(config);
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();
    let state = serialized_state(&classifier);

    let mut invalid_global = state.clone();
    invalid_global.global_model.layer_shapes[0] = 3;
    assert!(matches!(
        std::panic::catch_unwind(|| KMeansLogisticClassifier::from_json(invalid_global, config)),
        Ok(Err(ClassifierError::InvalidState(message)))
            if message == "global model input dimension must match centroid dimension 2"
    ));

    let mut invalid_cluster = state;
    invalid_cluster.cluster_models[0]
        .as_mut()
        .expect("single mixed cluster has a model")
        .layer_shapes[0] = 3;
    assert!(matches!(
        std::panic::catch_unwind(|| KMeansLogisticClassifier::from_json(invalid_cluster, config)),
        Ok(Err(ClassifierError::InvalidState(message)))
            if message == "cluster model 0 input dimension must match centroid dimension 2"
    ));
}
