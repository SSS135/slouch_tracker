use slouch_ml::ported::base_classifier::{BaseClassifier, ClassifierError};
use slouch_ml::ported::mlp_classifier::{MlpClassifier, MlpConfig};
use slouch_ml::ported::types::{SerializedClassifierState, SerializedMlp};

fn classifier(config: MlpConfig) -> MlpClassifier {
    MlpClassifier::new(config).unwrap()
}

fn two_cluster_features() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ]
}

fn with_max_iterations(max_iterations: usize) -> MlpConfig {
    MlpConfig {
        max_iterations,
        ..MlpConfig::default()
    }
}

fn serialized_state(classifier: &MlpClassifier) -> SerializedMlp {
    match classifier.to_json().unwrap() {
        SerializedClassifierState::Mlp(state) => state,
        _ => panic!("classifier returned the wrong serialized state"),
    }
}

#[test]
fn rejects_empty_dataset() {
    let mut classifier = classifier(MlpConfig::default());
    assert_eq!(
        classifier.train(&[], &[]),
        Err(ClassifierError::EmptyDataset)
    );
}

#[test]
fn rejects_feature_label_length_mismatch() {
    let mut classifier = classifier(MlpConfig::default());
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
    let mut classifier = classifier(with_max_iterations(100));
    assert!(classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .is_ok());
}

#[test]
fn handles_class_weight_balancing() {
    let mut classifier = classifier(MlpConfig {
        use_class_weights: true,
        max_iterations: 100,
        ..MlpConfig::default()
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
fn rejects_prediction_with_untrained_model() {
    let classifier = classifier(MlpConfig::default());
    assert_eq!(
        classifier.predict_proba(&[1.0, 2.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn returns_probabilities_in_the_unit_interval() {
    let mut classifier = classifier(with_max_iterations(100));
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
    let mut classifier = classifier(MlpConfig {
        max_iterations: 500,
        learning_rate: 0.05,
        ..MlpConfig::default()
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
fn works_with_zero_hidden_layers_logistic_regression_mode() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 0,
        max_iterations: 100,
        ..MlpConfig::default()
    });
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    classifier.train(&features, &[0, 1]).unwrap();

    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!((0.0..=1.0).contains(&probability));
}

#[test]
fn works_with_one_hidden_layer() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 1,
        hidden_size: 8,
        max_iterations: 100,
        ..MlpConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!((0.0..=1.0).contains(&probability));
}

#[test]
fn works_with_two_hidden_layers() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 2,
        hidden_size: 8,
        max_iterations: 100,
        ..MlpConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!((0.0..=1.0).contains(&probability));
}

#[test]
fn rejects_invalid_hidden_layer_values() {
    for hidden_layers in [3, usize::MAX] {
        assert!(matches!(
            MlpClassifier::new(MlpConfig {
                hidden_layers,
                ..MlpConfig::default()
            }),
            Err(ClassifierError::InvalidState(message))
                if message == "hiddenLayers must be 0, 1, or 2"
        ));
    }
    // The TypeScript -1 case cannot cross this native API as a usize. The
    // maximum usize exercises the same out-of-domain boundary after typing.
}

#[test]
fn rejects_serialization_of_an_untrained_model() {
    let classifier = classifier(MlpConfig::default());
    assert_eq!(classifier.to_json(), Err(ClassifierError::Untrained));
}

#[test]
fn serialized_state_has_the_expected_structure() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 1,
        hidden_size: 4,
        max_iterations: 50,
        ..MlpConfig::default()
    });
    classifier
        .train(&[vec![0.0, 0.0], vec![10.0, 10.0]], &[0, 1])
        .unwrap();

    let state = serialized_state(&classifier);
    assert_eq!(state.layer_weights.len(), 2);
    assert_eq!(state.layer_biases.len(), 2);
    assert_eq!(state.layer_shapes.len(), 3);
    assert_eq!(state.hidden_layers, 1);
    assert_eq!(state.hidden_size, 4);
    assert_eq!(state.class_weights.len(), 2);
}

#[test]
fn rejects_serialized_values_outside_the_finite_f32_range() {
    for state in [
        SerializedMlp {
            layer_weights: vec![vec![f64::MAX, 0.0]],
            layer_biases: vec![vec![0.0]],
            layer_shapes: vec![2, 1],
            hidden_layers: 0,
            hidden_size: 64,
            class_weights: [1.0, 1.0],
        },
        SerializedMlp {
            layer_weights: vec![vec![0.0, 0.0]],
            layer_biases: vec![vec![-f64::MAX]],
            layer_shapes: vec![2, 1],
            hidden_layers: 0,
            hidden_size: 64,
            class_weights: [1.0, 1.0],
        },
    ] {
        assert!(matches!(
            MlpClassifier::from_json(state, MlpConfig::default()),
            Err(ClassifierError::InvalidState(_))
        ));
    }
}

#[test]
fn restores_model_with_from_json() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 0,
        max_iterations: 100,
        ..MlpConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let state = serialized_state(&classifier);
    let restored = MlpClassifier::from_json(
        state,
        MlpConfig {
            hidden_layers: 0,
            ..MlpConfig::default()
        },
    )
    .unwrap();
    assert!(restored.predict_proba(&[5.0, 5.0]).is_ok());
}

#[test]
fn preserves_predictions_through_round_trip_serialization() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 1,
        hidden_size: 8,
        max_iterations: 100,
        ..MlpConfig::default()
    });
    classifier
        .train(&two_cluster_features(), &[0, 0, 1, 1])
        .unwrap();

    let test_input = [5.0, 5.0];
    let original_probability = classifier.predict_proba(&test_input).unwrap();
    let state = serialized_state(&classifier);
    let restored = MlpClassifier::from_json(
        state,
        MlpConfig {
            hidden_layers: 1,
            hidden_size: 8,
            ..MlpConfig::default()
        },
    )
    .unwrap();

    let restored_probability = restored.predict_proba(&test_input).unwrap();
    assert!((restored_probability - original_probability).abs() <= 1e-6);
}

#[test]
fn handles_edge_case_inputs_without_nan() {
    let mut classifier = classifier(MlpConfig {
        max_iterations: 100,
        learning_rate: 0.001,
        ..MlpConfig::default()
    });
    classifier
        .train(&[vec![0.0, 0.0], vec![1.0, 1.0]], &[0, 1])
        .unwrap();

    let probability = classifier.predict_proba(&[0.5, 0.5]).unwrap();
    assert!(probability.is_finite());
}

#[test]
fn produces_finite_weights_after_training() {
    let mut classifier = classifier(MlpConfig {
        hidden_layers: 1,
        hidden_size: 4,
        max_iterations: 50,
        ..MlpConfig::default()
    });
    classifier
        .train(&[vec![0.0, 0.0], vec![1.0, 1.0]], &[0, 1])
        .unwrap();

    let state = serialized_state(&classifier);
    assert!(state
        .layer_weights
        .iter()
        .flatten()
        .all(|weight| weight.is_finite()));
    assert!(state
        .layer_biases
        .iter()
        .flatten()
        .all(|bias| bias.is_finite()));
}

#[test]
fn handles_very_small_feature_values() {
    let mut classifier = classifier(with_max_iterations(100));
    classifier
        .train(&[vec![1e-10, 1e-10], vec![1e-9, 1e-9]], &[0, 1])
        .unwrap();

    let probability = classifier.predict_proba(&[5e-10, 5e-10]).unwrap();
    assert!(probability.is_finite());
}

#[test]
fn dispose_restores_untrained_state() {
    let mut classifier = classifier(with_max_iterations(50));
    classifier
        .train(&[vec![0.0, 0.0], vec![10.0, 10.0]], &[0, 1])
        .unwrap();
    classifier.dispose();

    assert_eq!(
        classifier.predict_proba(&[5.0, 5.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn dispose_is_idempotent() {
    let mut classifier = classifier(with_max_iterations(50));
    classifier
        .train(&[vec![0.0, 0.0], vec![10.0, 10.0]], &[0, 1])
        .unwrap();

    classifier.dispose();
    classifier.dispose();
    classifier.dispose();
}

#[test]
fn dispose_is_safe_on_an_untrained_model() {
    let mut classifier = classifier(MlpConfig::default());
    classifier.dispose();
}

#[test]
fn uses_default_hyperparameters_when_not_specified() {
    let mut classifier = classifier(MlpConfig::default());
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn respects_custom_learning_rate() {
    let mut classifier = classifier(MlpConfig {
        learning_rate: 0.001,
        max_iterations: 50,
        ..MlpConfig::default()
    });
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn respects_custom_weight_decay() {
    let mut classifier = classifier(MlpConfig {
        weight_decay: 0.1,
        max_iterations: 50,
        ..MlpConfig::default()
    });
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}

#[test]
fn respects_label_smoothing_parameter() {
    let mut classifier = classifier(MlpConfig {
        label_smoothing: 0.2,
        max_iterations: 50,
        ..MlpConfig::default()
    });
    let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    assert!(classifier.train(&features, &[0, 1]).is_ok());
}
