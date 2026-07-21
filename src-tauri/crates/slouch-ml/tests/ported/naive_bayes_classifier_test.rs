use slouch_ml::ported::base_classifier::{BaseClassifier, ClassifierError};
use slouch_ml::ported::naive_bayes_classifier::GaussianNbClassifier;
use slouch_ml::ported::types::SerializedGaussianNb;

fn create_feature(values: &[f32]) -> Vec<f32> {
    values.to_vec()
}

fn classifier(variance_smoothing: Option<f64>) -> GaussianNbClassifier {
    match variance_smoothing {
        Some(value) => GaussianNbClassifier::new(value).unwrap(),
        None => GaussianNbClassifier::default(),
    }
}

fn serialized_state(
    classifier: &GaussianNbClassifier,
) -> slouch_ml::ported::types::SerializedGaussianNb {
    match classifier.to_json().unwrap() {
        slouch_ml::ported::types::SerializedClassifierState::GaussianNb(state) => state,
        _ => panic!("classifier returned the wrong serialized state"),
    }
}

#[test]
fn rejects_epsilon_values_that_are_invalid_after_f32_conversion() {
    for epsilon in [f64::MAX, f64::from(f32::from_bits(1)) / 2.0] {
        assert!(matches!(
            GaussianNbClassifier::new(epsilon),
            Err(ClassifierError::InvalidEpsilon)
        ));
    }
}

#[test]
fn rejects_empty_dataset() {
    let mut classifier = classifier(None);
    assert_eq!(
        classifier.train(&[], &[]),
        Err(ClassifierError::EmptyDataset)
    );
}

#[test]
fn rejects_feature_label_length_mismatch() {
    let mut classifier = classifier(None);
    let features = vec![create_feature(&[1.0, 2.0])];
    assert_eq!(
        classifier.train(&features, &[0, 1]),
        Err(ClassifierError::LengthMismatch {
            features: 1,
            labels: 2,
        })
    );
}

#[test]
fn rejects_data_missing_either_class() {
    let mut classifier = classifier(None);
    let features = vec![create_feature(&[1.0, 2.0]), create_feature(&[2.0, 3.0])];
    assert_eq!(
        classifier.train(&features, &[0, 0]),
        Err(ClassifierError::MissingClass)
    );
}

#[test]
fn trains_successfully_with_separable_data() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
    ];
    assert!(classifier.train(&features, &[0, 0, 1, 1]).is_ok());
}

#[test]
fn handles_single_sample_per_class() {
    let mut classifier = classifier(Some(1e-3));
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();

    assert!(classifier.predict_proba(&[0.0, 0.0]).unwrap() > 0.5);
}

#[test]
fn rejects_prediction_with_untrained_model() {
    let classifier = classifier(None);
    assert_eq!(
        classifier.predict_proba(&[1.0, 2.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn returns_probabilities_in_the_unit_interval() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
    ];
    classifier.train(&features, &[0, 0, 1, 1]).unwrap();

    for probability in [
        classifier.predict_proba(&[0.0, 0.0]).unwrap(),
        classifier.predict_proba(&[10.0, 10.0]).unwrap(),
    ] {
        assert!((0.0..=1.0).contains(&probability));
    }
}

#[test]
fn predicts_high_probability_for_class_zero_samples() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[0.2, 0.0]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
        create_feature(&[10.2, 10.0]),
    ];
    classifier.train(&features, &[0, 0, 0, 1, 1, 1]).unwrap();

    assert!(classifier.predict_proba(&[0.0, 0.0]).unwrap() > 0.9);
}

#[test]
fn predicts_low_probability_for_class_one_samples() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[0.2, 0.0]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
        create_feature(&[10.2, 10.0]),
    ];
    classifier.train(&features, &[0, 0, 0, 1, 1, 1]).unwrap();

    assert!(classifier.predict_proba(&[10.0, 10.0]).unwrap() < 0.1);
}

#[test]
fn handles_intermediate_samples() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[1.0, 1.0]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[11.0, 11.0]),
    ];
    classifier.train(&features, &[0, 0, 1, 1]).unwrap();

    let probability = classifier.predict_proba(&[5.5, 5.5]).unwrap();
    assert!(probability > 0.3 && probability < 0.7);
}

#[test]
fn rejects_serialization_of_an_untrained_model() {
    let classifier = classifier(None);
    assert_eq!(classifier.to_json(), Err(ClassifierError::Untrained));
}

#[test]
fn serializes_trained_model() {
    let mut classifier = classifier(Some(1e-5));
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();

    let state = serialized_state(&classifier);
    assert_eq!(state.class_means.len(), 2);
    assert_eq!(state.class_variances.len(), 2);
    assert_eq!(state.class_priors.len(), 2);
    assert_eq!(state.epsilon, 1e-5);
}

#[test]
fn deserializes_and_produces_same_predictions() {
    let mut classifier = classifier(Some(1e-4));
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
    ];
    classifier.train(&features, &[0, 0, 1, 1]).unwrap();

    let test_input = [5.0, 5.0];
    let original_probability = classifier.predict_proba(&test_input).unwrap();
    let restored = GaussianNbClassifier::from_json(serialized_state(&classifier), 1e-4).unwrap();
    let restored_probability = restored.predict_proba(&test_input).unwrap();

    assert!((restored_probability - original_probability).abs() <= 1e-6);
}

#[test]
fn preserves_class_means_and_priors_through_serialization() {
    let mut classifier = classifier(None);
    let features = vec![
        create_feature(&[1.0, 2.0]),
        create_feature(&[2.0, 3.0]),
        create_feature(&[8.0, 9.0]),
        create_feature(&[9.0, 10.0]),
    ];
    classifier.train(&features, &[0, 0, 1, 1]).unwrap();
    let state = serialized_state(&classifier);

    assert!((state.class_means[0][0] - 1.5).abs() <= 1e-5);
    assert!((state.class_means[0][1] - 2.5).abs() <= 1e-5);
    assert!((state.class_means[1][0] - 8.5).abs() <= 1e-5);
    assert!((state.class_means[1][1] - 9.5).abs() <= 1e-5);
    assert!((state.class_priors[0] - 0.5).abs() <= 1e-5);
    assert!((state.class_priors[1] - 0.5).abs() <= 1e-5);
}

#[test]
fn dispose_restores_untrained_state() {
    let mut classifier = classifier(None);
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();
    classifier.dispose();

    assert_eq!(
        classifier.predict_proba(&[5.0, 5.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn high_dimensional_features_use_sqrt_dimension_normalization() {
    let dimensions = 200;
    let classifier = GaussianNbClassifier::from_state(SerializedGaussianNb {
        class_means: [vec![0.0; dimensions], vec![1.0; dimensions]],
        class_variances: [vec![1.0; dimensions], vec![1.0; dimensions]],
        class_priors: [0.5, 0.5],
        epsilon: 1e-6,
    })
    .unwrap();

    let probability = classifier.predict_proba(&vec![0.4; dimensions]).unwrap();
    assert!((probability - 0.804_429_682_506_956_9).abs() <= 1e-6);
}

#[test]
fn rejects_serialized_values_that_are_invalid_after_f32_conversion() {
    for state in [
        SerializedGaussianNb {
            class_means: [vec![f64::MAX], vec![1.0]],
            class_variances: [vec![1.0], vec![1.0]],
            class_priors: [0.5, 0.5],
            epsilon: 1e-6,
        },
        SerializedGaussianNb {
            class_means: [vec![0.0], vec![1.0]],
            class_variances: [vec![f64::from(f32::from_bits(1)) / 2.0], vec![1.0]],
            class_priors: [0.5, 0.5],
            epsilon: 1e-6,
        },
    ] {
        assert!(matches!(
            GaussianNbClassifier::from_state(state),
            Err(ClassifierError::InvalidState(_))
        ));
    }
}

#[test]
fn rejects_nonfinite_state_produced_by_extreme_but_finite_training_data() {
    let mut classifier = classifier(None);
    let result = classifier.train(
        &[vec![f32::MAX], vec![f32::MAX], vec![0.0], vec![1.0]],
        &[0, 0, 1, 1],
    );
    assert!(matches!(result, Err(ClassifierError::InvalidState(_))));
    assert_eq!(
        classifier.predict_proba(&[0.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn high_dimensional_well_separated_classes_remain_distinguishable() {
    let mut classifier = classifier(None);
    let dimensions = 150;
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..5 {
        features.push(
            (0..dimensions)
                .map(|column| if column % 2 == 0 { -0.05 } else { 0.05 })
                .collect::<Vec<_>>(),
        );
        labels.push(0);
    }
    for _ in 0..5 {
        features.push(
            (0..dimensions)
                .map(|column| if column % 2 == 0 { 9.95 } else { 10.05 })
                .collect::<Vec<_>>(),
        );
        labels.push(1);
    }

    classifier.train(&features, &labels).unwrap();
    assert!(classifier.predict_proba(&vec![0.0; 150]).unwrap() > 0.8);
    assert!(classifier.predict_proba(&vec![10.0; 150]).unwrap() < 0.2);
}

#[test]
fn default_variance_smoothing_is_one_e_minus_six() {
    let mut classifier = classifier(None);
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();

    assert_eq!(serialized_state(&classifier).epsilon, 1e-6);
}

#[test]
fn custom_variance_smoothing_is_preserved() {
    let mut classifier = classifier(Some(0.001));
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();

    assert_eq!(serialized_state(&classifier).epsilon, 0.001);
}

#[test]
fn smoothing_prevents_zero_variance_instability() {
    let mut classifier = classifier(Some(0.01));
    let features = vec![create_feature(&[5.0, 5.0]), create_feature(&[10.0, 10.0])];
    classifier.train(&features, &[0, 1]).unwrap();

    let probability = classifier.predict_proba(&[5.0, 5.0]).unwrap();
    assert!(probability.is_finite());
}
