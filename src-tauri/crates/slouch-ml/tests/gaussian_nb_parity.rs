use serde_json::Value;
use slouch_ml::ported::{
    base_classifier::{BaseClassifier, ClassifierError},
    naive_bayes_classifier::GaussianNbClassifier,
    types::SerializedGaussianNb,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/classifiers/gaussian-nb-v1.json"
    ))
    .unwrap()
}

fn matrix(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn f32_array(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

fn labels(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_i64().unwrap() as i32)
        .collect()
}

fn train_fixture() -> (GaussianNbClassifier, Value) {
    let fixture = fixture();
    let mut classifier = GaussianNbClassifier::new(fixture["epsilon"].as_f64().unwrap()).unwrap();
    classifier
        .train(&matrix(&fixture["features"]), &labels(&fixture["labels"]))
        .unwrap();
    (classifier, fixture)
}

#[test]
fn fixture_pins_current_sources() {
    let fixture = fixture();
    assert_eq!(
        fixture["sources"]["naiveBayesClassifier.ts"],
        "45ebc5d230064b25898cc64398dec787866a8a334da151d96b28edfa7466905e"
    );
    assert_eq!(
        fixture["sources"]["naiveBayesClassifier.test.ts"],
        "1d37b6fa8a1be13f7cedfb706789cc8d8895ae1a7c126df1c3176ff6923a2946"
    );
}

#[test]
fn rejects_empty_dataset() {
    assert_eq!(
        GaussianNbClassifier::default().train(&[], &[]),
        Err(ClassifierError::EmptyDataset)
    );
}

#[test]
fn rejects_feature_label_length_mismatch() {
    assert_eq!(
        GaussianNbClassifier::default().train(&[vec![1.0, 2.0]], &[0, 1]),
        Err(ClassifierError::LengthMismatch {
            features: 1,
            labels: 2
        })
    );
}

#[test]
fn rejects_data_missing_either_class() {
    let features = vec![vec![1.0, 2.0], vec![2.0, 3.0]];
    assert_eq!(
        GaussianNbClassifier::default().train(&features, &[0, 0]),
        Err(ClassifierError::MissingClass)
    );
}

#[test]
fn trains_successfully_with_separable_data() {
    let features = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    assert!(GaussianNbClassifier::default()
        .train(&features, &[0, 0, 1, 1])
        .is_ok());
}

#[test]
fn single_sample_class_variance_is_epsilon_by_f32_bits() {
    let mut classifier = GaussianNbClassifier::new(1e-3).unwrap();
    classifier
        .train(&[vec![0.0, 0.0], vec![10.0, 10.0]], &[0, 1])
        .unwrap();
    let state = classifier.to_state().unwrap();
    let expected = (1e-3_f64 as f32).to_bits();
    assert!(state
        .class_variances
        .iter()
        .flatten()
        .all(|value| (*value as f32).to_bits() == expected));
    assert!(classifier.predict_proba(&[0.0, 0.0]).unwrap() > 0.5);
}

#[test]
fn every_nonzero_label_enters_class_one() {
    let fixture = fixture();
    let case = &fixture["nonzeroLabelCase"];
    let mut classifier = GaussianNbClassifier::default();
    classifier
        .train(&matrix(&case["features"]), &labels(&case["labels"]))
        .unwrap();
    let state = classifier.to_state().unwrap();
    assert_eq!(
        (state.class_means[0][0] as f32).to_bits(),
        0.5_f32.to_bits()
    );
    assert_eq!(
        (state.class_means[1][0] as f32).to_bits(),
        9.5_f32.to_bits()
    );
    assert_eq!(state.class_priors, [0.5, 0.5]);
}

#[test]
fn means_and_population_variances_match_golden_f32_bits() {
    let (classifier, fixture) = train_fixture();
    let state = classifier.to_state().unwrap();
    let expected = &fixture["expectedState"];
    let expected_means = matrix(&expected["classMeans"]);
    let expected_variances = matrix(&expected["classVariances"]);
    for (actual, expected) in state
        .class_means
        .iter()
        .flatten()
        .zip(expected_means.iter().flatten())
    {
        assert_eq!((*actual as f32).to_bits(), expected.to_bits());
    }
    for (actual, expected) in state
        .class_variances
        .iter()
        .flatten()
        .zip(expected_variances.iter().flatten())
    {
        assert_eq!((*actual as f32).to_bits(), expected.to_bits());
    }
}

#[test]
fn priors_use_f64_sample_ratios() {
    let (classifier, _) = train_fixture();
    assert_eq!(
        classifier.to_state().unwrap().class_priors,
        [0.6_f64, 0.4_f64]
    );
}

#[test]
fn fixture_predictions_return_probability_of_good_with_log_scaling() {
    let (classifier, fixture) = train_fixture();
    for probe in fixture["probes"].as_array().unwrap() {
        let features: Vec<f32> = probe["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let actual = classifier.predict_proba(&features).unwrap();
        let expected = probe["probabilityGood"].as_f64().unwrap();
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn untrained_prediction_errors() {
    assert_eq!(
        GaussianNbClassifier::default().predict_proba(&[1.0, 2.0]),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn untrained_serialization_errors() {
    assert_eq!(
        GaussianNbClassifier::default().to_state(),
        Err(ClassifierError::Untrained)
    );
}

#[test]
fn serialization_fields_match_typescript_contract() {
    let (classifier, _) = train_fixture();
    let value = serde_json::to_value(classifier.to_state().unwrap()).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(object.len(), 4);
    for key in ["classMeans", "classVariances", "classPriors", "epsilon"] {
        assert!(object.contains_key(key));
    }
}

#[test]
fn serialization_round_trip_preserves_state_bits_and_prediction() {
    let (classifier, _) = train_fixture();
    let state = classifier.to_state().unwrap();
    let encoded = serde_json::to_vec(&state).unwrap();
    let decoded: SerializedGaussianNb = serde_json::from_slice(&encoded).unwrap();
    let restored = GaussianNbClassifier::from_state(decoded.clone()).unwrap();
    assert_eq!(restored.to_state().unwrap(), state);
    assert!(
        (restored.predict_proba(&[6.0, 9.0]).unwrap()
            - classifier.predict_proba(&[6.0, 9.0]).unwrap())
        .abs()
            <= 1e-12
    );
}

#[test]
fn high_dimensional_case_matches_deterministic_typescript_golden() {
    let fixture = fixture();
    let case = &fixture["highDimensionalCase"];
    let dimensions = case["dimensions"].as_u64().unwrap() as usize;
    let samples_per_class = case["samplesPerClass"].as_u64().unwrap() as usize;
    let class0 = f32_array(&case["class0Alternating"]);
    let class1 = f32_array(&case["class1Alternating"]);
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for index in 0..samples_per_class * 2 {
        let class = usize::from(index >= samples_per_class);
        let values = if class == 0 { &class0 } else { &class1 };
        features.push(vec![values[index % 2]; dimensions]);
        labels.push(class as i32);
    }

    let mut classifier = GaussianNbClassifier::default();
    classifier.train(&features, &labels).unwrap();
    let state = classifier.to_state().unwrap();
    let expected_means = f32_array(&case["expectedMean"]);
    let expected_variance = case["expectedVariance"].as_f64().unwrap() as f32;
    assert_eq!(
        (state.class_means[0][0] as f32).to_bits(),
        expected_means[0].to_bits()
    );
    assert_eq!(
        (state.class_means[1][0] as f32).to_bits(),
        expected_means[1].to_bits()
    );
    assert_eq!(
        (state.class_variances[0][0] as f32).to_bits(),
        expected_variance.to_bits()
    );
    assert_eq!(
        (state.class_variances[1][0] as f32).to_bits(),
        expected_variance.to_bits()
    );

    let probe = vec![case["probeValue"].as_f64().unwrap() as f32; dimensions];
    let actual = classifier.predict_proba(&probe).unwrap();
    let expected = case["probabilityGood"].as_f64().unwrap();
    assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
}

#[test]
fn high_dimensional_well_separated_classes_remain_distinguishable() {
    let dimensions = 150;
    let mut features = vec![vec![0.0; dimensions]; 5];
    features.extend(vec![vec![10.0; dimensions]; 5]);
    let mut classifier = GaussianNbClassifier::default();
    classifier
        .train(&features, &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        .unwrap();
    assert!(classifier.predict_proba(&vec![0.0; dimensions]).unwrap() > 0.8);
    assert!(classifier.predict_proba(&vec![10.0; dimensions]).unwrap() < 0.2);
}

#[test]
fn default_variance_smoothing_is_one_e_minus_six() {
    let mut classifier = GaussianNbClassifier::default();
    classifier.train(&[vec![0.0], vec![10.0]], &[0, 1]).unwrap();
    assert_eq!(classifier.to_state().unwrap().epsilon, 1e-6);
}

#[test]
fn custom_variance_smoothing_is_preserved() {
    let mut classifier = GaussianNbClassifier::new(0.001).unwrap();
    classifier.train(&[vec![0.0], vec![10.0]], &[0, 1]).unwrap();
    assert_eq!(classifier.to_state().unwrap().epsilon, 0.001);
}

#[test]
fn smoothing_prevents_zero_variance_instability() {
    let mut classifier = GaussianNbClassifier::new(0.01).unwrap();
    classifier
        .train(
            &[vec![5.0], vec![5.0], vec![10.0], vec![10.0]],
            &[0, 0, 1, 1],
        )
        .unwrap();
    assert!(classifier.predict_proba(&[5.0]).unwrap().is_finite());
}

#[test]
fn dispose_restores_untrained_state_and_is_idempotent() {
    let (mut classifier, _) = train_fixture();
    classifier.dispose();
    classifier.dispose();
    assert_eq!(
        classifier.predict_proba(&[5.0, 8.0]),
        Err(ClassifierError::Untrained)
    );
    assert_eq!(classifier.to_state(), Err(ClassifierError::Untrained));
}

#[test]
fn rejects_nonpositive_and_nonfinite_epsilon() {
    for epsilon in [0.0, -0.001, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            GaussianNbClassifier::new(epsilon),
            Err(ClassifierError::InvalidEpsilon)
        ));
    }
}

#[test]
fn rejects_empty_ragged_and_nonfinite_feature_vectors() {
    let mut classifier = GaussianNbClassifier::default();
    assert_eq!(
        classifier.train(&[vec![], vec![]], &[0, 1]),
        Err(ClassifierError::EmptyFeatureVector)
    );
    assert!(matches!(
        classifier.train(&[vec![0.0, 1.0], vec![2.0], vec![9.0, 10.0]], &[0, 0, 1]),
        Err(ClassifierError::RaggedFeatures { .. })
    ));
    assert_eq!(
        classifier.train(&[vec![f32::NAN], vec![1.0]], &[0, 1]),
        Err(ClassifierError::NonFiniteFeature { row: 0, column: 0 })
    );
}

#[test]
fn rejects_wrong_or_nonfinite_prediction_input() {
    let (classifier, _) = train_fixture();
    assert_eq!(
        classifier.predict_proba(&[1.0, 2.0, 3.0]),
        Err(ClassifierError::PredictionDimension {
            expected: 2,
            actual: 3
        })
    );
    assert_eq!(
        classifier.predict_proba(&[f32::INFINITY, 2.0]),
        Err(ClassifierError::NonFiniteFeature { row: 0, column: 0 })
    );
}

#[test]
fn serialized_state_validation_rejects_inconsistent_or_invalid_state() {
    let bad = SerializedGaussianNb {
        class_means: [vec![0.0], vec![1.0, 2.0]],
        class_variances: [vec![1.0], vec![1.0]],
        class_priors: [0.5, 0.5],
        epsilon: 1e-6,
    };
    assert!(matches!(
        GaussianNbClassifier::from_state(bad),
        Err(ClassifierError::InvalidState(_))
    ));
}

#[test]
fn classifier_id_matches_registry() {
    assert_eq!(
        GaussianNbClassifier::default().classifier_id(),
        "gaussian_nb"
    );
}
