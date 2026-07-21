use std::collections::BTreeMap;

use slouch_domain::{ClassifierConfig, ClassifierId, ClassifierLookupError, ParameterValue};
use slouch_ml::ported::base_classifier::{BaseClassifier, ClassifierError};
use slouch_ml::ported::classifier_factory::{create_classifier, deserialize_classifier};
use slouch_ml::ported::types::{
    KMeansPrototypeCluster, KnnKernel, SerializedClassifierState, SerializedKMeansPrototype,
    SerializedKnn, SerializedMlp, SerializedSvm,
};

fn config(
    classifier_id: &str,
    params: impl IntoIterator<Item = (&'static str, ParameterValue)>,
) -> ClassifierConfig {
    ClassifierConfig {
        classifier_id: classifier_id.parse::<ClassifierId>().unwrap(),
        params: params
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect::<BTreeMap<_, _>>(),
    }
}

fn number(value: f64) -> ParameterValue {
    ParameterValue::Number(value)
}

fn classifier_id(classifier: &dyn BaseClassifier) -> &'static str {
    classifier.classifier_id()
}

#[test]
fn creates_knn_classifier() {
    let config = config("knn", [("k", number(5.0))]);
    let classifier = create_classifier(config).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "knn");
}

#[test]
fn rejects_unknown_classifier_configuration() {
    assert!(matches!(
        "unknown_classifier".parse::<ClassifierId>(),
        Err(ClassifierLookupError::Unknown(id)) if id == "unknown_classifier"
    ));
}

#[test]
fn creates_mlp_classifier() {
    let config = config(
        "mlp",
        [
            ("hiddenLayers", number(1.0)),
            ("hiddenSize", number(64.0)),
            ("maxIterations", number(100.0)),
            ("learningRate", number(0.01)),
        ],
    );
    let classifier = create_classifier(config).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "mlp");
}

#[test]
fn creates_svm_classifier() {
    let config = config(
        "svm",
        [("C", number(1.0)), ("maxIterations", number(100.0))],
    );
    let classifier = create_classifier(config).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "svm");
}

#[test]
fn creates_kmeans_prototype_classifier() {
    let config = config("kmeans_prototype", [("temperature", number(1.0))]);
    let classifier = create_classifier(config).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "kmeans_prototype");
}

#[test]
fn deserializes_knn_classifier_from_legacy_class_name() {
    let state = SerializedClassifierState::Knn(SerializedKnn {
        training_data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        training_labels: vec![0.0, 1.0],
        k: 3,
        kernel: None,
        gamma: None,
    });
    let config = config("knn", [("k", number(3.0))]);

    let classifier = deserialize_classifier("KNNClassifier", state, &config.params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "knn");
}

#[test]
fn deserializes_knn_classifier_from_id() {
    let state = SerializedClassifierState::Knn(SerializedKnn {
        training_data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        training_labels: vec![0.0, 1.0],
        k: 3,
        kernel: None,
        gamma: None,
    });
    let config = config("knn", [("k", number(3.0))]);

    let classifier = deserialize_classifier("knn", state, &config.params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "knn");
}

fn mlp_state() -> SerializedClassifierState {
    SerializedClassifierState::Mlp(SerializedMlp {
        layer_weights: vec![vec![0.1, 0.2]],
        layer_biases: vec![vec![0.0]],
        layer_shapes: vec![2, 1],
        hidden_layers: 0,
        hidden_size: 64,
        class_weights: [1.0, 1.0],
    })
}

fn mlp_config() -> ClassifierConfig {
    config(
        "mlp",
        [
            ("hiddenLayers", number(0.0)),
            ("hiddenSize", number(64.0)),
            ("maxIterations", number(100.0)),
            ("learningRate", number(0.01)),
        ],
    )
}

#[test]
fn deserializes_mlp_classifier_from_legacy_class_name() {
    let classifier =
        deserialize_classifier("MLPClassifier", mlp_state(), &mlp_config().params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "mlp");
}

#[test]
fn deserializes_mlp_classifier_from_id() {
    let classifier = deserialize_classifier("mlp", mlp_state(), &mlp_config().params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "mlp");
}

fn svm_state() -> SerializedClassifierState {
    SerializedClassifierState::Svm(SerializedSvm {
        weights: vec![1.0, 2.0, 3.0],
        bias: 0.5,
        class_weights: [1.0, 1.0],
    })
}

fn svm_config() -> ClassifierConfig {
    config(
        "svm",
        [("C", number(1.0)), ("maxIterations", number(100.0))],
    )
}

#[test]
fn deserializes_svm_classifier_from_legacy_class_name() {
    let classifier =
        deserialize_classifier("SVMClassifier", svm_state(), &svm_config().params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "svm");
}

#[test]
fn deserializes_svm_classifier_from_id() {
    let classifier = deserialize_classifier("svm", svm_state(), &svm_config().params).unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "svm");
}

fn kmeans_prototype_state() -> SerializedClassifierState {
    SerializedClassifierState::KMeansPrototype(SerializedKMeansPrototype {
        clusters: vec![KMeansPrototypeCluster {
            centroid: vec![1.0, 2.0, 3.0],
            prototype_good: Some(vec![0.5, 0.5, 0.5]),
            prototype_bad: Some(vec![1.5, 1.5, 1.5]),
        }],
        global_prototype_good: vec![0.5, 0.5, 0.5],
        global_prototype_bad: vec![1.5, 1.5, 1.5],
        temperature: 1.0,
    })
}

fn kmeans_prototype_config() -> ClassifierConfig {
    config("kmeans_prototype", [("temperature", number(1.0))])
}

#[test]
fn deserializes_kmeans_prototype_from_legacy_class_name() {
    let classifier = deserialize_classifier(
        "PrototypicalClassifier",
        kmeans_prototype_state(),
        &kmeans_prototype_config().params,
    )
    .unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "kmeans_prototype");
}

#[test]
fn deserializes_kmeans_prototype_from_legacy_id() {
    let classifier = deserialize_classifier(
        "prototypical",
        kmeans_prototype_state(),
        &kmeans_prototype_config().params,
    )
    .unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "kmeans_prototype");
}

#[test]
fn deserializes_kmeans_prototype_from_current_id() {
    let classifier = deserialize_classifier(
        "kmeans_prototype",
        kmeans_prototype_state(),
        &kmeans_prototype_config().params,
    )
    .unwrap();

    assert_eq!(classifier_id(classifier.as_ref()), "kmeans_prototype");
}

#[test]
fn rejects_unknown_classifier_when_deserializing() {
    let state = SerializedClassifierState::Knn(SerializedKnn {
        training_data: vec![vec![1.0]],
        training_labels: vec![0.0],
        k: 1,
        kernel: Some(KnnKernel::Cosine),
        gamma: Some(1.0),
    });
    let config = config("knn", [("k", number(1.0))]);

    let error = match deserialize_classifier("unknown_classifier", state, &config.params) {
        Err(error) => error,
        Ok(_) => panic!("unknown classifier unexpectedly deserialized"),
    };
    assert!(matches!(
        error,
        ClassifierError::InvalidState(message)
            if message == "Unknown classifier ID: unknown_classifier"
    ));
}

#[test]
fn rejects_fractional_non_finite_and_out_of_bounds_parameters() {
    for invalid_k in [0.0, 20.5, 21.0, f64::NAN, f64::INFINITY] {
        let error = match create_classifier(config("knn", [("k", number(invalid_k))])) {
            Err(error) => error,
            Ok(_) => panic!("invalid k unexpectedly succeeded"),
        };
        assert!(!error.to_string().is_empty());
    }

    let error = match create_classifier(config(
        "svm",
        [("C", number(1.0)), ("maxIterations", number(3001.0))],
    )) {
        Err(error) => error,
        Ok(_) => panic!("out-of-bounds maxIterations unexpectedly succeeded"),
    };
    assert!(error.to_string().contains("bounds"));
}
