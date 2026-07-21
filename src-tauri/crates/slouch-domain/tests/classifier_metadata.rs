use slouch_domain::{
    classifier_registry, normalize_classifier_id, ClassifierId, ClassifierLookupError,
    ClassifierParameters, KnnDistance, ParameterValue,
};

#[test]
fn registry_has_exact_six_ids_and_runtime_knn_contract() {
    let registry = classifier_registry();
    assert_eq!(
        registry.iter().map(|item| item.id).collect::<Vec<_>>(),
        ClassifierId::ALL
    );
    let knn = registry
        .iter()
        .find(|item| item.id == ClassifierId::Knn)
        .unwrap();
    let kernel = knn.params.get("kernel").unwrap();
    assert_eq!(kernel.default, ParameterValue::String("cosine".into()));
    assert!(knn.params.contains_key("gamma"));
    assert!(!knn.params.contains_key("distance"));
    assert_eq!(
        serde_json::to_value(knn.params.get("gamma").unwrap()).unwrap()["showWhen"]["kernel"],
        "rbf"
    );
}

#[test]
fn registry_parameter_order_matches_typescript_declarations() {
    let registry = classifier_registry();
    let expected = [
        (
            ClassifierId::Mlp,
            vec![
                "hiddenLayers",
                "hiddenSize",
                "weightDecay",
                "maxIterations",
                "learningRate",
                "useClassWeights",
                "labelSmoothing",
            ],
        ),
        (ClassifierId::Knn, vec!["k", "kernel", "gamma"]),
        (
            ClassifierId::Svm,
            vec!["C", "maxIterations", "useClassWeights"],
        ),
        (
            ClassifierId::KmeansPrototype,
            vec!["nClusters", "temperature"],
        ),
        (ClassifierId::GaussianNb, vec!["varianceSmoothing"]),
        (
            ClassifierId::KmeansLogistic,
            vec![
                "nClusters",
                "temperature",
                "weightDecay",
                "maxIterations",
                "useClassWeights",
            ],
        ),
    ];

    for (id, keys) in expected {
        let metadata = registry.iter().find(|item| item.id == id).unwrap();
        assert_eq!(metadata.params.keys().collect::<Vec<_>>(), keys);
        assert!(serde_json::to_value(&metadata.params).unwrap().is_object());
    }
}

#[test]
fn ordered_params_keep_exact_json_object_shape() {
    let registry = classifier_registry();
    let params = &registry
        .iter()
        .find(|item| item.id == ClassifierId::GaussianNb)
        .unwrap()
        .params;
    assert_eq!(
        serde_json::to_string(params).unwrap(),
        r#"{"varianceSmoothing":{"type":"range","label":"Variance Smoothing","description":"Stability term added to variances to prevent division by zero","default":1e-6,"min":1e-12,"max":0.01,"scale":"exponential"}}"#
    );
}

#[test]
fn typed_parameter_dto_preserves_typescript_contract() {
    let params = ClassifierParameters::Knn {
        k: 3,
        distance: KnnDistance::Euclidean,
    };
    assert_eq!(
        serde_json::to_value(params).unwrap(),
        serde_json::json!({
            "classifierId": "knn",
            "k": 3,
            "distance": "euclidean"
        })
    );
}

#[test]
fn kmeans_metadata_includes_runtime_nclusters() {
    let registry = classifier_registry();
    for id in [ClassifierId::KmeansPrototype, ClassifierId::KmeansLogistic] {
        assert!(registry
            .iter()
            .find(|item| item.id == id)
            .unwrap()
            .params
            .contains_key("nClusters"));
    }
}

#[test]
fn legacy_aliases_and_deprecated_logistic_behavior_are_explicit() {
    for alias in [
        "PrototypicalClassifier",
        "prototypical",
        "KMeansPrototypeClassifier",
    ] {
        assert_eq!(
            normalize_classifier_id(alias).unwrap(),
            ClassifierId::KmeansPrototype
        );
    }
    assert_eq!(
        normalize_classifier_id("GaussianNBClassifier").unwrap(),
        ClassifierId::GaussianNb
    );
    assert_eq!(
        normalize_classifier_id("KMeansLogisticClassifier").unwrap(),
        ClassifierId::KmeansLogistic
    );
    assert_eq!(
        normalize_classifier_id("logistic_regression"),
        Err(ClassifierLookupError::DeprecatedLogisticRegression)
    );
}
