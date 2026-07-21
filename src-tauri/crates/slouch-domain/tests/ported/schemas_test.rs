use std::collections::BTreeMap;

use serde_json::{json, Value};
use slouch_domain::ported::schemas::{
    validate_posture_frame_schema, validate_serialized_classifier_state, validate_with_schema,
    FeatureTypeSchema, SchemaValidationResult, SerializedClassifierStateSchema,
};
use slouch_domain::ported::{
    classifier_registry, validate_posture_frame, BoundingBox, ClassifierId, FeatureId, FrameLabel,
    Keypoint, PostureFrame, Thumbnail,
};

fn valid_frame() -> PostureFrame {
    let features = FeatureId::ALL
        .into_iter()
        .map(|feature_id| (feature_id, vec![0.1; feature_id.metadata().dimensions]))
        .collect::<BTreeMap<_, _>>();

    PostureFrame {
        id: "frame-123".to_owned(),
        timestamp: 1_700_000_000_000.0,
        features,
        keypoints: (0..17).map(|_| Keypoint::new(10.5, 20.3, 0.95)).collect(),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: b"test".to_vec(),
        },
        label: FrameLabel::Good,
        bbox: BoundingBox {
            x1: 10.0,
            y1: 20.0,
            x2: 100.0,
            y2: 200.0,
            score: 0.95,
            width: 90.0,
            height: 180.0,
        },
    }
}

fn frame_value() -> Value {
    serde_json::to_value(valid_frame()).expect("valid frame should serialize")
}

fn decode_and_validate(value: Value) -> bool {
    let frame = match serde_json::from_value::<PostureFrame>(value) {
        Ok(frame) => frame,
        Err(_) => return false,
    };
    validate_posture_frame(&frame).is_ok()
}

fn serialized_state_for(id: ClassifierId) -> Value {
    match id {
        ClassifierId::Mlp => json!({
            "layerWeights": [[0.1, 0.2]],
            "layerBiases": [[0.0]],
            "layerShapes": [2, 1],
            "hiddenLayers": 0,
            "hiddenSize": 64,
            "classWeights": [1.0, 1.0]
        }),
        ClassifierId::Knn => json!({
            "trainingData": [[0.1, 0.2], [0.3, 0.4]],
            "trainingLabels": [0.0, 1.0],
            "k": 3,
            "kernel": "cosine",
            "gamma": 1.0
        }),
        ClassifierId::Svm => json!({
            "weights": [0.1, 0.2],
            "bias": 0.0,
            "classWeights": [1.0, 1.0]
        }),
        ClassifierId::KmeansPrototype => json!({
            "clusters": [{
                "centroid": [0.1, 0.2],
                "prototypeGood": [0.1, 0.2],
                "prototypeBad": [0.3, 0.4]
            }],
            "globalPrototypeGood": [0.1, 0.2],
            "globalPrototypeBad": [0.3, 0.4],
            "temperature": 1.0
        }),
        ClassifierId::GaussianNb => json!({
            "classMeans": [[0.1, 0.2], [0.3, 0.4]],
            "classVariances": [[1.0, 1.0], [1.0, 1.0]],
            "classPriors": [0.5, 0.5],
            "epsilon": 1e-6
        }),
        ClassifierId::KmeansLogistic => json!({
            "centroids": [[0.1, 0.2]],
            "clusterModels": [null],
            "globalModel": {
                "layerWeights": [[0.1, 0.2]],
                "layerBiases": [[0.0]],
                "layerShapes": [2, 1],
                "hiddenLayers": 0,
                "hiddenSize": 64,
                "classWeights": [1.0, 1.0]
            },
            "temperature": 1.0
        }),
    }
}

#[test]
fn parses_valid_posture_frame() {
    assert!(decode_and_validate(frame_value()));
}

#[test]
fn parses_frame_with_multiple_features() {
    assert_eq!(valid_frame().features.len(), FeatureId::ALL.len());
    assert!(decode_and_validate(frame_value()));
}

#[test]
fn round_trips_every_feature_type_with_exact_registry_wire_names() {
    for (schema_feature, feature_id) in FeatureTypeSchema::ALL.into_iter().zip(FeatureId::ALL) {
        let encoded = serde_json::to_value(schema_feature).unwrap();
        assert_eq!(encoded, Value::String(feature_id.as_str().to_owned()));
        let decoded = serde_json::from_value::<FeatureTypeSchema>(encoded).unwrap();
        assert_eq!(decoded, schema_feature);
    }
}

#[test]
fn rejects_string_thumbnail_and_invalid_thumbnail_metadata() {
    let mut string_thumbnail = frame_value();
    string_thumbnail["thumbnail"] = Value::String("data:image/webp;base64,xyz".to_owned());
    assert!(!decode_and_validate(string_thumbnail));

    let mut invalid_mime = valid_frame();
    invalid_mime.thumbnail.mime_type = "text/plain".to_owned();
    assert!(validate_posture_frame(&invalid_mime).is_err());

    let mut empty_thumbnail = valid_frame();
    empty_thumbnail.thumbnail.bytes.clear();
    assert!(validate_posture_frame(&empty_thumbnail).is_err());
}

#[test]
fn source_schema_accepts_arbitrary_feature_dimensions() {
    let mut frame = valid_frame();
    frame
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 1_000]);

    assert!(validate_posture_frame_schema(&frame).is_ok());
}

#[test]
fn native_boundary_rejects_registry_mismatched_feature_dimensions() {
    let mut frame = valid_frame();
    frame
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 1_000]);

    assert!(validate_posture_frame(&frame).is_err());
}

#[test]
fn rejects_invalid_label() {
    let mut value = frame_value();
    value["label"] = Value::String("invalid".to_owned());
    assert!(!decode_and_validate(value));
}

#[test]
fn rejects_missing_required_fields() {
    let mut value = frame_value();
    value.as_object_mut().unwrap().remove("features");
    assert!(!decode_and_validate(value));
}

#[test]
fn rejects_empty_id() {
    let mut frame = valid_frame();
    frame.id.clear();
    assert!(validate_posture_frame(&frame).is_err());
}

#[test]
fn rejects_missing_bbox() {
    let mut value = frame_value();
    value.as_object_mut().unwrap().remove("bbox");
    assert!(!decode_and_validate(value));
}

#[test]
fn rejects_incomplete_bbox() {
    let mut value = frame_value();
    value["bbox"] = json!({"x1": 10.0, "y1": 20.0});
    assert!(!decode_and_validate(value));
}

#[test]
fn rejects_missing_keypoints() {
    let mut value = frame_value();
    value.as_object_mut().unwrap().remove("keypoints");
    assert!(!decode_and_validate(value));
}

#[test]
fn rejects_invalid_keypoints_array() {
    let mut value = frame_value();
    value["keypoints"] = json!([{ "x": 10.0, "y": 20.0 }]);
    assert!(!decode_and_validate(value));
}

#[test]
fn validates_null_undefined_and_empty_objects_as_invalid() {
    assert!(serde_json::from_value::<PostureFrame>(Value::Null).is_err());
    assert!(serde_json::from_value::<PostureFrame>(json!({})).is_err());
}

#[test]
fn returns_an_error_for_invalid_frame_data() {
    let mut value = frame_value();
    value["id"] = Value::String(String::new());
    let frame = serde_json::from_value::<PostureFrame>(value).unwrap();
    let error = validate_posture_frame(&frame).unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn validate_with_schema_returns_success_and_failure_variants() {
    let success = validate_with_schema(validate_posture_frame_schema, valid_frame());
    match success {
        SchemaValidationResult::Success { data } => assert_eq!(data.id, "frame-123"),
        SchemaValidationResult::Failure { error } => panic!("unexpected failure: {error}"),
    }

    let mut invalid = valid_frame();
    invalid.id.clear();
    let failure = validate_with_schema(validate_posture_frame_schema, invalid);
    match failure {
        SchemaValidationResult::Failure { error } => assert!(!error.is_empty()),
        SchemaValidationResult::Success { .. } => panic!("invalid frame unexpectedly succeeded"),
    }
}

#[test]
fn enforces_required_nullable_and_optional_non_null_fields() {
    let valid_prototype = json!({
        "clusters": [{
            "centroid": [0.1, 0.2],
            "prototypeGood": null,
            "prototypeBad": null
        }],
        "globalPrototypeGood": [0.1, 0.2],
        "globalPrototypeBad": [0.3, 0.4],
        "temperature": 1.0
    });
    assert!(
        serde_json::from_value::<SerializedClassifierStateSchema>(valid_prototype.clone()).is_ok()
    );

    let mut missing_prototype = valid_prototype;
    missing_prototype["clusters"][0]
        .as_object_mut()
        .unwrap()
        .remove("prototypeGood");
    assert!(serde_json::from_value::<SerializedClassifierStateSchema>(missing_prototype).is_err());

    let knn_without_optionals = json!({
        "trainingData": [[0.1], [0.2]],
        "trainingLabels": [0.0, 1.0],
        "k": 3
    });
    assert!(
        serde_json::from_value::<SerializedClassifierStateSchema>(knn_without_optionals).is_ok()
    );

    let knn_with_null = json!({
        "trainingData": [[0.1], [0.2]],
        "trainingLabels": [0.0, 1.0],
        "k": 3,
        "kernel": null,
        "gamma": null
    });
    assert!(serde_json::from_value::<SerializedClassifierStateSchema>(knn_with_null).is_err());
}

#[test]
fn serialized_feature_extractor_requires_explicit_nullable_fields() {
    let valid = json!({
        "featureTypes": ["gau_features"],
        "normalizationMode": "none",
        "dimReductionConfig": { "method": "none", "components": 1 },
        "concatenatedDimensions": 256,
        "normalizationMean": null,
        "normalizationStd": null,
        "dimReductionTransformer": null
    });
    assert!(serde_json::from_value::<
        slouch_domain::ported::schemas::SerializedFeatureExtractorSchema,
    >(valid.clone())
    .is_ok());

    for field in [
        "normalizationMean",
        "normalizationStd",
        "dimReductionTransformer",
    ] {
        let mut missing = valid.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<
            slouch_domain::ported::schemas::SerializedFeatureExtractorSchema,
        >(missing)
        .is_err());
    }
}

#[test]
fn pca_explained_variance_rejects_explicit_null() {
    let base = json!({
        "components": [[0.1, 0.2]],
        "mean": [0.0, 0.0],
        "nComponents": 1,
        "nFeatures": 2
    });

    let mut with_null = base.clone();
    with_null
        .as_object_mut()
        .unwrap()
        .insert("explainedVariance".to_owned(), Value::Null);
    assert!(
        serde_json::from_value::<slouch_domain::ported::schemas::SerializedPcaSchema>(with_null)
            .is_err()
    );

    assert!(
        serde_json::from_value::<slouch_domain::ported::schemas::SerializedPcaSchema>(base).is_ok()
    );
}

#[test]
fn serialized_state_schema_covers_every_registered_classifier() {
    for classifier in classifier_registry() {
        let state = serde_json::from_value::<SerializedClassifierStateSchema>(
            serialized_state_for(classifier.id),
        )
        .unwrap_or_else(|error| {
            panic!(
                "classifier {} produced undecodable serialized state: {error}",
                classifier.id
            )
        });
        validate_serialized_classifier_state(&state).unwrap_or_else(|error| {
            panic!(
                "classifier {} produced invalid serialized state: {error}",
                classifier.id
            )
        });
    }
}
