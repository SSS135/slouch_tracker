use serde_json::{json, Value};
use slouch_domain::ported::messages::schemas::{InferenceResponseResult, InferenceWorkerResponse};
use slouch_domain::{BoundingBox, ExpandedBbox, FeatureId, Keypoint};

fn bbox(x1: f64, y1: f64, x2: f64, y2: f64) -> Value {
    json!({
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": 0.9,
        "width": x2 - x1,
        "height": y2 - y1
    })
}

fn keypoints() -> Value {
    Value::Array(
        (0..17)
            .map(|index| {
                json!({
                    "x": 0.2 + f64::from(index) * 0.02,
                    "y": 0.3,
                    "score": 0.9
                })
            })
            .collect(),
    )
}

fn valid_result_json() -> Value {
    json!({
        "personFound": true,
        "bbox": {
            "original": bbox(0.2, 0.2, 0.8, 0.8),
            "expanded": bbox(0.1, 0.1, 0.9, 0.9)
        },
        "keypoints": keypoints(),
        "features": {
            "rtmdet_extracted": vec![1.0_f32; 384],
            "backbone_features": vec![2.0_f32; 768]
        },
        "classification": {
            "presentProbability": 0.9,
            "goodProbability": 0.7
        }
    })
}

#[test]
fn initialized_response_round_trips_raw_json() {
    let raw = json!({ "type": "initialized", "provider": "wasm" });
    let response: InferenceWorkerResponse =
        serde_json::from_value(raw.clone()).expect("initialized response");
    assert_eq!(
        serde_json::to_value(response).expect("serialize initialized"),
        raw
    );
}

#[test]
fn result_response_round_trips_request_id_and_nested_expanded_bbox() {
    let raw = json!({
        "type": "result",
        "requestId": 123,
        "result": valid_result_json()
    });
    let response: InferenceWorkerResponse =
        serde_json::from_value(raw.clone()).expect("valid result response");
    let InferenceWorkerResponse::Result { request_id, result } = &response else {
        panic!("unexpected response: {response:?}")
    };
    assert_eq!(*request_id, 123);
    assert_ne!(
        result.bbox.expect("bbox").original,
        result.bbox.expect("bbox").expanded
    );
    result.validate_native().expect("native result validation");

    let serialized = serde_json::to_value(response).expect("serialize result response");
    assert_eq!(serialized["requestId"], 123);
    assert!(serialized.get("request_id").is_none());
    assert_eq!(serialized["result"]["bbox"]["original"]["x1"], 0.2);
    assert_eq!(serialized["result"]["bbox"]["expanded"]["x1"], 0.1);
}

#[test]
fn error_response_round_trips_camel_case_request_id_only() {
    let raw = json!({
        "type": "error",
        "error": "Test error message",
        "requestId": 789
    });
    let response: InferenceWorkerResponse =
        serde_json::from_value(raw.clone()).expect("error response");
    assert_eq!(
        serde_json::to_value(response).expect("serialize error"),
        raw
    );

    let snake: InferenceWorkerResponse = serde_json::from_value(json!({
        "type": "error",
        "error": "Test error message",
        "request_id": 789
    }))
    .expect("optional request ID may be absent");
    let serialized = serde_json::to_value(snake).expect("serialize snake input");
    assert!(serialized.get("request_id").is_none());
    assert!(serialized.get("requestId").is_none());
}

#[test]
fn raw_result_without_required_request_id_is_rejected() {
    let error = serde_json::from_value::<InferenceWorkerResponse>(json!({
        "type": "result",
        "request_id": 123,
        "result": valid_result_json()
    }))
    .expect_err("snake-case request ID must not satisfy the wire contract");
    assert!(error.to_string().contains("requestId"));
}

fn valid_typed_result() -> InferenceResponseResult {
    let original = BoundingBox {
        x1: 0.2,
        y1: 0.2,
        x2: 0.8,
        y2: 0.8,
        score: 0.9,
        width: 0.6,
        height: 0.6,
    };
    let expanded = BoundingBox {
        x1: 0.1,
        y1: 0.1,
        x2: 0.9,
        y2: 0.9,
        score: 0.9,
        width: 0.8,
        height: 0.8,
    };
    InferenceResponseResult {
        person_found: true,
        bbox: Some(ExpandedBbox { original, expanded }),
        keypoints: Some(
            (0..17)
                .map(|index| Keypoint::new(0.2 + f64::from(index) * 0.02, 0.3, 0.9))
                .collect(),
        ),
        features: [
            (FeatureId::RtmDetExtracted, vec![1.0; 384]),
            (FeatureId::BackboneFeatures, vec![2.0; 768]),
        ]
        .into_iter()
        .collect(),
        classification: None,
    }
}

#[test]
fn native_validation_rejects_wrong_length_and_non_finite_features() {
    let mut wrong_length = valid_typed_result();
    wrong_length
        .features
        .insert(FeatureId::RtmDetExtracted, vec![1.0; 3]);
    assert!(wrong_length
        .validate_native()
        .expect_err("wrong feature dimension")
        .contains("expected 384"));

    let mut non_finite = valid_typed_result();
    non_finite
        .features
        .get_mut(&FeatureId::BackboneFeatures)
        .expect("feature")[10] = f32::NAN;
    assert!(non_finite
        .validate_native()
        .expect_err("non-finite feature")
        .contains("finite"));
}

#[test]
fn native_validation_rejects_inconsistent_no_person_payload() {
    let mut result = valid_typed_result();
    result.person_found = false;
    assert!(result
        .validate_native()
        .expect_err("no-person payload must be null")
        .contains("must not contain"));

    result.bbox = None;
    result.keypoints = None;
    result.features.clear();
    result
        .validate_native()
        .expect("canonical no-person payload");
}
