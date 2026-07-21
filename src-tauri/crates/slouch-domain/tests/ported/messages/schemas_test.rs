use serde_json::{json, Value};
use slouch_domain::ported::messages::schemas::{
    ImageData, InferenceWorkerMessage, TrainingWorkerMessage,
};

#[test]
fn initialize_message_round_trips_raw_camel_case_json() {
    let raw = json!({
        "type": "initialize",
        "payload": {
            "rtmdetPath": "/path/to/rtmdet.onnx",
            "nlfPath": "/path/to/nlf.onnx"
        }
    });
    let message: InferenceWorkerMessage =
        serde_json::from_value(raw.clone()).expect("initialize message must deserialize");
    let InferenceWorkerMessage::Initialize { payload } = &message else {
        panic!("unexpected message: {message:?}")
    };
    assert_eq!(payload.rtmdet_path, "/path/to/rtmdet.onnx");
    assert_eq!(payload.nlf_path, "/path/to/nlf.onnx");
    assert_eq!(
        serde_json::to_value(message).expect("serialize initialize"),
        raw
    );
}

#[test]
fn initialize_message_rejects_empty_or_missing_paths() {
    for raw in [
        json!({
            "type": "initialize",
            "payload": { "rtmdetPath": "", "nlfPath": "/path/to/nlf.onnx" }
        }),
        json!({
            "type": "initialize",
            "payload": { "rtmdetPath": "/path/to/rtmdet.onnx", "nlfPath": "" }
        }),
        // nlfPath is now required; omitting it must be rejected.
        json!({
            "type": "initialize",
            "payload": { "rtmdetPath": "/path/to/rtmdet.onnx" }
        }),
    ] {
        let result: Result<InferenceWorkerMessage, _> = serde_json::from_value(raw);
        assert!(
            result.is_err(),
            "empty or missing path must be rejected (mirrors required z.string().min(1))"
        );
    }
}

#[test]
fn process_pixels_require_checked_binary_transport() {
    let image = ImageData::try_new(vec![0; 640 * 480 * 4], 640, 480).expect("valid 640x480 image");
    assert_eq!(image.data.len(), 640 * 480 * 4);

    for (data, width, height) in [
        (vec![], 0, 1),
        (vec![], 1, 0),
        (vec![0; 3], 1, 1),
        (vec![], u32::MAX, u32::MAX),
    ] {
        assert!(ImageData::try_new(data, width, height).is_err());
    }
    assert!(ImageData::try_new(vec![0; 1920 * 1080 * 4], 1920, 1080).is_ok());
    assert!(ImageData::try_new(vec![0; 1921 * 1080 * 4], 1921, 1080).is_err());

    let raw = json!({
        "type": "process",
        "payload": {
            "imageData": { "data": [0, 0, 0, 255], "width": 1, "height": 1 },
            "requestId": 42
        }
    });
    let error = serde_json::from_value::<InferenceWorkerMessage>(raw)
        .expect_err("JSON pixel arrays must be rejected");
    assert!(error.to_string().contains("binary transport"));
}

fn posture_model_json() -> Value {
    json!({
        "featureExtractor": {
            "featureTypes": ["backbone_features"],
            "normalizationMode": "z_score",
            "dimReductionConfig": { "method": "none", "components": 2 },
            "concatenatedDimensions": 2,
            "normalizationMean": [0.5, 0.5],
            "normalizationStd": [0.2, 0.2],
            "dimReductionTransformer": null
        },
        "classifier": {
            "classifierId": "logistic",
            "state": {
                "weights": [0.1, 0.2],
                "bias": 0.5,
                "classWeights": [1.0, 1.0]
            }
        },
        "trainedAt": 1700000000000.0,
        "version": 1.0
    })
}

fn presence_model_json() -> Value {
    json!({
        "featureExtractor": {
            "featureTypes": ["rtmdet_extracted"],
            "normalizationMode": "z_score",
            "dimReductionConfig": { "method": "none", "components": 1 },
            "concatenatedDimensions": 1,
            "normalizationMean": [0.5],
            "normalizationStd": [0.2],
            "dimReductionTransformer": null
        },
        "classifier": {
            "classifierId": "knn",
            "state": {
                "trainingData": [[0.1], [0.3]],
                "trainingLabels": [0.0, 1.0],
                "k": 1
            }
        },
        "trainedAt": 1700000000000.0,
        "version": 1.0
    })
}

#[test]
fn posture_model_message_preserves_exact_source_normalization_shape() {
    let raw = json!({
        "type": "loadPostureModel",
        "payload": { "postureModel": posture_model_json() }
    });
    let message: InferenceWorkerMessage =
        serde_json::from_value(raw).expect("posture model message must deserialize");
    let serialized = serde_json::to_value(message).expect("serialize posture model message");
    assert_eq!(
        serialized["payload"]["postureModel"]["featureExtractor"]["normalizationMean"],
        json!([0.5, 0.5])
    );
    assert_eq!(
        serialized["payload"]["postureModel"]["featureExtractor"]["normalizationStd"],
        json!([0.2, 0.2])
    );
}

#[test]
fn presence_model_message_preserves_one_element_normalization_shape() {
    let raw = json!({
        "type": "loadPresenceModel",
        "payload": { "presenceModel": presence_model_json() }
    });
    let message: InferenceWorkerMessage =
        serde_json::from_value(raw).expect("presence model message must deserialize");
    let serialized = serde_json::to_value(message).expect("serialize presence model message");
    assert_eq!(
        serialized["payload"]["presenceModel"]["featureExtractor"]["normalizationMean"],
        json!([0.5])
    );
    assert_eq!(
        serialized["payload"]["presenceModel"]["featureExtractor"]["normalizationStd"],
        json!([0.2])
    );
}

#[test]
fn training_message_round_trips_do_cv_wire_name() {
    let raw = json!({ "type": "train", "payload": { "doCV": true } });
    let message: TrainingWorkerMessage =
        serde_json::from_value(raw.clone()).expect("training message must deserialize");
    assert_eq!(serde_json::to_value(message).expect("serialize train"), raw);
}
