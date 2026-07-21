//! Native initialization tests for the NLF-only inference worker.
//!
//! The pose model (NLF-L on DirectML) is a HARD requirement: a failure to create
//! its session leaves the worker uninitialized and surfaces an actionable error.
//! RTMDet still loads on the CPU with the frozen retry/backoff policy.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, ProcessPayload,
};
use slouch_domain::FeatureId;
use slouch_vision::ported::inference_worker::{
    retry_delay, ExecutionProvider, InferenceWorker, SessionOptions, SessionOutput, WorkerResponse,
};

use super::support::{
    detector_outputs, image, nlf_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

/// The DirectML session options every NLF-L create must use.
fn nlf_options() -> SessionOptions {
    SessionOptions {
        execution_provider: ExecutionProvider::DirectMl,
        ..SessionOptions::default()
    }
}

fn person_detector() -> HashMap<String, SessionOutput> {
    detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25)
}

#[test]
fn actual_worker_retries_rtmdet_then_loads_nlf_on_directml() {
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Error("first".into()),
        CreateOutcome::Error("second".into()),
        CreateOutcome::Session(VecDeque::new()),
        CreateOutcome::Session(VecDeque::new()),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let responses = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            nlf_path: "nlf.onnx".into(),
        },
    });

    assert_eq!(
        responses,
        vec![WorkerResponse::Initialized {
            provider: "native".into()
        }]
    );
    let trace = trace.lock().expect("trace");
    assert_eq!(trace.creates.len(), 4);
    assert_eq!(
        trace.creates,
        vec![
            ("det.onnx".into(), "RTMDet".into(), Default::default()),
            ("det.onnx".into(), "RTMDet".into(), Default::default()),
            ("det.onnx".into(), "RTMDet".into(), Default::default()),
            ("nlf.onnx".into(), "NLF-L".into(), nlf_options()),
        ]
    );
    assert_eq!(
        trace.waits,
        vec![Duration::from_secs(1), Duration::from_secs(2)]
    );
}

#[test]
fn actual_worker_returns_last_error_after_exactly_three_rtmdet_attempts() {
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Error("first".into()),
        CreateOutcome::Error("second".into()),
        CreateOutcome::Error("last".into()),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let responses = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "bad.onnx".into(),
            nlf_path: "nlf.onnx".into(),
        },
    });
    // RTMDet fails all three attempts; NLF is never reached.
    assert_eq!(trace.lock().expect("trace").creates.len(), 3);
    assert_eq!(
        responses,
        vec![WorkerResponse::Error {
            error: "Failed to initialize models after 3 attempts: last. Please check your internet connection and reload the page.".into(),
            request_id: None,
            details: None,
            success: None,
        }]
    );
    assert_eq!(retry_delay(1), Duration::from_secs(1));
    assert_eq!(retry_delay(2), Duration::from_secs(2));
    assert_eq!(retry_delay(10), Duration::from_secs(8));
    assert_eq!(retry_delay(20), Duration::from_secs(8));
}

#[test]
fn nlf_create_failure_returns_actionable_directml_error_and_reinitializes_on_next_frame() {
    // RTMDet loads; the NLF-L DirectML session fails on its SINGLE attempt (no
    // retry — a missing GPU is not transient). Initialization must return the
    // actionable DirectML error and publish no half-initialized state. A later
    // frame re-initializes from the stored paths (proving is_initialized was
    // false), and this time NLF succeeds.
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::new()),
        CreateOutcome::Error("DirectML device not available".into()),
        CreateOutcome::Session(VecDeque::from([Ok(person_detector())])),
        CreateOutcome::Session(VecDeque::from([Ok(nlf_outputs())])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let initialized = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            nlf_path: "nlf.onnx".into(),
        },
    });
    assert_eq!(
        initialized,
        vec![WorkerResponse::Error {
            error: "Posture detection requires a DirectX 12-capable GPU. The NLF pose model failed to initialize on DirectML: DirectML device not available".into(),
            request_id: None,
            details: None,
            success: None,
        }]
    );

    let processed = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 77,
        },
    });
    let [WorkerResponse::Result {
        request_id: 77,
        result,
    }] = &processed[..]
    else {
        panic!("expected a re-initialized result: {processed:?}")
    };
    assert!(result.person_found);
    assert_eq!(result.keypoints.as_ref().expect("keypoints").len(), 17);

    let creates = &trace.lock().expect("trace").creates;
    assert_eq!(
        creates
            .iter()
            .map(|call| (call.0.as_str(), call.1.as_str(), call.2))
            .collect::<Vec<_>>(),
        vec![
            ("det.onnx", "RTMDet", Default::default()),
            ("nlf.onnx", "NLF-L", nlf_options()),
            ("det.onnx", "RTMDet", Default::default()),
            ("nlf.onnx", "NLF-L", nlf_options()),
        ]
    );
}

#[test]
fn skips_reinitialization_when_paths_are_not_stored() {
    let (runtime, trace) = TestRuntime::new([]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let responses = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 77,
        },
    });

    assert_eq!(
        responses,
        vec![WorkerResponse::Error {
            error: "Models failed to initialize. Please reload the page.".into(),
            request_id: Some(77),
            details: None,
            success: None,
        }]
    );
    assert!(trace.lock().expect("trace").creates.is_empty());
}

#[test]
fn invalid_model_path_is_preserved_in_initialization_error() {
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Error("404: Model file not found".into()),
        CreateOutcome::Error("404: Model file not found".into()),
        CreateOutcome::Error("404: Model file not found".into()),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let responses = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "/invalid/path.onnx".into(),
            nlf_path: "nlf.onnx".into(),
        },
    });

    assert_eq!(
        responses,
        vec![WorkerResponse::Error {
            error: "Failed to initialize models after 3 attempts: 404: Model file not found. Please check your internet connection and reload the page.".into(),
            request_id: None,
            details: None,
            success: None,
        }]
    );
    assert_eq!(trace.lock().expect("trace").creates.len(), 3);
}

#[test]
fn worker_response_serializes_types_and_camel_case_request_id() {
    let initialized = serde_json::to_value(WorkerResponse::Initialized {
        provider: "native".into(),
    })
    .expect("initialized response serializes");
    assert_eq!(initialized["type"], "initialized");
    assert_eq!(initialized["provider"], "native");

    let error = serde_json::to_value(WorkerResponse::Error {
        error: "processing failed".into(),
        request_id: Some(77),
        details: None,
        success: None,
    })
    .expect("error response serializes");
    assert_eq!(error["type"], "error");
    assert_eq!(error["requestId"], 77);
    assert!(error.get("request_id").is_none());
}

#[test]
fn nlf_present_path_yields_seventeen_keypoints_and_depth_feature() {
    // With RTMDet and a working NLF session, a person-found frame carries 17 COCO
    // keypoints plus a valid 14-dim NlfDepth feature. A successful Result (never an
    // Error) proves the native-result validation accepted them.
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(person_detector())])),
        CreateOutcome::Session(VecDeque::from([Ok(nlf_outputs())])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            nlf_path: "nlf.onnx".into(),
        },
    });

    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 2,
        },
    });
    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.person_found);
    assert_eq!(result.keypoints.as_ref().expect("keypoints").len(), 17);
    assert!(result
        .keypoints
        .as_ref()
        .expect("keypoints")
        .iter()
        .all(|keypoint| {
            keypoint.x.is_finite()
                && keypoint.y.is_finite()
                && (0.0..=1.0).contains(&keypoint.score)
        }));
    assert_eq!(result.features[&FeatureId::NlfDepth].len(), 14);
    assert!(result.features[&FeatureId::NlfDepth]
        .iter()
        .all(|value| value.is_finite()));
}
