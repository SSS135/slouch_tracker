//! Production-entry replacement for
//! `src/workers/__tests__/inference-worker-initialization.test.ts`
//! (frozen SHA-256 `43a2ae04885b26904a0bd96c922f76ed381a0db5e6c09f2a79bbb30fadf52789`).

use std::collections::VecDeque;
use std::time::Duration;

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, ProcessPayload,
};
use slouch_vision::ported::inference_worker::{retry_delay, InferenceWorker, WorkerResponse};

use super::support::{
    detector_outputs, image, pose_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

#[test]
fn actual_worker_retries_with_frozen_options_and_backoff() {
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
            rtmw3d_path: "pose.onnx".into(),
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
            ("det.onnx".into(), "RTMDet".into(), Default::default(),),
            ("det.onnx".into(), "RTMDet".into(), Default::default(),),
            ("det.onnx".into(), "RTMDet".into(), Default::default(),),
            ("pose.onnx".into(), "RTMPose-M".into(), Default::default(),),
        ]
    );
    assert_eq!(
        trace.waits,
        vec![Duration::from_secs(1), Duration::from_secs(2)]
    );
}

#[test]
fn actual_worker_returns_last_error_after_exactly_three_attempts() {
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
            rtmw3d_path: "unused.onnx".into(),
        },
    });
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
fn failed_second_session_publishes_no_half_initialized_state_and_process_retries_paths() {
    let detector = detector_outputs(vec![], vec![], 0.1);
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::new()),
        CreateOutcome::Error("pose one".into()),
        CreateOutcome::Error("pose two".into()),
        CreateOutcome::Error("pose last".into()),
        CreateOutcome::Session(VecDeque::from([Ok(detector)])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let initialized = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            rtmw3d_path: "pose.onnx".into(),
        },
    });
    assert_eq!(
        initialized,
        vec![WorkerResponse::Error {
            error: "Failed to initialize models after 3 attempts: pose last. Please check your internet connection and reload the page.".into(),
            request_id: None,
            details: None,
            success: None,
        }]
    );

    let processed = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 77,
        },
    });
    assert!(
        matches!(&processed[..], [WorkerResponse::Result { request_id: 77, result }] if !result.person_found)
    );
    let creates = &trace.lock().expect("trace").creates;
    assert_eq!(
        creates
            .iter()
            .map(|call| (call.0.as_str(), call.1.as_str(), call.2))
            .collect::<Vec<_>>(),
        vec![
            ("det.onnx", "RTMDet", Default::default()),
            ("pose.onnx", "RTMPose-M", Default::default()),
            ("pose.onnx", "RTMPose-M", Default::default()),
            ("pose.onnx", "RTMPose-M", Default::default()),
            ("det.onnx", "RTMDet", Default::default()),
            ("pose.onnx", "RTMPose-M", Default::default()),
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
            rtmw3d_path: "unused.onnx".into(),
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
