//! Production-entry replacement for
//! `src/workers/__tests__/inference-worker-initialization.test.ts`
//! (frozen SHA-256 `43a2ae04885b26904a0bd96c922f76ed381a0db5e6c09f2a79bbb30fadf52789`).

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, ProcessPayload,
};
use slouch_domain::FeatureId;
use slouch_vision::ported::inference_worker::{
    retry_delay, InferenceWorker, SessionOutput, WorkerError, WorkerResponse,
};

use super::support::{
    detector_outputs, image, pose_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

/// Synthetic NLF-L outputs for a realistic upright seated pose. Only the coco_19
/// posture joints consumed by `extract_nlf_depth_features` carry meaningful 3D
/// coordinates (meters, box/root-relative, matching the pilot's observed
/// magnitudes: z ~1.1-1.3, ~0.5 m torso); every other canonical joint stays zero.
/// Upper-body uncertainty is low (~0.03) and lower-body higher (~0.10), mirroring
/// the pilot's truncation signal so the extracted feature is non-degenerate.
fn nlf_outputs() -> HashMap<String, SessionOutput> {
    const NUM_CANONICAL: usize = 867;
    let mut coords = vec![0.0_f32; NUM_CANONICAL * 3];
    let mut set = |joint: usize, x: f32, y: f32, z: f32| {
        coords[joint * 3] = x;
        coords[joint * 3 + 1] = y;
        coords[joint * 3 + 2] = z;
    };
    // coco_19 posture indices from models/nlf_joint_map.json.
    set(75, 0.00, 0.60, 1.22); // neck
    set(76, 0.00, 0.68, 1.15); // nose
    set(77, -0.20, 0.50, 1.24); // lsho
    set(83, 0.20, 0.50, 1.24); // rsho
    set(80, -0.15, 0.00, 1.30); // lhip
    set(86, 0.15, 0.00, 1.30); // rhip
    set(90, -0.07, 0.72, 1.20); // lear
    set(92, 0.07, 0.72, 1.20); // rear
    set(89, -0.035, 0.74, 1.17); // leye
    set(91, 0.035, 0.74, 1.17); // reye
    set(93, 0.00, 0.00, 1.30); // pelvis
    set(81, -0.16, -0.45, 1.10); // lkne
    set(82, -0.16, -0.85, 1.05); // lank
    set(87, 0.16, -0.45, 1.10); // rkne
    set(88, 0.16, -0.85, 1.05); // rank

    let mut uncertainty = vec![0.03_f32; NUM_CANONICAL];
    for joint in [81, 82, 87, 88] {
        uncertainty[joint] = 0.10;
    }

    HashMap::from([
        (
            "coords2d".into(),
            SessionOutput::F32(vec![0.0; NUM_CANONICAL * 2]),
        ),
        ("coords3d_rel".into(), SessionOutput::F32(coords)),
        ("uncertainty".into(), SessionOutput::F32(uncertainty)),
    ])
}

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
            nlf_path: None,
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
            nlf_path: None,
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
            nlf_path: None,
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
            nlf_path: None,
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

fn person_detector() -> HashMap<String, SessionOutput> {
    detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25)
}

#[test]
fn nlf_session_failure_degrades_gracefully_and_omits_depth_feature() {
    // The DirectML NLF session fails to create (no GPU / DML runtime). Overall
    // initialization must still succeed, the worker must run frames normally, and
    // the result must simply carry no NlfDepth feature.
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(person_detector())])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
        CreateOutcome::Error("DirectML device not available".into()),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    let init = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            rtmw3d_path: "pose.onnx".into(),
            nlf_path: Some("nlf.onnx".into()),
        },
    });
    assert!(matches!(&init[..], [WorkerResponse::Initialized { .. }]));

    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 1,
        },
    });
    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.person_found);
    assert!(!result.features.contains_key(&FeatureId::NlfDepth));
}

#[test]
fn nlf_present_path_inserts_fourteen_dim_depth_feature() {
    // With a working NLF session, a person-found frame gains a valid 14-dim
    // NlfDepth feature. A successful Result (never an Error) proves the stricter
    // native-result validation accepted the extra feature.
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(person_detector())])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
        CreateOutcome::Session(VecDeque::from([Ok(nlf_outputs())])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            rtmw3d_path: "pose.onnx".into(),
            nlf_path: Some("nlf.onnx".into()),
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
    assert_eq!(result.features[&FeatureId::NlfDepth].len(), 14);
    assert!(result.features[&FeatureId::NlfDepth]
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn nlf_forward_error_is_non_fatal_and_omits_depth_feature() {
    // A working NLF session that errors on run must not fail the frame: the result
    // is still emitted, just without the NlfDepth feature.
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(person_detector())])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
        CreateOutcome::Session(VecDeque::from([Err(WorkerError::Inference(
            "injected NLF execution failure".into(),
        ))])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det.onnx".into(),
            rtmw3d_path: "pose.onnx".into(),
            nlf_path: Some("nlf.onnx".into()),
        },
    });

    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 3,
        },
    });
    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.person_found);
    assert!(!result.features.contains_key(&FeatureId::NlfDepth));
}
