//! Production-entry replacement for
//! `src/workers/__tests__/inference-worker-pipeline.test.ts`
//! (frozen SHA-256 `cab8a43e348ff2a7c6c516403783d1e01a45a80213b4daa1f6b8d09fa680d4b3`).

use std::collections::{HashMap, VecDeque};

use slouch_domain::ported::messages::schemas::{
    ImageData, InferenceWorkerMessage, InitializePayload, ProcessPayload,
};
use slouch_domain::FeatureId;
use slouch_vision::ported::inference_worker::{
    InferenceWorker, SessionOutput, WorkerError, WorkerResponse,
};

use super::support::{
    detector_outputs, image, pose_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

fn initialized_worker(
    detector_runs: VecDeque<Result<HashMap<String, SessionOutput>, WorkerError>>,
    pose_runs: VecDeque<Result<HashMap<String, SessionOutput>, WorkerError>>,
) -> (
    InferenceWorker<TestFactory, TestLogger, TestRuntime>,
    std::sync::Arc<std::sync::Mutex<super::support::RuntimeTrace>>,
) {
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Session(detector_runs),
        CreateOutcome::Session(pose_runs),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    assert!(matches!(
        &worker.handle_message(InferenceWorkerMessage::Initialize {
            payload: InitializePayload {
                rtmdet_path: "det".into(),
                rtmw3d_path: "pose".into(),
                nlf_path: None,
            },
        })[..],
        [WorkerResponse::Initialized { .. }]
    ));
    (worker, trace)
}

#[test]
fn actual_pipeline_returns_owned_seven_feature_buffers_bbox_and_keypoints() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25);
    let (mut worker, trace) = initialized_worker(
        VecDeque::from([Ok(detector)]),
        VecDeque::from([Ok(pose_outputs())]),
    );
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 42,
        },
    });
    let [WorkerResponse::Result { request_id, result }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert_eq!(*request_id, 42);
    assert!(result.person_found);
    assert_eq!(result.keypoints.as_ref().expect("keypoints").len(), 17);
    assert_eq!(result.features.len(), 7);
    assert_eq!(result.features[&FeatureId::BackboneFeatures].len(), 768);
    assert_eq!(result.features[&FeatureId::BackboneFeaturesMax].len(), 768);
    assert_eq!(result.features[&FeatureId::BackboneFeaturesStd].len(), 768);
    assert_eq!(result.features[&FeatureId::GauFeatures].len(), 256);
    assert_eq!(result.features[&FeatureId::GauFeaturesMax].len(), 256);
    assert_eq!(result.features[&FeatureId::GauFeaturesStd].len(), 256);
    assert_eq!(result.features[&FeatureId::RtmDetExtracted].len(), 384);
    assert_eq!(result.features[&FeatureId::BackboneFeatures][0], 0.25);
    assert_eq!(result.features[&FeatureId::BackboneFeaturesStd][0], 0.001);
    assert_eq!(
        result.bbox.as_ref().expect("bbox").original.score,
        0.8999999761581421
    );
    assert_eq!(trace.lock().expect("trace").runs, 2);
}

#[test]
fn no_person_and_threshold_adjacent_f32_skip_pose_but_preserve_detector_features() {
    // The ONNX tensor is f32; use the greatest representable value below the f64 0.3 policy.
    for score in [None, Some(f32::from_bits(0.3_f32.to_bits() - 1))] {
        let (dets, labels) = score.map_or((vec![], vec![]), |value| {
            (vec![20.0, 20.0, 200.0, 200.0, value], vec![0])
        });
        let (mut worker, trace) = initialized_worker(
            VecDeque::from([Ok(detector_outputs(dets, labels, 0.0))]),
            VecDeque::new(),
        );
        let response = worker.handle_message(InferenceWorkerMessage::Process {
            payload: ProcessPayload {
                image_data: image(4, 4),
                request_id: 9,
            },
        });
        assert!(
            matches!(&response[..], [WorkerResponse::Result { result, .. }] if !result.person_found && result.features.len() == 1 && result.features[&FeatureId::RtmDetExtracted].len() == 384)
        );
        assert_eq!(trace.lock().expect("trace").runs, 1);
    }
}

#[test]
fn malformed_images_fail_before_any_session_run_and_preserve_request_id() {
    let malformed = [
        ImageData {
            data: vec![],
            width: 0,
            height: 1,
        },
        ImageData {
            data: vec![],
            width: 1,
            height: 0,
        },
        ImageData {
            data: vec![0; 3],
            width: 1,
            height: 1,
        },
        ImageData {
            data: vec![],
            width: u32::MAX,
            height: u32::MAX,
        },
    ];
    for (index, input) in malformed.into_iter().enumerate() {
        let (mut worker, trace) = initialized_worker(VecDeque::new(), VecDeque::new());
        let request_id = 100 + index as u64;
        let response = worker.handle_message(InferenceWorkerMessage::Process {
            payload: ProcessPayload {
                image_data: input,
                request_id,
            },
        });
        assert!(
            matches!(&response[..], [WorkerResponse::Error { request_id: Some(id), error, .. }] if *id == request_id && (error.contains("dimensions") || error.contains("bytes") || error.contains("overflow")))
        );
        assert_eq!(trace.lock().expect("trace").runs, 0);
    }
}

#[test]
fn session_failures_and_missing_or_wrong_outputs_emit_only_typed_errors() {
    let cases = [
        Err(WorkerError::Inference("injected detector failure".into())),
        Ok(HashMap::new()),
        Ok(HashMap::from([("dets".into(), SessionOutput::I64(vec![]))])),
    ];
    for detector in cases {
        let (mut worker, _) = initialized_worker(VecDeque::from([detector]), VecDeque::new());
        let response = worker.handle_message(InferenceWorkerMessage::Process {
            payload: ProcessPayload {
                image_data: image(4, 4),
                request_id: 55,
            },
        });
        assert!(matches!(
            &response[..],
            [WorkerResponse::Error {
                request_id: Some(55),
                ..
            }]
        ));
    }
}

#[test]
fn pose_session_failure_emits_error_without_result() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.1);
    let (mut worker, trace) = initialized_worker(
        VecDeque::from([Ok(detector)]),
        VecDeque::from([Err(WorkerError::Inference(
            "injected pose execution failure".into(),
        ))]),
    );
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 77,
        },
    });
    assert!(
        matches!(&response[..], [WorkerResponse::Error { request_id: Some(77), error, .. }] if error.contains("injected pose execution failure"))
    );
    assert_eq!(trace.lock().expect("trace").runs, 2);
}

#[test]
fn malformed_pose_tensor_and_non_finite_output_do_not_publish_a_result() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.31], vec![0], 0.1);
    let mut pose = pose_outputs();
    pose.insert("simcc_x".into(), SessionOutput::F32(vec![f32::NAN]));
    let (mut worker, _) =
        initialized_worker(VecDeque::from([Ok(detector)]), VecDeque::from([Ok(pose)]));
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 88,
        },
    });
    assert!(
        matches!(&response[..], [WorkerResponse::Error { request_id: Some(88), error, .. }] if error.contains("non-finite"))
    );
}
