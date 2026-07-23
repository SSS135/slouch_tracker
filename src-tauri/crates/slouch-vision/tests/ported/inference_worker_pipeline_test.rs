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
    detector_outputs, image, nlf_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

fn initialized_worker(
    detector_runs: VecDeque<Result<HashMap<String, SessionOutput>, WorkerError>>,
    nlf_runs: VecDeque<Result<HashMap<String, SessionOutput>, WorkerError>>,
) -> (
    InferenceWorker<TestFactory, TestLogger, TestRuntime>,
    std::sync::Arc<std::sync::Mutex<super::support::RuntimeTrace>>,
) {
    let (runtime, trace) = TestRuntime::new([
        CreateOutcome::Session(detector_runs),
        CreateOutcome::Session(nlf_runs),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    assert!(matches!(
        &worker.handle_message(InferenceWorkerMessage::Initialize {
            payload: InitializePayload {
                rtmdet_path: "det".into(),
                nlf_path: "nlf".into(),
            },
        })[..],
        [WorkerResponse::Initialized { .. }]
    ));
    (worker, trace)
}

#[test]
fn actual_pipeline_returns_owned_detector_and_depth_features_bbox_and_keypoints() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25);
    let (mut worker, trace) = initialized_worker(
        VecDeque::from([Ok(detector)]),
        VecDeque::from([Ok(nlf_outputs())]),
    );
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 42,
            raw_image_data: None,
            crop_motion: None,
        },
    });
    let [WorkerResponse::Result { request_id, result }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert_eq!(*request_id, 42);
    assert!(result.person_found);
    // NLF supplies the 17 COCO keypoints from a single forward.
    assert_eq!(result.keypoints.as_ref().expect("keypoints").len(), 17);
    // The RTMDet presence feature, the NLF depth feature, the raw 3D keypoint
    // substrate, and the three pooled NLF backbone features (avg/max/std) are
    // produced on the person-found path.
    assert_eq!(result.features.len(), 6);
    assert_eq!(result.features[&FeatureId::RtmDetExtracted].len(), 384);
    assert_eq!(result.features[&FeatureId::NlfDepth].len(), 14);
    assert!(result.features[&FeatureId::NlfDepth]
        .iter()
        .all(|value| value.is_finite()));
    assert_eq!(result.features[&FeatureId::RawKeypoints3d].len(), 51);
    assert!(result.features[&FeatureId::RawKeypoints3d]
        .iter()
        .all(|value| value.is_finite()));
    for id in [
        FeatureId::NlfBackbone,
        FeatureId::NlfBackboneMax,
        FeatureId::NlfBackboneStd,
    ] {
        assert_eq!(result.features[&id].len(), 512);
        assert!(result.features[&id].iter().all(|value| value.is_finite()));
    }
    assert_eq!(
        result.bbox.as_ref().expect("bbox").original.score,
        0.8999999761581421
    );
    // One RTMDet run plus one NLF run per person-found frame.
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
                raw_image_data: None,
                crop_motion: None,
            },
        });
        assert!(
            matches!(&response[..], [WorkerResponse::Result { result, .. }] if !result.person_found && result.features.len() == 1 && result.features[&FeatureId::RtmDetExtracted].len() == 384)
        );
        // No person -> NLF is never run.
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
                raw_image_data: None,
                crop_motion: None,
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
                raw_image_data: None,
                crop_motion: None,
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
fn nlf_session_failure_emits_error_without_result() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.1);
    let (mut worker, trace) = initialized_worker(
        VecDeque::from([Ok(detector)]),
        VecDeque::from([Err(WorkerError::Inference(
            "injected nlf execution failure".into(),
        ))]),
    );
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 77,
            raw_image_data: None,
            crop_motion: None,
        },
    });
    assert!(
        matches!(&response[..], [WorkerResponse::Error { request_id: Some(77), error, .. }] if error.contains("injected nlf execution failure"))
    );
    // The NLF forward is on the critical path (it yields the keypoints), so both
    // the detector and NLF sessions ran before the frame failed.
    assert_eq!(trace.lock().expect("trace").runs, 2);
}

#[test]
fn malformed_nlf_tensor_non_finite_output_does_not_publish_a_result() {
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.31], vec![0], 0.1);
    let mut nlf = nlf_outputs();
    nlf.insert("coords2d".into(), SessionOutput::F32(vec![f32::NAN]));
    let (mut worker, _) =
        initialized_worker(VecDeque::from([Ok(detector)]), VecDeque::from([Ok(nlf)]));
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 88,
            raw_image_data: None,
            crop_motion: None,
        },
    });
    assert!(
        matches!(&response[..], [WorkerResponse::Error { request_id: Some(88), error, .. }] if error.contains("non-finite"))
    );
}

#[test]
fn nlf_backbone_pooling_yields_known_mean_max_and_population_std() {
    // Every one of the 512 backbone channels holds the same 144-element (12x12)
    // spatial ramp 0,1,...,143. Spatial pooling over axes [2,3] reduces each
    // channel to one scalar, so all 512 outputs of each pooling share a single
    // known mean / max / population-std (sqrt(pop_variance + 1e-6)).
    const CHANNELS: usize = 512;
    const SPATIAL: usize = 12 * 12;
    let mut backbone = Vec::with_capacity(CHANNELS * SPATIAL);
    for _channel in 0..CHANNELS {
        for spatial in 0..SPATIAL {
            backbone.push(spatial as f32);
        }
    }
    let mut nlf = nlf_outputs();
    nlf.insert("backbone_feats".into(), SessionOutput::F32(backbone));

    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25);
    let (mut worker, _) =
        initialized_worker(VecDeque::from([Ok(detector)]), VecDeque::from([Ok(nlf)]));
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 1,
            raw_image_data: None,
            crop_motion: None,
        },
    });
    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };

    let average = (0..SPATIAL).sum::<usize>() as f64 / SPATIAL as f64;
    let mean_expected = average as f32; // 71.5
    let max_expected = (SPATIAL - 1) as f32; // 143.0
    let variance = (0..SPATIAL)
        .map(|value| {
            let difference = value as f64 - average;
            difference * difference
        })
        .sum::<f64>()
        / SPATIAL as f64;
    let std_expected = (variance + 1e-6_f64).sqrt() as f32;

    let close = |actual: f32, expected: f32| {
        assert!(
            (actual - expected).abs() <= 1e-3 + 1e-4 * expected.abs(),
            "actual {actual}, expected {expected}"
        );
    };

    let mean = &result.features[&FeatureId::NlfBackbone];
    let max = &result.features[&FeatureId::NlfBackboneMax];
    let std = &result.features[&FeatureId::NlfBackboneStd];
    assert_eq!(mean.len(), 512);
    assert_eq!(max.len(), 512);
    assert_eq!(std.len(), 512);
    for &value in mean {
        close(value, mean_expected);
    }
    for &value in max {
        assert_eq!(value.to_bits(), max_expected.to_bits());
    }
    for &value in std {
        close(value, std_expected);
    }
}

#[test]
fn wrong_length_backbone_feats_fails_the_frame() {
    // A finite but wrong-length backbone tensor passes the non-finite output check
    // yet fails spatial pooling on its shape. Deliberately unlike NlfDepth's
    // None-on-degeneracy, that pooling error fails the whole frame.
    let mut nlf = nlf_outputs();
    nlf.insert(
        "backbone_feats".into(),
        SessionOutput::F32(vec![0.5_f32; 100]),
    );
    let detector = detector_outputs(vec![40.0, 40.0, 280.0, 280.0, 0.9], vec![0], 0.25);
    let (mut worker, _) =
        initialized_worker(VecDeque::from([Ok(detector)]), VecDeque::from([Ok(nlf)]));
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(640, 480),
            request_id: 99,
            raw_image_data: None,
            crop_motion: None,
        },
    });
    assert!(
        matches!(&response[..], [WorkerResponse::Error { request_id: Some(99), error, .. }] if error.contains("73728"))
    );
}
