//! Production-entry replacement for
//! `src/workers/__tests__/inference-worker-classifier.test.ts`
//! (frozen SHA-256 `739b1dc17897288f345a69afb3cfb74460b2242fee6d34080e7dccf553bdd9fd`).

use std::collections::VecDeque;

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, PostureModelPayload, PresenceModelPayload,
    ProcessPayload,
};
use slouch_vision::ported::inference_worker::{InferenceWorker, WorkerResponse};

use super::support::{
    detector_outputs, image, model, pose_outputs, CreateOutcome, TestFactory, TestLogger,
    TestRuntime,
};

fn worker_with_frames(
    factory: TestFactory,
    frames: usize,
) -> InferenceWorker<TestFactory, TestLogger, TestRuntime> {
    let detector_runs = (0..frames)
        .map(|_| {
            Ok(detector_outputs(
                vec![40.0, 40.0, 280.0, 280.0, 0.9],
                vec![0],
                0.1,
            ))
        })
        .collect::<VecDeque<_>>();
    let pose_runs = (0..frames)
        .map(|_| Ok(pose_outputs()))
        .collect::<VecDeque<_>>();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(detector_runs),
        CreateOutcome::Session(pose_runs),
    ]);
    let mut worker = InferenceWorker::with_runtime(factory, TestLogger::default(), runtime);
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
    worker
}

fn load_posture(
    worker: &mut InferenceWorker<TestFactory, TestLogger, TestRuntime>,
    probability: f64,
) -> Vec<WorkerResponse> {
    worker.handle_message(InferenceWorkerMessage::LoadPostureModel {
        payload: PostureModelPayload {
            posture_model: model(probability),
        },
    })
}

fn load_presence(
    worker: &mut InferenceWorker<TestFactory, TestLogger, TestRuntime>,
    probability: f64,
) -> Vec<WorkerResponse> {
    worker.handle_message(InferenceWorkerMessage::LoadPresenceModel {
        payload: PresenceModelPayload {
            presence_model: model(probability),
        },
    })
}

fn process(
    worker: &mut InferenceWorker<TestFactory, TestLogger, TestRuntime>,
    request_id: u64,
) -> slouch_domain::ClassificationResult {
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id,
        },
    });
    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    result.classification.expect("classification")
}

#[test]
fn actual_worker_disposes_roles_before_factory_load_and_clears_on_failure() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    let trace = factory.trace.clone();
    let mut worker = worker_with_frames(factory.clone(), 2);

    assert_eq!(
        load_posture(&mut worker, 0.7),
        vec![WorkerResponse::ClassifierLoaded { success: true }]
    );
    assert_eq!(
        load_presence(&mut worker, 0.8),
        vec![WorkerResponse::ClassifierLoaded { success: true }]
    );
    assert_eq!(
        load_posture(&mut worker, 0.6),
        vec![WorkerResponse::ClassifierLoaded { success: true }]
    );
    assert_eq!(stats.lock().expect("stats")[&70].disposes, 1);
    assert_eq!(stats.lock().expect("stats")[&80].disposes, 0);
    assert_eq!(process(&mut worker, 1).good_probability, Some(0.6));

    factory.load_failures.lock().expect("failures").insert(50);
    let failed = load_posture(&mut worker, 0.5);
    assert!(matches!(
        &failed[..],
        [WorkerResponse::Error {
            error,
            success: Some(false),
            ..
        }] if error.contains("injected load failure")
    ));
    factory.load_failures.lock().expect("failures").insert(40);
    let failed = load_presence(&mut worker, 0.4);
    assert!(matches!(
        &failed[..],
        [WorkerResponse::Error {
            error,
            success: Some(false),
            ..
        }] if error.contains("injected load failure")
    ));

    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 2,
        },
    });
    assert!(matches!(
        &response[..],
        [WorkerResponse::Result { result, .. }]
            if result.person_found && result.classification.is_none()
    ));
    assert_eq!(stats.lock().expect("stats")[&60].disposes, 1);
    assert_eq!(stats.lock().expect("stats")[&80].disposes, 1);
    assert_eq!(stats.lock().expect("stats")[&60].predicts, 1);
    assert_eq!(stats.lock().expect("stats")[&80].predicts, 1);

    let trace = trace.lock().expect("trace");
    assert!(trace
        .windows(2)
        .any(|events| events == ["dispose 70", "load 60"]));
    assert!(trace
        .windows(2)
        .any(|events| events == ["dispose 60", "load 50"]));
    assert!(trace
        .windows(2)
        .any(|events| events == ["dispose 80", "load 40"]));
}

#[test]
fn presence_threshold_cascades_posture_and_away_skips_it() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    let mut worker = worker_with_frames(factory, 2);
    load_posture(&mut worker, 0.7);
    load_presence(&mut worker, 0.5);
    let at_threshold = process(&mut worker, 1);
    assert_eq!(at_threshold.present_probability, 0.5);
    assert_eq!(at_threshold.good_probability, Some(0.7));
    assert_eq!(stats.lock().expect("stats")[&70].predicts, 1);

    load_presence(&mut worker, 0.49);
    // Oracle parity: loading a new presence model disposes the previously loaded
    // presence model (inference-worker-classifier.test.ts:130-144). Guard the
    // presence-role reload disposal symmetrically with the posture-role case.
    assert_eq!(stats.lock().expect("stats")[&50].disposes, 1);
    let away = process(&mut worker, 2);
    assert_eq!(away.present_probability, 0.49);
    assert_eq!(away.good_probability, None);
    assert_eq!(stats.lock().expect("stats")[&70].predicts, 1);
}

#[test]
fn no_person_reports_away_without_running_presence_and_posture_only_emits_no_classification() {
    for presence in [true, false] {
        let factory = TestFactory::default();
        let stats = factory.stats.clone();
        let (runtime, trace) = TestRuntime::new([
            CreateOutcome::Session(VecDeque::from([Ok(detector_outputs(vec![], vec![], 0.2))])),
            CreateOutcome::Session(VecDeque::new()),
        ]);
        let mut worker = InferenceWorker::with_runtime(factory, TestLogger::default(), runtime);
        worker.handle_message(InferenceWorkerMessage::Initialize {
            payload: InitializePayload {
                rtmdet_path: "det".into(),
                rtmw3d_path: "pose".into(),
                nlf_path: None,
            },
        });
        if presence {
            load_presence(&mut worker, 0.2);
        } else {
            load_posture(&mut worker, 0.7);
        }

        let response = worker.handle_message(InferenceWorkerMessage::Process {
            payload: ProcessPayload {
                image_data: image(4, 4),
                request_id: 10,
            },
        });
        let [WorkerResponse::Result { result, .. }] = &response[..] else {
            panic!("expected result: {response:?}")
        };
        assert!(!result.person_found);
        if presence {
            // No person -> "away" is reported directly; the presence model is
            // NOT invoked (its keypoint-derived features are absent on an empty
            // frame), so present_probability is 0.0 and its predict count stays 0.
            let classification = result.classification.as_ref().expect("away result");
            assert_eq!(classification.present_probability, 0.0);
            assert_eq!(classification.good_probability, None);
            assert_eq!(stats.lock().expect("stats")[&20].predicts, 0);
        } else {
            assert!(result.classification.is_none());
            assert_eq!(stats.lock().expect("stats")[&70].predicts, 0);
        }
        assert_eq!(trace.lock().expect("trace").runs, 1);
    }
}

#[test]
fn unload_disposes_both_roles_once_and_repeated_unload_is_idempotent() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    let mut worker = worker_with_frames(factory, 0);
    load_posture(&mut worker, 0.7);
    load_presence(&mut worker, 0.8);
    assert_eq!(
        worker.handle_message(InferenceWorkerMessage::UnloadClassifier),
        vec![WorkerResponse::ClassifierUnloaded]
    );
    assert_eq!(
        worker.handle_message(InferenceWorkerMessage::UnloadClassifier),
        vec![WorkerResponse::ClassifierUnloaded]
    );
    assert_eq!(stats.lock().expect("stats")[&70].disposes, 1);
    assert_eq!(stats.lock().expect("stats")[&80].disposes, 1);
}

#[test]
fn prediction_error_degrades_to_result_without_classification_and_logs() {
    // Oracle parity: on the person-found path classifyFeatures wraps prediction
    // in try/catch and returns null on any error
    // (src/workers/inference-worker.ts:780-783); processFrame then omits
    // classification and still emits a successful result (ts:1024-1026), while
    // logging the failure. It must NOT abort the frame with an Error response.
    let factory = TestFactory::default();
    factory
        .predict_failures
        .lock()
        .expect("failures")
        .insert(80);
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(detector_outputs(
            vec![40.0, 40.0, 280.0, 280.0, 0.9],
            vec![0],
            0.1,
        ))])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
    ]);
    let mut worker = InferenceWorker::with_runtime(factory, logger, runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det".into(),
            rtmw3d_path: "pose".into(),
            nlf_path: None,
        },
    });
    load_presence(&mut worker, 0.8);
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 3,
        },
    });
    assert!(matches!(
        &response[..],
        [WorkerResponse::Result { result, request_id: 3 }]
            if result.person_found && result.classification.is_none()
    ));
    assert!(logs
        .lock()
        .expect("logs")
        .iter()
        .any(|(level, message)| level == "error" && message.contains("prediction 80 failed")));

    let factory = TestFactory::default();
    factory
        .predict_failures
        .lock()
        .expect("failures")
        .insert(70);
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(detector_outputs(
            vec![40.0, 40.0, 280.0, 280.0, 0.9],
            vec![0],
            0.1,
        ))])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
    ]);
    let mut worker = InferenceWorker::with_runtime(factory, logger, runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det".into(),
            rtmw3d_path: "pose".into(),
            nlf_path: None,
        },
    });
    load_posture(&mut worker, 0.7);
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 4,
        },
    });
    assert!(matches!(
        &response[..],
        [WorkerResponse::Result { result, request_id: 4 }]
            if result.person_found && result.classification.is_none()
    ));
    assert!(logs
        .lock()
        .expect("logs")
        .iter()
        .any(|(level, message)| level == "error" && message.contains("prediction 70 failed")));
}

#[test]
fn no_person_reports_away_even_when_presence_model_would_error() {
    // Regression: the no-person branch previously ran the presence model bare, so
    // a model whose feature set needs keypoint-derived features (absent on an
    // empty frame) errored with "keypoint_scores not available in this container"
    // and aborted the whole frame. The no-person path now reports "away" without
    // running the presence model, so even a predict-failing presence model yields
    // a successful away result and the failing model is never invoked. A
    // predict-injected failure stands in for the missing-keypoint-feature error.
    let factory = TestFactory::default();
    factory
        .predict_failures
        .lock()
        .expect("failures")
        .insert(20);
    let stats = factory.stats.clone();
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(detector_outputs(vec![], vec![], 0.2))])),
        CreateOutcome::Session(VecDeque::new()),
    ]);
    let mut worker = InferenceWorker::with_runtime(factory, logger, runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det".into(),
            rtmw3d_path: "pose".into(),
            nlf_path: None,
        },
    });
    load_presence(&mut worker, 0.2);
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 7,
        },
    });
    let [WorkerResponse::Result {
        result,
        request_id: 7,
    }] = &response[..]
    else {
        panic!("expected an away result, got: {response:?}")
    };
    assert!(!result.person_found);
    let classification = result.classification.as_ref().expect("away result");
    assert_eq!(classification.present_probability, 0.0);
    assert_eq!(classification.good_probability, None);
    // The failing presence model must never be invoked on a no-person frame.
    assert_eq!(stats.lock().expect("stats")[&20].predicts, 0);
    assert!(!logs
        .lock()
        .expect("logs")
        .iter()
        .any(|(level, message)| level == "error" && message.contains("prediction 20 failed")));
}

#[test]
fn person_present_posture_only_uses_detector_confidence_and_runs_posture() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    let mut worker = worker_with_frames(factory, 1);
    load_posture(&mut worker, 0.8);

    let classification = process(&mut worker, 5);
    assert_eq!(classification.present_probability, f64::from(0.9_f32));
    assert_eq!(classification.good_probability, Some(0.8));
    assert_eq!(stats.lock().expect("stats")[&80].predicts, 1);
}

#[test]
fn dropping_worker_disposes_loaded_roles_once() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    {
        let mut worker = worker_with_frames(factory, 0);
        load_posture(&mut worker, 0.7);
        load_presence(&mut worker, 0.8);
    }

    assert_eq!(stats.lock().expect("stats")[&70].disposes, 1);
    assert_eq!(stats.lock().expect("stats")[&80].disposes, 1);
}

#[test]
fn pair_publication_failure_preserves_old_generation_and_disposes_only_candidate() {
    let factory = TestFactory::default();
    let stats = factory.stats.clone();
    let mut worker = worker_with_frames(factory.clone(), 1);
    worker
        .publish_model_pair(Some(model(0.7)), Some(model(0.8)))
        .expect("old pair");
    factory.load_failures.lock().expect("failures").insert(40);
    assert!(worker
        .publish_model_pair(Some(model(0.3)), Some(model(0.4)))
        .is_err());
    let classification = process(&mut worker, 4);
    assert_eq!(classification.present_probability, 0.8);
    assert_eq!(classification.good_probability, Some(0.7));
    let stats = stats.lock().expect("stats");
    assert_eq!(stats[&30].disposes, 1);
    assert_eq!(stats[&70].disposes, 0);
    assert_eq!(stats[&80].disposes, 0);
}
