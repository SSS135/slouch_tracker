//! Actor lifecycle contract tests.
//!
//! The app lib keeps its modules private, so these tests compile the exact
//! production sources via `#[path]` inclusion. Integration-test crates build
//! with `cfg(test)` active, which also makes the `#[cfg(test)]` seams inside
//! `actors.rs` (crash injection, event-stream validator) available here.

// This test binary exercises only the inference/training actors, so the camera
// surface (CameraActor/CameraCommand::Stop/stop_capture/CameraDeviceInfo — all
// used by api.rs in the real lib) is unused in this standalone compilation.
#[allow(dead_code, unused_imports)]
#[path = "../src/actors.rs"]
mod actors;
#[path = "../src/errors.rs"]
mod errors;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tauri::ipc::{Channel, InvokeResponseBody};

use actors::{
    inference_result, initialize_message, raw_inference_message, validate_training_event_stream,
    ActorHealth, ActorJoin, InferenceActor, InferenceCommand, TrainingActor, TrainingEvent,
    TrainingStage,
};
use errors::ApiError;
use slouch_domain::ported::messages::schemas::{
    DimensionalityReductionConfig as WireDimReductionConfig,
    DimensionalityReductionMethod as WireDimReductionMethod, ImageData, InferenceWorkerMessage,
    NormalizationMode as WireNormalizationMode, SerializedClassifier, SerializedClassifierState,
    SerializedFeatureExtractor, SerializedGaussianNb, SerializedModel as WireSerializedModel,
};
use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FeatureMap, FrameLabel, Keypoint, PostureFrame,
    Thumbnail, TrainingSettings,
};
use slouch_ml::ported::training_worker::TrainingWorkerResponse;
use slouch_store::ported::storage::DatasetStorage;
use slouch_vision::ported::inference_worker::{NativeInferenceResult, WorkerResponse};

fn empty_storage() -> Arc<DatasetStorage> {
    Arc::new(DatasetStorage::open_in_memory().expect("in-memory dataset storage"))
}

fn empty_inference_result() -> NativeInferenceResult {
    NativeInferenceResult {
        person_found: false,
        bbox: None,
        keypoints: None,
        features: FeatureMap::new(),
        classification: None,
    }
}

fn wire_gaussian_model(dimension: usize) -> WireSerializedModel {
    WireSerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: vec![FeatureId::RtmDetExtracted.as_str().into()],
            normalization_mode: WireNormalizationMode::None,
            dim_reduction_config: WireDimReductionConfig {
                method: WireDimReductionMethod::None,
                components: dimension,
            },
            concatenated_dimensions: dimension,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        },
        classifier: SerializedClassifier {
            classifier_id: "gaussian_nb".into(),
            state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                class_means: [vec![0.0; dimension], vec![1.0; dimension]],
                class_variances: [vec![1.0; dimension], vec![1.0; dimension]],
                class_priors: [0.5, 0.5],
                epsilon: 1e-9,
            }),
        },
        trained_at: 1.0,
        version: 1.0,
    }
}

fn valid_training_settings() -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id: ClassifierId::GaussianNb,
            params: BTreeMap::new(),
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: 1,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::RtmDetExtracted],
        feature_types: None,
        normalization_mode: None,
        cv_folds: 2,
        last_updated: 1.0,
    }
}

fn labeled_frame(id: &str, label: FrameLabel, timestamp: f64, level: f32) -> PostureFrame {
    let mut features = FeatureMap::new();
    features.insert(
        FeatureId::GauFeatures,
        vec![level; FeatureId::GauFeatures.metadata().dimensions],
    );
    PostureFrame {
        id: id.into(),
        timestamp,
        features,
        thumbnail: Thumbnail {
            mime_type: "image/png".into(),
            bytes: vec![1],
        },
        keypoints: (0..17).map(|_| Keypoint::new(0.4, 0.5, 0.9)).collect(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
        label,
    }
}

#[test]
fn api_error_wire_kinds_and_messages_cover_every_variant() {
    let cases = [
        (ApiError::InvalidRequest("m".into()), "invalidRequest"),
        (ApiError::NotFound("m".into()), "notFound"),
        (ApiError::NotReady("m".into()), "notReady"),
        (ApiError::Busy("m".into()), "busy"),
        (ApiError::Cancelled("m".into()), "cancelled"),
        (ApiError::DatasetChanged("m".into()), "datasetChanged"),
        (ApiError::Storage("m".into()), "storage"),
        (ApiError::Inference("m".into()), "inference"),
        (ApiError::Training("m".into()), "training"),
        (ApiError::Ipc("m".into()), "ipc"),
        (ApiError::Internal("m".into()), "internal"),
    ];
    for (error, kind) in cases {
        assert_eq!(error.to_string(), "m");
        let value = serde_json::to_value(&error).expect("serialize ApiError");
        assert_eq!(value["kind"], kind);
        assert_eq!(value["message"], "m");
    }
}

#[test]
fn full_mailbox_returns_busy_without_blocking_or_failing_health() {
    let (sender, receiver) = mpsc::sync_channel(1);
    sender
        .try_send(InferenceCommand::Shutdown)
        .expect("occupy the single mailbox slot");
    let actor = InferenceActor {
        sender,
        health: Arc::new(ActorHealth::new()),
        join: Mutex::new(None),
    };
    let started = Instant::now();
    let error = actor.clear_cache().expect_err("full mailbox");
    assert!(
        matches!(&error, ApiError::Busy(m) if m.contains("busy")),
        "{error:?}"
    );
    // backpressure is transient: health stays green and the call never blocks
    assert!(actor.is_healthy());
    assert!(started.elapsed() < Duration::from_secs(5));
    drop(receiver);
}

#[test]
fn a_disconnected_actor_fails_health_closed_and_reports_not_ready() {
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(receiver);
    let actor = InferenceActor {
        sender,
        health: Arc::new(ActorHealth::new()),
        join: Mutex::new(None),
    };
    assert!(actor.is_healthy());
    let error = actor.clear_cache().expect_err("disconnected mailbox");
    assert!(matches!(error, ApiError::NotReady(_)), "{error:?}");
    assert!(!actor.is_healthy(), "disconnect must fail health closed");
    // once failed, every later call short-circuits on the health gate
    assert!(matches!(
        actor.send(InferenceWorkerMessage::UnloadClassifier),
        Err(ApiError::NotReady(_))
    ));
}

#[test]
fn live_inference_actor_round_trips_one_time_tokens_and_worker_messages() {
    let actor = InferenceActor::start(empty_storage()).expect("start inference actor");
    assert!(actor.is_healthy());

    // an uninitialized worker rejects frames with an inference error while the
    // actor itself keeps serving
    let image = ImageData {
        data: vec![0; 16],
        width: 2,
        height: 2,
    };
    let error = actor
        .send_frame(raw_inference_message(image, 7))
        .expect_err("uninitialized worker");
    assert!(
        matches!(&error, ApiError::Inference(m) if m.contains("initialize")),
        "{error:?}"
    );
    assert!(
        actor.is_healthy(),
        "worker-level errors must not kill the actor"
    );

    // unloading with no loaded classifier is a clean typed response
    let responses = actor
        .send(InferenceWorkerMessage::UnloadClassifier)
        .expect("unload classifier");
    assert!(matches!(
        responses.as_slice(),
        [WorkerResponse::ClassifierUnloaded]
    ));

    // one-time token flow through the actor mailbox
    let token = actor
        .cache_result(9, empty_inference_result())
        .expect("cache result");
    assert_ne!(token, 9, "token must differ from the request id");
    assert!(matches!(
        actor.checkout_result(token, 8),
        Err(ApiError::InvalidRequest(_))
    ));
    let result = actor.checkout_result(token, 9).expect("checkout");
    actor
        .restore_result(token, 9, result)
        .expect("restore after a failed save");
    let result = actor.checkout_result(token, 9).expect("checkout again");
    drop(result);
    actor.commit_result(token, 9).expect("commit");
    let consumed = actor
        .checkout_result(token, 9)
        .expect_err("token already consumed");
    assert!(
        matches!(&consumed, ApiError::InvalidRequest(m) if m.contains("already consumed")),
        "{consumed:?}"
    );

    // clear_cache forgets live tokens outright
    let token = actor
        .cache_result(10, empty_inference_result())
        .expect("cache result");
    actor.clear_cache().expect("clear cache");
    let unknown = actor.checkout_result(token, 10).expect_err("cleared token");
    assert!(
        matches!(&unknown, ApiError::InvalidRequest(m) if m.contains("unknown")),
        "{unknown:?}"
    );

    actor.shutdown_until(Instant::now() + Duration::from_secs(5));
    assert!(!actor.is_healthy());
}

#[test]
fn publishing_models_accepts_valid_pairs_and_rejects_broken_ones_nonfatally() {
    let actor = InferenceActor::start(empty_storage()).expect("start inference actor");
    let dimension = FeatureId::RtmDetExtracted.metadata().dimensions;
    actor
        .publish_model_pair(
            Some(wire_gaussian_model(dimension)),
            Some(wire_gaussian_model(dimension)),
        )
        .expect("valid classifier pair");
    // A posture-only generation (no presence role) is accepted; runtime presence
    // falls back to the RTMDet detector confidence.
    actor
        .publish_model_pair(Some(wire_gaussian_model(dimension)), None)
        .expect("valid posture-only generation");
    let mut broken = wire_gaussian_model(dimension);
    broken.classifier.classifier_id = "no_such_classifier".into();
    let error = actor
        .publish_model_pair(Some(broken), Some(wire_gaussian_model(dimension)))
        .expect_err("unknown classifier id");
    assert!(matches!(error, ApiError::Inference(_)), "{error:?}");
    assert!(
        actor.is_healthy(),
        "a rejected model must leave the actor serving"
    );
    actor.shutdown_until(Instant::now() + Duration::from_secs(5));
}

#[test]
fn worker_message_constructors_and_response_mapping_preserve_the_ipc_contract() {
    let message = initialize_message(
        PathBuf::from("C:/models/rtmdet.onnx"),
        PathBuf::from("C:/models/rtmpose.onnx"),
    );
    match message {
        InferenceWorkerMessage::Initialize { payload } => {
            assert_eq!(payload.rtmdet_path, "C:/models/rtmdet.onnx");
            assert_eq!(payload.rtmw3d_path, "C:/models/rtmpose.onnx");
        }
        _ => panic!("initialize_message must build an Initialize message"),
    }

    let message = raw_inference_message(
        ImageData {
            data: vec![1, 2, 3, 4],
            width: 1,
            height: 1,
        },
        77,
    );
    match message {
        InferenceWorkerMessage::Process { payload } => {
            assert_eq!(payload.request_id, 77);
            assert_eq!(payload.image_data.width, 1);
            assert_eq!(payload.image_data.height, 1);
            assert_eq!(payload.image_data.data, vec![1, 2, 3, 4]);
        }
        _ => panic!("raw_inference_message must build a Process message"),
    }

    assert!(inference_result(WorkerResponse::Result {
        request_id: 1,
        result: empty_inference_result(),
    })
    .is_ok());
    assert!(matches!(
        inference_result(WorkerResponse::Error {
            error: "boom".into(),
            request_id: Some(1),
            details: None,
            success: None,
        }),
        Err(ApiError::Inference(_))
    ));
    // a non-result response must never be treated as a usable inference
    assert!(matches!(
        inference_result(WorkerResponse::Initialized {
            provider: "native".into(),
        }),
        Err(ApiError::Inference(_))
    ));
}

#[test]
fn an_injected_panic_is_contained_health_fails_closed_and_shutdown_completes() {
    let actor = InferenceActor::start(empty_storage()).expect("start inference actor");
    actor
        .sender
        .try_send(InferenceCommand::CrashForTests)
        .expect("enqueue the crash command");
    let deadline = Instant::now() + Duration::from_secs(10);
    while actor.is_healthy() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(!actor.is_healthy(), "panic must fail health closed");
    // subsequent sends observe NotReady instead of hanging or propagating
    assert!(matches!(actor.clear_cache(), Err(ApiError::NotReady(_))));
    assert!(matches!(
        actor.cache_result(1, empty_inference_result()),
        Err(ApiError::NotReady(_))
    ));
    // shutdown after the panic must neither hang nor poison the join mutex
    let started = Instant::now();
    actor.shutdown_until(Instant::now() + Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(actor
        .join
        .lock()
        .expect("join mutex must not be poisoned")
        .is_none());
}

#[test]
fn idle_actor_shutdown_joins_cleanly_and_then_refuses_work() {
    let actor = InferenceActor::start(empty_storage()).expect("start inference actor");
    let started = Instant::now();
    actor.shutdown_until(started + Duration::from_secs(5));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "an idle actor must join well before the deadline"
    );
    assert!(!actor.is_healthy());
    assert!(actor.join.lock().expect("join state").is_none());
    assert!(matches!(actor.clear_cache(), Err(ApiError::NotReady(_))));
}

#[test]
fn shutdown_of_a_wedged_actor_returns_at_the_deadline_instead_of_hanging() {
    let (sender, receiver) = mpsc::sync_channel(1);
    sender
        .try_send(InferenceCommand::Shutdown)
        .expect("wedge the mailbox");
    let (done_sender, done) = mpsc::channel();
    let handle = thread::spawn(|| {});
    let actor = InferenceActor {
        sender,
        health: Arc::new(ActorHealth::new()),
        join: Mutex::new(Some(ActorJoin { handle, done })),
    };
    let started = Instant::now();
    actor.shutdown_until(started + Duration::from_millis(300));
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(200),
        "shutdown must keep retrying through the cooperative window, returned after {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "shutdown must not hang past the deadline, returned after {elapsed:?}"
    );
    assert!(!actor.is_healthy());
    assert!(actor.join.lock().expect("join state").is_none());
    drop(done_sender);
    drop(receiver);
}

#[test]
fn cancelling_a_running_training_yields_cancelled_and_keeps_the_actor_serving() {
    let actor = Arc::new(TrainingActor::start(empty_storage()).expect("start training actor"));
    let barrier = Arc::new(Barrier::new(2));
    let trainer = {
        let actor = Arc::clone(&actor);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            actor.train(31, false, None)
        })
    };
    barrier.wait();
    // Acknowledge-cancel loop: Busy means the mailbox still holds the Train
    // command, NotReady means it has not been processed yet; both resolve as
    // soon as the actor enters its running state.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match actor.cancel() {
            Ok(()) => break,
            Err(ApiError::Busy(_) | ApiError::NotReady(_)) if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(other) => panic!("unexpected cancel outcome: {other}"),
        }
    }
    // the cancelled job is still draining, so a new training is refused
    let busy = actor.train(32, false, None).expect_err("second training");
    assert!(matches!(busy, ApiError::Busy(_)), "{busy:?}");
    let outcome = trainer.join().expect("trainer thread");
    assert!(
        matches!(outcome, Err(ApiError::Cancelled(_))),
        "cancelled training must report Cancelled, got {outcome:?}"
    );
    assert!(actor.is_healthy(), "cancellation must not fail the actor");
    // the actor keeps serving real work afterwards: an empty store fails with
    // the training worker's own storage error, not a lifecycle error
    let error = actor.train(33, false, None).expect_err("empty dataset");
    assert!(
        matches!(&error, ApiError::Training(m) if m.contains("No training settings")),
        "{error:?}"
    );
    assert!(matches!(actor.cancel(), Err(ApiError::NotReady(_))));
    actor.shutdown();
    assert!(!actor.is_healthy());
    assert!(matches!(
        actor.train(34, false, None),
        Err(ApiError::NotReady(_))
    ));
}

#[test]
fn a_real_training_run_emits_the_monotonic_actor_event_prefix_and_a_model() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    storage
        .save_training_settings(valid_training_settings())
        .expect("training settings");
    storage
        .save_frame(labeled_frame("good-1", FrameLabel::Good, 1_000.0, 0.2))
        .expect("good frame");
    storage
        .save_frame(labeled_frame("bad-1", FrameLabel::Bad, 2_000.0, 0.8))
        .expect("bad frame");
    let actor = TrainingActor::start(Arc::new(storage)).expect("start training actor");

    // Channel::new works without a running Tauri app; events arrive as the
    // serialized JSON the webview would receive, so this pins the wire shape.
    let events: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let channel = Channel::new(move |body| {
        let InvokeResponseBody::Json(json) = body else {
            panic!("training events must serialize as JSON");
        };
        sink.lock()
            .expect("event sink")
            .push(serde_json::from_str(&json).expect("event JSON"));
        Ok(())
    });

    let response = actor
        .train(41, false, Some(channel))
        .expect("training succeeds on a minimal good/bad dataset");
    match response {
        TrainingWorkerResponse::Result { result, models } => {
            assert!(result.success, "training must succeed: {:?}", result.errors);
            assert!(models.posture.is_some(), "a posture model must be produced");
        }
        TrainingWorkerResponse::Error { error, .. } => {
            panic!("unexpected training error: {error}")
        }
    }

    let events = events.lock().expect("event sink");
    let expected = [
        serde_json::json!({"type": "started", "jobId": 41, "sequence": 0}),
        serde_json::json!({
            "type": "progress", "jobId": 41, "sequence": 1,
            "stage": "processing", "progress": 5,
        }),
        serde_json::json!({
            "type": "progress", "jobId": 41, "sequence": 2,
            "stage": "evaluating", "progress": 85,
        }),
    ];
    assert_eq!(
        events.as_slice(),
        &expected,
        "the actor-level event prefix is a wire contract"
    );
    drop(events);
    actor.shutdown();
}

#[test]
fn event_stream_validator_rejects_duplicate_and_out_of_order_sequences() {
    let started = TrainingEvent::Started {
        job_id: 1,
        sequence: 0,
    };
    let progress = |sequence, progress| TrainingEvent::Progress {
        job_id: 1,
        sequence,
        stage: TrainingStage::Processing,
        progress,
    };
    let cancelled = |sequence| TrainingEvent::Cancelled {
        job_id: 1,
        sequence,
    };

    // baseline: a contiguous, single-terminal stream passes
    assert!(
        validate_training_event_stream(&[started.clone(), progress(1, 5), cancelled(2)]).is_ok()
    );
    // duplicate sequence number
    assert!(validate_training_event_stream(&[
        started.clone(),
        progress(1, 5),
        progress(1, 6),
        cancelled(2),
    ])
    .is_err());
    // skipped sequence number
    assert!(
        validate_training_event_stream(&[started.clone(), progress(2, 5), cancelled(3)]).is_err()
    );
    // stream not starting at sequence zero
    assert!(validate_training_event_stream(&[
        TrainingEvent::Started {
            job_id: 1,
            sequence: 1,
        },
        cancelled(2),
    ])
    .is_err());
    // job id changes mid-stream
    assert!(validate_training_event_stream(&[
        started.clone(),
        TrainingEvent::Cancelled {
            job_id: 2,
            sequence: 1,
        },
    ])
    .is_err());
    // empty stream and missing terminal event
    assert!(validate_training_event_stream(&[]).is_err());
    assert!(validate_training_event_stream(&[started, progress(1, 5)]).is_err());
}
