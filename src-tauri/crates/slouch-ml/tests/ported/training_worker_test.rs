//! Production-entry replacement for `src/workers/__tests__/training-worker.test.ts`
//! (frozen SHA-256 `52f26cace4a5e5783200fb2cb9314a20f8a3c7d8678bc71acffba106650eaaeb`).

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use slouch_domain::ported::messages::schemas::{
    LogLevelPayload, TrainPayload, TrainingWorkerMessage,
};
use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FrameLabel, Keypoint, NormalizationMode,
    PostureDataset, PostureFrame, Thumbnail, TrainingMetrics, TrainingSettings,
};
use slouch_ml::ported::training_worker::{
    FeatureContainer, FeatureExtractorConfig, ReservoirSample, TrainingBackend, TrainingClock,
    TrainingLogger, TrainingStorage, TrainingWorker, TrainingWorkerResponse,
};
use slouch_ml::ported::types::{
    SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
    SerializedGaussianNb, SerializedModel,
};

#[derive(Clone, Default)]
struct StorageState {
    calls: Vec<String>,
    saved_posture: Vec<SerializedModel>,
    saved_presence: Vec<SerializedModel>,
}

struct TestStorage {
    state: Arc<Mutex<StorageState>>,
    datasets: VecDeque<Result<Option<PostureDataset>, String>>,
    settings: Result<Option<TrainingSettings>, String>,
    reservoir: Result<Vec<ReservoirSample>, String>,
    fail_posture_save: bool,
    fail_presence_save: bool,
}

impl TrainingStorage for TestStorage {
    fn load_dataset(&mut self) -> Result<Option<PostureDataset>, String> {
        self.state
            .lock()
            .expect("storage")
            .calls
            .push("load_dataset".into());
        self.datasets
            .pop_front()
            .unwrap_or_else(|| Err("no dataset outcome".into()))
    }
    fn load_training_settings(&mut self) -> Result<Option<TrainingSettings>, String> {
        self.state
            .lock()
            .expect("storage")
            .calls
            .push("load_settings".into());
        self.settings.clone()
    }
    fn load_reservoir_samples(&mut self) -> Result<Vec<ReservoirSample>, String> {
        self.state
            .lock()
            .expect("storage")
            .calls
            .push("load_reservoir".into());
        self.reservoir.clone()
    }
    fn save_posture_model(&mut self, model: &SerializedModel) -> Result<(), String> {
        self.state
            .lock()
            .expect("storage")
            .calls
            .push("save_posture".into());
        if self.fail_posture_save {
            Err("posture storage failed".into())
        } else {
            self.state
                .lock()
                .expect("storage")
                .saved_posture
                .push(model.clone());
            Ok(())
        }
    }
    fn save_presence_model(&mut self, model: &SerializedModel) -> Result<(), String> {
        self.state
            .lock()
            .expect("storage")
            .calls
            .push("save_presence".into());
        if self.fail_presence_save {
            Err("presence storage failed".into())
        } else {
            self.state
                .lock()
                .expect("storage")
                .saved_presence
                .push(model.clone());
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
struct CvCall {
    features: Vec<FeatureId>,
    frame_labels: Vec<FrameLabel>,
    labels: Vec<i32>,
    folds: usize,
}

#[derive(Clone, Debug, Default)]
struct BackendState {
    calibrations: usize,
    cv_calls: Vec<CvCall>,
    fit_calls: Vec<Vec<FeatureId>>,
    releases: usize,
}

struct TestBackend {
    state: Arc<Mutex<BackendState>>,
    fail_cv_for: Option<FeatureId>,
    fail_fit_for: Option<FeatureId>,
}

impl TrainingBackend for TestBackend {
    fn calibrate_feature_bins(
        &mut self,
        _samples: &[FeatureContainer],
        _log_engineered: bool,
        _logger: &dyn TrainingLogger,
    ) {
        self.state.lock().expect("backend").calibrations += 1;
    }
    fn cross_validate(
        &mut self,
        config: &FeatureExtractorConfig,
        _classifier: &ClassifierConfig,
        frames: &[PostureFrame],
        labels: &[i32],
        cv_folds: usize,
    ) -> Result<Option<TrainingMetrics>, String> {
        self.state.lock().expect("backend").cv_calls.push(CvCall {
            features: config.feature_types.clone(),
            frame_labels: frames.iter().map(|frame| frame.label).collect(),
            labels: labels.to_vec(),
            folds: cv_folds,
        });
        if self
            .fail_cv_for
            .is_some_and(|feature| config.feature_types.contains(&feature))
        {
            return Err("cv failed".into());
        }
        Ok(Some(metrics()))
    }
    fn fit(
        &mut self,
        config: &FeatureExtractorConfig,
        _classifier: &ClassifierConfig,
        _frames: &[PostureFrame],
        _labels: &[i32],
    ) -> Result<SerializedModel, String> {
        self.state
            .lock()
            .expect("backend")
            .fit_calls
            .push(config.feature_types.clone());
        if self
            .fail_fit_for
            .is_some_and(|feature| config.feature_types.contains(&feature))
        {
            return Err("fit failed".into());
        }
        Ok(serialized_model(&config.feature_types))
    }
    fn release_training_buffers(&mut self) {
        self.state.lock().expect("backend").releases += 1;
    }
}

#[derive(Clone, Default)]
struct TestLogger(Arc<Mutex<Vec<(String, String)>>>);
impl TrainingLogger for TestLogger {
    fn info(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("info".into(), message.into()));
    }
    fn warn(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("warn".into(), message.into()));
    }
    fn error(&self, message: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("error".into(), message.into()));
    }
    fn set_from_url_param(&self, value: &str) {
        self.0
            .lock()
            .expect("logs")
            .push(("set".into(), value.into()));
    }
}

#[derive(Clone, Copy)]
struct FixedClock(f64);
impl TrainingClock for FixedClock {
    fn now_millis(&self) -> f64 {
        self.0
    }
}

fn frame(id: &str, label: FrameLabel) -> PostureFrame {
    PostureFrame {
        id: id.into(),
        label,
        timestamp: 1.0,
        features: BTreeMap::from([
            (
                FeatureId::GauFeatures,
                vec![0.2; FeatureId::GauFeatures.metadata().dimensions],
            ),
            (
                FeatureId::RtmDetEngineered,
                vec![0.3; FeatureId::RtmDetEngineered.metadata().dimensions],
            ),
        ]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![1],
        },
        keypoints: (0..17)
            .map(|index| Keypoint::new(0.2 + f64::from(index) * 0.02, 0.3, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

fn dataset() -> PostureDataset {
    PostureDataset {
        frames: vec![
            frame("good", FrameLabel::Good),
            frame("bad", FrameLabel::Bad),
            frame("away", FrameLabel::Away),
            frame("unused", FrameLabel::Unused),
        ],
        version: 1,
        last_modified: 1.0,
    }
}

fn settings() -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id: ClassifierId::GaussianNb,
            params: BTreeMap::new(),
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: 0,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::RtmDetEngineered],
        feature_types: None,
        normalization_mode: Some(NormalizationMode::None),
        cv_folds: 5,
        last_updated: 1.0,
    }
}

fn metrics() -> TrainingMetrics {
    TrainingMetrics {
        cv_accuracy: 0.75,
        cv_std: 0.1,
        mcc: 0.5,
        f1_score: 0.7,
        confusion_matrix: vec![vec![1, 0], vec![0, 1]],
        fold_accuracies: vec![0.75],
        balanced_accuracy: 0.75,
        accuracy_ci_low: 0.5,
        accuracy_ci_high: 0.9,
        worst_fold_accuracy: 0.75,
        cv_type: None,
    }
}

fn serialized_model(features: &[FeatureId]) -> SerializedModel {
    let dimensions = features
        .iter()
        .map(|feature| feature.metadata().dimensions)
        .sum();
    SerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: features
                .iter()
                .map(|feature| feature.as_str().to_owned())
                .collect(),
            normalization_mode: slouch_ml::ported::types::NormalizationMode::None,
            dim_reduction_config: slouch_ml::ported::types::DimensionalityReductionConfig {
                method: slouch_ml::ported::types::DimensionalityReductionMethod::None,
                components: 0,
            },
            concatenated_dimensions: dimensions,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        },
        classifier: SerializedClassifier {
            classifier_id: "gaussian_nb".into(),
            state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                class_means: [vec![0.0; dimensions], vec![1.0; dimensions]],
                class_variances: [vec![1.0; dimensions], vec![1.0; dimensions]],
                class_priors: [0.5, 0.5],
                epsilon: 1e-9,
            }),
        },
        trained_at: 0.0,
        version: 0.0,
    }
}

fn make_worker(
    storage: TestStorage,
    backend: TestBackend,
    logger: TestLogger,
) -> TrainingWorker<TestStorage, TestBackend, TestLogger, FixedClock> {
    TrainingWorker::with_clock(storage, backend, logger, FixedClock(1234.0))
}

fn normal_storage(state: Arc<Mutex<StorageState>>) -> TestStorage {
    TestStorage {
        state,
        datasets: VecDeque::from([Ok(Some(dataset()))]),
        settings: Ok(Some(settings())),
        reservoir: Ok(vec![]),
        fail_posture_save: false,
        fail_presence_save: false,
    }
}

fn dataset_of(labels: &[FrameLabel]) -> PostureDataset {
    PostureDataset {
        frames: labels
            .iter()
            .enumerate()
            .map(|(index, &label)| frame(&format!("f{index}"), label))
            .collect(),
        version: 1,
        last_modified: 1.0,
    }
}

fn storage_with_dataset(state: Arc<Mutex<StorageState>>, ds: PostureDataset) -> TestStorage {
    TestStorage {
        state,
        datasets: VecDeque::from([Ok(Some(ds))]),
        settings: Ok(Some(settings())),
        reservoir: Ok(vec![]),
        fail_posture_save: false,
        fail_presence_save: false,
    }
}

#[test]
fn actual_worker_preserves_dataset_order_labels_cv_config_metadata_and_logging() {
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let mut worker = make_worker(
        normal_storage(storage_state.clone()),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        logger,
    );

    let response = worker.handle_message(TrainingWorkerMessage::Train {
        payload: Some(TrainPayload { do_cv: Some(true) }),
    });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.success);
    assert!(result.posture_result.is_some() && result.presence_result.is_some());
    assert_eq!(models.posture.as_ref().expect("posture").trained_at, 1234.0);
    assert_eq!(models.presence.as_ref().expect("presence").version, 1.0);
    let backend = backend_state.lock().expect("backend");
    assert_eq!(backend.cv_calls.len(), 2);
    assert_eq!(
        backend.cv_calls[0].features,
        vec![FeatureId::RtmDetEngineered]
    );
    assert_eq!(
        backend.cv_calls[0].frame_labels,
        vec![FrameLabel::Good, FrameLabel::Good, FrameLabel::Bad]
    );
    assert_eq!(backend.cv_calls[0].labels, vec![0, 0, 1]);
    assert_eq!(backend.cv_calls[0].folds, 5);
    assert_eq!(
        backend.cv_calls[1].frame_labels,
        vec![FrameLabel::Good, FrameLabel::Bad]
    );
    assert_eq!(backend.cv_calls[1].labels, vec![0, 1]);
    assert_eq!(backend.releases, 2);
    assert_eq!(
        storage_state.lock().expect("storage").calls,
        vec![
            "load_dataset",
            "load_settings",
            "load_reservoir",
            "save_presence",
            "save_posture"
        ]
    );
    assert!(logs
        .lock()
        .expect("logs")
        .iter()
        .any(|(_, message)| message.contains("Dual training complete. Success: true")));
}

#[test]
fn custom_cv_folds_propagate_to_backend_on_the_do_cv_path() {
    // Oracle 'should respect custom cvFolds from settings' (training-worker.test.ts:360-367):
    // a non-default cvFolds must flow from stored settings through `doCV ? cvFolds : 1` to the
    // backend. Use a value distinct from every other test's default (5) so a regression that
    // hardcoded 5 on the CV path would fail here.
    let mut custom = settings();
    custom.cv_folds = 10;
    let storage = TestStorage {
        state: Arc::new(Mutex::new(StorageState::default())),
        datasets: VecDeque::from([Ok(Some(dataset()))]),
        settings: Ok(Some(custom)),
        reservoir: Ok(vec![]),
        fail_posture_save: false,
        fail_presence_save: false,
    };
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut worker = make_worker(
        storage,
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    worker.handle_message(TrainingWorkerMessage::Train {
        payload: Some(TrainPayload { do_cv: Some(true) }),
    });
    let backend = backend_state.lock().expect("backend");
    assert_eq!(backend.cv_calls.len(), 2);
    assert!(
        backend.cv_calls.iter().all(|call| call.folds == 10),
        "stored cv_folds must propagate to every CV call: {:?}",
        backend.cv_calls.iter().map(|c| c.folds).collect::<Vec<_>>()
    );
}

#[test]
fn do_cv_false_uses_one_fold_and_log_level_dispatches_to_actual_logger() {
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let mut worker = make_worker(
        normal_storage(Arc::new(Mutex::new(StorageState::default()))),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        logger,
    );
    assert!(worker
        .handle_message(TrainingWorkerMessage::SetLogLevel {
            payload: Some(LogLevelPayload {
                log_param: Some("training:debug".into())
            })
        })
        .is_empty());
    worker.handle_message(TrainingWorkerMessage::Train {
        payload: Some(TrainPayload { do_cv: Some(false) }),
    });
    {
        let backend = backend_state.lock().expect("backend");
        // Both roles train on the full dataset, so CV must actually have run twice; without
        // this length check the `.all()` below would pass vacuously if CV were skipped entirely.
        assert_eq!(backend.cv_calls.len(), 2);
        assert!(backend.cv_calls.iter().all(|call| call.folds == 1));
    }
    assert!(logs
        .lock()
        .expect("logs")
        .iter()
        .any(|entry| entry == &("set".into(), "training:debug".into())));
}

#[test]
fn storage_load_failures_return_errors_and_reset_training_for_a_followup_request() {
    for first in [Err("dataset read failed".into()), Ok(None)] {
        let state = Arc::new(Mutex::new(StorageState::default()));
        let storage = TestStorage {
            state,
            datasets: VecDeque::from([first, Ok(Some(dataset()))]),
            settings: Ok(Some(settings())),
            reservoir: Ok(vec![]),
            fail_posture_save: false,
            fail_presence_save: false,
        };
        let backend_state = Arc::new(Mutex::new(BackendState::default()));
        let mut worker = make_worker(
            storage,
            TestBackend {
                state: backend_state,
                fail_cv_for: None,
                fail_fit_for: None,
            },
            TestLogger::default(),
        );
        assert!(matches!(
            &worker.handle_message(TrainingWorkerMessage::Train { payload: None })[..],
            [TrainingWorkerResponse::Error { .. }]
        ));
        assert!(matches!(
            &worker.handle_message(TrainingWorkerMessage::Train { payload: None })[..],
            [TrainingWorkerResponse::Result { .. }]
        ));
    }
}

#[test]
fn settings_and_reservoir_failures_stop_before_backend_training() {
    let cases = [
        (Err("settings read failed".into()), Ok(vec![])),
        (Ok(None), Ok(vec![])),
        (Ok(Some(settings())), Err("reservoir read failed".into())),
    ];
    for (settings_result, reservoir) in cases {
        let backend_state = Arc::new(Mutex::new(BackendState::default()));
        let storage = TestStorage {
            state: Arc::new(Mutex::new(StorageState::default())),
            datasets: VecDeque::from([Ok(Some(dataset()))]),
            settings: settings_result,
            reservoir,
            fail_posture_save: false,
            fail_presence_save: false,
        };
        let mut worker = make_worker(
            storage,
            TestBackend {
                state: backend_state.clone(),
                fail_cv_for: None,
                fail_fit_for: None,
            },
            TestLogger::default(),
        );
        assert!(matches!(
            &worker.handle_message(TrainingWorkerMessage::Train { payload: None })[..],
            [TrainingWorkerResponse::Error { .. }]
        ));
        assert!(backend_state.lock().expect("backend").cv_calls.is_empty());
    }
}

#[test]
fn explicit_empty_feature_lists_are_rejected_without_backend_calls() {
    for role in ["posture", "presence"] {
        let storage_state = Arc::new(Mutex::new(StorageState::default()));
        let backend_state = Arc::new(Mutex::new(BackendState::default()));
        let mut invalid = settings();
        if role == "posture" {
            invalid.posture_feature_types.clear();
            invalid.feature_types = Some(vec![FeatureId::GauFeatures]);
        } else {
            invalid.presence_feature_types.clear();
        }
        let storage = TestStorage {
            state: storage_state.clone(),
            datasets: VecDeque::from([Ok(Some(dataset()))]),
            settings: Ok(Some(invalid)),
            reservoir: Ok(vec![]),
            fail_posture_save: false,
            fail_presence_save: false,
        };
        let mut worker = make_worker(
            storage,
            TestBackend {
                state: backend_state.clone(),
                fail_cv_for: None,
                fail_fit_for: None,
            },
            TestLogger::default(),
        );

        let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
        assert!(
            matches!(&response[..], [TrainingWorkerResponse::Error { error, .. }] if error.contains(&format!("{role}FeatureTypes is empty or invalid"))),
            "unexpected response for {role}: {response:?}",
        );
        let backend = backend_state.lock().expect("backend");
        assert_eq!(backend.calibrations, 0);
        assert!(backend.cv_calls.is_empty());
        assert!(backend.fit_calls.is_empty());
        assert_eq!(backend.releases, 0);
        assert_eq!(
            storage_state.lock().expect("storage").calls,
            vec!["load_dataset", "load_settings"],
        );
    }
}

#[test]
fn role_storage_save_failures_are_non_fatal_and_keep_models_and_success() {
    // Oracle (training-worker.ts:432-453, 520-543) assigns presenceResult/presenceModel
    // (and posture equivalents) BEFORE awaiting the save. A save failure lands in the catch,
    // which only appends an error string; the result/model stay set and are returned, and the
    // overall success flag (line 546) is unaffected. So a save failure must NOT drop the model
    // nor flip success.
    for (fail_presence, fail_posture) in [(true, false), (false, true), (true, true)] {
        let storage_state = Arc::new(Mutex::new(StorageState::default()));
        let mut storage = normal_storage(storage_state.clone());
        storage.fail_presence_save = fail_presence;
        storage.fail_posture_save = fail_posture;
        let backend_state = Arc::new(Mutex::new(BackendState::default()));
        let mut worker = make_worker(
            storage,
            TestBackend {
                state: backend_state.clone(),
                fail_cv_for: None,
                fail_fit_for: None,
            },
            TestLogger::default(),
        );
        let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
        let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
            panic!("result")
        };
        // Both models trained; save failure keeps result/model set, so success stays true.
        assert!(result.success);
        assert!(result.presence_result.as_ref().is_some_and(|r| r.success));
        assert!(result.posture_result.as_ref().is_some_and(|r| r.success));
        assert!(models.presence.is_some());
        assert!(models.posture.is_some());
        // The stored copy is still absent when the save itself failed.
        assert_eq!(
            storage_state
                .lock()
                .expect("storage")
                .saved_presence
                .is_empty(),
            fail_presence
        );
        assert_eq!(
            storage_state
                .lock()
                .expect("storage")
                .saved_posture
                .is_empty(),
            fail_posture
        );
        assert_eq!(backend_state.lock().expect("backend").releases, 2);
        if fail_presence {
            assert!(result
                .errors
                .iter()
                .any(|error| error
                    .contains("Presence model training failed: presence storage failed")));
        }
        if fail_posture {
            assert!(result.errors.iter().any(
                |error| error.contains("Posture model training failed: posture storage failed")
            ));
        }
    }
}

#[test]
fn backend_cv_and_fit_failures_release_once_per_attempt_and_do_not_block_other_role() {
    for (fail_cv, fail_fit) in [
        (Some(FeatureId::RtmDetEngineered), None),
        (None, Some(FeatureId::RtmDetEngineered)),
        (Some(FeatureId::GauFeatures), None),
        (None, Some(FeatureId::GauFeatures)),
    ] {
        let backend_state = Arc::new(Mutex::new(BackendState::default()));
        let mut worker = make_worker(
            normal_storage(Arc::new(Mutex::new(StorageState::default()))),
            TestBackend {
                state: backend_state.clone(),
                fail_cv_for: fail_cv,
                fail_fit_for: fail_fit,
            },
            TestLogger::default(),
        );
        let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
        let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
            panic!("result")
        };
        assert!(!result.success);
        // Assert the SPECIFIC role was dropped: RtmDetEngineered drives the presence model and
        // GauFeatures drives the posture model. A regression that failed the wrong role would
        // still satisfy a bare XOR, so pin each case to the role that owns the failing feature.
        let failing_feature = fail_cv.or(fail_fit).expect("one failure configured");
        match failing_feature {
            FeatureId::RtmDetEngineered => {
                assert!(models.presence.is_none());
                assert!(models.posture.is_some());
            }
            FeatureId::GauFeatures => {
                assert!(models.posture.is_none());
                assert!(models.presence.is_some());
            }
            other => panic!("unexpected failing feature: {other:?}"),
        }
        assert_eq!(backend_state.lock().expect("backend").releases, 2);
    }
}

#[test]
fn good_and_bad_without_away_trains_posture_only_and_warns_no_away_frames() {
    // Oracle training-worker.ts:333-347,454-456: with no AWAY frames the presence model is
    // skipped, emitting both the "Skipping presence model" and "No AWAY frames collected"
    // warnings, while posture trains and overall success stays true (presence not required).
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut worker = make_worker(
        storage_with_dataset(
            storage_state.clone(),
            dataset_of(&[FrameLabel::Good, FrameLabel::Bad]),
        ),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.success);
    assert!(models.posture.is_some());
    assert!(models.presence.is_none());
    assert!(result.presence_result.is_none());
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("No AWAY frames collected")));
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("Skipping presence model")));
    let backend = backend_state.lock().expect("backend");
    assert_eq!(backend.cv_calls.len(), 1);
    assert_eq!(backend.cv_calls[0].features, vec![FeatureId::GauFeatures]);
    assert!(!storage_state
        .lock()
        .expect("storage")
        .calls
        .iter()
        .any(|call| call == "save_presence"));
}

#[test]
fn missing_posture_class_trains_presence_only_and_warns_skipping_posture() {
    // Oracle training-worker.ts:333-347: GOOD present but no BAD -> posture skipped with the
    // "Skipping posture model" warning, presence trains, overall success stays true.
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut worker = make_worker(
        storage_with_dataset(
            storage_state.clone(),
            dataset_of(&[FrameLabel::Good, FrameLabel::Away]),
        ),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.success);
    assert!(models.presence.is_some());
    assert!(models.posture.is_none());
    assert!(result.posture_result.is_none());
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.contains("Skipping posture model")));
    let backend = backend_state.lock().expect("backend");
    assert_eq!(backend.cv_calls.len(), 1);
    assert_eq!(
        backend.cv_calls[0].features,
        vec![FeatureId::RtmDetEngineered]
    );
    assert!(!storage_state
        .lock()
        .expect("storage")
        .calls
        .iter()
        .any(|call| call == "save_posture"));
}

#[test]
fn no_trainable_class_returns_insufficient_data_without_backend_training_or_saves() {
    // Oracle training-worker.ts:349-362: when neither model can train, the worker returns a
    // failed result (success=false) carrying the "Insufficient data" message with null models
    // and never runs CV/fit or persists anything.
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut worker = make_worker(
        storage_with_dataset(
            storage_state.clone(),
            dataset_of(&[FrameLabel::Unused, FrameLabel::Unused]),
        ),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(!result.success);
    assert!(models.posture.is_none() && models.presence.is_none());
    assert!(result.posture_result.is_none() && result.presence_result.is_none());
    assert!(result
        .errors
        .iter()
        .any(|error| error.contains("Insufficient data")));
    let backend = backend_state.lock().expect("backend");
    assert!(backend.cv_calls.is_empty());
    assert!(backend.fit_calls.is_empty());
    let calls = storage_state.lock().expect("storage").calls.clone();
    assert!(!calls.iter().any(|call| call == "save_posture"));
    assert!(!calls.iter().any(|call| call == "save_presence"));
}

#[test]
fn frames_missing_features_fail_the_affected_role_before_backend_training() {
    // Oracle 'should error when frames missing features' (training-worker.test.ts:539-552):
    // a frame whose features map is empty trips the missing-features guard in `train_one`
    // (training_worker.rs:540-551) BEFORE any backend call. Such a frame still passes
    // `validate_posture_frame` (validation.rs only iterates present features), so the guard is
    // the sole defense. Build a GOOD frame with empty features and no AWAY frames (only posture
    // trains): assert the posture role errors with 'missing features' and the backend never sees
    // the posture feature set. A regression that weakened the guard would let a featureless frame
    // reach the backend and fail this test.
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut empty_good = frame("good", FrameLabel::Good);
    empty_good.features = BTreeMap::new();
    let ds = PostureDataset {
        frames: vec![empty_good, frame("bad", FrameLabel::Bad)],
        version: 1,
        last_modified: 1.0,
    };
    let mut worker = make_worker(
        storage_with_dataset(storage_state.clone(), ds),
        TestBackend {
            state: backend_state.clone(),
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(!result.success);
    assert!(models.posture.is_none());
    assert!(result.posture_result.is_none());
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.contains("missing features")),
        "expected a missing-features error: {:?}",
        result.errors
    );
    let backend = backend_state.lock().expect("backend");
    assert!(
        !backend
            .cv_calls
            .iter()
            .any(|call| call.features.contains(&FeatureId::GauFeatures)),
        "posture feature must never reach cross_validate: {:?}",
        backend.cv_calls
    );
    assert!(
        !backend
            .fit_calls
            .iter()
            .any(|call| call.contains(&FeatureId::GauFeatures)),
        "posture feature must never reach fit: {:?}",
        backend.fit_calls
    );
    assert!(!storage_state
        .lock()
        .expect("storage")
        .calls
        .iter()
        .any(|call| call == "save_posture"));
}

/// Backend that mimics the real one clamping PCA to the selected features' rank: it
/// returns a fitted model whose serialized config records the EFFECTIVE component
/// count (7) while the requested count in settings stays 30.
struct ClampingPcaBackend;

impl TrainingBackend for ClampingPcaBackend {
    fn calibrate_feature_bins(
        &mut self,
        _samples: &[FeatureContainer],
        _log_engineered: bool,
        _logger: &dyn TrainingLogger,
    ) {
    }
    fn cross_validate(
        &mut self,
        _config: &FeatureExtractorConfig,
        _classifier: &ClassifierConfig,
        _frames: &[PostureFrame],
        _labels: &[i32],
        _cv_folds: usize,
    ) -> Result<Option<TrainingMetrics>, String> {
        Ok(Some(metrics()))
    }
    fn fit(
        &mut self,
        config: &FeatureExtractorConfig,
        _classifier: &ClassifierConfig,
        _frames: &[PostureFrame],
        _labels: &[i32],
    ) -> Result<SerializedModel, String> {
        let dims = 7;
        Ok(SerializedModel {
            feature_extractor: SerializedFeatureExtractor {
                feature_types: config
                    .feature_types
                    .iter()
                    .map(|feature| feature.as_str().to_owned())
                    .collect(),
                normalization_mode: slouch_ml::ported::types::NormalizationMode::None,
                dim_reduction_config: slouch_ml::ported::types::DimensionalityReductionConfig {
                    method: slouch_ml::ported::types::DimensionalityReductionMethod::Pca,
                    components: dims,
                },
                concatenated_dimensions: dims,
                normalization_mean: None,
                normalization_std: None,
                dim_reduction_transformer: None,
            },
            classifier: SerializedClassifier {
                classifier_id: "gaussian_nb".into(),
                state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                    class_means: [vec![0.0; dims], vec![1.0; dims]],
                    class_variances: [vec![1.0; dims], vec![1.0; dims]],
                    class_priors: [0.5, 0.5],
                    epsilon: 1e-9,
                }),
            },
            trained_at: 0.0,
            version: 0.0,
        })
    }
    fn release_training_buffers(&mut self) {}
}

#[test]
fn pca_component_clamp_surfaces_as_a_posture_warning() {
    // torso_invariant (7 dims) with PCA components 30 clamps to 7. The pipeline must
    // append a visible degradation warning to the posture result rather than failing.
    let mut pca_settings = settings();
    pca_settings.dim_reduction_config = DimensionalityReductionConfig {
        method: DimensionalityReductionMethod::Pca,
        components: 30,
    };
    pca_settings.posture_feature_types = vec![FeatureId::TorsoInvariant];
    let storage = TestStorage {
        state: Arc::new(Mutex::new(StorageState::default())),
        datasets: VecDeque::from([Ok(Some(dataset_of(&[FrameLabel::Good, FrameLabel::Bad])))]),
        settings: Ok(Some(pca_settings)),
        reservoir: Ok(vec![]),
        fail_posture_save: false,
        fail_presence_save: false,
    };
    let mut worker = TrainingWorker::with_clock(
        storage,
        ClampingPcaBackend,
        TestLogger::default(),
        FixedClock(1234.0),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    let posture = result.posture_result.as_ref().expect("posture result");
    assert!(
        posture.warnings.iter().any(|warning| warning
            == "PCA components reduced 30\u{2192}7 (selected features have only 7 dimensions)"),
        "expected PCA clamp warning, got {:?}",
        posture.warnings
    );
}

/// A reservoir sample carrying the pooled RTMPose/RTMDet stored features (and keypoints)
/// but NOT the GPU-only `nlf_depth` stored feature — exactly the shape the PCA guard
/// must detect and drop.
fn nlf_reservoir_sample() -> ReservoirSample {
    ReservoirSample {
        backbone_avg: vec![0.0; FeatureId::BackboneFeatures.metadata().dimensions],
        backbone_max: vec![0.0; FeatureId::BackboneFeaturesMax.metadata().dimensions],
        backbone_std: vec![0.0; FeatureId::BackboneFeaturesStd.metadata().dimensions],
        gau_avg: vec![0.0; FeatureId::GauFeatures.metadata().dimensions],
        gau_max: vec![0.0; FeatureId::GauFeaturesMax.metadata().dimensions],
        gau_std: vec![0.0; FeatureId::GauFeaturesStd.metadata().dimensions],
        rtmdet: vec![0.0; FeatureId::RtmDetExtracted.metadata().dimensions],
        keypoints: vec![Keypoint::new(0.5, 0.5, 0.5); 17],
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

fn nlf_frame(id: &str, label: FrameLabel) -> PostureFrame {
    let mut posture_frame = frame(id, label);
    posture_frame.features = BTreeMap::from([
        (
            FeatureId::NlfDepth,
            vec![0.1; FeatureId::NlfDepth.metadata().dimensions],
        ),
        (
            FeatureId::BackboneFeatures,
            vec![0.2; FeatureId::BackboneFeatures.metadata().dimensions],
        ),
    ]);
    posture_frame
}

#[test]
fn pca_drops_reservoir_for_a_posture_feature_absent_from_reservoir_samples() {
    // Selecting [nlf_depth, backbone_features] + PCA + a non-empty reservoir: nlf_depth is a
    // GPU-only stored feature the reservoir never carries, so the guard must drop the reservoir
    // (rather than hard-error in concatenate_features) and surface a warning. backbone_features
    // IS in the reservoir, so it must NOT trigger a warning.
    let storage_state = Arc::new(Mutex::new(StorageState::default()));
    let backend_state = Arc::new(Mutex::new(BackendState::default()));
    let mut pca_settings = settings();
    pca_settings.dim_reduction_config = DimensionalityReductionConfig {
        method: DimensionalityReductionMethod::Pca,
        components: 5,
    };
    pca_settings.posture_feature_types = vec![FeatureId::NlfDepth, FeatureId::BackboneFeatures];
    let storage = TestStorage {
        state: storage_state,
        datasets: VecDeque::from([Ok(Some(PostureDataset {
            frames: vec![
                nlf_frame("good", FrameLabel::Good),
                nlf_frame("bad", FrameLabel::Bad),
            ],
            version: 1,
            last_modified: 1.0,
        }))]),
        settings: Ok(Some(pca_settings)),
        reservoir: Ok(vec![nlf_reservoir_sample()]),
        fail_posture_save: false,
        fail_presence_save: false,
    };
    let mut worker = make_worker(
        storage,
        TestBackend {
            state: backend_state,
            fail_cv_for: None,
            fail_fit_for: None,
        },
        TestLogger::default(),
    );
    let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
    let [TrainingWorkerResponse::Result { result, models }] = &response[..] else {
        panic!("expected result: {response:?}")
    };
    assert!(result.success);
    assert!(models.posture.is_some());
    assert!(
        result.warnings.iter().any(|warning| warning
            == "PCA fitted on labeled frames only: feature 'nlf_depth' is absent from reservoir samples."),
        "expected nlf_depth reservoir-drop warning, got {:?}",
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|warning| warning.contains("'backbone_features' is absent")),
        "backbone_features is present in the reservoir and must not warn: {:?}",
        result.warnings
    );
}
