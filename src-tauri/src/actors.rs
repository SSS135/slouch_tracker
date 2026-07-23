use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::{query, Buffer, Camera};
use tauri::ipc::Channel;

use crate::errors::ApiError;
use slouch_domain::ported::messages::schemas::{
    ImageData, InferenceWorkerMessage, ProcessPayload, TrainPayload, TrainingWorkerMessage,
};
use slouch_domain::{
    BoundingBox, CameraSettings, ClassificationResult, ExpandedBbox, FrameLabel, InferenceResult,
    Keypoint, PostureDataset, PostureFrame, Thumbnail, TrainingSettings,
};
use slouch_ml::ported::training_worker::{
    FeatureContainer, FeatureExtractorConfig, NoopLogger as TrainingNoopLogger,
    ReservoirSample as TrainingReservoirSample, TrainingBackend, TrainingLogger, TrainingStorage,
    TrainingWorker, TrainingWorkerResponse,
};
use slouch_ml::ported::types::{
    DimensionalityReductionConfig as MlDimensionalityReductionConfig,
    DimensionalityReductionMethod as MlDimensionalityReductionMethod,
    NormalizationMode as MlNormalizationMode, SerializedModel,
};
use slouch_ml::ported::{
    classifier_registry, evaluation,
    feature_extractor::{FeatureExtractor, FeatureExtractorConfig as MlFeatureExtractorConfig},
    model::Model,
};
use slouch_store::ported::{feature_reservoir::ReservoirSample, storage::DatasetStorage};
use slouch_vision::ported::inference_worker::{
    ClassificationInput, ClassifierModel, InferenceWorker, ModelFactory, NativeInferenceResult,
    WorkerLogger, WorkerResponse,
};
use slouch_vision::NativePreprocessor;

const ACTOR_WAIT: Duration = Duration::from_secs(30);
const SHUTDOWN_WAIT: Duration = Duration::from_secs(5);
// Control-plane sends (model publish) must reach the actor even when a
// background detection momentarily occupies its single mailbox slot. A detection
// round-trip is ~130 ms, so briefly retrying the enqueue drains the slot and lets
// the publish through instead of failing training as "busy"; the ceiling keeps it
// from ever hanging if the actor is wedged.
const PUBLISH_ENQUEUE_WAIT: Duration = Duration::from_secs(5);

// The one-time inference token cache lives in its own source file so the
// ipc_security integration suite can compile the exact same code via #[path]
// inclusion without pulling in the whole actor module.
#[path = "inference_cache.rs"]
mod inference_cache;
use self::inference_cache::InferenceCache;

// pub(crate) visibility on the actor internals below exists for the
// actor_contracts integration suite, which includes this source file at its
// test-crate root and constructs actors around hand-made mailboxes.
pub(crate) struct ActorHealth {
    accepting: AtomicBool,
    alive: AtomicBool,
}

impl ActorHealth {
    pub(crate) fn new() -> Self {
        Self {
            accepting: AtomicBool::new(true),
            alive: AtomicBool::new(true),
        }
    }

    pub(crate) fn is_healthy(&self) -> bool {
        self.accepting.load(Ordering::Acquire) && self.alive.load(Ordering::Acquire)
    }

    fn fail(&self) {
        self.accepting.store(false, Ordering::Release);
        self.alive.store(false, Ordering::Release);
    }
}

pub(crate) struct ActorJoin {
    pub(crate) handle: JoinHandle<()>,
    pub(crate) done: Receiver<()>,
}

#[derive(Debug)]
pub(crate) enum InferenceCommand {
    Message {
        message: InferenceWorkerMessage,
        reply: Sender<Result<Vec<WorkerResponse>, ApiError>>,
    },
    PublishModelPair {
        models: Box<(
            Option<slouch_domain::ported::messages::schemas::SerializedModel>,
            Option<slouch_domain::ported::messages::schemas::SerializedModel>,
        )>,
        reply: Sender<Result<(), ApiError>>,
    },
    CacheResult {
        request_id: u64,
        result: NativeInferenceResult,
        reply: Sender<Result<u64, ApiError>>,
    },
    CheckoutResult {
        token: u64,
        request_id: u64,
        reply: Sender<Result<NativeInferenceResult, ApiError>>,
    },
    RestoreResult {
        token: u64,
        request_id: u64,
        result: NativeInferenceResult,
        reply: Sender<Result<(), ApiError>>,
    },
    CommitResult {
        token: u64,
        request_id: u64,
        reply: Sender<Result<(), ApiError>>,
    },
    ClearCache {
        reply: Sender<Result<(), ApiError>>,
    },
    Shutdown,
    // Test-only seam: a command whose handler panics inside the actor loop so
    // the catch_unwind isolation path can be exercised deterministically.
    #[cfg(test)]
    CrashForTests,
}

pub struct InferenceActor {
    pub(crate) sender: SyncSender<InferenceCommand>,
    pub(crate) health: Arc<ActorHealth>,
    pub(crate) join: Mutex<Option<ActorJoin>>,
}

impl InferenceActor {
    pub fn start(storage: Arc<DatasetStorage>) -> Result<Self, ApiError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let health = Arc::new(ActorHealth::new());
        let child_health = health.clone();
        let (done_sender, done) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("slouch-inference".to_owned())
            .spawn(move || {
                if catch_unwind(AssertUnwindSafe(|| inference_loop(receiver, storage))).is_err() {
                    log::error!(target: "inference", "inference actor panicked and was stopped");
                }
                child_health.fail();
                let _ = done_sender.send(());
            })
            .map_err(|error| {
                ApiError::Internal(format!("failed to start inference actor: {error}"))
            })?;
        Ok(Self {
            sender,
            health,
            join: Mutex::new(Some(ActorJoin { handle, done })),
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.health.is_healthy()
    }

    fn enqueue(&self, command: InferenceCommand) -> Result<(), ApiError> {
        if !self.health.is_healthy() {
            return Err(ApiError::NotReady("inference actor is stopped".into()));
        }
        self.sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => ApiError::Busy("inference actor is busy".into()),
            TrySendError::Disconnected(_) => {
                self.health.fail();
                ApiError::NotReady("inference actor is stopped".into())
            }
        })
    }

    // Reliable variant of `enqueue` for control-plane commands (model publish).
    // The best-effort `enqueue` is correct for the droppable detection data path,
    // but a model swap must not fail just because a background detection holds the
    // single mailbox slot for the moment. This waits for the slot to drain (one
    // inference round-trip), re-checking health each retry, and only reports Busy
    // if the actor stays saturated past the ceiling.
    fn enqueue_reliable(&self, command: InferenceCommand) -> Result<(), ApiError> {
        let deadline = Instant::now() + PUBLISH_ENQUEUE_WAIT;
        let mut command = command;
        loop {
            if !self.health.is_healthy() {
                return Err(ApiError::NotReady("inference actor is stopped".into()));
            }
            match self.sender.try_send(command) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    if Instant::now() >= deadline {
                        return Err(ApiError::Busy("inference actor is busy".into()));
                    }
                    command = returned;
                    thread::sleep(Duration::from_millis(5));
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.health.fail();
                    return Err(ApiError::NotReady("inference actor is stopped".into()));
                }
            }
        }
    }

    pub fn send(&self, message: InferenceWorkerMessage) -> Result<Vec<WorkerResponse>, ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::Message { message, reply })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn send_frame(
        &self,
        message: InferenceWorkerMessage,
    ) -> Result<Vec<WorkerResponse>, ApiError> {
        self.send(message)
    }

    pub fn publish_model_pair(
        &self,
        posture: Option<slouch_domain::ported::messages::schemas::SerializedModel>,
        presence: Option<slouch_domain::ported::messages::schemas::SerializedModel>,
    ) -> Result<(), ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue_reliable(InferenceCommand::PublishModelPair {
            models: Box::new((posture, presence)),
            reply,
        })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn cache_result(
        &self,
        request_id: u64,
        result: NativeInferenceResult,
    ) -> Result<u64, ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::CacheResult {
            request_id,
            result,
            reply,
        })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn checkout_result(
        &self,
        token: u64,
        request_id: u64,
    ) -> Result<NativeInferenceResult, ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::CheckoutResult {
            token,
            request_id,
            reply,
        })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn restore_result(
        &self,
        token: u64,
        request_id: u64,
        result: NativeInferenceResult,
    ) -> Result<(), ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::RestoreResult {
            token,
            request_id,
            result,
            reply,
        })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn commit_result(&self, token: u64, request_id: u64) -> Result<(), ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::CommitResult {
            token,
            request_id,
            reply,
        })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn clear_cache(&self) -> Result<(), ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(InferenceCommand::ClearCache { reply })?;
        receive_inference_reply(receiver, &self.health)
    }

    pub fn shutdown_until(&self, deadline: Instant) {
        self.health.accepting.store(false, Ordering::Release);
        let mut command = InferenceCommand::Shutdown;
        loop {
            match self.sender.try_send(command) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => break,
                Err(TrySendError::Full(returned)) if Instant::now() < deadline => {
                    command = returned;
                    thread::sleep(Duration::from_millis(5));
                }
                Err(TrySendError::Full(_)) => break,
            }
        }
        if let Ok(mut join) = self.join.lock() {
            if let Some(actor_join) = join.take() {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if actor_join.done.recv_timeout(remaining).is_ok() {
                    let _ = actor_join.handle.join();
                }
            }
        }
        self.health.fail();
    }
}

fn receive_inference_reply<T>(
    receiver: Receiver<Result<T, ApiError>>,
    health: &ActorHealth,
) -> Result<T, ApiError> {
    match receiver.recv_timeout(ACTOR_WAIT) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => {
            health.fail();
            Err(ApiError::NotReady("inference actor timed out".into()))
        }
        Err(RecvTimeoutError::Disconnected) => {
            health.fail();
            Err(ApiError::NotReady(
                "inference actor stopped before replying".into(),
            ))
        }
    }
}

fn inference_loop(receiver: Receiver<InferenceCommand>, storage: Arc<DatasetStorage>) {
    let mut worker =
        InferenceWorker::with_logger(NativeModelFactory, NativeWorkerLogger::default());
    let mut cache = InferenceCache::new();
    while let Ok(command) = receiver.recv() {
        match command {
            InferenceCommand::Message { message, reply } => {
                let result = worker
                    .handle_message(message)
                    .into_iter()
                    .map(|response| match response {
                        WorkerResponse::Error { error, .. } => Err(ApiError::Inference(error)),
                        response => Ok(response),
                    })
                    .collect::<Result<Vec<_>, _>>();
                if let Ok(responses) = &result {
                    for response in responses {
                        if let WorkerResponse::Result { result, .. } = response {
                            if let Some(sample) = reservoir_sample_from_inference(result) {
                                match unix_time_ms().and_then(|time| {
                                    storage
                                        .sample_reservoir(&sample, time)
                                        .map_err(|error| error.to_string())
                                }) {
                                    Ok(_) => {}
                                    Err(error) => {
                                        log::warn!(target: "storage", "inference reservoir sample was not persisted: {error}")
                                    }
                                }
                            }
                        }
                    }
                }
                let _ = reply.send(result);
            }
            InferenceCommand::PublishModelPair { models, reply } => {
                let (posture, presence) = *models;
                let result = worker
                    .publish_model_pair(posture, presence)
                    .map_err(|error| ApiError::Inference(error.to_string()));
                let _ = reply.send(result);
            }
            InferenceCommand::CacheResult {
                request_id,
                result,
                reply,
            } => {
                let _ = reply.send(cache.insert(request_id, result));
            }
            InferenceCommand::CheckoutResult {
                token,
                request_id,
                reply,
            } => {
                let _ = reply.send(cache.checkout(token, request_id));
            }
            InferenceCommand::RestoreResult {
                token,
                request_id,
                result,
                reply,
            } => {
                let _ = reply.send(cache.restore(token, request_id, result));
            }
            InferenceCommand::CommitResult {
                token,
                request_id,
                reply,
            } => {
                let _ = reply.send(cache.commit(token, request_id));
            }
            InferenceCommand::ClearCache { reply } => {
                cache.clear();
                let _ = reply.send(Ok(()));
            }
            InferenceCommand::Shutdown => break,
            #[cfg(test)]
            InferenceCommand::CrashForTests => panic!("inference actor crash injected by a test"),
        }
    }
}

pub(crate) fn reservoir_sample_from_inference(
    result: &NativeInferenceResult,
) -> Option<ReservoirSample> {
    if !result.person_found {
        return None;
    }
    let keypoints = result.keypoints.clone()?;
    let bbox = result.bbox.as_ref()?.original;
    // Persist every stored feature the inference produced; computed features are
    // rederived from keypoints/bbox at training time, so they never enter the reservoir.
    let features: slouch_domain::FeatureMap = result
        .features
        .iter()
        .filter(|&(id, _)| !id.metadata().computed)
        .map(|(id, values)| (*id, values.clone()))
        .collect();
    if features.is_empty() {
        return None;
    }
    Some(ReservoirSample {
        features,
        keypoints,
        bbox,
    })
}

fn unix_time_ms() -> Result<i64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())
        .and_then(|duration| {
            i64::try_from(duration.as_millis()).map_err(|_| "system time exceeds i64".to_owned())
        })
}

#[derive(Debug)]
struct NativeWorkerLogger {
    level: AtomicU8,
}

impl Default for NativeWorkerLogger {
    fn default() -> Self {
        Self {
            level: AtomicU8::new(3),
        }
    }
}

impl NativeWorkerLogger {
    fn enabled(&self, level: u8) -> bool {
        self.level.load(Ordering::Relaxed) >= level
    }
}

impl WorkerLogger for NativeWorkerLogger {
    fn set_from_url_param(&self, log_param: &str) {
        let mut configured =
            if log_param.trim().is_empty() || log_param.eq_ignore_ascii_case("none") {
                0
            } else {
                3
            };
        for token in log_param
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            let (category, level) = token.split_once(':').unwrap_or(("inference", token));
            if !matches!(category, "inference" | "worker" | "detection") {
                continue;
            }
            configured = match level.to_ascii_lowercase().as_str() {
                "none" => 0,
                "error" => 1,
                "warn" => 2,
                "info" => 3,
                "debug" => 4,
                _ => configured,
            };
        }
        self.level.store(configured, Ordering::Relaxed);
    }

    fn is_debug_enabled(&self) -> bool {
        self.enabled(4)
    }

    fn debug(&self, message: &str) {
        if self.enabled(4) {
            log::debug!(target: "inference", "{message}");
        }
    }
    fn info(&self, message: &str) {
        if self.enabled(3) {
            log::info!(target: "inference", "{message}");
        }
    }
    fn warn(&self, message: &str) {
        if self.enabled(2) {
            log::warn!(target: "inference", "{message}");
        }
    }
    fn error(&self, message: &str) {
        if self.enabled(1) {
            log::error!(target: "inference", "{message}");
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeModelFactory;

impl ModelFactory for NativeModelFactory {
    fn load(
        &self,
        serialized: slouch_domain::ported::messages::schemas::SerializedModel,
    ) -> Result<Box<dyn ClassifierModel>, String> {
        let value = serde_json::to_value(serialized).map_err(|error| error.to_string())?;
        let model: SerializedModel =
            serde_json::from_value(value).map_err(|error| error.to_string())?;
        let model =
            Model::<InferenceResult>::from_json(model).map_err(|error| error.to_string())?;
        Ok(Box::new(NativeClassifierModel { model: Some(model) }))
    }
}

struct NativeClassifierModel {
    model: Option<Model<InferenceResult>>,
}

impl ClassifierModel for NativeClassifierModel {
    fn predict(&mut self, input: &ClassificationInput<'_>) -> Result<f64, String> {
        // Presence models are intentionally invoked for no-person frames. A
        // zero-area box preserves the feature extractor's source behavior:
        // RTMDet engineered features become the all-zero vector while stored
        // RTMDet tensors remain available directly from `features`.
        let bbox = input.bbox.copied().unwrap_or_else(|| {
            let empty = BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.0,
                y2: 0.0,
                score: 0.0,
                width: 0.0,
                height: 0.0,
            };
            ExpandedBbox {
                original: empty,
                expanded: empty,
            }
        });
        let source = InferenceResult {
            features: input.features.clone(),
            keypoints: input.keypoints.unwrap_or_default().to_vec(),
            bbox,
            classification: None,
        };
        self.model
            .as_ref()
            .ok_or_else(|| "classifier model is disposed".to_owned())?
            .predict(&source)
            .map_err(|error| error.to_string())
    }

    fn dispose(&mut self) {
        if let Some(model) = self.model.as_mut() {
            model.dispose();
        }
        self.model = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum TrainingStage {
    Processing,
    Evaluating,
    Deploying,
}

#[derive(Debug, Clone, serde::Serialize, specta::Type)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum TrainingEvent {
    Started {
        #[serde(rename = "jobId")]
        #[specta(type = specta_typescript::Number)]
        job_id: u64,
        #[specta(type = specta_typescript::Number)]
        sequence: u64,
    },
    Progress {
        #[serde(rename = "jobId")]
        #[specta(type = specta_typescript::Number)]
        job_id: u64,
        #[specta(type = specta_typescript::Number)]
        sequence: u64,
        stage: TrainingStage,
        progress: u8,
    },
    Completed {
        #[serde(rename = "jobId")]
        #[specta(type = specta_typescript::Number)]
        job_id: u64,
        #[specta(type = specta_typescript::Number)]
        sequence: u64,
        result: Box<slouch_ml::ported::training_worker::TrainingResultResponse>,
    },
    Failed {
        #[serde(rename = "jobId")]
        #[specta(type = specta_typescript::Number)]
        job_id: u64,
        #[specta(type = specta_typescript::Number)]
        sequence: u64,
        error: String,
    },
    Cancelled {
        #[serde(rename = "jobId")]
        #[specta(type = specta_typescript::Number)]
        job_id: u64,
        #[specta(type = specta_typescript::Number)]
        sequence: u64,
    },
}

#[cfg(test)]
impl TrainingEvent {
    fn job_id(&self) -> u64 {
        match self {
            Self::Started { job_id, .. }
            | Self::Progress { job_id, .. }
            | Self::Completed { job_id, .. }
            | Self::Failed { job_id, .. }
            | Self::Cancelled { job_id, .. } => *job_id,
        }
    }

    fn sequence(&self) -> u64 {
        match self {
            Self::Started { sequence, .. }
            | Self::Progress { sequence, .. }
            | Self::Completed { sequence, .. }
            | Self::Failed { sequence, .. }
            | Self::Cancelled { sequence, .. } => *sequence,
        }
    }

    fn progress(&self) -> Option<u8> {
        match self {
            Self::Progress { progress, .. } => Some(*progress),
            _ => None,
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled { .. }
        )
    }
}

#[cfg(test)]
pub(crate) fn validate_training_event_stream(events: &[TrainingEvent]) -> Result<(), String> {
    let first = events
        .first()
        .ok_or_else(|| "training event stream is empty".to_owned())?;
    if !matches!(first, TrainingEvent::Started { sequence: 0, .. }) {
        return Err("training event stream must start at sequence zero".into());
    }
    let job_id = first.job_id();
    let mut previous_sequence = None;
    let mut previous_progress = 0;
    let mut terminal_count = 0;
    for event in events {
        if event.job_id() != job_id {
            return Err("training event stream contains a different job ID".into());
        }
        if previous_sequence.is_some_and(|previous| event.sequence() != previous + 1) {
            return Err("training event sequence is not contiguous".into());
        }
        if terminal_count > 0 {
            return Err("training event occurred after the terminal event".into());
        }
        if let Some(progress) = event.progress() {
            if progress < previous_progress || progress >= 100 {
                return Err("training progress is not monotonic within 0..100".into());
            }
            previous_progress = progress;
        }
        terminal_count += usize::from(event.is_terminal());
        previous_sequence = Some(event.sequence());
    }
    if terminal_count != 1 {
        return Err("training event stream must contain exactly one terminal event".into());
    }
    Ok(())
}

type RunningTraining = (
    u64,
    Receiver<Result<TrainingWorkerResponse, String>>,
    Arc<AtomicBool>,
    Sender<Result<TrainingWorkerResponse, ApiError>>,
    Option<Channel<TrainingEvent>>,
    JoinHandle<()>,
);

enum TrainingCommand {
    Train {
        job_id: u64,
        do_cv: bool,
        events: Option<Channel<TrainingEvent>>,
        reply: Sender<Result<TrainingWorkerResponse, ApiError>>,
    },
    Cancel {
        reply: Sender<Result<(), ApiError>>,
    },
    Shutdown,
}

pub struct TrainingActor {
    sender: SyncSender<TrainingCommand>,
    health: Arc<ActorHealth>,
    join: Mutex<Option<ActorJoin>>,
}

impl TrainingActor {
    pub fn start(storage: Arc<DatasetStorage>) -> Result<Self, ApiError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let health = Arc::new(ActorHealth::new());
        let child_health = health.clone();
        let (done_sender, done) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("slouch-training".to_owned())
            .spawn(move || {
                if catch_unwind(AssertUnwindSafe(|| training_loop(receiver, storage))).is_err() {
                    log::error!(target: "training", "training actor panicked and was stopped");
                }
                child_health.fail();
                let _ = done_sender.send(());
            })
            .map_err(|error| {
                ApiError::Internal(format!("failed to start training actor: {error}"))
            })?;
        Ok(Self {
            sender,
            health,
            join: Mutex::new(Some(ActorJoin { handle, done })),
        })
    }

    pub fn is_healthy(&self) -> bool {
        self.health.is_healthy()
    }

    fn enqueue(&self, command: TrainingCommand) -> Result<(), ApiError> {
        if !self.health.is_healthy() {
            return Err(ApiError::NotReady("training actor is stopped".into()));
        }
        self.sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => ApiError::Busy("training actor is busy".into()),
            TrySendError::Disconnected(_) => {
                self.health.fail();
                ApiError::NotReady("training actor is stopped".into())
            }
        })
    }

    pub fn train(
        &self,
        job_id: u64,
        do_cv: bool,
        events: Option<Channel<TrainingEvent>>,
    ) -> Result<TrainingWorkerResponse, ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(TrainingCommand::Train {
            job_id,
            do_cv,
            events,
            reply,
        })?;
        match receiver.recv_timeout(Duration::from_secs(30 * 60)) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                self.health.fail();
                Err(ApiError::NotReady("training actor timed out".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.health.fail();
                Err(ApiError::NotReady("training actor stopped".into()))
            }
        }
    }

    pub fn cancel(&self) -> Result<(), ApiError> {
        let (reply, receiver) = mpsc::channel();
        self.enqueue(TrainingCommand::Cancel { reply })?;
        match receiver.recv_timeout(SHUTDOWN_WAIT) {
            Ok(result) => result,
            Err(_) => {
                self.health.fail();
                Err(ApiError::NotReady("training cancellation timed out".into()))
            }
        }
    }

    pub fn shutdown(&self) {
        self.shutdown_until(Instant::now() + SHUTDOWN_WAIT);
    }

    pub fn shutdown_until(&self, deadline: Instant) {
        self.health.accepting.store(false, Ordering::Release);
        let mut command = TrainingCommand::Shutdown;
        loop {
            match self.sender.try_send(command) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => break,
                Err(TrySendError::Full(returned)) if Instant::now() < deadline => {
                    command = returned;
                    thread::sleep(Duration::from_millis(5));
                }
                Err(TrySendError::Full(_)) => break,
            }
        }
        if let Ok(mut join) = self.join.lock() {
            if let Some(actor_join) = join.take() {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if actor_join.done.recv_timeout(remaining).is_ok() {
                    let _ = actor_join.handle.join();
                }
            }
        }
        self.health.fail();
    }
}

fn training_loop(receiver: Receiver<TrainingCommand>, storage: Arc<DatasetStorage>) {
    let mut running: Option<RunningTraining> = None;
    loop {
        if let Some((job_id, result_receiver, cancel, reply, events, job)) = running.take() {
            match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(TrainingCommand::Cancel {
                    reply: cancel_reply,
                }) => {
                    cancel.store(true, Ordering::Release);
                    let _ = cancel_reply.send(Ok(()));
                    running = Some((job_id, result_receiver, cancel, reply, events, job));
                    continue;
                }
                Ok(TrainingCommand::Train {
                    reply: busy_reply, ..
                }) => {
                    let _ =
                        busy_reply.send(Err(ApiError::Busy("training already in progress".into())));
                    running = Some((job_id, result_receiver, cancel, reply, events, job));
                    continue;
                }
                Ok(TrainingCommand::Shutdown) => {
                    cancel.store(true, Ordering::Release);
                    if result_receiver.recv_timeout(SHUTDOWN_WAIT).is_ok() {
                        let _ = job.join();
                    }
                    let _ = reply.send(Err(ApiError::Cancelled(
                        "training stopped during shutdown".into(),
                    )));
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    match result_receiver.try_recv() {
                        Ok(result) => {
                            finish_training(
                                job_id,
                                result,
                                cancel.load(Ordering::Acquire),
                                reply,
                                events,
                            );
                            let _ = job.join();
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            running = Some((job_id, result_receiver, cancel, reply, events, job));
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            let _ = reply.send(Err(ApiError::Training(
                                "training worker disconnected".into(),
                            )));
                            let _ = job.join();
                        }
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    cancel.store(true, Ordering::Release);
                    if result_receiver.recv_timeout(SHUTDOWN_WAIT).is_ok() {
                        let _ = job.join();
                    }
                    let _ = reply.send(Err(ApiError::Cancelled(
                        "training stopped because the actor was dropped".into(),
                    )));
                    break;
                }
            }
        }

        match receiver.recv() {
            Ok(TrainingCommand::Train {
                job_id,
                do_cv,
                events,
                reply,
            }) => {
                if let Some(channel) = &events {
                    let _ = channel.send(TrainingEvent::Started {
                        job_id,
                        sequence: 0,
                    });
                    let _ = channel.send(TrainingEvent::Progress {
                        job_id,
                        sequence: 1,
                        stage: TrainingStage::Processing,
                        progress: 5,
                    });
                }
                let cancel = Arc::new(AtomicBool::new(false));
                let child_cancel = cancel.clone();
                let child_storage = storage.clone();
                let (result_sender, result_receiver) =
                    mpsc::channel::<Result<TrainingWorkerResponse, String>>();
                let job = thread::Builder::new()
                    .name("slouch-training-job".to_owned())
                    .spawn(move || {
                        let worker = TrainingWorker::with_logger(
                            NativeTrainingStorage::new(child_storage),
                            NativeTrainingBackend::new(child_cancel),
                            TrainingNoopLogger,
                        );
                        let mut worker = worker;
                        let response = worker.handle_message(TrainingWorkerMessage::Train {
                            payload: Some(TrainPayload { do_cv: Some(do_cv) }),
                        });
                        let result = response.into_iter().next().map_or_else(
                            || Err("training worker returned no response".to_owned()),
                            Ok,
                        );
                        let _ = result_sender.send(result);
                    });
                match job {
                    Ok(job) => {
                        running = Some((job_id, result_receiver, cancel, reply, events, job));
                    }
                    Err(error) => {
                        let message = format!("failed to start training job: {error}");
                        let _ = reply.send(Err(ApiError::Internal(message)));
                    }
                }
            }
            Ok(TrainingCommand::Cancel { reply }) => {
                let _ = reply.send(Err(ApiError::NotReady("no training is running".into())));
            }
            Ok(TrainingCommand::Shutdown) | Err(_) => break,
        }
    }
}

fn finish_training(
    job_id: u64,
    result: Result<TrainingWorkerResponse, String>,
    cancelled: bool,
    reply: Sender<Result<TrainingWorkerResponse, ApiError>>,
    events: Option<Channel<TrainingEvent>>,
) {
    if cancelled {
        let _ = reply.send(Err(ApiError::Cancelled("training cancelled".into())));
        return;
    }
    match result {
        Ok(TrainingWorkerResponse::Result { result, models }) => {
            if let Some(channel) = events {
                let _ = channel.send(TrainingEvent::Progress {
                    job_id,
                    sequence: 2,
                    stage: TrainingStage::Evaluating,
                    progress: 85,
                });
            }
            let response = TrainingWorkerResponse::Result { result, models };
            let _ = reply.send(Ok(response));
        }
        Ok(TrainingWorkerResponse::Error { error, .. }) | Err(error) => {
            let _ = reply.send(Err(ApiError::Training(error)));
        }
    }
}

#[derive(Clone)]
pub(crate) struct NativeTrainingStorage {
    storage: Arc<DatasetStorage>,
}

impl NativeTrainingStorage {
    pub(crate) fn new(storage: Arc<DatasetStorage>) -> Self {
        Self { storage }
    }
}

impl TrainingStorage for NativeTrainingStorage {
    fn load_dataset(&mut self) -> Result<Option<PostureDataset>, String> {
        self.storage
            .load_dataset()
            .map(Some)
            .map_err(|error| error.to_string())
    }
    fn load_training_settings(&mut self) -> Result<Option<TrainingSettings>, String> {
        self.storage
            .get_training_settings()
            .map_err(|error| error.to_string())
    }
    fn load_reservoir_samples(&mut self) -> Result<Vec<TrainingReservoirSample>, String> {
        self.storage
            .load_reservoir_samples()
            .map(|samples| {
                samples
                    .into_iter()
                    .map(|sample| TrainingReservoirSample {
                        features: sample.features,
                        keypoints: sample.keypoints,
                        bbox: sample.bbox,
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
    fn save_posture_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
        Ok(())
    }
    fn save_presence_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
        Ok(())
    }
}

struct NativeTrainingBackend {
    cancelled: Arc<AtomicBool>,
}

impl NativeTrainingBackend {
    fn new(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }
    fn check_cancelled(&self) -> Result<(), String> {
        if self.cancelled.load(Ordering::Acquire) {
            Err("training cancelled".into())
        } else {
            Ok(())
        }
    }
}

impl TrainingBackend for NativeTrainingBackend {
    fn calibrate_feature_bins(
        &mut self,
        _samples: &[FeatureContainer],
        _log_engineered: bool,
        logger: &dyn TrainingLogger,
    ) {
        logger.info("[Training Worker] Native feature extractor owns bin fitting");
    }

    fn cross_validate(
        &mut self,
        config: &FeatureExtractorConfig,
        classifier_config: &slouch_domain::ClassifierConfig,
        frames: &[PostureFrame],
        labels: &[i32],
        cv_folds: usize,
    ) -> Result<Option<slouch_domain::TrainingMetrics>, String> {
        self.check_cancelled()?;
        let extractor = to_ml_extractor_config(config);
        let metrics = evaluation::cross_validate_fallible(
            &extractor,
            || {
                self.check_cancelled()?;
                classifier_registry::create_classifier(classifier_config)
                    .map_err(|error| error.to_string())
            },
            frames,
            labels,
            evaluation::CrossValidationOptions {
                cv_folds: Some(cv_folds),
                timestamps: Some(frames.iter().map(|frame| frame.timestamp).collect()),
                ..Default::default()
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(metrics.map(|metrics| slouch_domain::TrainingMetrics {
            cv_accuracy: metrics.cv_accuracy,
            cv_std: metrics.cv_std,
            mcc: metrics.mcc,
            f1_score: metrics.f1_score,
            confusion_matrix: metrics
                .confusion_matrix
                .into_iter()
                .map(|row| row.into_iter().map(|value| value as u64).collect())
                .collect(),
            fold_accuracies: metrics.fold_accuracies,
            balanced_accuracy: metrics.balanced_accuracy,
            accuracy_ci_low: metrics.accuracy_ci_low,
            accuracy_ci_high: metrics.accuracy_ci_high,
            worst_fold_accuracy: metrics.worst_fold_accuracy,
            cv_type: Some(match metrics.cv_type {
                evaluation::CvType::TemporalBlock => {
                    slouch_domain::CrossValidationType::TemporalBlock
                }
                evaluation::CvType::ShuffledStratified => {
                    slouch_domain::CrossValidationType::ShuffledStratified
                }
            }),
        }))
    }

    fn fit(
        &mut self,
        config: &FeatureExtractorConfig,
        classifier_config: &slouch_domain::ClassifierConfig,
        frames: &[PostureFrame],
        labels: &[i32],
    ) -> Result<SerializedModel, String> {
        self.check_cancelled()?;
        let extractor = FeatureExtractor::new(to_ml_extractor_config(config));
        let classifier = classifier_registry::create_classifier(classifier_config)
            .map_err(|error| error.to_string())?;
        let mut model = Model::new(extractor, classifier);
        model
            .fit(frames, labels)
            .map_err(|error| error.to_string())?;
        self.check_cancelled()?;
        model
            .to_serialized_model(now_millis(), 1.0)
            .map_err(|error| error.to_string())
    }

    fn release_training_buffers(&mut self) {}
}

fn to_ml_extractor_config(
    config: &FeatureExtractorConfig,
) -> MlFeatureExtractorConfig<PostureFrame> {
    MlFeatureExtractorConfig {
        feature_types: config.feature_types.clone(),
        normalization_mode: match config.normalization_mode {
            slouch_domain::NormalizationMode::None => MlNormalizationMode::None,
            slouch_domain::NormalizationMode::Layer => MlNormalizationMode::Layer,
            slouch_domain::NormalizationMode::ZScore => MlNormalizationMode::ZScore,
            slouch_domain::NormalizationMode::Calibrated => MlNormalizationMode::Calibrated,
        },
        dim_reduction_config: MlDimensionalityReductionConfig {
            method: match config.dim_reduction_config.method {
                slouch_domain::DimensionalityReductionMethod::RandomProjection => {
                    MlDimensionalityReductionMethod::RandomProjection
                }
                slouch_domain::DimensionalityReductionMethod::Pca => {
                    MlDimensionalityReductionMethod::Pca
                }
                slouch_domain::DimensionalityReductionMethod::None => {
                    MlDimensionalityReductionMethod::None
                }
            },
            components: config.dim_reduction_config.components,
        },
        unlabeled_samples: config
            .unlabeled_samples
            .iter()
            .enumerate()
            .map(feature_container_to_frame)
            .collect(),
    }
}

fn feature_container_to_frame((index, sample): (usize, &FeatureContainer)) -> PostureFrame {
    PostureFrame {
        id: format!("reservoir-{index}"),
        timestamp: index as f64,
        features: sample.features.clone(),
        thumbnail: Thumbnail {
            mime_type: "image/png".into(),
            bytes: vec![0],
        },
        keypoints: sample.keypoints.clone(),
        bbox: sample.bbox.unwrap_or(BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 1.0,
            width: 1.0,
            height: 1.0,
        }),
        label: FrameLabel::Unused,
    }
}

fn now_millis() -> f64 {
    // Whole integer milliseconds (oracle parity with Date.now()); a fractional
    // trained_at is rejected by model_format::encode_model during persistence.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |value| value.as_millis() as f64)
}

pub fn raw_inference_message(image: ImageData, request_id: u64) -> InferenceWorkerMessage {
    InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image,
            request_id,
        },
    }
}

pub fn initialize_message(rtmdet_path: PathBuf, nlf_path: PathBuf) -> InferenceWorkerMessage {
    InferenceWorkerMessage::Initialize {
        payload: slouch_domain::ported::messages::schemas::InitializePayload {
            rtmdet_path: rtmdet_path.to_string_lossy().into_owned(),
            nlf_path: nlf_path.to_string_lossy().into_owned(),
        },
    }
}

pub fn inference_result(response: WorkerResponse) -> Result<NativeInferenceResult, ApiError> {
    match response {
        WorkerResponse::Result { result, .. } => Ok(result),
        WorkerResponse::Error { error, .. } => Err(ApiError::Inference(error)),
        _ => Err(ApiError::Inference(
            "inference command returned an unexpected response".into(),
        )),
    }
}

// ======================== Native camera capture (CameraActor) ========================
//
// CameraActor owns the webcam on a dedicated "slouch-camera" thread, mirroring
// InferenceActor's mailbox/health/join lifecycle. It captures MJPEG, keeps the
// freshest frame in a drop-oldest cell for the preview (foreground only), and
// feeds the InferenceActor in-process at the configured detection cadence so
// save_capture's one-use token handshake keeps working. AppState owns it; the
// start/stop/list commands drive it, the `slouchcam` URI protocol serves the
// preview frame, and window-focus events switch it between foreground/background.

pub use camera::{CameraActor, CameraDeviceInfo, CameraMode, InferenceUiResult};

mod camera {
    use super::*;
    use std::collections::VecDeque;

    const MIN_INTERVAL_SECONDS: f64 = 0.05;
    // When settings are momentarily unreadable, back off briefly instead of hot-looping.
    const DETECTION_RETRY: Duration = Duration::from_millis(500);
    // Ring of the most recent encoded camera buffers kept for temporal smoothing.
    // Sized to the maximum smoothing_frames (validated 1..=10) so a detection can
    // always pull the configured number of *consecutive* frames: Foreground fills
    // the ring at capture rate; Background bursts to top it up before a detection.
    const FRAME_RING_CAPACITY: usize = 10;
    // A processed-view pull within this window counts as live demand: the capture
    // loop keeps the processed preview fresh at capture rate. ~2s absorbs the
    // unfocused ~2fps pump and brief scheduling gaps without ever latching on.
    const PROCESSED_DEMAND_WINDOW: Duration = Duration::from_secs(2);

    /// UI-facing inference summary pushed after each native detection. No pixels:
    /// only the one-use token, presence/posture verdict, and overlay geometry cross
    /// to the webview. Produced by the legacy `infer_frame` path and the CameraActor.
    #[derive(Debug, Clone, serde::Serialize, specta::Type)]
    #[serde(rename_all = "camelCase")]
    pub struct InferenceUiResult {
        #[specta(type = specta_typescript::Number)]
        pub request_id: u64,
        #[specta(type = specta_typescript::Number)]
        pub token: u64,
        pub person_found: bool,
        pub bbox: Option<ExpandedBbox>,
        pub keypoints: Option<Vec<Keypoint>>,
        pub classification: Option<ClassificationResult>,
    }

    /// Foreground streams frames for the smooth preview; Background powers detection
    /// only (preview cell left empty so the webview renders nothing).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CameraMode {
        Foreground,
        Background,
    }

    /// Coarse capture state the actor reports directly, off the command mailbox.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CameraStatus {
        Idle,
        Streaming(CameraMode),
        /// Generic, platform-agnostic device open / stream / disconnect / permission
        /// failure. Platform-specific permission deep-links are a later-wave concern.
        Error(String),
    }

    /// One enumerated capture device from `nokhwa::query`.
    #[derive(Debug, Clone, serde::Serialize, specta::Type)]
    #[serde(rename_all = "camelCase")]
    pub struct CameraDeviceInfo {
        pub index: String,
        pub name: String,
        pub description: String,
    }

    type ResultSink = Box<dyn Fn(InferenceUiResult) + Send>;

    /// State the capture thread publishes and callers read without the mailbox.
    struct CameraShared {
        latest_frame: Mutex<Option<Vec<u8>>>,
        // JPEG of the exact preprocessed RGBA the detector sees (post
        // CLAHE/blur/temporal smoothing), paired with the monotonic capture
        // sequence of the newest source frame it was built from. Two producers
        // write this cell — the capture-rate preview preprocessor (~30fps while the
        // processed view is watched) and the ~1fps dispatcher fallback — so the
        // sequence is the ordering key: a write only lands if it is strictly newer,
        // keeping the served frame monotonic in capture time (no stale flicker).
        processed_frame: Mutex<Option<(u64, Vec<u8>)>>,
        // Instant of the most recent `slouchcam …/processed` pull. The capture loop
        // treats a stamp within PROCESSED_DEMAND_WINDOW as live demand and drives
        // the processed-view preview at capture rate; stale/absent means zero extra
        // work. Touched ~30x/s from the protocol thread.
        processed_requested_at: Mutex<Option<Instant>>,
        latest_result: Mutex<Option<InferenceUiResult>>,
        status: Mutex<CameraStatus>,
        sink: Mutex<Option<ResultSink>>,
    }

    impl CameraShared {
        fn new() -> Self {
            Self {
                latest_frame: Mutex::new(None),
                processed_frame: Mutex::new(None),
                processed_requested_at: Mutex::new(None),
                latest_result: Mutex::new(None),
                status: Mutex::new(CameraStatus::Idle),
                sink: Mutex::new(None),
            }
        }

        /// Records that the webview just pulled the processed-view frame, marking
        /// the view as actively watched so the capture loop keeps it fresh.
        fn note_processed_request(&self) {
            if let Ok(mut guard) = self.processed_requested_at.lock() {
                *guard = Some(Instant::now());
            }
        }

        /// True when a processed-view pull landed within `window` — i.e. the view
        /// is being watched right now. A single cheap check per captured frame.
        fn processed_demand_active(&self, window: Duration) -> bool {
            self.processed_requested_at
                .lock()
                .ok()
                .and_then(|guard| *guard)
                .is_some_and(|at| at.elapsed() < window)
        }

        fn set_status(&self, status: CameraStatus) {
            if let Ok(mut guard) = self.status.lock() {
                *guard = status;
            }
        }

        fn publish(&self, result: InferenceUiResult) {
            if let Ok(mut latest) = self.latest_result.lock() {
                *latest = Some(result.clone());
            }
            // Wave 2 connects a Tauri Channel here; the callback only forwards, so
            // invoking it under the lock cannot re-enter this mutex.
            if let Ok(sink) = self.sink.lock() {
                if let Some(callback) = sink.as_ref() {
                    callback(result);
                }
            }
        }
    }

    pub struct CameraActor {
        sender: SyncSender<CameraCommand>,
        health: Arc<ActorHealth>,
        join: Mutex<Option<ActorJoin>>,
        shared: Arc<CameraShared>,
    }

    enum CameraCommand {
        Start {
            reply: Sender<Result<(), ApiError>>,
        },
        Stop {
            reply: Sender<Result<(), ApiError>>,
        },
        SetMode {
            mode: CameraMode,
            reply: Sender<Result<(), ApiError>>,
        },
        ListDevices {
            reply: Sender<Result<Vec<CameraDeviceInfo>, ApiError>>,
        },
        Shutdown,
    }

    impl CameraActor {
        pub fn start(
            inference: Arc<InferenceActor>,
            storage: Arc<DatasetStorage>,
        ) -> Result<Self, ApiError> {
            let (sender, receiver) = mpsc::sync_channel(1);
            let health = Arc::new(ActorHealth::new());
            let child_health = health.clone();
            let shared = Arc::new(CameraShared::new());
            let child_shared = shared.clone();
            let (done_sender, done) = mpsc::channel();
            let handle = thread::Builder::new()
                .name("slouch-camera".to_owned())
                .spawn(move || {
                    if catch_unwind(AssertUnwindSafe(|| {
                        camera_loop(receiver, inference, storage, child_shared)
                    }))
                    .is_err()
                    {
                        log::error!(target: "camera", "camera actor panicked and was stopped");
                    }
                    child_health.fail();
                    let _ = done_sender.send(());
                })
                .map_err(|error| {
                    ApiError::Internal(format!("failed to start camera actor: {error}"))
                })?;
            Ok(Self {
                sender,
                health,
                join: Mutex::new(Some(ActorJoin { handle, done })),
                shared,
            })
        }

        // Health/status/last-result getters are part of the actor's tested
        // surface; the command path pushes results over the sink channel, so the
        // pull-style accessors are exercised only by the test suites.
        #[allow(dead_code)]
        pub fn is_healthy(&self) -> bool {
            self.health.is_healthy()
        }

        fn enqueue(&self, command: CameraCommand) -> Result<(), ApiError> {
            if !self.health.is_healthy() {
                return Err(ApiError::NotReady("camera actor is stopped".into()));
            }
            self.sender.try_send(command).map_err(|error| match error {
                TrySendError::Full(_) => ApiError::Busy("camera actor is busy".into()),
                TrySendError::Disconnected(_) => {
                    self.health.fail();
                    ApiError::NotReady("camera actor is stopped".into())
                }
            })
        }

        /// Opens the device (lazily) and begins the capture loop.
        pub fn start_capture(&self) -> Result<(), ApiError> {
            let (reply, receiver) = mpsc::channel();
            self.enqueue(CameraCommand::Start { reply })?;
            receive_camera_reply(receiver, &self.health)
        }

        /// Stops the stream and releases the device; the actor stays ready to retry.
        pub fn stop_capture(&self) -> Result<(), ApiError> {
            let (reply, receiver) = mpsc::channel();
            self.enqueue(CameraCommand::Stop { reply })?;
            receive_camera_reply(receiver, &self.health)
        }

        pub fn set_mode(&self, mode: CameraMode) -> Result<(), ApiError> {
            let (reply, receiver) = mpsc::channel();
            self.enqueue(CameraCommand::SetMode { mode, reply })?;
            receive_camera_reply(receiver, &self.health)
        }

        pub fn list_devices(&self) -> Result<Vec<CameraDeviceInfo>, ApiError> {
            let (reply, receiver) = mpsc::channel();
            self.enqueue(CameraCommand::ListDevices { reply })?;
            receive_camera_reply(receiver, &self.health)
        }

        /// Registers the sink the `start_camera` command wires to a Tauri
        /// `Channel<InferenceUiResult>` so each detection is pushed to the webview.
        pub fn set_result_sink<F>(&self, sink: F)
        where
            F: Fn(InferenceUiResult) + Send + 'static,
        {
            if let Ok(mut guard) = self.shared.sink.lock() {
                *guard = Some(Box::new(sink));
            }
        }

        /// The freshest raw MJPEG frame (foreground only); `None` in background.
        pub fn latest_frame_bytes(&self) -> Option<Vec<u8>> {
            self.shared.latest_frame.lock().ok()?.clone()
        }

        /// The freshest processed (detector-input) JPEG frame paired with its
        /// monotonic capture sequence; `None` until a detection has run since the
        /// camera started. The sequence is echoed to the webview so both layers
        /// agree on frame ordering.
        pub fn processed_frame_snapshot(&self) -> Option<(u64, Vec<u8>)> {
            self.shared.processed_frame.lock().ok()?.clone()
        }

        /// Records a `slouchcam …/processed` pull so the capture loop keeps the
        /// processed preview fresh at capture rate while the view is watched.
        pub fn note_processed_request(&self) {
            self.shared.note_processed_request();
        }

        #[allow(dead_code)]
        pub fn latest_result(&self) -> Option<InferenceUiResult> {
            self.shared.latest_result.lock().ok()?.clone()
        }

        #[allow(dead_code)]
        pub fn status(&self) -> CameraStatus {
            match self.shared.status.lock() {
                Ok(guard) => guard.clone(),
                Err(_) => CameraStatus::Idle,
            }
        }

        // Reused verbatim from InferenceActor: cooperatively drain the mailbox, then
        // join the capture thread (which stops the stream and closes the device on
        // its way out) before failing health closed.
        pub fn shutdown_until(&self, deadline: Instant) {
            self.health.accepting.store(false, Ordering::Release);
            let mut command = CameraCommand::Shutdown;
            loop {
                match self.sender.try_send(command) {
                    Ok(()) | Err(TrySendError::Disconnected(_)) => break,
                    Err(TrySendError::Full(returned)) if Instant::now() < deadline => {
                        command = returned;
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(TrySendError::Full(_)) => break,
                }
            }
            if let Ok(mut join) = self.join.lock() {
                if let Some(actor_join) = join.take() {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if actor_join.done.recv_timeout(remaining).is_ok() {
                        let _ = actor_join.handle.join();
                    }
                }
            }
            self.health.fail();
        }
    }

    fn receive_camera_reply<T>(
        receiver: Receiver<Result<T, ApiError>>,
        health: &ActorHealth,
    ) -> Result<T, ApiError> {
        match receiver.recv_timeout(ACTOR_WAIT) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                health.fail();
                Err(ApiError::NotReady("camera actor timed out".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                health.fail();
                Err(ApiError::NotReady(
                    "camera actor stopped before replying".into(),
                ))
            }
        }
    }

    /// A single detection unit of work handed from the capture thread to the
    /// detection dispatcher: the last `smoothing_frames` *consecutive* camera
    /// buffers (oldest→newest) plus the settings snapshot used to preprocess them.
    /// The dispatcher ingests them in order, so temporal smoothing averages frames
    /// captured ~33 ms apart (camera rate) rather than the ~1 fps detection cadence.
    /// The final buffer is the frame detection actually runs on. These clones are
    /// the only per-detection copies the capture thread pays.
    struct DetectionJob {
        buffers: Vec<Buffer>,
        settings: CameraSettings,
        // Monotonic capture sequence of the newest buffer (the frame detection runs
        // on). Carried so the dispatcher's processed-frame store is ordered against
        // the capture-rate preview writes and can never regress the served frame.
        capture_seq: u64,
    }

    /// Bounded(1), drop-oldest slot between the capture thread (producer) and the
    /// dispatcher thread (consumer). A new `Pending` overwrites an older one, so a
    /// stale frame is never processed ahead of a newer one.
    enum DispatchSlot {
        Empty,
        Pending(DetectionJob),
        Shutdown,
    }

    /// State shared by the capture thread and the detection dispatcher thread.
    struct DispatcherShared {
        slot: Mutex<DispatchSlot>,
        signal: Condvar,
        // Set the instant a job is submitted, cleared when the dispatcher finishes
        // it: guarantees exactly one detection in flight. The capture thread skips
        // new frames while it is set instead of backlogging them.
        in_flight: AtomicBool,
        // Requested by the capture thread on camera (re)start; the dispatcher clears
        // its preprocessor before the next detection so temporal smoothing restarts
        // from an empty ring.
        reset: AtomicBool,
    }

    /// Owns the detection thread. Preprocessing plus the blocking inference
    /// round-trip run here, off the capture loop, so the preview keeps updating at
    /// ~30 fps while a detection is in flight.
    struct DetectionDispatcher {
        shared: Arc<DispatcherShared>,
        handle: Option<JoinHandle<()>>,
    }

    impl DetectionDispatcher {
        fn spawn(inference: Arc<InferenceActor>, camera_shared: Arc<CameraShared>) -> Self {
            let shared = Arc::new(DispatcherShared {
                slot: Mutex::new(DispatchSlot::Empty),
                signal: Condvar::new(),
                in_flight: AtomicBool::new(false),
                reset: AtomicBool::new(false),
            });
            let worker_shared = shared.clone();
            let handle = match thread::Builder::new()
                .name("slouch-detection".to_owned())
                .spawn(move || {
                    if catch_unwind(AssertUnwindSafe(|| {
                        dispatcher_loop(&worker_shared, &inference, &camera_shared)
                    }))
                    .is_err()
                    {
                        log::error!(target: "camera", "detection dispatcher panicked and was stopped");
                    }
                    // A panicked dispatcher must not wedge the capture thread into
                    // skipping every future detection.
                    worker_shared.in_flight.store(false, Ordering::Release);
                }) {
                Ok(handle) => Some(handle),
                Err(error) => {
                    log::error!(target: "camera", "failed to start detection dispatcher: {error}");
                    None
                }
            };
            Self { shared, handle }
        }

        /// True while a detection is being processed. The capture thread checks this
        /// before cloning a frame, so it neither backlogs work nor copies a buffer
        /// that would only be dropped.
        fn is_busy(&self) -> bool {
            self.shared.in_flight.load(Ordering::Acquire)
        }

        /// Hands the freshest frame to the dispatcher. Callers gate on `is_busy()`
        /// first, so the slot is normally empty here; the drop-oldest overwrite is a
        /// safeguard against a stale queued frame.
        fn submit(&self, job: DetectionJob) {
            self.shared.in_flight.store(true, Ordering::Release);
            if let Ok(mut slot) = self.shared.slot.lock() {
                if !matches!(*slot, DispatchSlot::Shutdown) {
                    *slot = DispatchSlot::Pending(job);
                    self.shared.signal.notify_one();
                    return;
                }
            }
            // Shutting down (or a poisoned lock): nothing will consume the job, so
            // release the gate we just took.
            self.shared.in_flight.store(false, Ordering::Release);
        }

        /// Asks the dispatcher to reset its preprocessor before the next detection.
        fn request_reset(&self) {
            self.shared.reset.store(true, Ordering::Release);
        }

        /// Signals the dispatcher to stop and joins it. Any in-flight detection
        /// finishes first (the inference actor outlives the camera on shutdown).
        fn shutdown(&mut self) {
            if let Ok(mut slot) = self.shared.slot.lock() {
                *slot = DispatchSlot::Shutdown;
            }
            self.shared.signal.notify_all();
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn dispatcher_loop(
        shared: &DispatcherShared,
        inference: &InferenceActor,
        camera_shared: &CameraShared,
    ) {
        // The preprocessor lives here alone, so its temporal smoothing state stays
        // single-threaded and correct; request ids are minted here for the same
        // reason.
        let mut preprocessor = NativePreprocessor::default();
        let mut request_id: u64 = 0;
        loop {
            let job = {
                let mut slot = match shared.slot.lock() {
                    Ok(slot) => slot,
                    Err(_) => return,
                };
                loop {
                    match std::mem::replace(&mut *slot, DispatchSlot::Empty) {
                        DispatchSlot::Pending(job) => break job,
                        DispatchSlot::Shutdown => {
                            *slot = DispatchSlot::Shutdown;
                            return;
                        }
                        DispatchSlot::Empty => {
                            slot = match shared.signal.wait(slot) {
                                Ok(slot) => slot,
                                Err(_) => return,
                            };
                        }
                    }
                }
            };
            if shared.reset.swap(false, Ordering::AcqRel) {
                preprocessor.reset();
            }
            if let Err(error) = run_detection(
                &job,
                &mut preprocessor,
                &mut request_id,
                inference,
                camera_shared,
            ) {
                log::warn!(target: "camera", "native detection failed: {error}");
            }
            shared.in_flight.store(false, Ordering::Release);
        }
    }

    fn camera_loop(
        receiver: Receiver<CameraCommand>,
        inference: Arc<InferenceActor>,
        storage: Arc<DatasetStorage>,
        shared: Arc<CameraShared>,
    ) {
        // Detection runs on its own thread so the ~100-150 ms inference round-trip
        // never blocks this capture loop: it keeps pulling and storing preview
        // frames at ~30 fps while a detection is in flight. The dispatcher owns the
        // preprocessor and mints request ids.
        let mut dispatcher = DetectionDispatcher::spawn(inference, shared.clone());
        let mut mode = CameraMode::Background;
        let mut camera: Option<Camera> = None;
        // The format the open device negotiated. The preview cell must always hold
        // JPEG (slouchcam serves it as image/jpeg), so a non-MJPEG camera's frames
        // are re-encoded before storing; MJPEG cameras store the buffer verbatim.
        let mut frame_format = FrameFormat::MJPEG;
        let mut next_detection = Instant::now();
        // Processed-view fast path (Foreground, demand-driven). Its own preprocessor
        // keeps temporal-smoothing state independent of the dispatcher's detection
        // preprocessor. `preview_active` tracks the demand edge so the ring is reset
        // and settings refreshed exactly when demand (re)starts; `cached_settings`
        // feeds this per-frame path without a storage read every frame (refreshed at
        // each detection tick and on demand start).
        let mut preview_preprocessor = NativePreprocessor::default();
        let mut preview_active = false;
        let mut cached_settings: Option<CameraSettings> = None;
        // Ring of the most recent encoded camera buffers (oldest→newest). Foreground
        // pushes every captured frame (~30 fps) so it naturally holds consecutive
        // frames; Background pushes ~1 fps and bursts to top it up just before a
        // detection. A detection dispatches the last `smoothing_frames` entries so
        // temporal smoothing averages true camera-rate frames.
        let mut frame_ring: VecDeque<Buffer> = VecDeque::with_capacity(FRAME_RING_CAPACITY);
        // Monotonic capture counter: one tick per frame this thread pulls (main loop
        // + background bursts), never reset for the actor's life. It tags every
        // processed-frame store so out-of-order writes from the two producers are
        // rejected and the served processed frame stays monotonic in capture time.
        let mut capture_seq: u64 = 0;

        loop {
            if camera.is_some() {
                // The device always streams one ~30fps MJPEG format; the effective read
                // rate is set here by cadence, not by the device. Foreground polls the
                // command mailbox without blocking and reads every delivered frame
                // (~30fps preview). Background sleeps on the mailbox until the next
                // detection is due and reads ~once per interval (~1fps), so unrendered
                // frames aren't pulled while Start/Stop/SetMode/Shutdown stay responsive
                // (they wake the recv early). On Windows this pull-driven cadence
                // naturally throttles Background; on macOS the device delivers frames
                // continuously, so Background still reads ~1fps but the device keeps
                // running — acceptable for now (a later wave may re-add a device throttle).
                let pending = if mode == CameraMode::Foreground {
                    match receiver.try_recv() {
                        Ok(command) => Some(command),
                        Err(mpsc::TryRecvError::Empty) => None,
                        Err(mpsc::TryRecvError::Disconnected) => break,
                    }
                } else {
                    let wait = next_detection.saturating_duration_since(Instant::now());
                    match receiver.recv_timeout(wait) {
                        Ok(command) => Some(command),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                };
                if let Some(command) = pending {
                    if handle_command(
                        command,
                        &mut camera,
                        &mut mode,
                        &storage,
                        &shared,
                        &dispatcher,
                        &mut next_detection,
                        &mut frame_format,
                        &mut preview_active,
                        &mut frame_ring,
                    ) {
                        break;
                    }
                    continue;
                }

                // No command pending: pull one frame. Foreground reaches here every
                // loop (preview + detection when due); Background only after the
                // recv_timeout above elapsed, i.e. a detection is due.
                // Binding the &mut borrow to this expression frees `camera` before the
                // match arms run, so a read error can drop the device in place.
                let frame = match camera.as_mut() {
                    Some(active) => active.frame(),
                    None => continue,
                };
                match frame {
                    Ok(buffer) => {
                        // Tag this capture. Every processed-frame store carries the
                        // sequence of the newest source frame it was built from, so
                        // the served processed cell can never regress in capture time.
                        capture_seq = capture_seq.wrapping_add(1);

                        // Keep the freshest frame for the preview in BOTH modes:
                        // Foreground stores at ~30fps for a smooth feed; Background
                        // stores each detection frame (~1fps) so a visible-but-unfocused
                        // window still shows the near-free interval preview. The store
                        // guarantees valid JPEG. (The frontend stops fetching when the
                        // window is minimized/hidden, so nothing renders then.)
                        store_preview_frame(&buffer, frame_format, &shared);

                        // Record every captured frame in the temporal ring. Foreground
                        // reaches here ~30x/s so the ring holds consecutive frames;
                        // Background reaches here ~1x/s and is topped up by a burst
                        // just before dispatch (below).
                        push_frame_ring(&mut frame_ring, buffer.clone());

                        // Demand-driven processed-view fast path: while the webview is
                        // actively pulling /processed (Foreground only), also
                        // preprocess + JPEG-encode this frame into the processed cell
                        // at capture rate so the processed view stays smooth. Outside
                        // the demand window this costs a single timestamp check — the
                        // dispatcher's ~1fps store remains the fallback.
                        let demand = mode == CameraMode::Foreground
                            && shared.processed_demand_active(PROCESSED_DEMAND_WINDOW);
                        if demand {
                            if !preview_active {
                                // Demand just (re)started: drop stale temporal history
                                // and take a fresh settings snapshot so the first
                                // processed frame is correct.
                                preview_preprocessor.reset();
                                cached_settings = storage.get_camera_settings().ok();
                                preview_active = true;
                            }
                            if let Some(settings) = cached_settings.as_ref() {
                                if let Err(error) = run_preview_processing(
                                    &buffer,
                                    &mut preview_preprocessor,
                                    settings,
                                    capture_seq,
                                    &shared,
                                ) {
                                    log::debug!(target: "camera", "processed-view preview update skipped: {error}");
                                }
                            }
                        } else if preview_active {
                            // Demand lapsed (view off or backgrounded): reset so stale
                            // frames can't bleed into a later re-enable.
                            preview_preprocessor.reset();
                            preview_active = false;
                        }

                        if Instant::now() >= next_detection {
                            match storage.get_camera_settings() {
                                Ok(settings) => {
                                    next_detection = Instant::now()
                                        + Duration::from_secs_f64(
                                            settings
                                                .capture_interval_seconds
                                                .max(MIN_INTERVAL_SECONDS),
                                        );
                                    // Keep the per-frame preview path on fresh settings
                                    // without a storage read every frame.
                                    cached_settings = Some(settings.clone());
                                    // Hand the freshest consecutive frames to the
                                    // detection dispatcher, but only when it is idle:
                                    // exactly one detection in flight, always the
                                    // newest frames, never a backlog. While the
                                    // dispatcher runs inference this loop keeps storing
                                    // preview frames at ~30 fps.
                                    if !dispatcher.is_busy() {
                                        let want = usize::from(settings.smoothing_frames).max(1);
                                        // Background wakes only ~once per detection, so
                                        // the ring holds ~1 fps-spaced frames. Pull the
                                        // remaining consecutive frames now (bounded:
                                        // ~33 ms each, N-1 reads once per detection) so
                                        // smoothing averages true camera-rate frames.
                                        // Foreground already filled the ring at capture
                                        // rate, so it never bursts.
                                        if mode == CameraMode::Background && want > 1 {
                                            burst_fill_ring(
                                                &mut camera,
                                                &mut frame_ring,
                                                want,
                                                frame_format,
                                                &shared,
                                                &mut capture_seq,
                                            );
                                        }
                                        // Dispatch the last `want` consecutive frames
                                        // (oldest→newest); the newest is the frame
                                        // detection runs on. These clones are the sole
                                        // per-detection copies. `capture_seq` now tags
                                        // that newest frame (post-burst), so the
                                        // dispatcher's store is ordered against the
                                        // capture-rate preview writes.
                                        let start = frame_ring.len().saturating_sub(want);
                                        let buffers: Vec<Buffer> =
                                            frame_ring.iter().skip(start).cloned().collect();
                                        dispatcher.submit(DetectionJob {
                                            buffers,
                                            settings,
                                            capture_seq,
                                        });
                                    }
                                }
                                Err(error) => {
                                    log::warn!(target: "camera", "camera settings unavailable, deferring detection: {error}");
                                    next_detection = Instant::now() + DETECTION_RETRY;
                                }
                            }
                        }
                    }
                    Err(error) => {
                        // Device read failure / disconnect: surface it generically as
                        // an error status and exit the loop so the wrapper fails the
                        // actor's health closed. Recovery and platform-specific
                        // permission deep-links are a later-wave concern.
                        let message = error.to_string();
                        log::error!(target: "camera", "camera capture stopped: {message}");
                        shared.set_status(CameraStatus::Error(message));
                        break;
                    }
                }
            } else {
                // Idle: block until the next command (no device open, no frame pump).
                match receiver.recv() {
                    Ok(command) => {
                        if handle_command(
                            command,
                            &mut camera,
                            &mut mode,
                            &storage,
                            &shared,
                            &dispatcher,
                            &mut next_detection,
                            &mut frame_format,
                            &mut preview_active,
                            &mut frame_ring,
                        ) {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        // Release the device on the way out; the exit-point status (Error on failure,
        // Idle/Streaming on Stop/Shutdown) is left intact.
        stop_camera(&mut camera, &shared);
        // Tear down the detection thread last so an in-flight inference finishes
        // (the inference actor outlives the camera during app shutdown).
        dispatcher.shutdown();
    }

    // Threads the capture loop's mutable state (device, mode, timing, negotiated
    // format) through each command; grouping into a state struct would be cleaner.
    #[allow(clippy::too_many_arguments)]
    fn handle_command(
        command: CameraCommand,
        camera: &mut Option<Camera>,
        mode: &mut CameraMode,
        storage: &DatasetStorage,
        shared: &CameraShared,
        dispatcher: &DetectionDispatcher,
        next_detection: &mut Instant,
        frame_format: &mut FrameFormat,
        preview_active: &mut bool,
        frame_ring: &mut VecDeque<Buffer>,
    ) -> bool {
        match command {
            CameraCommand::Start { reply } => {
                if camera.is_some() {
                    let _ = reply.send(Ok(()));
                    return false;
                }
                let settings = match storage.get_camera_settings() {
                    Ok(settings) => settings,
                    Err(error) => {
                        // A settings-read glitch is not a camera-device fault: report
                        // it but keep the actor alive to retry.
                        let error = ApiError::Storage(error.to_string());
                        shared.set_status(CameraStatus::Error(error.to_string()));
                        let _ = reply.send(Err(error));
                        return false;
                    }
                };
                match open_camera(&settings) {
                    Ok((opened, negotiated)) => {
                        *camera = Some(opened);
                        *frame_format = negotiated;
                        dispatcher.request_reset();
                        // Re-seed the processed-view preview on the next demand frame
                        // so a restart never bleeds pre-restart frames into it.
                        *preview_active = false;
                        // Drop pre-restart frames so the first post-restart detection
                        // never smooths across the stream break.
                        frame_ring.clear();
                        *next_detection = Instant::now();
                        shared.set_status(CameraStatus::Streaming(*mode));
                        let _ = reply.send(Ok(()));
                        false
                    }
                    Err(message) => {
                        // Generic open failure: report it and terminate so the wrapper
                        // fails the actor's health closed.
                        log::error!(target: "camera", "failed to open camera: {message}");
                        let _ = reply.send(Err(ApiError::NotReady(format!(
                            "failed to open camera: {message}"
                        ))));
                        shared.set_status(CameraStatus::Error(message));
                        true
                    }
                }
            }
            CameraCommand::Stop { reply } => {
                stop_camera(camera, shared);
                shared.set_status(CameraStatus::Idle);
                let _ = reply.send(Ok(()));
                false
            }
            CameraCommand::SetMode {
                mode: requested,
                reply,
            } => {
                *mode = requested;
                // The device streams one ~30fps MJPEG format in both modes; the read
                // cadence in camera_loop sets the effective rate, so a mode switch
                // never touches the device. The preview cell (populated in both modes)
                // is left intact, so there's no blank flash on the switch.
                if camera.is_some() {
                    shared.set_status(CameraStatus::Streaming(requested));
                }
                let _ = reply.send(Ok(()));
                false
            }
            CameraCommand::ListDevices { reply } => {
                let _ = reply.send(list_devices());
                false
            }
            CameraCommand::Shutdown => true,
        }
    }

    const NEGOTIATION_TARGET_WIDTH: u32 = 1280;
    const NEGOTIATION_TARGET_HEIGHT: u32 = 720;

    /// Ranks an MJPEG capture mode for negotiation; higher tuples win under
    /// `max_by_key`. Ordered priorities: (1) the resolution closest to 720p wins,
    /// preferring at-or-above 720p over anything below it; (2) the highest fps that
    /// is still <= 30.
    fn mjpeg_format_rank(width: u32, height: u32, fps: u32) -> (i64, i64, i64, i64) {
        let fps = i64::from(fps);
        let area = i64::from(width) * i64::from(height);
        let target_area =
            i64::from(NEGOTIATION_TARGET_WIDTH) * i64::from(NEGOTIATION_TARGET_HEIGHT);

        let at_or_above = area >= target_area;
        // At-or-above 720p: prefer the smallest such mode (closest from above);
        // below 720p: prefer the largest (closest from below). The group bit keeps
        // the two orderings from crossing.
        let res_group = i64::from(at_or_above);
        let res_tiebreak = if at_or_above { -area } else { area };
        let (fps_group, fps_key) = if fps <= 30 { (1, fps) } else { (0, -fps) };

        (res_group, res_tiebreak, fps_group, fps_key)
    }

    /// Opens the device at the best MJPEG mode the webcam advertises, preferring
    /// 1280x720 at the highest fps <= 30 (see `mjpeg_format_rank`), falling back to
    /// the nearest lower resolution only when no 720p-class mode is offered.
    fn open_camera(settings: &CameraSettings) -> Result<(Camera, FrameFormat), String> {
        let index = CameraIndex::Index(0);
        let base = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(index, base).map_err(|error| error.to_string())?;

        // Enumerate the device's supported formats and pick the best MJPEG mode by
        // the 720p-preferring policy. `compatible_camera_formats` needs &mut and may
        // fail on some backends; on failure we fall back to a settings-resolution
        // MJPEG request.
        let chosen = camera.compatible_camera_formats().ok().and_then(|formats| {
            formats
                .into_iter()
                .filter(|candidate| candidate.format() == FrameFormat::MJPEG)
                .max_by_key(|candidate| {
                    mjpeg_format_rank(
                        candidate.width(),
                        candidate.height(),
                        candidate.frame_rate(),
                    )
                })
        });

        match chosen {
            Some(format) => {
                log::info!(
                    target: "camera",
                    "selected MJPEG capture format {}x{} @ {}fps",
                    format.width(),
                    format.height(),
                    format.frame_rate()
                );
                let exact = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(format));
                if let Err(error) = camera.set_camera_requset(exact) {
                    // Exact can fail when the enumerated list and live negotiation
                    // disagree; retry the same format as Closest so a near match opens.
                    log::warn!(target: "camera", "exact MJPEG format rejected, retrying closest: {error}");
                    let closest =
                        RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
                    if let Err(error) = camera.set_camera_requset(closest) {
                        log::warn!(target: "camera", "closest MJPEG request rejected, using default format: {error}");
                    }
                }
            }
            None => {
                // No MJPEG format enumerated: fall back to requesting MJPEG at the
                // settings resolution and let nokhwa pick the closest supported rate.
                let (width, height) = usable_resolution(settings);
                let request = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
                    CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, 30),
                ));
                if let Err(error) = camera.set_camera_requset(request) {
                    log::warn!(target: "camera", "MJPEG request rejected, using default format: {error}");
                }
            }
        }

        let negotiated = camera.camera_format();
        let format = negotiated.format();
        log::info!(
            target: "camera",
            "camera opened at {}x{} {:?} @ {}fps",
            negotiated.width(),
            negotiated.height(),
            format,
            negotiated.frame_rate()
        );
        if format != FrameFormat::MJPEG {
            // slouchcam serves the preview cell as image/jpeg; a non-MJPEG source
            // would paint black in the webview. The preview store re-encodes such
            // frames to JPEG, but surface the negotiation so it never passes
            // silently — it means the fast zero-copy path was lost.
            log::error!(
                target: "camera",
                "camera negotiated non-MJPEG format {format:?}; preview frames will be re-encoded to JPEG"
            );
        }
        camera.open_stream().map_err(|error| error.to_string())?;
        Ok((camera, format))
    }

    /// Foreground preview store. The `slouchcam` protocol serves these bytes as
    /// image/jpeg, so a non-MJPEG camera's raw buffer must be re-encoded to JPEG
    /// first (rare fallback, off the detection hot path). MJPEG cameras — the
    /// common/fast path — store the compressed buffer verbatim, zero re-encode.
    fn store_preview_frame(buffer: &Buffer, frame_format: FrameFormat, shared: &CameraShared) {
        let bytes = if frame_format == FrameFormat::MJPEG {
            buffer.buffer().to_vec()
        } else {
            match encode_preview_jpeg(buffer) {
                Ok(jpeg) => jpeg,
                Err(error) => {
                    log::error!(target: "camera", "preview JPEG re-encode failed: {error}");
                    return;
                }
            }
        };
        if let Ok(mut cell) = shared.latest_frame.lock() {
            *cell = Some(bytes);
        }
    }

    /// Decode a non-MJPEG camera frame to RGB (the same decode detection uses) and
    /// JPEG-encode it so the preview cell always holds valid JPEG.
    fn encode_preview_jpeg(buffer: &Buffer) -> Result<Vec<u8>, String> {
        let decoded = buffer
            .decode_image::<RgbFormat>()
            .map_err(|error| error.to_string())?;
        let width = decoded.width();
        let height = decoded.height();
        let rgb = decoded.into_raw();
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 75)
            .encode(&rgb, width, height, image::ExtendedColorType::Rgb8)
            .map_err(|error| error.to_string())?;
        Ok(jpeg)
    }

    fn usable_resolution(settings: &CameraSettings) -> (u32, u32) {
        let width = settings.camera_width;
        let height = settings.camera_height;
        if (160..=1920).contains(&width) && (120..=1080).contains(&height) {
            (width, height)
        } else {
            (1280, 720)
        }
    }

    fn stop_camera(camera: &mut Option<Camera>, shared: &CameraShared) {
        if let Some(mut active) = camera.take() {
            if let Err(error) = active.stop_stream() {
                log::warn!(target: "camera", "stop_stream failed: {error}");
            }
            // `active` drops here, releasing the native capture device.
        }
        if let Ok(mut cell) = shared.latest_frame.lock() {
            *cell = None;
        }
        if let Ok(mut cell) = shared.processed_frame.lock() {
            *cell = None;
        }
    }

    /// Processed-view preview store: JPEG-encode the exact preprocessed RGBA frame
    /// the detector is about to see so `slouchcam …/processed` can serve it, tagged
    /// with `capture_seq` (the newest source frame's capture sequence). Two producers
    /// write this cell — the ~30fps preview preprocessor and the ~1fps dispatcher —
    /// so the store only lands when `capture_seq` is strictly newer than the frame
    /// already there; a stale (older-source) write from either producer is dropped.
    /// This is what keeps the served processed frame monotonic in capture time and
    /// stops the ~1fps dispatcher write from flickering an older average over the
    /// smooth preview stream (the effect grows with `smoothing_frames`). Returns
    /// whether the frame was stored. Best-effort: a failed encode skips this preview
    /// update, never the detection.
    fn store_processed_frame(frame: &ImageData, capture_seq: u64, shared: &CameraShared) -> bool {
        // The image crate's JPEG encoder rejects Rgba8 input; strip the (always
        // opaque) alpha channel and encode as Rgb8.
        let mut rgb = Vec::with_capacity(frame.data.len() / 4 * 3);
        for pixel in frame.data.chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
        }
        let mut jpeg = Vec::new();
        match image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 80).encode(
            &rgb,
            frame.width,
            frame.height,
            image::ExtendedColorType::Rgb8,
        ) {
            Ok(()) => {
                if let Ok(mut cell) = shared.processed_frame.lock() {
                    // Strictly-newer guard: never let an older-source frame overwrite
                    // a newer one, regardless of which producer or thread wrote first.
                    if cell
                        .as_ref()
                        .is_none_or(|(stored, _)| capture_seq > *stored)
                    {
                        *cell = Some((capture_seq, jpeg));
                        return true;
                    }
                }
                false
            }
            Err(error) => {
                log::warn!(target: "camera", "processed preview JPEG encode failed: {error}");
                false
            }
        }
    }

    /// Processed-view preview update: preprocess the current camera frame through
    /// the preview-only preprocessor (temporal state independent of detection) and
    /// store the JPEG in the processed cell. Mirrors `run_detection` minus the
    /// inference round-trip; runs at capture rate while the view is watched.
    fn run_preview_processing(
        buffer: &Buffer,
        preprocessor: &mut NativePreprocessor,
        settings: &CameraSettings,
        capture_seq: u64,
        shared: &CameraShared,
    ) -> Result<(), ApiError> {
        let image = decode_rgba(buffer).map_err(ApiError::InvalidRequest)?;
        preprocessor
            .ingest_camera_frame(image, settings)
            .map_err(ApiError::InvalidRequest)?;
        let processed = preprocessor
            .process_latest(settings)
            .map_err(ApiError::InvalidRequest)?;
        store_processed_frame(&processed, capture_seq, shared);
        Ok(())
    }

    /// Pushes one captured buffer into the temporal ring, evicting the oldest so it
    /// never exceeds `FRAME_RING_CAPACITY` (the maximum smoothing window).
    fn push_frame_ring(ring: &mut VecDeque<Buffer>, frame: Buffer) {
        ring.push_back(frame);
        while ring.len() > FRAME_RING_CAPACITY {
            ring.pop_front();
        }
    }

    /// Background top-up: the capture loop wakes only ~once per detection, so the
    /// ring lacks consecutive frames. Pulls extra frames back-to-back at the
    /// device's pacing until the ring holds `want` fresh consecutive frames (the
    /// just-captured frame is already at the back, so this reads `want - 1` more),
    /// so temporal smoothing averages ~33 ms-apart frames. Bounded to `want - 1`
    /// reads (≤ 9), once per detection. A transient read failure just shortens this
    /// detection's window; the next main-loop read surfaces a real disconnect.
    fn burst_fill_ring(
        camera: &mut Option<Camera>,
        ring: &mut VecDeque<Buffer>,
        want: usize,
        frame_format: FrameFormat,
        shared: &CameraShared,
        capture_seq: &mut u64,
    ) {
        let Some(active) = camera.as_mut() else {
            return;
        };
        for _ in 1..want {
            match active.frame() {
                Ok(buffer) => {
                    // Each burst frame advances the capture sequence so the newest
                    // ring entry (the frame detection runs on) carries the highest
                    // sequence, keeping the dispatcher's store ordered.
                    *capture_seq = capture_seq.wrapping_add(1);
                    // Keep the preview on the freshest (detected-on) frame too.
                    store_preview_frame(&buffer, frame_format, shared);
                    push_frame_ring(ring, buffer);
                }
                Err(error) => {
                    log::warn!(target: "camera", "smoothing burst frame read failed: {error}");
                    break;
                }
            }
        }
    }

    fn run_detection(
        job: &DetectionJob,
        preprocessor: &mut NativePreprocessor,
        request_id: &mut u64,
        inference: &InferenceActor,
        shared: &CameraShared,
    ) -> Result<(), ApiError> {
        // Ingest the consecutive camera-rate frames in capture order, then process
        // once. A ring of exactly `smoothing_frames` frames is dispatched and the
        // preprocessor keeps the same window size, so `process_latest` averages this
        // detection's own consecutive frames — the previous detection's frames are
        // fully evicted by the time the last one is ingested.
        if job.buffers.is_empty() {
            return Err(ApiError::InvalidRequest(
                "detection job carried no frames".into(),
            ));
        }
        for buffer in &job.buffers {
            let image = decode_rgba(buffer).map_err(ApiError::InvalidRequest)?;
            preprocessor
                .ingest_camera_frame(image, &job.settings)
                .map_err(ApiError::InvalidRequest)?;
        }
        let processed = preprocessor
            .process_latest(&job.settings)
            .map_err(ApiError::InvalidRequest)?;
        // Tagged with the job's newest capture sequence: while the processed view is
        // actively watched the capture-rate preview writes carry a newer sequence, so
        // this ~1fps store is dropped and can't flicker an older average over the
        // smooth stream. When the view is not watched, this is the only writer and
        // always advances the sequence.
        store_processed_frame(&processed, job.capture_seq, shared);
        let id = *request_id;
        *request_id = request_id.wrapping_add(1);
        let result = inference
            .send_frame(raw_inference_message(processed, id))?
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::Inference("inference actor returned no response".into()))
            .and_then(inference_result)?;
        // Copy the overlay fields out (bbox/classification are Copy) before the
        // result is moved into the token cache, matching the infer_frame idiom.
        let person_found = result.person_found;
        let bbox = result.bbox;
        let keypoints = result.keypoints.clone();
        let classification = result.classification;
        let token = inference.cache_result(id, result)?;
        shared.publish(InferenceUiResult {
            request_id: id,
            token,
            person_found,
            bbox,
            keypoints,
            classification,
        });
        Ok(())
    }

    fn decode_rgba(buffer: &Buffer) -> Result<ImageData, String> {
        // Spike-validated path: decode MJPEG to RGB, then widen to the RGBA layout
        // ImageData/NativePreprocessor require (opaque alpha).
        let decoded = buffer
            .decode_image::<RgbFormat>()
            .map_err(|error| error.to_string())?;
        let width = decoded.width();
        let height = decoded.height();
        let rgb = decoded.into_raw();
        let mut rgba = vec![0_u8; rgb.len() / 3 * 4];
        for (source, target) in rgb.chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
            target[0] = source[0];
            target[1] = source[1];
            target[2] = source[2];
            target[3] = 255;
        }
        Ok(ImageData {
            data: rgba,
            width,
            height,
        })
    }

    fn list_devices() -> Result<Vec<CameraDeviceInfo>, ApiError> {
        let cameras = query(ApiBackend::Auto)
            .map_err(|error| ApiError::Internal(format!("failed to enumerate cameras: {error}")))?;
        Ok(cameras
            .iter()
            .map(|info| CameraDeviceInfo {
                index: info.index().to_string(),
                name: info.human_name(),
                description: info.description().to_string(),
            })
            .collect())
    }

    #[cfg(test)]
    mod ordering_tests {
        use super::*;

        fn best_mjpeg(candidates: &[(u32, u32, u32)]) -> (u32, u32, u32) {
            candidates
                .iter()
                .copied()
                .max_by_key(|&(width, height, fps)| mjpeg_format_rank(width, height, fps))
                .expect("at least one candidate")
        }

        #[test]
        fn negotiation_prefers_720p_over_vga_at_equal_fps() {
            assert_eq!(
                best_mjpeg(&[(640, 480, 30), (1280, 720, 30)]),
                (1280, 720, 30)
            );
        }

        #[test]
        fn negotiation_prefers_720p_over_higher_resolution() {
            // 720p is the closest mode at-or-above the target; larger modes are farther.
            assert_eq!(
                best_mjpeg(&[(1920, 1080, 30), (1280, 720, 30)]),
                (1280, 720, 30)
            );
        }

        #[test]
        fn negotiation_takes_highest_fps_up_to_30_at_720p() {
            assert_eq!(
                best_mjpeg(&[(1280, 720, 15), (1280, 720, 60), (1280, 720, 30)]),
                (1280, 720, 30)
            );
        }

        #[test]
        fn negotiation_falls_back_to_closest_lower_when_no_720p() {
            // No 720p-class mode offered: pick the largest resolution below it.
            assert_eq!(
                best_mjpeg(&[(320, 240, 30), (640, 480, 30), (800, 600, 30)]),
                (800, 600, 30)
            );
        }

        #[test]
        fn negotiation_prefers_above_720p_over_vga_when_no_exact_720p() {
            // With 720p itself absent, an above-720p mode still beats a VGA one.
            assert_eq!(
                best_mjpeg(&[(1920, 1080, 30), (640, 480, 30)]),
                (1920, 1080, 30)
            );
        }

        #[test]
        fn negotiation_tie_breaks_on_higher_fps_within_lower_fallback() {
            assert_eq!(
                best_mjpeg(&[(640, 480, 15), (640, 480, 30)]),
                (640, 480, 30)
            );
        }

        fn solid_frame(width: u32, height: u32, value: u8) -> ImageData {
            ImageData {
                data: vec![value; (width as usize) * (height as usize) * 4],
                width,
                height,
            }
        }

        fn stored_seq(shared: &CameraShared) -> Option<u64> {
            shared
                .processed_frame
                .lock()
                .unwrap()
                .as_ref()
                .map(|(seq, _)| *seq)
        }

        fn stored_bytes(shared: &CameraShared) -> Option<Vec<u8>> {
            shared
                .processed_frame
                .lock()
                .unwrap()
                .as_ref()
                .map(|(_, bytes)| bytes.clone())
        }

        #[test]
        fn processed_store_rejects_stale_and_equal_capture_sequences() {
            let shared = CameraShared::new();
            // First write into an empty cell always lands.
            assert!(store_processed_frame(&solid_frame(2, 2, 10), 5, &shared));
            assert_eq!(stored_seq(&shared), Some(5));

            // A strictly newer write lands.
            assert!(store_processed_frame(&solid_frame(2, 2, 20), 6, &shared));
            assert_eq!(stored_seq(&shared), Some(6));
            let newest = stored_bytes(&shared);

            // An older-source write is dropped (the ~1fps dispatcher stale case).
            assert!(!store_processed_frame(&solid_frame(2, 2, 30), 4, &shared));
            assert_eq!(stored_seq(&shared), Some(6));
            // The equal-sequence write is dropped too (already served).
            assert!(!store_processed_frame(&solid_frame(2, 2, 40), 6, &shared));
            assert_eq!(stored_seq(&shared), Some(6));
            // The cell still holds the newest frame's bytes, never the stale ones.
            assert_eq!(stored_bytes(&shared), newest);
        }

        #[test]
        fn interleaved_preview_and_dispatcher_writes_never_regress_served_frame() {
            // Models smoothing_frames > 1 steady state: the capture-rate preview
            // preprocessor advances the sequence every frame while the dispatcher
            // periodically tries to store a frame built from older source captures.
            let shared = CameraShared::new();
            let mut served: Vec<u64> = Vec::new();
            let mut preview_seq = 0_u64;
            for step in 0..30_u64 {
                // Preview writes at capture rate — always the newest source frame.
                preview_seq += 1;
                store_processed_frame(&solid_frame(2, 2, (step % 200) as u8), preview_seq, &shared);
                if let Some(seq) = stored_seq(&shared) {
                    served.push(seq);
                }
                // ~Every 10th frame the dispatcher tries to store a frame whose newest
                // source capture is several frames stale (thread lag under smoothing).
                if step % 10 == 9 {
                    let stale = preview_seq.saturating_sub(4);
                    assert!(
                        !store_processed_frame(&solid_frame(2, 2, 200), stale, &shared),
                        "stale dispatcher write must be rejected while preview leads"
                    );
                    if let Some(seq) = stored_seq(&shared) {
                        served.push(seq);
                    }
                }
            }
            // The served sequence never regresses — no older frame is ever displayed.
            for pair in served.windows(2) {
                assert!(pair[1] >= pair[0], "served frame regressed: {served:?}");
            }
            assert_eq!(stored_seq(&shared), Some(preview_seq));
        }

        #[test]
        fn dispatcher_write_resumes_once_preview_demand_stops() {
            // When the processed view is closed the preview stops writing; the next
            // detection's newest capture is necessarily newer than the last preview
            // frame (the capture loop keeps advancing the sequence), so the
            // dispatcher's store resumes driving the cell.
            let shared = CameraShared::new();
            assert!(store_processed_frame(&solid_frame(2, 2, 10), 20, &shared));
            // Capture kept advancing while the view was off; the detection stores with
            // a newer capture sequence and lands.
            assert!(store_processed_frame(&solid_frame(2, 2, 50), 25, &shared));
            assert_eq!(stored_seq(&shared), Some(25));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{mpsc, Arc, Mutex},
        thread,
        time::{Duration, Instant},
    };

    use super::{
        finish_training, initialize_message, receive_inference_reply,
        validate_training_event_stream, ActorHealth, ActorJoin, CameraActor, CameraMode,
        InferenceActor, InferenceCache, InferenceCommand, InferenceUiResult, NativeInferenceResult,
        NativeModelFactory, TrainingEvent, TrainingStage, TrainingWorkerResponse,
    };
    use slouch_domain::ported::messages::schemas::{
        DimensionalityReductionConfig, DimensionalityReductionMethod, NormalizationMode,
        SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
        SerializedGaussianNb, SerializedModel,
    };
    use slouch_domain::{FeatureId, FeatureMap};
    use slouch_ml::ported::training_worker::{TrainingModelsResponse, TrainingResultResponse};
    use slouch_vision::ported::inference_worker::{ClassificationInput, ModelFactory};

    fn result() -> NativeInferenceResult {
        NativeInferenceResult {
            person_found: false,
            bbox: None,
            keypoints: None,
            features: Default::default(),
            classification: None,
        }
    }

    #[test]
    fn native_presence_classifier_accepts_no_person_feature_only_input() {
        let dimension = FeatureId::RtmDetExtracted.metadata().dimensions;
        let serialized = SerializedModel {
            feature_extractor: SerializedFeatureExtractor {
                feature_types: vec![FeatureId::RtmDetExtracted.as_str().into()],
                normalization_mode: NormalizationMode::None,
                dim_reduction_config: DimensionalityReductionConfig {
                    method: DimensionalityReductionMethod::None,
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
        };
        let mut model = NativeModelFactory.load(serialized).expect("load model");
        let features = FeatureMap::from([(FeatureId::RtmDetExtracted, vec![0.0; dimension])]);
        let probability = model
            .predict(&ClassificationInput {
                features: &features,
                bbox: None,
                keypoints: None,
            })
            .expect("feature-only presence prediction");
        assert!(probability.is_finite());
    }

    #[test]
    fn inference_tokens_are_rust_generated_distinct_and_one_time() {
        let mut cache = InferenceCache::with_seed(1);
        let token = cache.insert(7, result()).unwrap();
        assert_ne!(token, 7);
        assert!(cache.checkout(7, 7).is_err());
        assert!(cache.checkout(token, 8).is_err());
        assert!(cache.checkout(token, 7).is_ok());
        cache.commit(token, 7).unwrap();
        assert!(cache.checkout(token, 7).is_err());
    }

    #[test]
    fn inference_tokens_are_entry_bounded() {
        let mut cache = InferenceCache::with_seed(2);
        let mut tokens = Vec::new();
        for request_id in 0..=32 {
            tokens.push(cache.insert(request_id, result()).unwrap());
        }
        assert!(cache.checkout(tokens[0], 0).is_err());
        assert!(cache.checkout(tokens[32], 32).is_ok());
    }

    #[test]
    fn inference_tokens_are_byte_bounded_with_deterministic_lru_eviction() {
        let mut cache = InferenceCache::with_seed(3);
        let mut large = result();
        let mut features = FeatureMap::new();
        features.insert(FeatureId::BackboneFeatures, vec![0.0; 8_500_000]);
        large.features = features;
        let first = cache.insert(1, large.clone()).unwrap();
        let second = cache.insert(2, large).unwrap();
        assert!(cache.retained_bytes <= 64 * 1024 * 1024);
        assert!(cache.checkout(first, 1).is_err());
        assert!(cache.checkout(second, 2).is_ok());
    }

    #[test]
    fn failed_capture_restores_the_same_opaque_token() {
        let mut cache = InferenceCache::with_seed(4);
        let token = cache.insert(9, result()).unwrap();
        let checked_out = cache.checkout(token, 9).unwrap();
        cache.restore(token, 9, checked_out).unwrap();
        assert!(cache.checkout(token, 9).is_ok());
    }

    #[test]
    fn disconnected_reply_fails_actor_health_closed() {
        let health = ActorHealth::new();
        let (sender, receiver) = mpsc::channel::<Result<(), crate::errors::ApiError>>();
        drop(sender);
        assert!(receive_inference_reply(receiver, &health).is_err());
        assert!(!health.is_healthy());
    }

    #[test]
    fn acknowledged_cancellation_discards_a_racing_success() {
        let response = TrainingWorkerResponse::Result {
            result: Box::new(TrainingResultResponse {
                posture_result: None,
                presence_result: None,
                success: true,
                errors: Vec::new(),
                warnings: Vec::new(),
            }),
            models: Box::new(TrainingModelsResponse {
                posture: None,
                presence: None,
            }),
        };
        let (reply_sender, reply_receiver) = mpsc::channel();
        finish_training(1, Ok(response), true, reply_sender, None);
        assert!(matches!(
            reply_receiver.recv().unwrap(),
            Err(crate::errors::ApiError::Cancelled(_))
        ));
    }

    #[test]
    fn training_event_streams_are_keyed_monotonic_and_exactly_terminal() {
        let job_id = 42;
        let result = TrainingResultResponse {
            posture_result: None,
            presence_result: None,
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };
        let completed = vec![
            TrainingEvent::Started {
                job_id,
                sequence: 0,
            },
            TrainingEvent::Progress {
                job_id,
                sequence: 1,
                stage: TrainingStage::Processing,
                progress: 5,
            },
            TrainingEvent::Progress {
                job_id,
                sequence: 2,
                stage: TrainingStage::Evaluating,
                progress: 85,
            },
            TrainingEvent::Progress {
                job_id,
                sequence: 3,
                stage: TrainingStage::Deploying,
                progress: 95,
            },
            TrainingEvent::Completed {
                job_id,
                sequence: 4,
                result: Box::new(result),
            },
        ];
        assert!(validate_training_event_stream(&completed).is_ok());

        let cancelled = vec![
            TrainingEvent::Started {
                job_id,
                sequence: 0,
            },
            TrainingEvent::Progress {
                job_id,
                sequence: 1,
                stage: TrainingStage::Processing,
                progress: 5,
            },
            TrainingEvent::Cancelled {
                job_id,
                sequence: 2,
            },
        ];
        assert!(validate_training_event_stream(&cancelled).is_ok());
    }

    #[test]
    fn training_event_validation_rejects_regression_duplicate_terminal_and_post_terminal_events() {
        let invalid = vec![
            TrainingEvent::Started {
                job_id: 1,
                sequence: 0,
            },
            TrainingEvent::Progress {
                job_id: 1,
                sequence: 1,
                stage: TrainingStage::Evaluating,
                progress: 85,
            },
            TrainingEvent::Progress {
                job_id: 1,
                sequence: 2,
                stage: TrainingStage::Processing,
                progress: 5,
            },
            TrainingEvent::Failed {
                job_id: 1,
                sequence: 3,
                error: "failed".into(),
            },
            TrainingEvent::Cancelled {
                job_id: 1,
                sequence: 4,
            },
        ];
        assert!(validate_training_event_stream(&invalid).is_err());
    }

    #[test]
    fn health_state_is_shared_across_actor_callers() {
        let health = Arc::new(ActorHealth::new());
        let observer = health.clone();
        health.fail();
        assert!(!observer.is_healthy());
    }

    #[test]
    fn full_actor_mailbox_rejects_without_blocking() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        sender.try_send(InferenceCommand::Shutdown).unwrap();
        let actor = InferenceActor {
            sender,
            health: Arc::new(ActorHealth::new()),
            join: Mutex::new(None),
        };
        assert!(matches!(
            actor.clear_cache(),
            Err(crate::errors::ApiError::Busy(_))
        ));
    }

    #[test]
    fn an_injected_panic_is_caught_and_fails_the_inference_actor_closed() {
        let storage = Arc::new(
            slouch_store::ported::storage::DatasetStorage::open_in_memory()
                .expect("in-memory dataset storage"),
        );
        let actor = InferenceActor::start(storage).expect("start inference actor");
        actor
            .sender
            .try_send(InferenceCommand::CrashForTests)
            .expect("mailbox accepts the crash command");
        let deadline = Instant::now() + Duration::from_secs(5);
        while actor.is_healthy() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(!actor.is_healthy(), "panic must fail actor health closed");
        assert!(matches!(
            actor.clear_cache(),
            Err(crate::errors::ApiError::NotReady(_))
        ));
    }

    #[test]
    fn shutdown_with_an_expired_deadline_does_not_join_a_non_cooperative_actor() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        sender.try_send(InferenceCommand::Shutdown).unwrap();
        let (_done_sender, done) = mpsc::channel();
        let handle = thread::spawn(|| thread::sleep(Duration::from_millis(10)));
        let actor = InferenceActor {
            sender,
            health: Arc::new(ActorHealth::new()),
            join: Mutex::new(Some(ActorJoin { handle, done })),
        };

        actor.shutdown_until(Instant::now());

        assert!(actor.join.lock().expect("join state").is_none());
    }

    #[test]
    fn camera_actor_enumerates_devices_and_shuts_down_without_hardware() {
        let storage = Arc::new(
            slouch_store::ported::storage::DatasetStorage::open_in_memory()
                .expect("in-memory dataset storage"),
        );
        let inference =
            Arc::new(InferenceActor::start(storage.clone()).expect("start inference actor"));
        let camera = CameraActor::start(inference, storage).expect("start camera actor");
        assert!(camera.is_healthy());
        // Enumeration never opens a device: it returns a typed result (an empty
        // list on a machine with no webcam, or a typed backend error) and leaves
        // the actor serving either way.
        match camera.list_devices() {
            Ok(_) | Err(crate::errors::ApiError::Internal(_)) => {}
            Err(other) => panic!("unexpected list_devices outcome: {other}"),
        }
        assert!(camera.is_healthy());
        assert!(camera.latest_frame_bytes().is_none());
        assert!(camera.latest_result().is_none());
        camera.shutdown_until(Instant::now() + Duration::from_secs(5));
        assert!(!camera.is_healthy());
    }

    #[test]
    #[ignore = "requires a physical webcam and the packaged ONNX Runtime"]
    fn camera_actor_opens_a_real_camera_and_publishes_an_inference_result() {
        use std::path::PathBuf;

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dll = manifest.join("resources/onnxruntime/windows-x86_64/onnxruntime.dll");
        assert!(
            ort::init_from(&dll).unwrap().commit(),
            "packaged ONNX Runtime must load"
        );
        let storage = Arc::new(
            slouch_store::ported::storage::DatasetStorage::open_in_memory()
                .expect("in-memory dataset storage"),
        );
        let inference =
            Arc::new(InferenceActor::start(storage.clone()).expect("start inference actor"));
        inference
            .send(initialize_message(
                manifest.join("resources/models/rtmdet-nano.onnx"),
                manifest.join("resources/models/nlf_l_crop_fp16.onnx"),
            ))
            .expect("initialize native inference models");

        let camera = CameraActor::start(inference, storage).expect("start camera actor");
        let captured: Arc<Mutex<Option<InferenceUiResult>>> = Arc::new(Mutex::new(None));
        let sink = captured.clone();
        camera.set_result_sink(move |result| {
            *sink.lock().expect("sink lock") = Some(result);
        });
        camera
            .set_mode(CameraMode::Foreground)
            .expect("foreground mode");
        camera.start_capture().expect("start capture");

        let deadline = Instant::now() + Duration::from_secs(20);
        let result = loop {
            if let Some(result) = captured.lock().expect("captured lock").clone() {
                break Some(result);
            }
            if Instant::now() >= deadline {
                break None;
            }
            thread::sleep(Duration::from_millis(100));
        };
        // Read the preview cell before shutdown clears it.
        let preview_present = camera.latest_frame_bytes().is_some();
        camera.shutdown_until(Instant::now() + Duration::from_secs(5));

        let result = result.expect("a real detection must publish an InferenceUiResult");
        assert_ne!(
            result.token, result.request_id,
            "one-use token must be minted"
        );
        assert!(
            preview_present,
            "foreground capture must fill the MJPEG preview cell"
        );
        println!(
            "real-camera detection: person_found={} token={} request_id={}",
            result.person_found, result.token, result.request_id
        );
    }
}
