//! Native port of `src/workers/inference-worker.ts`.
//!
//! The browser worker's structured-clone boundary is represented here by typed
//! messages and responses. Image pixels and feature vectors remain native byte
//! buffers; the application boundary is responsible for transporting them
//! without JSON array conversion.

use std::collections::HashMap;
use std::fmt;
use std::thread;
use std::time::{Duration, Instant};

use ndarray::Array4;
use ort::{
    ep::DirectML,
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use slouch_domain::ported::messages::schemas::{
    ImageData, InferenceWorkerMessage, JsonValue, SerializedModel,
};
use slouch_domain::{
    BoundingBox, ClassificationResult, ExpandedBbox, FeatureId, FeatureMap, Keypoint,
};
use slouch_ml::ported::constants::{
    PERSON_DETECTION_CONFIDENCE, RTMDET_INPUT_SIZE, RTMDET_OUTPUT_NAMES, RTMDET_RAW_DIMS,
    RTMPOSE_BACKBONE_RAW_DIMS, RTMPOSE_GAU_RAW_DIMS, RTMPOSE_INPUT_SIZE, RTMPOSE_MEAN_RGB,
    RTMPOSE_NUM_KEYPOINTS, RTMPOSE_SIMCC_SPLIT_RATIO, RTMPOSE_STD_RGB,
};
use slouch_ml::ported::nlf_features::extract_nlf_depth_features;
use slouch_ml::ported::rtmdet_features::extract_rtm_det_features as extract_ported_rtmdet_features;
use slouch_ml::ported::rtmpose_features::{
    pool_backbone_features, pool_backbone_features_max, pool_backbone_features_std,
    pool_gau_features, pool_gau_features_max, pool_gau_features_std,
};

const RTMDET_INPUT_WIDTH: usize = RTMDET_INPUT_SIZE.width;
const RTMDET_INPUT_HEIGHT: usize = RTMDET_INPUT_SIZE.height;
const RTMDET_CONFIDENCE: f64 = PERSON_DETECTION_CONFIDENCE;
const RTMPOSE_INPUT_WIDTH: usize = RTMPOSE_INPUT_SIZE.width;
const RTMPOSE_INPUT_HEIGHT: usize = RTMPOSE_INPUT_SIZE.height;
const RTMPOSE_SIMCC_X_WIDTH: usize = RTMPOSE_INPUT_SIZE.width * 2;
const RTMPOSE_SIMCC_Y_WIDTH: usize = RTMPOSE_INPUT_SIZE.height * 2;
const RTMPOSE_SPLIT_RATIO: f64 = RTMPOSE_SIMCC_SPLIT_RATIO;
const RTMPOSE_BACKBONE_VALUES: usize = RTMPOSE_BACKBONE_RAW_DIMS;
const RTMPOSE_GAU_VALUES: usize = RTMPOSE_GAU_RAW_DIMS;
const RTMDET_FEATURE_VALUES: usize = RTMDET_RAW_DIMS;
const MARK_CLEANUP_INTERVAL: u64 = 100;
const MODEL_RETRIES: usize = 3;

// NLF-L's square crop side (proc_side). The supplementary depth model consumes a
// `[1,3,384,384]` RGB tensor; kept local like the other preprocessing geometry.
const NLF_INPUT_SIZE: usize = 384;

// Graph output names of the NLF-L model consumed by the depth-feature tap. The
// third output (`coords2d`) is copied but not read here.
const NLF_COORDS3D_OUTPUT: &str = "coords3d_rel";
const NLF_UNCERTAINTY_OUTPUT: &str = "uncertainty";

const RTMDET_CLS_P5: &str = RTMDET_OUTPUT_NAMES.cls_p5;
const RTMDET_REG_P5: &str = RTMDET_OUTPUT_NAMES.reg_p5;

// RTMDet's BGR normalization is worker-specific in the source oracle; the
// centralized constants module only owns RTMPose's RGB normalization.
const RTMDET_MEAN_BGR: [f64; 3] = [103.53, 116.28, 123.675];
const RTMDET_STD_BGR: [f64; 3] = [57.375, 57.12, 58.395];

/// A model loaded by the native classifier layer.
///
/// The trait deliberately accepts the complete inference result. This keeps
/// feature selection, normalization, K-Means Logistic, and K-Means Prototype
/// in the shared model implementation rather than duplicating classifier logic
/// in the vision boundary.
pub trait ClassifierModel: Send {
    fn predict(&mut self, input: &ClassificationInput<'_>) -> Result<f64, String>;
    fn dispose(&mut self);
}

/// Creates a classifier model from the serialized model envelope.
pub trait ModelFactory: Send + Sync {
    fn load(&self, serialized: SerializedModel) -> Result<Box<dyn ClassifierModel>, String>;
}

/// Inputs supplied to a loaded classifier.
#[derive(Debug, Clone, Copy)]
pub struct ClassificationInput<'a> {
    pub features: &'a FeatureMap,
    pub bbox: Option<&'a ExpandedBbox>,
    pub keypoints: Option<&'a [Keypoint]>,
}

/// A native response result. Unlike the browser response, the native result
/// keeps the expanded and original boxes typed instead of using `z.any()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeInferenceResult {
    pub person_found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ExpandedBbox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keypoints: Option<Vec<Keypoint>>,
    pub features: FeatureMap,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationResult>,
}

/// Responses emitted by [`InferenceWorker::handle_message`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerResponse {
    #[serde(rename = "initialized")]
    Initialized { provider: String },
    #[serde(rename = "result")]
    Result {
        #[serde(rename = "requestId")]
        request_id: u64,
        result: NativeInferenceResult,
    },
    #[serde(rename = "error")]
    Error {
        error: String,
        #[serde(rename = "requestId", skip_serializing_if = "Option::is_none")]
        request_id: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        #[serde(skip_serializing_if = "Option::is_none")]
        success: Option<bool>,
    },
    #[serde(rename = "classifierLoaded")]
    ClassifierLoaded { success: bool },
    #[serde(rename = "classifierUnloaded")]
    ClassifierUnloaded,
}

#[derive(Debug)]
pub enum WorkerError {
    InvalidInput(String),
    Model(String),
    Inference(String),
    MissingOutput(String),
    Tensor(String),
    Feature(String),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::Model(message)
            | Self::Inference(message)
            | Self::MissingOutput(message)
            | Self::Tensor(message)
            | Self::Feature(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for WorkerError {}

/// Small logging boundary used by the native worker.
pub trait WorkerLogger: Send + Sync {
    fn set_from_url_param(&self, _log_param: &str) {}
    fn is_debug_enabled(&self) -> bool {
        false
    }
    fn debug(&self, _message: &str) {}
    fn info(&self, _message: &str) {}
    fn warn(&self, _message: &str) {}
    fn error(&self, _message: &str) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLogger;

impl WorkerLogger for NoopLogger {}

/// Execution provider a session is created on. RTMDet/RTMPose stay on the CPU
/// kernels (byte-identical across the CPU and DirectML runtime builds); only the
/// supplementary NLF-L depth session runs on DirectML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    DirectMl,
}

/// Frozen native session settings passed through the injectable runtime boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionOptions {
    pub intra_threads: usize,
    pub graph_optimization_all: bool,
    pub execution_provider: ExecutionProvider,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            // Single core on purpose: these are tiny models run ~1fps in the background.
            // All-core intra-op parallelism only added thread-pool spinning, not speed, and
            // pinned foreground CPU. One thread keeps the app battery-friendly.
            intra_threads: 1,
            graph_optimization_all: true,
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

/// Owned named outputs returned by an inference session.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionOutput {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

pub type SessionOutputMap = HashMap<String, SessionOutput>;

/// Minimal session seam used by production and deterministic worker tests.
pub trait InferenceSession: Send {
    fn run(&mut self, input: &Array4<f32>) -> Result<SessionOutputMap, WorkerError>;
}

/// Runtime seam for session creation and retry waiting.
pub trait InferenceRuntime: Send {
    fn create_session(
        &mut self,
        path: &str,
        model_name: &str,
        options: SessionOptions,
    ) -> Result<Box<dyn InferenceSession>, WorkerError>;

    fn wait_before_retry(&mut self, delay: Duration);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OrtRuntime;

struct OrtSession(Session);

impl InferenceSession for OrtSession {
    fn run(&mut self, input: &Array4<f32>) -> Result<SessionOutputMap, WorkerError> {
        let input = TensorRef::from_array_view(input)
            .map_err(|error| WorkerError::Tensor(error.to_string()))?;
        let outputs = self
            .0
            .run(ort::inputs![input])
            .map_err(|error| WorkerError::Inference(error.to_string()))?;
        copy_session_outputs(&outputs)
    }
}

impl InferenceRuntime for OrtRuntime {
    fn create_session(
        &mut self,
        path: &str,
        _model_name: &str,
        options: SessionOptions,
    ) -> Result<Box<dyn InferenceSession>, WorkerError> {
        let builder = Session::builder()
            .map_err(|error| WorkerError::Model(error.to_string()))?
            .with_intra_threads(options.intra_threads)
            .map_err(|error| WorkerError::Model(error.to_string()))?
            // Stop the single intra-op thread from spin-waiting between sparse ~1fps
            // inferences; spinning is the classic ONNX Runtime source of high idle CPU.
            .with_intra_op_spinning(false)
            .map_err(|error| WorkerError::Model(error.to_string()))?;
        let mut builder = if options.graph_optimization_all {
            builder
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|error| WorkerError::Model(error.to_string()))?
        } else {
            builder
        };
        if options.execution_provider == ExecutionProvider::DirectMl {
            // DirectML requires memory-pattern optimization disabled and sequential
            // execution (pilot harness recipe). The DML API is resolved at runtime
            // from the loaded onnxruntime.dll via the load-dynamic path.
            builder = builder
                .with_memory_pattern(false)
                .map_err(|error| WorkerError::Model(error.to_string()))?
                .with_parallel_execution(false)
                .map_err(|error| WorkerError::Model(error.to_string()))?
                .with_execution_providers([DirectML::default().build()])
                .map_err(|error| WorkerError::Model(error.to_string()))?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|error| WorkerError::Model(error.to_string()))?;
        Ok(Box::new(OrtSession(session)))
    }

    fn wait_before_retry(&mut self, delay: Duration) {
        thread::sleep(delay);
    }
}

#[derive(Debug, Default)]
struct PerformanceMarks {
    marks: HashMap<String, Instant>,
}

impl PerformanceMarks {
    fn mark(&mut self, name: String) -> String {
        self.marks.insert(name.clone(), Instant::now());
        name
    }

    fn measure(&self, name: &str, start: &str, end: &str) {
        let (Some(start), Some(end)) = (self.marks.get(start), self.marks.get(end)) else {
            return;
        };
        let elapsed_ms = end.duration_since(*start).as_secs_f64() * 1_000.0;
        log::debug!(target: "detection", "[Unified Worker] {name}: {elapsed_ms:.2}ms");
    }

    fn cleanup(&mut self, frame: u64, keep_frames: u64) {
        let cutoff = frame.saturating_sub(keep_frames);
        self.marks.retain(|name, _| {
            name.strip_prefix("frame-")
                .and_then(|rest| rest.split('-').next())
                .and_then(|value| value.parse::<u64>().ok())
                .is_none_or(|mark_frame| mark_frame >= cutoff)
        });
    }
}

/// Unified native inference worker. It owns both ONNX sessions and the two
/// optional classifier models, matching the browser worker's state machine.
pub struct InferenceWorker<F, L = NoopLogger, R = OrtRuntime>
where
    F: ModelFactory,
    L: WorkerLogger,
    R: InferenceRuntime,
{
    model_factory: F,
    logger: L,
    runtime: R,
    rtmdet_session: Option<Box<dyn InferenceSession>>,
    rtmpose_session: Option<Box<dyn InferenceSession>>,
    // Supplementary NLF-L depth session on DirectML. Absent whenever the GPU/DML
    // runtime is unavailable; its absence never fails init or a frame.
    nlf_session: Option<Box<dyn InferenceSession>>,
    is_initialized: bool,
    last_rtmdet_path: String,
    last_rtmpose_path: String,
    last_nlf_path: Option<String>,
    loaded_posture_model: Option<Box<dyn ClassifierModel>>,
    loaded_presence_model: Option<Box<dyn ClassifierModel>>,
    frame_counter: u64,
    performance: PerformanceMarks,
}

impl<F> InferenceWorker<F, NoopLogger, OrtRuntime>
where
    F: ModelFactory,
{
    pub fn new(model_factory: F) -> Self {
        Self::with_logger(model_factory, NoopLogger)
    }
}

impl<F, L> InferenceWorker<F, L, OrtRuntime>
where
    F: ModelFactory,
    L: WorkerLogger,
{
    pub fn with_logger(model_factory: F, logger: L) -> Self {
        Self::with_runtime(model_factory, logger, OrtRuntime)
    }
}

impl<F, L, R> InferenceWorker<F, L, R>
where
    F: ModelFactory,
    L: WorkerLogger,
    R: InferenceRuntime,
{
    pub fn with_runtime(model_factory: F, logger: L, runtime: R) -> Self {
        Self {
            model_factory,
            logger,
            runtime,
            rtmdet_session: None,
            rtmpose_session: None,
            nlf_session: None,
            is_initialized: false,
            last_rtmdet_path: String::new(),
            last_rtmpose_path: String::new(),
            last_nlf_path: None,
            loaded_posture_model: None,
            loaded_presence_model: None,
            frame_counter: 0,
            performance: PerformanceMarks::default(),
        }
    }

    /// Handles one typed worker message and returns the emitted response(s).
    pub fn handle_message(&mut self, message: InferenceWorkerMessage) -> Vec<WorkerResponse> {
        let response = match message {
            InferenceWorkerMessage::Initialize { payload } => self.initialize(
                &payload.rtmdet_path,
                &payload.rtmw3d_path,
                payload.nlf_path.as_deref(),
            ),
            InferenceWorkerMessage::Process { payload } => {
                self.process_frame(payload.image_data, payload.request_id)
            }
            InferenceWorkerMessage::LoadPostureModel { payload } => {
                match self.load_posture_model(payload.posture_model) {
                    Ok(()) => WorkerResponse::ClassifierLoaded { success: true },
                    Err(error) => WorkerResponse::Error {
                        error: format!("Failed to load posture model: {error}"),
                        request_id: None,
                        details: None,
                        success: Some(false),
                    },
                }
            }
            InferenceWorkerMessage::LoadPresenceModel { payload } => {
                match self.load_presence_model(payload.presence_model) {
                    Ok(()) => WorkerResponse::ClassifierLoaded { success: true },
                    Err(error) => WorkerResponse::Error {
                        error: format!("Failed to load presence model: {error}"),
                        request_id: None,
                        details: None,
                        success: Some(false),
                    },
                }
            }
            InferenceWorkerMessage::UnloadClassifier => self.unload_classifier(),
            InferenceWorkerMessage::SetLogLevel { payload } => {
                if let Some(payload) = payload {
                    if let Some(log_param) = payload.log_param {
                        self.logger.set_from_url_param(&log_param);
                        let normalized = if log_param.is_empty() {
                            "none"
                        } else {
                            log_param.as_str()
                        };
                        self.logger
                            .info(&format!("[Unified Worker] Log level updated: {normalized}"));
                    }
                }
                return Vec::new();
            }
        };
        vec![response]
    }

    fn initialize(
        &mut self,
        rtmdet_path: &str,
        rtmpose_path: &str,
        nlf_path: Option<&str>,
    ) -> WorkerResponse {
        self.last_rtmdet_path = rtmdet_path.to_owned();
        self.last_rtmpose_path = rtmpose_path.to_owned();
        self.last_nlf_path = nlf_path.map(str::to_owned);
        self.logger
            .info("[Unified Worker] Initializing ONNX Runtime");

        let result = (|| {
            self.rtmdet_session = Some(load_model_with_retry(
                &mut self.runtime,
                &self.logger,
                rtmdet_path,
                "RTMDet",
            )?);
            self.rtmpose_session = Some(load_model_with_retry(
                &mut self.runtime,
                &self.logger,
                rtmpose_path,
                "RTMPose-M",
            )?);
            self.is_initialized = true;
            Ok::<(), WorkerError>(())
        })();

        match result {
            Ok(()) => {
                self.logger
                    .info("[Unified Worker] Both models loaded successfully");
                // The NLF-L depth session is a supplementary tap: a failed load must
                // never fail overall initialization. It is attempted only after the
                // required detector/pose models are up.
                self.nlf_session = self.try_load_nlf_session(nlf_path);
                WorkerResponse::Initialized {
                    provider: "native".to_owned(),
                }
            }
            Err(error) => {
                self.is_initialized = false;
                self.rtmdet_session = None;
                self.rtmpose_session = None;
                self.nlf_session = None;
                self.logger
                    .error(&format!("[Unified Worker] Initialization error: {error}"));
                WorkerResponse::Error {
                    error: format!(
                        "Failed to initialize models after 3 attempts: {error}. Please check your internet connection and reload the page."
                    ),
                    request_id: None,
                    details: None,
                    success: None,
                }
            }
        }
    }

    /// Attempts a single DirectML session load for the optional NLF-L depth model.
    /// Any failure (no path, empty path, absent GPU/DML runtime, load error) yields
    /// `None` with a warning and never propagates — the depth feature is a
    /// best-effort supplementary tap. No retry: a missing GPU is not transient.
    fn try_load_nlf_session(
        &mut self,
        nlf_path: Option<&str>,
    ) -> Option<Box<dyn InferenceSession>> {
        let path = nlf_path?;
        if path.is_empty() {
            return None;
        }
        let options = SessionOptions {
            execution_provider: ExecutionProvider::DirectMl,
            ..SessionOptions::default()
        };
        match self.runtime.create_session(path, "NLF-L", options) {
            Ok(session) => {
                self.logger
                    .info("[Unified Worker] NLF-L depth session initialized on DirectML");
                Some(session)
            }
            Err(error) => {
                self.logger.warn(&format!(
                    "[Unified Worker] NLF-L depth session unavailable ({error}); depth features disabled"
                ));
                None
            }
        }
    }

    fn load_posture_model(&mut self, serialized: SerializedModel) -> Result<(), WorkerError> {
        self.logger
            .info("[Unified Worker] Loading posture model into worker");
        if let Some(mut model) = self.loaded_posture_model.take() {
            model.dispose();
        }
        self.loaded_posture_model = Some(
            self.model_factory
                .load(serialized)
                .map_err(WorkerError::Model)?,
        );
        Ok(())
    }

    fn load_presence_model(&mut self, serialized: SerializedModel) -> Result<(), WorkerError> {
        self.logger
            .info("[Unified Worker] Loading presence model into worker");
        if let Some(mut model) = self.loaded_presence_model.take() {
            model.dispose();
        }
        self.loaded_presence_model = Some(
            self.model_factory
                .load(serialized)
                .map_err(WorkerError::Model)?,
        );
        Ok(())
    }

    /// Builds an immutable classifier generation before publishing it. Each role
    /// is optional: a posture-only generation carries `presence = None`, and a
    /// presence-only generation carries `posture = None`. If any provided model
    /// fails to load, the currently active generation is left intact.
    ///
    /// Publishing is generation-atomic: the new generation replaces the whole
    /// active generation, so a role absent from the new generation is unloaded.
    /// An unloaded presence model makes runtime presence fall back to the RTMDet
    /// detector confidence; an unloaded posture model yields no good-probability.
    pub fn publish_model_pair(
        &mut self,
        posture: Option<SerializedModel>,
        presence: Option<SerializedModel>,
    ) -> Result<(), WorkerError> {
        let posture_replacement = match posture {
            Some(model) => Some(self.model_factory.load(model).map_err(WorkerError::Model)?),
            None => None,
        };
        let presence_replacement = match presence {
            Some(model) => match self.model_factory.load(model) {
                Ok(replacement) => Some(replacement),
                Err(error) => {
                    if let Some(mut model) = posture_replacement {
                        model.dispose();
                    }
                    return Err(WorkerError::Model(error));
                }
            },
            None => None,
        };

        let old_posture = std::mem::replace(&mut self.loaded_posture_model, posture_replacement);
        let old_presence = std::mem::replace(&mut self.loaded_presence_model, presence_replacement);
        if let Some(mut model) = old_posture {
            model.dispose();
        }
        if let Some(mut model) = old_presence {
            model.dispose();
        }
        Ok(())
    }

    fn unload_classifier(&mut self) -> WorkerResponse {
        if let Some(mut model) = self.loaded_posture_model.take() {
            model.dispose();
        }
        if let Some(mut model) = self.loaded_presence_model.take() {
            model.dispose();
        }
        WorkerResponse::ClassifierUnloaded
    }

    fn classify_features(
        &mut self,
        result: &NativeInferenceResult,
    ) -> Result<Option<ClassificationResult>, WorkerError> {
        if self.loaded_presence_model.is_none() && self.loaded_posture_model.is_none() {
            return Ok(None);
        }

        // No person detected: RTMDet returned no box, so the frame is "away" by
        // construction. The presence model is deliberately NOT run here — it
        // needs keypoint-derived features that an empty frame does not have
        // (e.g. "keypoint_scores not available in this container"), so running
        // it would error. Report away (present_probability = 0.0) directly and
        // skip posture, mirroring the presence<0.5 gate on the person-found
        // path. A posture-only worker still emits no classification for an empty
        // frame: there is no presence verdict to report.
        if !result.person_found {
            if self.loaded_presence_model.is_none() {
                return Ok(None);
            }
            return Ok(Some(ClassificationResult {
                present_probability: 0.0,
                good_probability: None,
            }));
        }

        let input = ClassificationInput {
            features: &result.features,
            bbox: result.bbox.as_ref(),
            keypoints: result.keypoints.as_deref(),
        };

        let presence_was_classified = self.loaded_presence_model.is_some();
        let present_probability = if let Some(model) = self.loaded_presence_model.as_mut() {
            model.predict(&input).map_err(|error| {
                self.logger
                    .error(&format!("[Unified Worker] Classification error: {error}"));
                WorkerError::Inference(error)
            })?
        } else {
            result
                .bbox
                .as_ref()
                .map(|bbox| bbox.original.score)
                .unwrap_or(0.0)
        };
        validate_probability("presentProbability", present_probability)?;

        if presence_was_classified && !should_run_posture_for_presence(present_probability) {
            return Ok(Some(ClassificationResult {
                present_probability,
                good_probability: None,
            }));
        }

        let good_probability = if let Some(model) = self.loaded_posture_model.as_mut() {
            let probability = model.predict(&input).map_err(|error| {
                self.logger
                    .error(&format!("[Unified Worker] Classification error: {error}"));
                WorkerError::Inference(error)
            })?;
            validate_probability("goodProbability", probability)?;
            Some(probability)
        } else {
            None
        };

        Ok(Some(ClassificationResult {
            present_probability,
            good_probability,
        }))
    }

    fn process_frame(&mut self, image_data: ImageData, request_id: u64) -> WorkerResponse {
        let frame_start = self.mark("total-start");
        self.frame_counter = self.frame_counter.saturating_add(1);

        if !self.is_initialized || self.rtmdet_session.is_none() || self.rtmpose_session.is_none() {
            if self.last_rtmdet_path.is_empty() || self.last_rtmpose_path.is_empty() {
                self.cleanup_performance_if_due();
                return WorkerResponse::Error {
                    error: "Models failed to initialize. Please reload the page.".to_owned(),
                    request_id: Some(request_id),
                    details: None,
                    success: None,
                };
            }
            let rtmdet_path = self.last_rtmdet_path.clone();
            let rtmpose_path = self.last_rtmpose_path.clone();
            let nlf_path = self.last_nlf_path.clone();
            let init = self.initialize(&rtmdet_path, &rtmpose_path, nlf_path.as_deref());
            if !matches!(init, WorkerResponse::Initialized { .. }) {
                self.cleanup_performance_if_due();
                return WorkerResponse::Error {
                    error: "Models failed to initialize. Please reload the page.".to_owned(),
                    request_id: Some(request_id),
                    details: None,
                    success: None,
                };
            }
        }

        let result = self.process_frame_inner(&image_data);
        let frame_end = self.mark("total-end");
        self.measure("frame_total", &frame_start, &frame_end);
        // Cleanup is frame-scoped, not success-scoped: malformed input,
        // classifier failures, and validation errors must not retain marks.
        self.cleanup_performance_if_due();

        match result {
            Ok(mut result) => {
                if result.person_found
                    && (self.loaded_posture_model.is_some() || self.loaded_presence_model.is_some())
                {
                    let classifier_start = self.mark("classifier-start");
                    // Oracle parity: on the person-found path classifyFeatures wraps
                    // prediction in try/catch and returns null on any error
                    // (inference-worker.ts:780-783); processFrame then omits
                    // classification and still emits a successful result
                    // (inference-worker.ts:1024-1026). Degrade gracefully instead of
                    // aborting the whole frame. classify_features already logs the
                    // failure, so no additional log here.
                    match self.classify_features(&result) {
                        Ok(classification) => result.classification = classification,
                        Err(_) => result.classification = None,
                    }
                    let classifier_end = self.mark("classifier-end");
                    self.measure("classifier_inference", &classifier_start, &classifier_end);
                }
                if let Err(error) = validate_native_result(&result) {
                    self.logger
                        .error(&format!("[Unified Worker] Processing error: {error}"));
                    return WorkerResponse::Error {
                        error: error.to_string(),
                        request_id: Some(request_id),
                        details: None,
                        success: None,
                    };
                }
                WorkerResponse::Result { request_id, result }
            }
            Err(error) => {
                self.logger
                    .error(&format!("[Unified Worker] Processing error: {error}"));
                WorkerResponse::Error {
                    error: error.to_string(),
                    request_id: Some(request_id),
                    details: None,
                    success: None,
                }
            }
        }
    }

    fn cleanup_performance_if_due(&mut self) {
        if self.frame_counter.is_multiple_of(MARK_CLEANUP_INTERVAL) {
            self.performance
                .cleanup(self.frame_counter, MARK_CLEANUP_INTERVAL * 2);
        }
    }

    fn process_frame_inner(
        &mut self,
        image_data: &ImageData,
    ) -> Result<NativeInferenceResult, WorkerError> {
        let rtmdet_start = self.mark("rtmdet-start");
        let preprocessed = preprocess_rtmdet(image_data)?;
        let rtm_det_preprocess_end = self.mark("rtmdet-preprocess-end");
        self.measure("rtmdet_preprocess", &rtmdet_start, &rtm_det_preprocess_end);

        let rtmdet_input = Array4::from_shape_vec(
            (1, 3, RTMDET_INPUT_HEIGHT, RTMDET_INPUT_WIDTH),
            preprocessed.tensor,
        )
        .map_err(|error| WorkerError::Tensor(error.to_string()))?;
        let (dets, labels, raw_cls_p5, raw_reg_p5) = {
            let session = self
                .rtmdet_session
                .as_mut()
                .ok_or_else(|| WorkerError::Inference("RTMDet session is not loaded".to_owned()))?;
            let outputs = session.run(&rtmdet_input)?;
            (
                take_f32_output(&outputs, "dets")?,
                take_i64_output(&outputs, "labels")?,
                take_f32_output(&outputs, RTMDET_CLS_P5)?,
                take_f32_output(&outputs, RTMDET_REG_P5)?,
            )
        };
        let rtm_det_end = self.mark("rtmdet-end");
        self.measure("rtmdet_total", &rtmdet_start, &rtm_det_end);

        let bbox = select_person_bbox(
            &dets,
            &labels,
            preprocessed.scale,
            preprocessed.pad_w,
            preprocessed.pad_h,
            image_data.width,
            image_data.height,
        )?;
        let rtm_det_features = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5)?;

        let Some(bbox) = bbox else {
            let mut features = FeatureMap::new();
            features.insert(FeatureId::RtmDetExtracted, rtm_det_features);
            // Deliberate deviation from oracle parity: the browser worker ran the
            // presence model on this no-person branch (inference-worker.ts:894-900),
            // but that model's keypoint-derived features do not exist on an empty
            // frame and make it error ("keypoint_scores not available in this
            // container"). classify_features now short-circuits a no-person result
            // to "away" without running the presence model, so no prediction error
            // can propagate from this path.
            let classification = self.classify_features(&NativeInferenceResult {
                person_found: false,
                bbox: None,
                keypoints: None,
                features: features.clone(),
                classification: None,
            })?;
            return Ok(NativeInferenceResult {
                person_found: false,
                bbox: None,
                keypoints: None,
                features,
                classification,
            });
        };

        let rtmpose_start = self.mark("rtmpose-start");
        let expanded = expand_bbox(&bbox, 0.2, image_data.width, image_data.height)?;
        let cropped = crop_image_data(image_data, &expanded.expanded)?;
        let pose_tensor = preprocess_rtmpose(&cropped)?;
        let pose_input = Array4::from_shape_vec(
            (1, 3, RTMPOSE_INPUT_HEIGHT, RTMPOSE_INPUT_WIDTH),
            pose_tensor,
        )
        .map_err(|error| WorkerError::Tensor(error.to_string()))?;
        let (simcc_x, simcc_y, backbone, gau) = {
            let session = self.rtmpose_session.as_mut().ok_or_else(|| {
                WorkerError::Inference("RTMPose session is not loaded".to_owned())
            })?;
            let outputs = session.run(&pose_input)?;
            (
                take_f32_output(&outputs, "simcc_x")?,
                take_f32_output(&outputs, "simcc_y")?,
                take_f32_output(&outputs, "backbone_features")?,
                take_f32_output(&outputs, "gau_features")?,
            )
        };

        let keypoints = decode_simcc(&simcc_x, &simcc_y)?;
        let mut features = FeatureMap::new();
        features.insert(
            FeatureId::BackboneFeatures,
            pool_backbone_features(&backbone)
                .map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(
            FeatureId::BackboneFeaturesMax,
            pool_backbone_features_max(&backbone)
                .map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(
            FeatureId::BackboneFeaturesStd,
            pool_backbone_features_std(&backbone)
                .map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(
            FeatureId::GauFeatures,
            pool_gau_features(&gau).map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(
            FeatureId::GauFeaturesMax,
            pool_gau_features_max(&gau).map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(
            FeatureId::GauFeaturesStd,
            pool_gau_features_std(&gau).map_err(|error| WorkerError::Feature(error.to_string()))?,
        );
        features.insert(FeatureId::RtmDetExtracted, rtm_det_features);
        let rtmpose_end = self.mark("rtmpose-end");
        self.measure("rtmpose_total", &rtmpose_start, &rtmpose_end);

        // Supplementary NLF-L depth tap. Uses the ORIGINAL (un-expanded) RTMDet
        // bbox. Never fatal: any failure logs and skips the feature insert so the
        // primary detection/pose pipeline always returns a valid result.
        if self.nlf_session.is_some() {
            match self.run_nlf_depth(image_data, &bbox) {
                Ok(Some(values)) => {
                    features.insert(FeatureId::NlfDepth, values);
                }
                Ok(None) => {}
                Err(error) => {
                    self.logger.warn(&format!(
                        "[Unified Worker] NLF depth features skipped for this frame: {error}"
                    ));
                }
            }
        }

        let normalized_bbox =
            normalize_expanded_bbox(&expanded, image_data.width, image_data.height)?;
        let transformed_keypoints = transform_keypoints(
            &keypoints,
            &expanded.expanded,
            cropped.width,
            cropped.height,
            image_data.width,
            image_data.height,
        )?;

        Ok(NativeInferenceResult {
            person_found: true,
            bbox: Some(normalized_bbox),
            keypoints: Some(transformed_keypoints),
            features,
            classification: None,
        })
    }

    /// Preprocesses the original-bbox square crop, runs the NLF-L depth session,
    /// and derives the body-intrinsic depth features. Returns `Ok(None)` when the
    /// pose is too degenerate for a meaningful feature (extractor's decision).
    fn run_nlf_depth(
        &mut self,
        image_data: &ImageData,
        bbox: &BoundingBox,
    ) -> Result<Option<Vec<f32>>, WorkerError> {
        let input = preprocess_nlf(image_data, bbox)?;
        let session = self
            .nlf_session
            .as_mut()
            .ok_or_else(|| WorkerError::Inference("NLF session is not loaded".to_owned()))?;
        let outputs = session.run(&input)?;
        let coords3d = take_f32_output(&outputs, NLF_COORDS3D_OUTPUT)?;
        let uncertainty = take_f32_output(&outputs, NLF_UNCERTAINTY_OUTPUT)?;
        extract_nlf_depth_features(&coords3d, &uncertainty)
            .map_err(|error| WorkerError::Feature(error.to_string()))
    }

    fn mark(&mut self, stage: &str) -> String {
        self.performance
            .mark(format!("frame-{}-{stage}", self.frame_counter))
    }

    fn measure(&self, name: &str, start: &str, end: &str) {
        self.performance.measure(name, start, end);
    }
}

impl<F, L, R> Drop for InferenceWorker<F, L, R>
where
    F: ModelFactory,
    L: WorkerLogger,
    R: InferenceRuntime,
{
    fn drop(&mut self) {
        if let Some(mut model) = self.loaded_posture_model.take() {
            model.dispose();
        }
        if let Some(mut model) = self.loaded_presence_model.take() {
            model.dispose();
        }
    }
}

/// Exponential retry policy shared by production loading and compatibility tests.
pub fn retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(
        (1_000_u64.saturating_mul(2_u64.saturating_pow((attempt.saturating_sub(1)) as u32)))
            .min(8_000),
    )
}

fn load_model_with_retry<R: InferenceRuntime, L: WorkerLogger>(
    runtime: &mut R,
    logger: &L,
    path: &str,
    model_name: &str,
) -> Result<Box<dyn InferenceSession>, WorkerError> {
    let mut last_error = None;
    for attempt in 1..=MODEL_RETRIES {
        logger.info(&format!(
            "[Worker] Loading {model_name} (attempt {attempt}/{MODEL_RETRIES})"
        ));
        match runtime.create_session(path, model_name, SessionOptions::default()) {
            Ok(session) => {
                logger.info(&format!("[Worker] {model_name} loaded successfully"));
                return Ok(session);
            }
            Err(error) => {
                logger.warn(&format!(
                    "[Worker] {model_name} load attempt {attempt} failed: {error}"
                ));
                last_error = Some(error);
                if attempt < MODEL_RETRIES {
                    let delay = retry_delay(attempt);
                    logger.warn(&format!(
                        "[Worker] Retrying {model_name} in {}ms",
                        delay.as_millis()
                    ));
                    runtime.wait_before_retry(delay);
                }
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| WorkerError::Model(format!("{model_name} model loading failed"))))
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityPreprocessInfo {
    pub tensor: Vec<f32>,
    pub scale: f64,
    pub scaled_width: usize,
    pub scaled_height: usize,
    pub pad_width: f64,
    pub pad_height: f64,
}

#[derive(Debug)]
struct PreprocessInfo {
    tensor: Vec<f32>,
    scale: f64,
    pad_w: f64,
    pad_h: f64,
}

fn preprocess_rtmdet(image: &ImageData) -> Result<PreprocessInfo, WorkerError> {
    validate_image(image)?;
    let orig_w = image.width as usize;
    let orig_h = image.height as usize;
    let scale =
        (RTMDET_INPUT_WIDTH as f64 / orig_w as f64).min(RTMDET_INPUT_HEIGHT as f64 / orig_h as f64);
    let scaled_w = (orig_w as f64 * scale).round() as usize;
    let scaled_h = (orig_h as f64 * scale).round() as usize;
    let pad_w = ((RTMDET_INPUT_WIDTH - scaled_w) / 2) as f64;
    let pad_h = ((RTMDET_INPUT_HEIGHT - scaled_h) / 2) as f64;
    let plane = RTMDET_INPUT_WIDTH * RTMDET_INPUT_HEIGHT;
    let mut tensor = vec![0.0_f32; plane * 3];

    for i in 0..plane {
        tensor[i] = ((114.0 - RTMDET_MEAN_BGR[0]) / RTMDET_STD_BGR[0]) as f32;
        tensor[plane + i] = ((114.0 - RTMDET_MEAN_BGR[1]) / RTMDET_STD_BGR[1]) as f32;
        tensor[2 * plane + i] = ((114.0 - RTMDET_MEAN_BGR[2]) / RTMDET_STD_BGR[2]) as f32;
    }

    for h in 0..scaled_h {
        for w in 0..scaled_w {
            let src_x = w as f64 / scale;
            let src_y = h as f64 / scale;
            let src_x0 = src_x.floor() as usize;
            let src_y0 = src_y.floor() as usize;
            let src_x1 = (src_x0 + 1).min(orig_w - 1);
            let src_y1 = (src_y0 + 1).min(orig_h - 1);
            let dx = src_x - src_x0 as f64;
            let dy = src_y - src_y0 as f64;
            let p00 = (src_y0 * orig_w + src_x0) * 4;
            let p01 = (src_y0 * orig_w + src_x1) * 4;
            let p10 = (src_y1 * orig_w + src_x0) * 4;
            let p11 = (src_y1 * orig_w + src_x1) * 4;
            let channel = |offset: usize| {
                (1.0 - dx) * (1.0 - dy) * image.data[p00 + offset] as f64
                    + dx * (1.0 - dy) * image.data[p01 + offset] as f64
                    + (1.0 - dx) * dy * image.data[p10 + offset] as f64
                    + dx * dy * image.data[p11 + offset] as f64
            };
            let index = (h + pad_h as usize) * RTMDET_INPUT_WIDTH + w + pad_w as usize;
            tensor[index] = ((channel(2) - RTMDET_MEAN_BGR[0]) / RTMDET_STD_BGR[0]) as f32;
            tensor[plane + index] = ((channel(1) - RTMDET_MEAN_BGR[1]) / RTMDET_STD_BGR[1]) as f32;
            tensor[2 * plane + index] =
                ((channel(0) - RTMDET_MEAN_BGR[2]) / RTMDET_STD_BGR[2]) as f32;
        }
    }

    Ok(PreprocessInfo {
        tensor,
        scale,
        pad_w,
        pad_h,
    })
}

/// Returns whether the production presence cascade proceeds to posture.
pub fn should_run_posture_for_presence(present_probability: f64) -> bool {
    present_probability >= 0.5
}

/// Executes production detector selection for synthetic compatibility rows.
pub fn compatibility_select_person_bbox(
    dets: &[f32],
    labels: &[i64],
    image_width: u32,
    image_height: u32,
) -> Result<Option<BoundingBox>, WorkerError> {
    select_person_bbox(dets, labels, 1.0, 0.0, 0.0, image_width, image_height)
}

/// Executes production SimCC tie handling for synthetic compatibility rows.
pub fn compatibility_decode_simcc(
    simcc_x: &[f32],
    simcc_y: &[f32],
) -> Result<Vec<Keypoint>, WorkerError> {
    decode_simcc(simcc_x, simcc_y)
}

/// Deterministic compatibility seam over the production RTMDet preprocessor.
pub fn compatibility_preprocess_rtmdet(
    image: &ImageData,
) -> Result<CompatibilityPreprocessInfo, WorkerError> {
    let processed = preprocess_rtmdet(image)?;
    Ok(CompatibilityPreprocessInfo {
        tensor: processed.tensor,
        scale: processed.scale,
        scaled_width: (image.width as f64 * processed.scale).round() as usize,
        scaled_height: (image.height as f64 * processed.scale).round() as usize,
        pad_width: processed.pad_w,
        pad_height: processed.pad_h,
    })
}

fn preprocess_rtmpose(image: &ImageData) -> Result<Vec<f32>, WorkerError> {
    validate_image(image)?;
    let width = image.width as usize;
    let height = image.height as usize;
    let plane = RTMPOSE_INPUT_WIDTH * RTMPOSE_INPUT_HEIGHT;
    let scale_x = width as f64 / RTMPOSE_INPUT_WIDTH as f64;
    let scale_y = height as f64 / RTMPOSE_INPUT_HEIGHT as f64;
    let mut tensor = vec![0.0_f32; plane * 3];

    for h in 0..RTMPOSE_INPUT_HEIGHT {
        for w in 0..RTMPOSE_INPUT_WIDTH {
            let src_x = (w as f64 * scale_x).floor() as usize;
            let src_y = (h as f64 * scale_y).floor() as usize;
            let pixel = (src_y * width + src_x) * 4;
            let index = h * RTMPOSE_INPUT_WIDTH + w;
            tensor[index] =
                (f32::from(image.data[pixel]) - RTMPOSE_MEAN_RGB[0]) / RTMPOSE_STD_RGB[0];
            tensor[plane + index] =
                (f32::from(image.data[pixel + 1]) - RTMPOSE_MEAN_RGB[1]) / RTMPOSE_STD_RGB[1];
            tensor[2 * plane + index] =
                (f32::from(image.data[pixel + 2]) - RTMPOSE_MEAN_RGB[2]) / RTMPOSE_STD_RGB[2];
        }
    }
    Ok(tensor)
}

/// Deterministic compatibility seam over the production RTMPose preprocessor.
pub fn compatibility_preprocess_rtmpose(image: &ImageData) -> Result<Vec<f32>, WorkerError> {
    preprocess_rtmpose(image)
}

/// Builds the NLF-L input tensor: a square crop centered on the ORIGINAL RTMDet
/// bbox, side `max(width, height)`, bilinearly resampled to `384x384` with
/// edge-replicated borders, RGB channels scaled by `/255.0` into `[1,3,384,384]`.
/// Mirrors the pilot crop (`cv2.warpAffine`, `INTER_LINEAR`, `BORDER_REPLICATE`);
/// plain `/255.0` preprocessing per the integration addendum (no gamma).
fn preprocess_nlf(image: &ImageData, bbox: &BoundingBox) -> Result<Array4<f32>, WorkerError> {
    validate_image(image)?;
    let width = image.width as usize;
    let height = image.height as usize;
    let side = (bbox.x2 - bbox.x1).max(bbox.y2 - bbox.y1);
    if !side.is_finite() || side <= 0.0 {
        return Err(WorkerError::InvalidInput(
            "NLF crop square has a non-positive side".to_owned(),
        ));
    }
    let center_x = (bbox.x1 + bbox.x2) / 2.0;
    let center_y = (bbox.y1 + bbox.y2) / 2.0;
    let origin_x = center_x - side / 2.0;
    let origin_y = center_y - side / 2.0;
    let step = side / NLF_INPUT_SIZE as f64;
    let plane = NLF_INPUT_SIZE * NLF_INPUT_SIZE;
    let max_x = (width - 1) as i64;
    let max_y = (height - 1) as i64;
    let mut tensor = vec![0.0_f32; plane * 3];

    for out_y in 0..NLF_INPUT_SIZE {
        let src_y = origin_y + out_y as f64 * step;
        let floor_y = src_y.floor();
        let weight_y = src_y - floor_y;
        let y_lo = (floor_y as i64).clamp(0, max_y) as usize;
        let y_hi = (floor_y as i64 + 1).clamp(0, max_y) as usize;
        for out_x in 0..NLF_INPUT_SIZE {
            let src_x = origin_x + out_x as f64 * step;
            let floor_x = src_x.floor();
            let weight_x = src_x - floor_x;
            let x_lo = (floor_x as i64).clamp(0, max_x) as usize;
            let x_hi = (floor_x as i64 + 1).clamp(0, max_x) as usize;
            let index = out_y * NLF_INPUT_SIZE + out_x;
            for channel in 0..3 {
                let p00 = image.data[(y_lo * width + x_lo) * 4 + channel] as f64;
                let p01 = image.data[(y_lo * width + x_hi) * 4 + channel] as f64;
                let p10 = image.data[(y_hi * width + x_lo) * 4 + channel] as f64;
                let p11 = image.data[(y_hi * width + x_hi) * 4 + channel] as f64;
                let top = p00 * (1.0 - weight_x) + p01 * weight_x;
                let bottom = p10 * (1.0 - weight_x) + p11 * weight_x;
                let value = top * (1.0 - weight_y) + bottom * weight_y;
                tensor[channel * plane + index] = (value / 255.0) as f32;
            }
        }
    }

    Array4::from_shape_vec((1, 3, NLF_INPUT_SIZE, NLF_INPUT_SIZE), tensor)
        .map_err(|error| WorkerError::Tensor(error.to_string()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityCropOutputs {
    pub expanded_pixels: ExpandedBbox,
    pub cropped: ImageData,
    pub normalized_bbox: ExpandedBbox,
    pub normalized_keypoints: Vec<Keypoint>,
}

/// Exercises the same expansion, RGBA crop, keypoint transform, and output
/// normalization used by the production detector-to-pose cascade.
pub fn compatibility_crop_pipeline(
    image: &ImageData,
    bbox: &BoundingBox,
    pose_keypoints: &[Keypoint],
) -> Result<CompatibilityCropOutputs, WorkerError> {
    let expanded_pixels = expand_bbox(bbox, 0.2, image.width, image.height)?;
    let cropped = crop_image_data(image, &expanded_pixels.expanded)?;
    let normalized_keypoints = transform_keypoints(
        pose_keypoints,
        &expanded_pixels.expanded,
        cropped.width,
        cropped.height,
        image.width,
        image.height,
    )?;
    let normalized_bbox = normalize_expanded_bbox(&expanded_pixels, image.width, image.height)?;
    Ok(CompatibilityCropOutputs {
        expanded_pixels,
        cropped,
        normalized_bbox,
        normalized_keypoints,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityModelOutputs {
    pub pose_runs: usize,
    pub bbox: Option<ExpandedBbox>,
    pub rtmdet_pooled: Vec<f32>,
    pub keypoints: Vec<Keypoint>,
    pub backbone_avg: Vec<f32>,
    pub backbone_max: Vec<f32>,
    pub backbone_std: Vec<f32>,
    pub gau_avg: Vec<f32>,
    pub gau_max: Vec<f32>,
    pub gau_std: Vec<f32>,
}

struct CompatibilityModelFactory;

impl ModelFactory for CompatibilityModelFactory {
    fn load(&self, _serialized: SerializedModel) -> Result<Box<dyn ClassifierModel>, String> {
        Err("compatibility model factory does not load classifiers".to_owned())
    }
}

/// Runs the actual production frame-processing entry with independently loaded
/// ORT sessions, then exposes its owned result buffers for fixture comparison.
pub fn compatibility_run_models(
    rtmdet_path: &str,
    rtmpose_path: &str,
    image: &ImageData,
) -> Result<CompatibilityModelOutputs, WorkerError> {
    let mut runtime = OrtRuntime;
    let logger = NoopLogger;
    let detector = load_model_with_retry(&mut runtime, &logger, rtmdet_path, "RTMDet")?;
    let pose = load_model_with_retry(&mut runtime, &logger, rtmpose_path, "RTMPose-M")?;
    let mut worker = InferenceWorker::new(CompatibilityModelFactory);
    worker.rtmdet_session = Some(detector);
    worker.rtmpose_session = Some(pose);
    let result = worker.process_frame_inner(image)?;
    let pose_runs = usize::from(result.person_found);
    let mut features = result.features;
    Ok(CompatibilityModelOutputs {
        pose_runs,
        bbox: result.bbox,
        rtmdet_pooled: features
            .remove(&FeatureId::RtmDetExtracted)
            .unwrap_or_default(),
        keypoints: result.keypoints.unwrap_or_default(),
        backbone_avg: features
            .remove(&FeatureId::BackboneFeatures)
            .unwrap_or_default(),
        backbone_max: features
            .remove(&FeatureId::BackboneFeaturesMax)
            .unwrap_or_default(),
        backbone_std: features
            .remove(&FeatureId::BackboneFeaturesStd)
            .unwrap_or_default(),
        gau_avg: features.remove(&FeatureId::GauFeatures).unwrap_or_default(),
        gau_max: features
            .remove(&FeatureId::GauFeaturesMax)
            .unwrap_or_default(),
        gau_std: features
            .remove(&FeatureId::GauFeaturesStd)
            .unwrap_or_default(),
    })
}

fn select_person_bbox(
    dets: &[f32],
    labels: &[i64],
    scale: f64,
    pad_w: f64,
    pad_h: f64,
    image_width: u32,
    image_height: u32,
) -> Result<Option<BoundingBox>, WorkerError> {
    if !dets.len().is_multiple_of(5) || labels.len() != dets.len() / 5 {
        return Err(WorkerError::Tensor(
            "RTMDet dets and labels have inconsistent lengths".to_owned(),
        ));
    }
    if !scale.is_finite() || scale <= 0.0 || image_width == 0 || image_height == 0 {
        return Err(WorkerError::InvalidInput(
            "RTMDet preprocessing geometry is invalid".to_owned(),
        ));
    }
    let mut selected = None;
    let mut max_area = 0.0_f64;
    for (index, label) in labels.iter().enumerate() {
        let offset = index * 5;
        let score = dets[offset + 4] as f64;
        let x1 = dets[offset] as f64;
        let y1 = dets[offset + 1] as f64;
        let x2 = dets[offset + 2] as f64;
        let y2 = dets[offset + 3] as f64;
        if !(0.0..=1.0).contains(&score) || x1 > x2 || y1 > y2 {
            return Err(WorkerError::InvalidInput(format!(
                "RTMDet detection {index} is invalid"
            )));
        }
        if *label == 0 && score > RTMDET_CONFIDENCE {
            let area = (x2 - x1) * (y2 - y1);
            if area > max_area {
                max_area = area;
                let x1_original = (x1 - pad_w) / scale;
                let y1_original = (y1 - pad_h) / scale;
                let x2_original = (x2 - pad_w) / scale;
                let y2_original = (y2 - pad_h) / scale;
                let bounded_x1 = x1_original.clamp(0.0, image_width as f64);
                let bounded_y1 = y1_original.clamp(0.0, image_height as f64);
                let bounded_x2 = x2_original.clamp(0.0, image_width as f64);
                let bounded_y2 = y2_original.clamp(0.0, image_height as f64);
                selected = Some(BoundingBox {
                    x1: bounded_x1,
                    y1: bounded_y1,
                    x2: bounded_x2,
                    y2: bounded_y2,
                    score,
                    // Oracle derives width/height from the UNclamped mapped-back
                    // coords (inference-worker.ts:655-656), so a detection that
                    // extends past the image boundary keeps its full extent.
                    width: (x2_original - x1_original).max(0.0),
                    height: (y2_original - y1_original).max(0.0),
                });
            }
        }
    }
    Ok(selected)
}

fn expand_bbox(
    bbox: &BoundingBox,
    padding: f64,
    image_width: u32,
    image_height: u32,
) -> Result<ExpandedBbox, WorkerError> {
    let width = bbox.x2 - bbox.x1;
    let height = bbox.y2 - bbox.y1;
    let expanded_x1 = (bbox.x1 - width * padding).max(0.0);
    let expanded_y1 = (bbox.y1 - height * padding).max(0.0);
    let expanded_x2 = (bbox.x2 + width * padding).min(image_width as f64);
    let expanded_y2 = (bbox.y2 + height * padding).min(image_height as f64);
    Ok(ExpandedBbox {
        original: *bbox,
        expanded: BoundingBox {
            x1: expanded_x1,
            y1: expanded_y1,
            x2: expanded_x2,
            y2: expanded_y2,
            score: bbox.score,
            width: expanded_x2 - expanded_x1,
            height: expanded_y2 - expanded_y1,
        },
    })
}

fn crop_image_data(image: &ImageData, bbox: &BoundingBox) -> Result<ImageData, WorkerError> {
    validate_image(image)?;
    let x1 = bbox.x1.max(0.0).floor() as usize;
    let y1 = bbox.y1.max(0.0).floor() as usize;
    let x2 = bbox.x2.min(image.width as f64).ceil() as usize;
    let y2 = bbox.y2.min(image.height as f64).ceil() as usize;
    if x2 <= x1 || y2 <= y1 {
        return Err(WorkerError::InvalidInput(
            "bounding box crop is empty".to_owned(),
        ));
    }
    let width = x2 - x1;
    let height = y2 - y1;
    let mut data = vec![0_u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let source = ((y1 + y) * image.width as usize + x1 + x) * 4;
            let target = (y * width + x) * 4;
            data[target..target + 4].copy_from_slice(&image.data[source..source + 4]);
        }
    }
    Ok(ImageData {
        data,
        width: width as u32,
        height: height as u32,
    })
}

fn decode_simcc(simcc_x: &[f32], simcc_y: &[f32]) -> Result<Vec<Keypoint>, WorkerError> {
    let expected_x = RTMPOSE_NUM_KEYPOINTS * RTMPOSE_SIMCC_X_WIDTH;
    let expected_y = RTMPOSE_NUM_KEYPOINTS * RTMPOSE_SIMCC_Y_WIDTH;
    if simcc_x.len() != expected_x || simcc_y.len() != expected_y {
        return Err(WorkerError::Tensor(format!(
            "invalid SimCC lengths: x={}, y={}, expected x={}, y={}",
            simcc_x.len(),
            simcc_y.len(),
            expected_x,
            expected_y
        )));
    }
    let mut keypoints = Vec::with_capacity(RTMPOSE_NUM_KEYPOINTS);
    for keypoint in 0..RTMPOSE_NUM_KEYPOINTS {
        let x_lane =
            &simcc_x[keypoint * RTMPOSE_SIMCC_X_WIDTH..(keypoint + 1) * RTMPOSE_SIMCC_X_WIDTH];
        let y_lane =
            &simcc_y[keypoint * RTMPOSE_SIMCC_Y_WIDTH..(keypoint + 1) * RTMPOSE_SIMCC_Y_WIDTH];
        let (argmax_x, max_x) = argmax(x_lane);
        let (argmax_y, max_y) = argmax(y_lane);
        // SimCC lane maxima are raw model activations, not probabilities: real frames
        // routinely exceed 1.0, and the oracle derived the keypoint score from them
        // without any range restriction. Finiteness is already enforced when the
        // tensor is read (take_f32_output).
        keypoints.push(Keypoint::new(
            argmax_x as f64 / RTMPOSE_SPLIT_RATIO,
            argmax_y as f64 / RTMPOSE_SPLIT_RATIO,
            (f64::from(max_x) + f64::from(max_y)) / 2.0,
        ));
    }
    Ok(keypoints)
}

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, current| {
            if current.1 > best.1 {
                current
            } else {
                best
            }
        })
}

fn transform_keypoints(
    keypoints: &[Keypoint],
    crop_bbox: &BoundingBox,
    crop_width: u32,
    crop_height: u32,
    image_width: u32,
    image_height: u32,
) -> Result<Vec<Keypoint>, WorkerError> {
    if image_width == 0 || image_height == 0 {
        return Err(WorkerError::InvalidInput(
            "image dimensions must be positive".to_owned(),
        ));
    }
    let scale_x = crop_width as f64 / RTMPOSE_INPUT_WIDTH as f64;
    let scale_y = crop_height as f64 / RTMPOSE_INPUT_HEIGHT as f64;
    Ok(keypoints
        .iter()
        .map(|keypoint| {
            Keypoint::new(
                (keypoint.x * scale_x + crop_bbox.x1) / image_width as f64,
                (keypoint.y * scale_y + crop_bbox.y1) / image_height as f64,
                keypoint.score,
            )
        })
        .collect())
}

fn normalize_expanded_bbox(
    bbox: &ExpandedBbox,
    image_width: u32,
    image_height: u32,
) -> Result<ExpandedBbox, WorkerError> {
    if image_width == 0 || image_height == 0 {
        return Err(WorkerError::InvalidInput(
            "image dimensions must be positive".to_owned(),
        ));
    }
    let normalize = |value: BoundingBox| BoundingBox {
        x1: value.x1 / image_width as f64,
        y1: value.y1 / image_height as f64,
        x2: value.x2 / image_width as f64,
        y2: value.y2 / image_height as f64,
        score: value.score,
        width: value.width / image_width as f64,
        height: value.height / image_height as f64,
    };
    Ok(ExpandedBbox {
        original: normalize(bbox.original),
        expanded: normalize(bbox.expanded),
    })
}

fn extract_rtm_det_features(cls: &[f32], reg: &[f32]) -> Result<Vec<f32>, WorkerError> {
    extract_ported_rtmdet_features(cls, reg)
        .map_err(|error| WorkerError::Feature(error.to_string()))
}

fn validate_probability(name: &str, value: f64) -> Result<(), WorkerError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(WorkerError::InvalidInput(format!(
            "{name} must be finite and between 0 and 1"
        )))
    }
}

fn validate_normalized_bbox(
    name: &str,
    bbox: &BoundingBox,
    check_derived: bool,
) -> Result<(), WorkerError> {
    // The oracle's `original` box clamps x1/x2 to the image but keeps width/height
    // from the UNclamped extent (inference-worker.ts:650-656), so width == x2-x1
    // only holds for the `expanded` box. Enforce the derived-dimension identity
    // only when the caller guarantees clamp-consistent coords.
    let derived_mismatch = check_derived
        && ((bbox.width - (bbox.x2 - bbox.x1)).abs() > 1e-9
            || (bbox.height - (bbox.y2 - bbox.y1)).abs() > 1e-9);
    if ![
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.score,
        bbox.width,
        bbox.height,
    ]
    .iter()
    .all(|value| value.is_finite())
        || !(0.0..=1.0).contains(&bbox.score)
        || bbox.x1 < 0.0
        || bbox.y1 < 0.0
        || bbox.x2 > 1.0
        || bbox.y2 > 1.0
        || bbox.x1 > bbox.x2
        || bbox.y1 > bbox.y2
        || bbox.width < 0.0
        || bbox.height < 0.0
        || derived_mismatch
    {
        return Err(WorkerError::InvalidInput(format!(
            "{name} bounding box is invalid"
        )));
    }
    Ok(())
}

fn validate_native_result(result: &NativeInferenceResult) -> Result<(), WorkerError> {
    match (
        result.person_found,
        result.bbox.as_ref(),
        result.keypoints.as_ref(),
    ) {
        (true, Some(bbox), Some(keypoints)) => {
            validate_normalized_bbox("original", &bbox.original, false)?;
            validate_normalized_bbox("expanded", &bbox.expanded, true)?;
            if keypoints.len() != RTMPOSE_NUM_KEYPOINTS
                // Keypoint scores are SimCC activation means and may exceed 1.0;
                // only finiteness is part of the contract.
                || keypoints.iter().any(|keypoint| {
                    !keypoint.x.is_finite()
                        || !keypoint.y.is_finite()
                        || !keypoint.score.is_finite()
                })
            {
                return Err(WorkerError::InvalidInput(
                    "native keypoints are invalid".to_owned(),
                ));
            }
        }
        (false, None, None) => {}
        _ => {
            return Err(WorkerError::InvalidInput(
                "native result geometry is inconsistent".to_owned(),
            ));
        }
    }

    for (feature, values) in &result.features {
        if values.len() != feature.metadata().dimensions
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(WorkerError::Feature(format!(
                "{} feature output is invalid",
                feature.as_str()
            )));
        }
    }
    if let Some(classification) = &result.classification {
        validate_probability("presentProbability", classification.present_probability)?;
        if let Some(probability) = classification.good_probability {
            validate_probability("goodProbability", probability)?;
        }
    }
    Ok(())
}

fn validate_image(image: &ImageData) -> Result<(), WorkerError> {
    if image.width == 0 || image.height == 0 {
        return Err(WorkerError::InvalidInput(
            "image dimensions must be positive".to_owned(),
        ));
    }
    let expected = usize::try_from(image.width)
        .ok()
        .and_then(|width| {
            usize::try_from(image.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| WorkerError::InvalidInput("image dimensions overflow".to_owned()))?;
    if image.data.len() != expected {
        return Err(WorkerError::InvalidInput(format!(
            "image data has {} bytes, expected {expected}",
            image.data.len()
        )));
    }
    Ok(())
}

fn take_f32_output(outputs: &SessionOutputMap, name: &str) -> Result<Vec<f32>, WorkerError> {
    let output = outputs
        .get(name)
        .ok_or_else(|| WorkerError::MissingOutput(name.to_owned()))?;
    let SessionOutput::F32(values) = output else {
        return Err(WorkerError::Tensor(format!("{name}: expected f32 tensor")));
    };
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(WorkerError::Tensor(format!(
            "{name} contains a non-finite value at index {index}"
        )));
    }
    Ok(values.clone())
}

fn take_i64_output(outputs: &SessionOutputMap, name: &str) -> Result<Vec<i64>, WorkerError> {
    let output = outputs
        .get(name)
        .ok_or_else(|| WorkerError::MissingOutput(name.to_owned()))?;
    let SessionOutput::I64(values) = output else {
        return Err(WorkerError::Tensor(format!("{name}: expected i64 tensor")));
    };
    Ok(values.clone())
}

fn copy_session_outputs(outputs: &SessionOutputs<'_>) -> Result<SessionOutputMap, WorkerError> {
    // Copy every named output generically (f32 or i64), covering RTMDet/RTMPose and
    // the NLF-L outputs (coords2d/coords3d_rel/uncertainty) without a per-model
    // allowlist. Outputs the worker never reads by name are simply carried along;
    // outputs of an unsupported dtype are skipped, and a missing/wrong-typed output
    // needed downstream still surfaces at the typed `take_*_output` read site.
    let mut copied = SessionOutputMap::new();
    for (name, value) in outputs.iter() {
        if let Ok((_, values)) = value.try_extract_tensor::<f32>() {
            copied.insert(name.to_owned(), SessionOutput::F32(values.to_vec()));
        } else if let Ok((_, values)) = value.try_extract_tensor::<i64>() {
            copied.insert(name.to_owned(), SessionOutput::I64(values.to_vec()));
        }
    }
    Ok(copied)
}

const _: () = {
    assert!(RTMPOSE_BACKBONE_VALUES == 36_864);
    assert!(RTMPOSE_GAU_VALUES == 4_352);
    assert!(RTMDET_FEATURE_VALUES == 6_400);
};

#[cfg(test)]
mod tests {
    use super::{
        validate_native_result, ClassificationInput, ClassifierModel, InferenceWorker,
        ModelFactory, NativeInferenceResult,
    };
    use slouch_domain::ported::messages::schemas::{
        DimensionalityReductionConfig, DimensionalityReductionMethod, NormalizationMode,
        SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
        SerializedGaussianNb, SerializedModel,
    };
    use slouch_domain::{BoundingBox, ExpandedBbox, Keypoint};

    struct FixedModel(f64);

    impl ClassifierModel for FixedModel {
        fn predict(&mut self, _input: &ClassificationInput<'_>) -> Result<f64, String> {
            Ok(self.0)
        }

        fn dispose(&mut self) {}
    }

    struct FailingFactory;

    impl ModelFactory for FailingFactory {
        fn load(&self, serialized: SerializedModel) -> Result<Box<dyn ClassifierModel>, String> {
            if serialized.trained_at < 0.0 {
                Err("injected model load failure".into())
            } else {
                Ok(Box::new(FixedModel(serialized.trained_at)))
            }
        }
    }

    fn model(value: f64) -> SerializedModel {
        SerializedModel {
            feature_extractor: SerializedFeatureExtractor {
                feature_types: vec!["gau_features".into()],
                normalization_mode: NormalizationMode::None,
                dim_reduction_config: DimensionalityReductionConfig {
                    method: DimensionalityReductionMethod::None,
                    components: 1,
                },
                concatenated_dimensions: 1,
                normalization_mean: None,
                normalization_std: None,
                dim_reduction_transformer: None,
            },
            classifier: SerializedClassifier {
                classifier_id: "gaussian_nb".into(),
                state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                    class_means: [vec![0.0], vec![1.0]],
                    class_variances: [vec![1.0], vec![1.0]],
                    class_priors: [0.5, 0.5],
                    epsilon: 1e-9,
                }),
            },
            trained_at: value,
            version: 1.0,
        }
    }

    fn inference(score: f64) -> NativeInferenceResult {
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score,
            width: 1.0,
            height: 1.0,
        };
        NativeInferenceResult {
            person_found: true,
            bbox: Some(ExpandedBbox {
                original: bbox,
                expanded: bbox,
            }),
            keypoints: None,
            features: Default::default(),
            classification: None,
        }
    }

    #[test]
    fn pair_publication_preserves_the_old_generation_on_second_model_failure() {
        let mut worker = InferenceWorker::new(FailingFactory);
        worker
            .publish_model_pair(Some(model(0.8)), Some(model(0.9)))
            .expect("initial pair");
        assert!(worker
            .publish_model_pair(Some(model(0.2)), Some(model(-1.0)))
            .is_err());
        let classification = worker
            .classify_features(&inference(1.0))
            .expect("prediction")
            .expect("classification");
        assert_eq!(classification.present_probability, 0.9);
        assert_eq!(classification.good_probability, Some(0.8));
    }

    #[test]
    fn complete_pair_publication_uses_presence_classifier() {
        let mut worker = InferenceWorker::new(FailingFactory);
        worker
            .publish_model_pair(Some(model(0.7)), Some(model(0.6)))
            .expect("complete pair");
        let classification = worker
            .classify_features(&inference(0.85))
            .expect("prediction")
            .expect("classification");
        assert_eq!(classification.present_probability, 0.6);
        assert_eq!(classification.good_probability, Some(0.7));
    }

    #[test]
    fn posture_only_publication_unloads_presence_and_uses_detector_confidence() {
        let mut worker = InferenceWorker::new(FailingFactory);
        // Start from a full pair, then publish a posture-only generation.
        worker
            .publish_model_pair(Some(model(0.3)), Some(model(0.6)))
            .expect("initial pair");
        worker
            .publish_model_pair(Some(model(0.8)), None)
            .expect("posture-only generation");
        // Presence is unloaded, so present_probability falls back to the
        // detector/bbox score (0.85), while posture still runs.
        let classification = worker
            .classify_features(&inference(0.85))
            .expect("prediction")
            .expect("classification");
        assert_eq!(classification.present_probability, 0.85);
        assert_eq!(classification.good_probability, Some(0.8));
    }

    #[test]
    fn posture_only_classification_does_not_gate_detector_score_below_half() {
        let mut worker = InferenceWorker::new(FailingFactory);
        worker.load_posture_model(model(0.8)).unwrap();
        let classification = worker
            .classify_features(&inference(0.4))
            .expect("prediction")
            .expect("classification");
        assert_eq!(classification.present_probability, 0.4);
        assert_eq!(classification.good_probability, Some(0.8));
    }

    #[test]
    fn invalid_classifier_probabilities_are_rejected() {
        let mut worker = InferenceWorker::new(FailingFactory);
        worker.load_posture_model(model(2.0)).unwrap();
        assert!(worker.classify_features(&inference(0.9)).is_err());
    }

    #[test]
    fn selected_bbox_width_height_use_unclamped_extent() {
        // Detection extends past the image on the top-left; oracle clamps x1/y1
        // to 0 but keeps width/height from the UNclamped mapped-back extent
        // (inference-worker.ts:650-656).
        let dets = [-5.0_f32, -5.0, 50.0, 60.0, 0.9];
        let labels = [0_i64];
        let selected = super::compatibility_select_person_bbox(&dets, &labels, 100, 100)
            .expect("selection succeeds")
            .expect("person found");
        assert_eq!(selected.x1, 0.0);
        assert_eq!(selected.y1, 0.0);
        assert_eq!(selected.x2, 50.0);
        assert_eq!(selected.y2, 60.0);
        assert_eq!(selected.width, 55.0);
        assert_eq!(selected.height, 65.0);
    }

    #[test]
    fn native_keypoint_beyond_unit_range_is_not_rejected() {
        // The oracle's transformKeypoints (inference-worker.ts:602-624) performs no
        // positional validation and can return a normalized coordinate marginally
        // above 1.0 for an edge-of-frame pose. Boundary validation must not reject
        // it: PORTING.md line 30 tightens score/probability ranges, not keypoint
        // positions.
        let unit_box = BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.9,
            y2: 0.9,
            score: 0.8,
            width: 0.8,
            height: 0.8,
        };
        let mut keypoints = vec![Keypoint::new(0.5, 0.5, 0.9); 17];
        keypoints[0] = Keypoint::new(1.006, 0.5, 0.9);
        // SimCC-derived scores are raw activation means and exceed 1.0 on real
        // frames; only finiteness is validated.
        keypoints[7] = Keypoint::new(0.5, 0.5, 3.2);
        let result = NativeInferenceResult {
            person_found: true,
            bbox: Some(ExpandedBbox {
                original: unit_box,
                expanded: unit_box,
            }),
            keypoints: Some(keypoints),
            features: Default::default(),
            classification: None,
        };
        assert!(validate_native_result(&result).is_ok());
    }
}
