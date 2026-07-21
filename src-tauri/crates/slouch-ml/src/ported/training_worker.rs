//! Native port of `src/workers/training-worker.ts`.
//!
//! The browser worker owns message validation, dataset/settings loading, dual
//! model orchestration, cross-validation, persistence, and the training lock.
//! Native storage and ML implementations are injected through the two traits
//! below so the worker remains independent of the Tauri command/runtime layer.

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use slouch_domain::ported::messages::schemas::{JsonValue, TrainingWorkerMessage};
use slouch_domain::{
    validate_posture_frame, ClassifierConfig, CrossValidationType, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FrameLabel, NormalizationMode, PostureDataset,
    PostureFrame, TrainingMetrics, TrainingResult, TrainingSettings,
};

use super::feature_extraction::is_feature_available;
use super::types::SerializedModel;

const MIN_FRAMES_PER_CLASS: usize = 1;
const MODEL_VERSION: f64 = 1.0;
const MAX_TRAINING_FRAMES: usize = 100_000;
const MAX_RESERVOIR_SAMPLES: usize = 1_000;

const KEYPOINT_BASED_FEATURES: [FeatureId; 5] = [
    FeatureId::EngineeredFeatures,
    FeatureId::Joint2d,
    FeatureId::Joint3d,
    FeatureId::Joint4d,
    FeatureId::PostureRaw,
];

/// A reservoir item in the native training boundary. The storage crate owns the
/// actual reservoir implementation; this DTO is the worker's stable input contract.
/// It carries a generic `FeatureMap` so any stored feature (RTMPose/RTMDet pooled,
/// NLF backbone/depth, future additions) flows through with no per-field wiring.
/// The `features`/`keypoints`/`bbox` field names and `rmp_serde::to_vec_named`
/// encoding must stay byte-identical to `slouch_store`'s twin struct for the
/// cross-crate reservoir round-trip to hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReservoirSample {
    pub features: slouch_domain::FeatureMap,
    pub keypoints: Vec<slouch_domain::Keypoint>,
    pub bbox: slouch_domain::BoundingBox,
}

/// Feature data supplied to a fitted extractor for PCA/random-projection
/// fitting. Stored vectors stay as `f32`, matching inference buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureContainer {
    pub features: slouch_domain::FeatureMap,
    pub keypoints: Vec<slouch_domain::Keypoint>,
    pub bbox: Option<slouch_domain::BoundingBox>,
}

impl From<ReservoirSample> for FeatureContainer {
    fn from(sample: ReservoirSample) -> Self {
        Self {
            features: sample.features,
            keypoints: sample.keypoints,
            bbox: Some(sample.bbox),
        }
    }
}

/// The small data access surface used by the worker. Implementations may use
/// SQLite, an in-memory test store, or a Tauri command-backed repository.
pub trait TrainingStorage {
    fn load_dataset(&mut self) -> Result<Option<PostureDataset>, String>;
    fn load_training_settings(&mut self) -> Result<Option<TrainingSettings>, String>;
    fn load_reservoir_samples(&mut self) -> Result<Vec<ReservoirSample>, String>;
    fn save_posture_model(&mut self, model: &SerializedModel) -> Result<(), String>;
    fn save_presence_model(&mut self, model: &SerializedModel) -> Result<(), String>;
}

/// Native replacement for the TensorFlow.js Model/classifier/evaluation
/// combination. The worker still owns all sequencing and error semantics; a
/// backend owns numeric implementation and model buffer disposal.
pub trait TrainingBackend {
    fn calibrate_feature_bins(
        &mut self,
        samples: &[FeatureContainer],
        log_engineered: bool,
        logger: &dyn TrainingLogger,
    );

    fn cross_validate(
        &mut self,
        config: &FeatureExtractorConfig,
        classifier_config: &ClassifierConfig,
        frames: &[PostureFrame],
        labels: &[i32],
        cv_folds: usize,
    ) -> Result<Option<TrainingMetrics>, String>;

    fn fit(
        &mut self,
        config: &FeatureExtractorConfig,
        classifier_config: &ClassifierConfig,
        frames: &[PostureFrame],
        labels: &[i32],
    ) -> Result<SerializedModel, String>;

    fn release_training_buffers(&mut self);
}

/// Logging boundary matching the source logger's training category.
pub trait TrainingLogger: Send + Sync {
    fn info(&self, message: &str);
    fn warn(&self, message: &str);
    fn error(&self, message: &str);
    fn set_from_url_param(&self, _value: &str) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLogger;

impl TrainingLogger for NoopLogger {
    fn info(&self, _message: &str) {}
    fn warn(&self, _message: &str) {}
    fn error(&self, _message: &str) {}
}

pub trait TrainingClock {
    fn now_millis(&self) -> f64;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl TrainingClock for SystemClock {
    fn now_millis(&self) -> f64 {
        // Oracle parity: Date.now() yields whole integer milliseconds, and the
        // persisted `trained_at` must be a JS-safe *integer* (model_format::
        // encode_model rejects fractional timestamps). Truncate to whole ms
        // instead of scaling fractional seconds.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |duration| duration.as_millis() as f64)
    }
}

/// Configuration passed to the native feature extractor backend.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureExtractorConfig {
    pub feature_types: Vec<FeatureId>,
    pub normalization_mode: NormalizationMode,
    pub dim_reduction_config: DimensionalityReductionConfig,
    pub unlabeled_samples: Vec<FeatureContainer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainingResultResponse {
    pub posture_result: Option<TrainingResult>,
    pub presence_result: Option<TrainingResult>,
    pub success: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingModelsResponse {
    pub posture: Option<SerializedModel>,
    pub presence: Option<SerializedModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingWorkerResponse {
    #[serde(rename = "result")]
    Result {
        result: Box<TrainingResultResponse>,
        models: Box<TrainingModelsResponse>,
    },
    #[serde(rename = "error")]
    Error {
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrainingWorkerError {
    Storage(String),
    InvalidSettings(String),
    Training(String),
}

impl fmt::Display for TrainingWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(message) | Self::InvalidSettings(message) | Self::Training(message) => {
                formatter.write_str(message)
            }
        }
    }
}

impl std::error::Error for TrainingWorkerError {}

/// Stateful native equivalent of the browser worker.
pub struct TrainingWorker<S, B, L = NoopLogger, C = SystemClock>
where
    S: TrainingStorage,
    B: TrainingBackend,
    L: TrainingLogger,
    C: TrainingClock,
{
    storage: S,
    backend: B,
    logger: L,
    clock: C,
    training: Arc<Mutex<bool>>,
}

/// RAII guard mirroring the oracle's `finally { isTraining = false }`. Resetting
/// the flag from `Drop` guarantees it runs on the normal return path AND when a
/// backend call unwinds via panic, so a panicking ML backend can never wedge the
/// worker into a permanent "Training already in progress" state.
struct TrainingLockGuard(Arc<Mutex<bool>>);

impl Drop for TrainingLockGuard {
    fn drop(&mut self) {
        if let Ok(mut state) = self.0.lock() {
            *state = false;
        }
    }
}

impl<S, B> TrainingWorker<S, B, NoopLogger, SystemClock>
where
    S: TrainingStorage,
    B: TrainingBackend,
{
    pub fn new(storage: S, backend: B) -> Self {
        Self::with_logger(storage, backend, NoopLogger)
    }
}

impl<S, B, L> TrainingWorker<S, B, L, SystemClock>
where
    S: TrainingStorage,
    B: TrainingBackend,
    L: TrainingLogger,
{
    pub fn with_logger(storage: S, backend: B, logger: L) -> Self {
        Self::with_clock(storage, backend, logger, SystemClock)
    }
}

impl<S, B, L, C> TrainingWorker<S, B, L, C>
where
    S: TrainingStorage,
    B: TrainingBackend,
    L: TrainingLogger,
    C: TrainingClock,
{
    pub fn with_clock(storage: S, backend: B, logger: L, clock: C) -> Self {
        Self {
            storage,
            backend,
            logger,
            clock,
            training: Arc::new(Mutex::new(false)),
        }
    }

    /// Handles one already-deserialized and schema-validated worker message.
    /// JSON/Zod-equivalent validation belongs at the application transport
    /// edge, where malformed payload details can be represented naturally.
    pub fn handle_message(
        &mut self,
        message: TrainingWorkerMessage,
    ) -> Vec<TrainingWorkerResponse> {
        match message {
            TrainingWorkerMessage::Train { payload } => {
                let do_cv = payload.and_then(|value| value.do_cv).unwrap_or(true);
                if self.is_training() {
                    return vec![error_response("Training already in progress")];
                }

                self.set_training(true);
                // Guard resets the flag on every exit path, including a panic
                // unwinding out of the injected backend, matching the oracle's
                // `finally { isTraining = false }`.
                let _training_guard = TrainingLockGuard(Arc::clone(&self.training));
                let response = match self.run_training(do_cv) {
                    Ok(result) => vec![TrainingWorkerResponse::Result {
                        result: Box::new(TrainingResultResponse {
                            posture_result: result.posture_result,
                            presence_result: result.presence_result,
                            success: result.success,
                            errors: result.errors,
                            warnings: result.warnings,
                        }),
                        models: Box::new(TrainingModelsResponse {
                            posture: result.posture_model,
                            presence: result.presence_model,
                        }),
                    }],
                    Err(error) => {
                        self.logger.error(&format!(
                            "[Training Worker] Training failed with exception: {error}"
                        ));
                        vec![error_response(error.to_string())]
                    }
                };
                response
            }
            TrainingWorkerMessage::SetLogLevel { payload } => {
                if let Some(log_param) = payload.and_then(|value| value.log_param) {
                    self.logger.set_from_url_param(&log_param);
                    self.logger.info(&format!(
                        "[Training Worker] Log level updated: {}",
                        if log_param.is_empty() {
                            "none"
                        } else {
                            &log_param
                        }
                    ));
                }
                Vec::new()
            }
        }
    }

    fn run_training(&mut self, do_cv: bool) -> Result<TrainingRun, TrainingWorkerError> {
        self.logger.info(&format!(
            "[Training Worker] Received training request (doCV: {do_cv})"
        ));

        let dataset = self
            .storage
            .load_dataset()
            .map_err(TrainingWorkerError::Storage)?
            .ok_or_else(|| TrainingWorkerError::Storage("No dataset found in storage".into()))?;
        self.logger.info(&format!(
            "[Training Worker] Loaded dataset: {} frames",
            dataset.frames.len()
        ));

        let settings = self
            .storage
            .load_training_settings()
            .map_err(TrainingWorkerError::Storage)?
            .ok_or_else(|| {
                TrainingWorkerError::Storage("No training settings found in storage".into())
            })?;
        validate_training_inputs(&dataset, &settings)?;

        // The persisted native DTO always distinguishes the current explicit
        // selections from the deprecated optional `feature_types` field. Match
        // the TypeScript worker's nullish fallback: an explicit empty list is
        // invalid and must not be treated as an absent legacy setting.
        let posture_features = settings.posture_feature_types.clone();
        let presence_features = settings.presence_feature_types.clone();
        if posture_features.is_empty() {
            return Err(TrainingWorkerError::InvalidSettings(
                "Invalid training settings: postureFeatureTypes is empty or invalid. Please check your training configuration in the Training tab.".into(),
            ));
        }
        if presence_features.is_empty() {
            return Err(TrainingWorkerError::InvalidSettings(
                "Invalid training settings: presenceFeatureTypes is empty or invalid. Please check your training configuration in the Training tab.".into(),
            ));
        }

        let reservoir_samples = self
            .storage
            .load_reservoir_samples()
            .map_err(TrainingWorkerError::Storage)?;
        validate_training_reservoir(&reservoir_samples)?;
        let reservoir_sources: Vec<FeatureContainer> = reservoir_samples
            .into_iter()
            .map(FeatureContainer::from)
            .collect();
        let mut calibration_sources = dataset
            .frames
            .iter()
            .map(|frame| FeatureContainer {
                features: frame.features.clone(),
                keypoints: frame.keypoints.clone(),
                bbox: Some(frame.bbox),
            })
            .collect::<Vec<_>>();
        calibration_sources.extend(reservoir_sources.iter().cloned());
        self.logger.info(&format!(
            "[Training Worker] Bin calibration: {} labeled + {} reservoir = {} total samples",
            dataset.frames.len(),
            reservoir_sources.len(),
            calibration_sources.len()
        ));
        self.backend.calibrate_feature_bins(
            &calibration_sources,
            posture_features
                .iter()
                .any(|feature| KEYPOINT_BASED_FEATURES.contains(feature)),
            &self.logger,
        );

        let good = frames_with_label(&dataset.frames, FrameLabel::Good);
        let bad = frames_with_label(&dataset.frames, FrameLabel::Bad);
        let away = frames_with_label(&dataset.frames, FrameLabel::Away);
        let should_train_posture = has_minimum_per_class(good.len(), bad.len());
        let present_count = good.len() + bad.len();
        let should_train_presence = has_minimum_per_class(away.len(), present_count);

        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        if !should_train_posture && (!good.is_empty() || !bad.is_empty()) {
            warnings.push(format!(
                "Skipping posture model: need at least {MIN_FRAMES_PER_CLASS} GOOD and {MIN_FRAMES_PER_CLASS} BAD frames (have {} GOOD, {} BAD)",
                good.len(), bad.len()
            ));
        }
        if !should_train_presence && (!away.is_empty() || present_count > 0) {
            warnings.push(format!(
                "Skipping presence model: need at least {MIN_FRAMES_PER_CLASS} AWAY and {MIN_FRAMES_PER_CLASS} PRESENT frames (have {} AWAY, {} PRESENT)",
                away.len(), present_count
            ));
        }
        if !should_train_posture && !should_train_presence {
            errors.push(format!(
                "Insufficient data: need at least {MIN_FRAMES_PER_CLASS} frames per class. Posture model requires {MIN_FRAMES_PER_CLASS} GOOD + {MIN_FRAMES_PER_CLASS} BAD. Presence model requires {MIN_FRAMES_PER_CLASS} AWAY + {MIN_FRAMES_PER_CLASS} PRESENT (GOOD+BAD combined)."
            ));
            return Ok(TrainingRun::failed(errors, warnings));
        }

        let normalization = settings
            .normalization_mode
            .unwrap_or(NormalizationMode::None);
        let cv_folds = if do_cv { settings.cv_folds } else { 1 };
        let mut run = TrainingRun::default();

        if should_train_presence {
            self.logger
                .info("[Training Worker] Training presence classifier (PRESENT vs AWAY)");
            let frames = present_and_away_frames(&good, &bad, &away);
            let labels = frames
                .iter()
                .map(|frame| i32::from(frame.label == FrameLabel::Bad))
                .collect::<Vec<_>>();
            match self.train_one(
                (&frames, &labels, &[]),
                &presence_features,
                &settings,
                normalization,
                cv_folds,
                true,
            ) {
                // Oracle assigns presenceResult/presenceModel before awaiting the
                // save, so a save failure only appends an error while both stay set
                // and the overall success flag is unaffected.
                Ok((result, model)) => {
                    run.presence_result = Some(result);
                    run.presence_model = Some(model.clone());
                    if let Err(error) = self.storage.save_presence_model(&model) {
                        errors.push(format!("Presence model training failed: {error}"));
                    }
                }
                Err(error) => errors.push(format!("Presence model training failed: {error}")),
            }
        } else if away.is_empty() && present_count > 0 {
            warnings.push(
                "No AWAY frames collected. Using RTMDet confidence for presence detection.".into(),
            );
        }

        if should_train_posture {
            self.logger
                .info("[Training Worker] Training posture classifier (GOOD vs BAD)");
            let frames = good.iter().chain(bad.iter()).cloned().collect::<Vec<_>>();
            let labels = frames
                .iter()
                .map(|frame| i32::from(frame.label != FrameLabel::Good))
                .collect::<Vec<_>>();
            let unlabeled = if settings.dim_reduction_config.method
                == DimensionalityReductionMethod::Pca
            {
                drop_reservoir_absent_from_pca(reservoir_sources, &posture_features, &mut warnings)
            } else {
                Vec::new()
            };
            match self.train_one(
                (&frames, &labels, &unlabeled),
                &posture_features,
                &settings,
                normalization,
                cv_folds,
                false,
            ) {
                // Oracle assigns postureResult/postureModel before awaiting the
                // save, so a save failure only appends an error while both stay set
                // and the overall success flag is unaffected.
                Ok((result, model)) => {
                    run.posture_result = Some(result);
                    run.posture_model = Some(model.clone());
                    if let Err(error) = self.storage.save_posture_model(&model) {
                        errors.push(format!("Posture model training failed: {error}"));
                    }
                }
                Err(error) => errors.push(format!("Posture model training failed: {error}")),
            }
        }

        run.errors = errors;
        run.warnings = warnings;
        run.success = (run.presence_result.as_ref().is_some_and(|r| r.success)
            || !should_train_presence)
            && (run.posture_result.as_ref().is_some_and(|r| r.success) || !should_train_posture);
        self.logger.info(&format!(
            "[Training Worker] Dual training complete. Success: {}",
            run.success
        ));
        Ok(run)
    }

    fn train_one(
        &mut self,
        samples: (&[PostureFrame], &[i32], &[FeatureContainer]),
        feature_types: &[FeatureId],
        settings: &TrainingSettings,
        normalization_mode: NormalizationMode,
        cv_folds: usize,
        presence: bool,
    ) -> Result<(TrainingResult, SerializedModel), String> {
        let (frames, labels, unlabeled_samples) = samples;
        if frames.len() != labels.len() {
            return Err("frame/label count mismatch".into());
        }
        let missing = frames
            .iter()
            .filter(|frame| frame.features.is_empty())
            .count();
        if missing != 0 {
            self.logger.error(&format!(
                "[Training Worker] {missing} frames are missing features"
            ));
            return Err(format!(
                "{missing} frames are missing features. Clear dataset and recapture."
            ));
        }

        self.logger.info(&format!(
            "[Training Worker] {} classifier will use features: {}",
            if presence { "Presence" } else { "Posture" },
            feature_types
                .iter()
                .map(|feature| feature.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let config = FeatureExtractorConfig {
            feature_types: feature_types.to_vec(),
            normalization_mode,
            dim_reduction_config: settings.dim_reduction_config.clone(),
            unlabeled_samples: unlabeled_samples.to_vec(),
        };
        let metrics = match self.backend.cross_validate(
            &config,
            &settings.classifier_config,
            frames,
            labels,
            cv_folds,
        ) {
            Ok(metrics) => metrics,
            Err(error) => {
                self.backend.release_training_buffers();
                return Err(error);
            }
        };
        if let Some(metrics) = &metrics {
            self.logger.info(&format!(
                "[Training Worker] {} CV: {:.1}% accuracy",
                if presence { "Presence" } else { "Posture" },
                metrics.cv_accuracy * 100.0
            ));
        } else {
            self.logger.warn(&format!(
                "[Training Worker] {} CV skipped (insufficient data for a time-ordered holdout)",
                if presence { "Presence" } else { "Posture" }
            ));
        }

        let mut model = match self
            .backend
            .fit(&config, &settings.classifier_config, frames, labels)
        {
            Ok(model) => model,
            Err(error) => {
                self.backend.release_training_buffers();
                return Err(error);
            }
        };
        model.trained_at = self.clock.now_millis();
        model.version = MODEL_VERSION;
        let mut warnings = vec![match &metrics {
            Some(_) => format!("Using {cv_folds}-fold CV ({} samples)", frames.len()),
            None => "CV skipped - not enough frames for a purged time-ordered holdout. Metrics unavailable; collect more frames spread over time.".to_owned(),
        }];
        // Surface a soft degradation when PCA could not honor the requested component
        // count. The fitted extractor records the EFFECTIVE count it clamped to; compare
        // it against the requested count from settings so the user sees why their model
        // has fewer dimensions than they asked for instead of a hard training failure.
        if settings.dim_reduction_config.method == DimensionalityReductionMethod::Pca {
            if let Some(warning) = pca_component_clamp_warning(
                settings.dim_reduction_config.components,
                model.feature_extractor.dim_reduction_config.components,
                model.feature_extractor.concatenated_dimensions,
            ) {
                warnings.push(warning);
            }
        }
        let result = TrainingResult {
            success: true,
            metrics: metrics.clone().unwrap_or_else(empty_metrics),
            dim_reduction_method: settings.dim_reduction_config.method,
            warnings,
            errors: Vec::new(),
        };
        self.backend.release_training_buffers();
        Ok((result, model))
    }

    fn is_training(&self) -> bool {
        self.training.lock().map(|state| *state).unwrap_or(true)
    }

    fn set_training(&self, value: bool) {
        if let Ok(mut state) = self.training.lock() {
            *state = value;
        }
    }
}

#[derive(Debug, Default)]
struct TrainingRun {
    posture_result: Option<TrainingResult>,
    presence_result: Option<TrainingResult>,
    posture_model: Option<SerializedModel>,
    presence_model: Option<SerializedModel>,
    success: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl TrainingRun {
    fn failed(errors: Vec<String>, warnings: Vec<String>) -> Self {
        Self {
            success: false,
            errors,
            warnings,
            ..Self::default()
        }
    }
}

fn validate_training_inputs(
    dataset: &PostureDataset,
    settings: &TrainingSettings,
) -> Result<(), TrainingWorkerError> {
    if dataset.frames.len() > MAX_TRAINING_FRAMES {
        return Err(TrainingWorkerError::InvalidSettings(format!(
            "dataset exceeds the {MAX_TRAINING_FRAMES}-frame training limit"
        )));
    }
    // The oracle imposes no lower bound on cv_folds: `effectiveCvFolds = doCV ? cvFolds : 1`
    // and crossValidate explicitly skips CV for cvFolds <= 1. Only an upper sanity bound is
    // kept here so a stored config with cv_folds in {0,1} trains (with CV skipped) as it does
    // in the browser worker.
    if !dataset.last_modified.is_finite()
        || dataset.last_modified <= 0.0
        || settings.cv_folds > 100
        || !settings.last_updated.is_finite()
        || settings.last_updated <= 0.0
        || (settings.dim_reduction_config.method != DimensionalityReductionMethod::None
            && settings.dim_reduction_config.components == 0)
        || settings.dim_reduction_config.components > 1_048_576
    {
        return Err(TrainingWorkerError::InvalidSettings(
            "training settings contain invalid bounds or timestamps".into(),
        ));
    }
    for frame in &dataset.frames {
        validate_posture_frame(frame).map_err(|error| {
            TrainingWorkerError::InvalidSettings(format!(
                "training frame {} is invalid: {error}",
                frame.id
            ))
        })?;
    }
    Ok(())
}

fn validate_training_reservoir(samples: &[ReservoirSample]) -> Result<(), TrainingWorkerError> {
    if samples.len() > MAX_RESERVOIR_SAMPLES {
        return Err(TrainingWorkerError::InvalidSettings(format!(
            "reservoir exceeds the {MAX_RESERVOIR_SAMPLES}-sample training limit"
        )));
    }
    for (index, sample) in samples.iter().enumerate() {
        if sample.features.is_empty() {
            return Err(TrainingWorkerError::InvalidSettings(format!(
                "reservoir sample {index} carries no features"
            )));
        }
        for (feature, values) in &sample.features {
            let metadata = feature.metadata();
            if metadata.computed {
                return Err(TrainingWorkerError::InvalidSettings(format!(
                    "reservoir sample {index} feature {} is computed, not stored",
                    feature.as_str(),
                )));
            }
            if values.len() != metadata.dimensions || values.iter().any(|value| !value.is_finite())
            {
                return Err(TrainingWorkerError::InvalidSettings(format!(
                    "reservoir sample {index} feature {} is invalid",
                    feature.as_str(),
                )));
            }
        }
        if sample.keypoints.len() != 17
            || sample.keypoints.iter().any(|point| {
                !point.x.is_finite() || !point.y.is_finite() || !point.score.is_finite()
            })
        {
            return Err(TrainingWorkerError::InvalidSettings(format!(
                "reservoir sample {index} keypoints are invalid"
            )));
        }
        slouch_domain::validate_bbox(&sample.bbox).map_err(|_| {
            TrainingWorkerError::InvalidSettings(format!(
                "reservoir sample {index} bounding box is invalid"
            ))
        })?;
    }
    Ok(())
}

/// Drops the reservoir from PCA fitting when a selected posture feature cannot be
/// extracted from the reservoir samples. Reservoir containers carry only the pooled
/// RTMPose/RTMDet stored features plus keypoints, never GPU-only stored features such as
/// `nlf_depth`; feeding one to `concatenate_features` for an absent feature hard-errors,
/// so PCA falls back to the labeled frames only and a visible warning is surfaced.
fn drop_reservoir_absent_from_pca(
    reservoir_sources: Vec<FeatureContainer>,
    posture_features: &[FeatureId],
    warnings: &mut Vec<String>,
) -> Vec<FeatureContainer> {
    let Some(representative) = reservoir_sources.first() else {
        return reservoir_sources;
    };
    let probe = reservoir_probe_frame(representative);
    let absent = posture_features
        .iter()
        .copied()
        .filter(|feature| !matches!(is_feature_available(&probe, *feature), Ok(true)))
        .collect::<Vec<_>>();
    if absent.is_empty() {
        return reservoir_sources;
    }
    for feature in absent {
        warnings.push(format!(
            "PCA fitted on labeled frames only: feature '{}' is absent from reservoir samples.",
            feature.as_str()
        ));
    }
    Vec::new()
}

/// Wraps a reservoir container as a `PostureFrame` so feature availability can be probed
/// through the shared extractor. Posture-feature availability reads only the feature map
/// and keypoints, never the bbox, so a placeholder bbox is used when the sample had none.
fn reservoir_probe_frame(container: &FeatureContainer) -> PostureFrame {
    PostureFrame {
        id: String::new(),
        timestamp: 0.0,
        features: container.features.clone(),
        thumbnail: slouch_domain::Thumbnail {
            mime_type: String::new(),
            bytes: Vec::new(),
        },
        keypoints: container.keypoints.clone(),
        bbox: container.bbox.unwrap_or(slouch_domain::BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.0,
            width: 1.0,
            height: 1.0,
        }),
        label: FrameLabel::Unused,
    }
}

fn frames_with_label(frames: &[PostureFrame], label: FrameLabel) -> Vec<PostureFrame> {
    frames
        .iter()
        .filter(|frame| frame.label == label)
        .cloned()
        .collect()
}

fn present_and_away_frames(
    good: &[PostureFrame],
    bad: &[PostureFrame],
    away: &[PostureFrame],
) -> Vec<PostureFrame> {
    good.iter()
        .chain(bad)
        .map(|frame| PostureFrame {
            label: FrameLabel::Good,
            ..frame.clone()
        })
        .chain(away.iter().map(|frame| PostureFrame {
            label: FrameLabel::Bad,
            ..frame.clone()
        }))
        .collect()
}

fn empty_metrics() -> TrainingMetrics {
    TrainingMetrics {
        cv_accuracy: 0.0,
        cv_std: 0.0,
        mcc: 0.0,
        f1_score: 0.0,
        confusion_matrix: vec![vec![0, 0], vec![0, 0]],
        fold_accuracies: Vec::new(),
        balanced_accuracy: 0.0,
        accuracy_ci_low: 0.0,
        accuracy_ci_high: 0.0,
        worst_fold_accuracy: 0.0,
        cv_type: Some(CrossValidationType::ShuffledStratified),
    }
}

fn has_minimum_per_class(first: usize, second: usize) -> bool {
    first >= MIN_FRAMES_PER_CLASS && second >= MIN_FRAMES_PER_CLASS
}

/// Builds the user-facing warning when PCA reduced the requested component count to
/// the effective rank the data supported. PCA fits at most `min(requested, fit_rows
/// - 1, n_features)` components; when the result equals `n_features` the selected
/// features were too low-dimensional, otherwise the fitted sample count was the
/// limit. Returns `None` when the request was honored in full.
fn pca_component_clamp_warning(
    requested: usize,
    effective: usize,
    n_features: usize,
) -> Option<String> {
    if effective >= requested {
        return None;
    }
    let reason = if effective >= n_features {
        format!("selected features have only {n_features} dimensions")
    } else {
        "limited by dataset size".to_owned()
    };
    Some(format!(
        "PCA components reduced {requested}\u{2192}{effective} ({reason})"
    ))
}

fn error_response(error: impl Into<String>) -> TrainingWorkerResponse {
    TrainingWorkerResponse::Error {
        error: error.into(),
        details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posture_requires_good_and_bad() {
        assert!(!has_minimum_per_class(0, 1));
        assert!(has_minimum_per_class(1, 1));
    }

    #[test]
    fn pca_clamp_warning_distinguishes_dims_and_samples_limits() {
        // Requested honored in full: no warning.
        assert_eq!(pca_component_clamp_warning(7, 7, 7), None);
        assert_eq!(pca_component_clamp_warning(5, 16, 256), None);

        // Feature-dimension limited: effective equals the concatenated feature width
        // (the user's torso_invariant + pca 30 case: 7-dim selection caps rank at 7).
        assert_eq!(
            pca_component_clamp_warning(30, 7, 7).as_deref(),
            Some("PCA components reduced 30\u{2192}7 (selected features have only 7 dimensions)")
        );

        // Sample limited: effective is below the feature width, so the fitted row
        // count (rows - 1) was the binding constraint.
        assert_eq!(
            pca_component_clamp_warning(50, 9, 256).as_deref(),
            Some("PCA components reduced 50\u{2192}9 (limited by dataset size)")
        );
    }

    #[test]
    fn system_clock_yields_a_positive_whole_millisecond() {
        // model_format::encode_model rejects a fractional trained_at, so the
        // production clock must emit whole integer milliseconds.
        let now = SystemClock.now_millis();
        assert!(now > 0.0, "timestamp must be positive: {now}");
        assert_eq!(now.fract(), 0.0, "timestamp must be a whole integer: {now}");
    }

    fn labeled_frame(id: &str, label: FrameLabel) -> PostureFrame {
        let mut features = slouch_domain::FeatureMap::new();
        features.insert(
            FeatureId::GauFeatures,
            vec![0.2; FeatureId::GauFeatures.metadata().dimensions],
        );
        features.insert(
            FeatureId::RtmDetEngineered,
            vec![0.3; FeatureId::RtmDetEngineered.metadata().dimensions],
        );
        PostureFrame {
            id: id.into(),
            label,
            timestamp: 1.0,
            features,
            thumbnail: slouch_domain::Thumbnail {
                mime_type: "image/webp".into(),
                bytes: vec![1],
            },
            keypoints: (0..17)
                .map(|_| slouch_domain::Keypoint::new(0.2, 0.3, 0.9))
                .collect(),
            bbox: slouch_domain::BoundingBox {
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

    #[test]
    fn presence_maps_present_to_zero_and_away_to_one() {
        // Exercise the real presence label-mapping pipeline: present_and_away_frames
        // relabels PRESENT (GOOD+BAD) -> Good and AWAY -> Bad, then run_training maps
        // `label == Bad` to the integer class. So present -> 0, away -> 1.
        let good = vec![labeled_frame("g", FrameLabel::Good)];
        let bad = vec![labeled_frame("b", FrameLabel::Bad)];
        let away = vec![
            labeled_frame("a0", FrameLabel::Away),
            labeled_frame("a1", FrameLabel::Away),
        ];
        let frames = present_and_away_frames(&good, &bad, &away);
        let presence_labels = frames
            .iter()
            .map(|frame| i32::from(frame.label == FrameLabel::Bad))
            .collect::<Vec<_>>();
        assert_eq!(presence_labels, vec![0, 0, 1, 1]);

        // Posture mapping (run_training): GOOD then BAD, mapping `label != Good` -> class.
        let posture_frames = good.iter().chain(bad.iter()).cloned().collect::<Vec<_>>();
        let posture_labels = posture_frames
            .iter()
            .map(|frame| i32::from(frame.label != FrameLabel::Good))
            .collect::<Vec<_>>();
        assert_eq!(posture_labels, vec![0, 1]);
    }

    struct PanicBackend;

    impl TrainingBackend for PanicBackend {
        fn calibrate_feature_bins(
            &mut self,
            _samples: &[FeatureContainer],
            _log_engineered: bool,
            _logger: &dyn TrainingLogger,
        ) {
            panic!("backend calibration panicked");
        }
        fn cross_validate(
            &mut self,
            _config: &FeatureExtractorConfig,
            _classifier_config: &ClassifierConfig,
            _frames: &[PostureFrame],
            _labels: &[i32],
            _cv_folds: usize,
        ) -> Result<Option<TrainingMetrics>, String> {
            panic!("backend cross_validate panicked");
        }
        fn fit(
            &mut self,
            _config: &FeatureExtractorConfig,
            _classifier_config: &ClassifierConfig,
            _frames: &[PostureFrame],
            _labels: &[i32],
        ) -> Result<SerializedModel, String> {
            panic!("backend fit panicked");
        }
        fn release_training_buffers(&mut self) {}
    }

    struct PanicTestStorage;

    impl TrainingStorage for PanicTestStorage {
        fn load_dataset(&mut self) -> Result<Option<PostureDataset>, String> {
            Ok(Some(PostureDataset {
                frames: vec![
                    labeled_frame("g", FrameLabel::Good),
                    labeled_frame("b", FrameLabel::Bad),
                ],
                version: 1,
                last_modified: 1.0,
            }))
        }
        fn load_training_settings(&mut self) -> Result<Option<TrainingSettings>, String> {
            Ok(Some(TrainingSettings {
                classifier_config: ClassifierConfig {
                    classifier_id: slouch_domain::ClassifierId::GaussianNb,
                    params: std::collections::BTreeMap::new(),
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
            }))
        }
        fn load_reservoir_samples(&mut self) -> Result<Vec<ReservoirSample>, String> {
            Ok(Vec::new())
        }
        fn save_posture_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
            Ok(())
        }
        fn save_presence_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn backend_panic_still_resets_training_lock_for_a_followup_request() {
        let mut worker = TrainingWorker::new(PanicTestStorage, PanicBackend);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            worker.handle_message(TrainingWorkerMessage::Train { payload: None })
        }));
        assert!(result.is_err(), "backend was expected to panic");
        // The RAII guard must have reset the lock during unwind, so a second
        // request is NOT wrongly rejected with 'Training already in progress'.
        assert!(!worker.is_training());
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            worker.handle_message(TrainingWorkerMessage::Train { payload: None })
        }));
        // The second attempt panics again (backend still panics) rather than
        // returning the 'already in progress' rejection, proving the lock cleared.
        assert!(second.is_err());
    }

    #[test]
    fn empty_metrics_have_binary_confusion_matrix() {
        let metrics = empty_metrics();
        assert_eq!(metrics.confusion_matrix, vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(
            metrics.cv_type,
            Some(CrossValidationType::ShuffledStratified),
        );
    }

    struct BusyStorage;

    impl TrainingStorage for BusyStorage {
        fn load_dataset(&mut self) -> Result<Option<PostureDataset>, String> {
            Err("storage must not be called while busy".into())
        }
        fn load_training_settings(&mut self) -> Result<Option<TrainingSettings>, String> {
            Err("storage must not be called while busy".into())
        }
        fn load_reservoir_samples(&mut self) -> Result<Vec<ReservoirSample>, String> {
            Err("storage must not be called while busy".into())
        }
        fn save_posture_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
            Err("storage must not be called while busy".into())
        }
        fn save_presence_model(&mut self, _model: &SerializedModel) -> Result<(), String> {
            Err("storage must not be called while busy".into())
        }
    }

    struct BusyBackend;

    impl TrainingBackend for BusyBackend {
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
            _classifier_config: &ClassifierConfig,
            _frames: &[PostureFrame],
            _labels: &[i32],
            _cv_folds: usize,
        ) -> Result<Option<TrainingMetrics>, String> {
            Err("backend must not be called while busy".into())
        }
        fn fit(
            &mut self,
            _config: &FeatureExtractorConfig,
            _classifier_config: &ClassifierConfig,
            _frames: &[PostureFrame],
            _labels: &[i32],
        ) -> Result<SerializedModel, String> {
            Err("backend must not be called while busy".into())
        }
        fn release_training_buffers(&mut self) {}
    }

    #[test]
    fn busy_actual_worker_rejects_a_second_training_request() {
        let mut worker = TrainingWorker::new(BusyStorage, BusyBackend);
        worker.set_training(true);
        let response = worker.handle_message(TrainingWorkerMessage::Train { payload: None });
        assert_eq!(
            response,
            vec![error_response("Training already in progress")]
        );
    }

    #[test]
    fn reservoir_samples_preserve_all_feature_kinds() {
        // `From` moves the generic feature map through verbatim, so any stored
        // feature the reservoir captured survives the training-boundary conversion.
        let mut features = slouch_domain::FeatureMap::new();
        features.insert(FeatureId::RtmDetExtracted, vec![7.0]);
        features.insert(FeatureId::NlfBackbone, vec![1.0]);
        features.insert(FeatureId::NlfBackboneMax, vec![2.0]);
        features.insert(FeatureId::NlfBackboneStd, vec![3.0]);
        let sample = ReservoirSample {
            features: features.clone(),
            keypoints: Vec::new(),
            bbox: slouch_domain::BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                score: 1.0,
                width: 1.0,
                height: 1.0,
            },
        };
        let container = FeatureContainer::from(sample);
        assert_eq!(container.features, features);
        assert_eq!(container.features[&FeatureId::RtmDetExtracted], vec![7.0]);
    }

    fn full_reservoir_sample(bbox: slouch_domain::BoundingBox) -> ReservoirSample {
        let mut features = slouch_domain::FeatureMap::new();
        for feature in [
            FeatureId::RtmDetExtracted,
            FeatureId::NlfDepth,
            FeatureId::NlfBackbone,
            FeatureId::NlfBackboneMax,
            FeatureId::NlfBackboneStd,
        ] {
            features.insert(feature, vec![0.0; feature.metadata().dimensions]);
        }
        ReservoirSample {
            features,
            keypoints: vec![slouch_domain::Keypoint::new(0.5, 0.5, 0.5); 17],
            bbox,
        }
    }

    #[test]
    fn reservoir_bbox_accepts_unclamped_extent_at_frame_edge() {
        // Inference clamps x1..y2 to the frame but keeps the UNCLAMPED detector
        // extent in width/height (matching the frozen TS oracle), so a subject
        // filling the frame yields width > x2 - x1. Training must accept this —
        // demanding width == x2 - x1 aborted every real-data training with
        // "reservoir sample N bounding box is invalid".
        let sample = full_reservoir_sample(slouch_domain::BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.4,
            height: 1.3,
        });
        assert!(validate_training_reservoir(&[sample]).is_ok());
    }

    #[test]
    fn reservoir_bbox_rejects_only_genuine_invariants() {
        let base = slouch_domain::BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.8,
            y2: 0.8,
            score: 0.9,
            width: 0.7,
            height: 0.7,
        };
        assert!(validate_training_reservoir(&[full_reservoir_sample(base)]).is_ok());

        let non_finite = slouch_domain::BoundingBox {
            x1: f64::NAN,
            ..base
        };
        assert!(validate_training_reservoir(&[full_reservoir_sample(non_finite)]).is_err());

        let reversed = slouch_domain::BoundingBox { x2: 0.05, ..base };
        assert!(validate_training_reservoir(&[full_reservoir_sample(reversed)]).is_err());

        let bad_score = slouch_domain::BoundingBox { score: 1.5, ..base };
        assert!(validate_training_reservoir(&[full_reservoir_sample(bad_score)]).is_err());

        let negative_width = slouch_domain::BoundingBox {
            width: -0.1,
            ..base
        };
        assert!(validate_training_reservoir(&[full_reservoir_sample(negative_width)]).is_err());
    }
}
