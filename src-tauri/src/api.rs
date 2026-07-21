use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
};

use serde::Serialize;
use tauri::{
    ipc::{InvokeBody, Request, Response},
    AppHandle, Emitter, State,
};
use tauri_plugin_dialog::DialogExt;

use crate::{
    actors::{
        self, CameraActor, CameraDeviceInfo, CameraMode, InferenceActor, InferenceUiResult,
        TrainingActor, TrainingEvent, TrainingStage,
    },
    errors::ApiError,
};
use slouch_domain::ported::messages::schemas::ImageData;
use slouch_domain::{
    classifier_registry, feature_registry, BoundingBox, CameraSettings, DatasetStats, FrameLabel,
    Keypoint, Thumbnail, TrainingSettings, UiSettings,
};
use slouch_ml::ported::{model::Model, training_worker::TrainingWorkerResponse};
use slouch_store::ported::{
    archive::ArchiveSummary,
    storage::{DatasetStorage, StorageError, StorageInfo},
};
use slouch_vision::{ported::inference_worker::NativeInferenceResult, NativePreprocessor};

// Pure validation/parsing helpers live in a separate source file so the
// ipc_security integration suite can compile the exact same code via #[path]
// inclusion while this module keeps the Request-facing wrappers.
#[path = "ipc_validation.rs"]
mod ipc_validation;

#[cfg(test)]
use ipc_validation::validate_image_layout;
use ipc_validation::{
    ensure_js_safe_timestamp, ensure_js_safe_u64, ensure_js_safe_usize, parse_frame_label,
    validate_id, validate_page_limit, validate_thumbnail_size, validate_training_settings,
    MAX_PAGE_SIZE, MAX_SAFE_JS_INTEGER,
};

pub struct AppState {
    pub storage: Arc<DatasetStorage>,
    // Declared before `inference` so field-drop stops the capture thread (which
    // borrows the shared InferenceActor) before the inference actor is released.
    pub camera: CameraActor,
    pub inference: Arc<InferenceActor>,
    pub training: TrainingActor,
    model_paths: ModelPaths,
    inference_ready: AtomicBool,
    shutdown_started: AtomicBool,
    training_running: AtomicBool,
    next_training_job_id: AtomicU64,
    import_reserved: AtomicBool,
    importing: AtomicBool,
    undo_history: Mutex<VecDeque<UndoAction>>,
    undo_revision: AtomicU64,
    preprocessor: Mutex<NativePreprocessor>,
    lifecycle: Mutex<()>,
}

#[derive(Debug, Clone)]
enum UndoAction {
    DeleteFrame(String),
    RestoreFrame(slouch_domain::PostureFrame),
}

impl AppState {
    pub fn shutdown(&self) {
        if self.shutdown_started.swap(true, Ordering::AcqRel) {
            return;
        }
        self.inference_ready.store(false, Ordering::Release);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        self.training.shutdown_until(deadline);
        self.camera.shutdown_until(deadline);
        self.inference.shutdown_until(deadline);
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, Clone)]
struct ModelPaths {
    rtmdet: Option<PathBuf>,
    rtmpose: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct StorageInfoDto {
    #[specta(type = specta_typescript::Number)]
    pub used: u64,
    #[specta(type = specta_typescript::Number)]
    pub available: u64,
    #[specta(type = specta_typescript::Number)]
    pub quota: u64,
}

impl TryFrom<StorageInfo> for StorageInfoDto {
    type Error = ApiError;

    fn try_from(value: StorageInfo) -> Result<Self, Self::Error> {
        ensure_js_safe_u64(value.used, "storage used bytes")?;
        ensure_js_safe_u64(value.available, "storage available bytes")?;
        ensure_js_safe_u64(value.quota, "storage quota bytes")?;
        Ok(Self {
            used: value.used,
            available: value.available,
            quota: value.quota,
        })
    }
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct AppStatus {
    pub ready: bool,
    pub inference_ready: bool,
    #[specta(type = specta_typescript::Number)]
    pub dataset_version: u64,
    pub storage: StorageInfoDto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub enum UndoActionKind {
    RemoveCapture,
    RestoreFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct UndoStatus {
    pub available: bool,
    #[specta(type = specta_typescript::Number)]
    pub depth: usize,
    pub next_action: Option<UndoActionKind>,
    #[specta(type = specta_typescript::Number)]
    pub revision: u64,
}

#[derive(
    Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, specta::Type, tauri_specta::Event,
)]
#[serde(rename_all = "camelCase")]
#[tauri_specta(event_name = "undo-status-changed")]
pub struct UndoStatusChangedEvent {
    pub status: UndoStatus,
}

#[derive(Debug, Clone, Serialize, specta::Type, tauri_specta::Event)]
#[serde(rename_all = "camelCase")]
#[tauri_specta(event_name = "native-state-changed")]
pub struct NativeStateChangedEvent {
    pub reason: String,
    pub state: NativeStateSnapshot,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct NativeStateSnapshot {
    pub app: AppStatus,
    pub camera_settings: CameraSettings,
    pub ui_settings: UiSettings,
    pub training_settings: Option<TrainingSettings>,
    pub active_models: ActiveModelMetadata,
    pub undo: UndoStatus,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct DatasetChangedEvent {
    version: u64,
    reason: String,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct FrameMetadataDto {
    pub id: String,
    pub timestamp: f64,
    pub keypoints: Vec<Keypoint>,
    pub bbox: BoundingBox,
    pub label: FrameLabel,
    pub thumbnail_mime_type: String,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ReservoirMetadata {
    pub total_seen: u32,
    pub count: u32,
    pub max_samples: u32,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct DatasetPage {
    pub frames: Vec<FrameMetadataDto>,
    #[specta(type = specta_typescript::Number)]
    pub offset: usize,
    #[specta(type = specta_typescript::Number)]
    pub limit: usize,
    #[specta(type = specta_typescript::Number)]
    pub total: usize,
    #[specta(type = specta_typescript::Number)]
    pub version: u64,
    pub last_modified: f64,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    pub classifier_id: String,
    pub trained_at: f64,
    pub feature_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ActiveModelMetadata {
    pub posture: Option<ModelMetadata>,
    pub presence: Option<ModelMetadata>,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveSummaryDto {
    #[specta(type = specta_typescript::Number)]
    pub frame_count: usize,
    #[specta(type = specta_typescript::Number)]
    pub dataset_version: u64,
}

impl TryFrom<ArchiveSummary> for ArchiveSummaryDto {
    type Error = ApiError;

    fn try_from(value: ArchiveSummary) -> Result<Self, Self::Error> {
        ensure_js_safe_usize(value.frame_count, "archive frame count")?;
        ensure_js_safe_u64(value.dataset_version, "archive dataset version")?;
        Ok(Self {
            frame_count: value.frame_count,
            dataset_version: value.dataset_version,
        })
    }
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveImportResult {
    #[specta(type = specta_typescript::Number)]
    pub frame_count: usize,
    #[specta(type = specta_typescript::Number)]
    pub dataset_version: u64,
    pub state: NativeStateSnapshot,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ShortcutStatus {
    pub registered: bool,
}

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainingStatus {
    pub running: bool,
}

fn publish_dataset_changed(app: &AppHandle, storage: &DatasetStorage, reason: &str) {
    let version = match storage.get_dataset_metadata() {
        Ok(metadata) => metadata.version,
        Err(error) => {
            log::error!(target: "storage", "dataset mutation committed but version reload failed: {error}");
            return;
        }
    };
    if let Err(error) = app.emit(
        "dataset-changed",
        DatasetChangedEvent {
            version,
            reason: reason.to_owned(),
        },
    ) {
        log::error!(target: "storage", "dataset mutation committed but event delivery failed: {error}");
    }
}

fn app_status_state(state: &AppState) -> Result<AppStatus, ApiError> {
    let dataset = state
        .storage
        .get_dataset_metadata()
        .map_err(map_storage_read)?;
    let storage =
        StorageInfoDto::try_from(state.storage.get_storage_info().map_err(map_storage_read)?)?;
    let inference_ready = inference_is_ready(state);
    Ok(AppStatus {
        ready: inference_ready,
        inference_ready,
        dataset_version: dataset.version,
        storage,
    })
}

#[tauri::command]
#[specta::specta]
pub fn app_status(state: State<'_, AppState>) -> Result<AppStatus, ApiError> {
    app_status_state(&state)
}

#[tauri::command]
#[specta::specta]
pub fn initialize_inference(state: State<'_, AppState>) -> Result<(), ApiError> {
    initialize_inference_state(&state)
}

fn inference_is_ready(state: &AppState) -> bool {
    state.inference_ready.load(Ordering::Acquire)
        && state.inference.is_healthy()
        && state.training.is_healthy()
        && !state.importing.load(Ordering::Acquire)
        && !state.shutdown_started.load(Ordering::Acquire)
}

fn initialize_inference_state(state: &AppState) -> Result<(), ApiError> {
    state.inference_ready.store(false, Ordering::Release);
    let rtmdet = state
        .model_paths
        .rtmdet
        .clone()
        .ok_or_else(|| ApiError::NotReady("RTMDet model resource is unavailable".into()))?;
    let rtmpose = state
        .model_paths
        .rtmpose
        .clone()
        .ok_or_else(|| ApiError::NotReady("RTMPose model resource is unavailable".into()))?;
    let responses = state
        .inference
        .send(actors::initialize_message(rtmdet, rtmpose))?;
    if !responses.iter().any(|response| {
        matches!(
            response,
            slouch_vision::ported::inference_worker::WorkerResponse::Initialized { .. }
        )
    }) {
        return Err(ApiError::Inference(
            "native models failed to initialize".into(),
        ));
    }

    // A corrupt or incompatible active model must never take the whole app down
    // at startup. Native detection (RTMDet/RTMPose) is already initialized above;
    // if the trained classifier pair cannot be loaded or published, surface it in
    // the log and continue with no active classifier so the user can retrain,
    // rather than aborting the setup hook (which crash-loops the application).
    match state.storage.load_active_model_pair() {
        Ok(Some((posture, presence))) => {
            if let Err(error) = load_runtime_pair(state, posture.as_ref(), presence.as_ref()) {
                log::error!(
                    target: "inference",
                    "active model could not be published to the inference runtime; continuing with no trained model (retrain to restore): {error}"
                );
                let _ = state.inference.send(
                    slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier,
                );
            }
        }
        Ok(None) => {}
        Err(error) => {
            log::error!(
                target: "inference",
                "active model could not be loaded from storage; continuing with no trained model (retrain to restore): {error}"
            );
            let _ = state.inference.send(
                slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier,
            );
        }
    }
    state.inference_ready.store(true, Ordering::Release);
    Ok(())
}

#[tauri::command]
pub fn infer_frame(state: State<'_, AppState>, request: Request<'_>) -> Result<Response, ApiError> {
    let image = parse_raw_image(&request)?;
    if !inference_is_ready(&state) {
        return Err(ApiError::NotReady(
            "inference models are not initialized".into(),
        ));
    }
    let settings = state
        .storage
        .get_camera_settings()
        .map_err(map_storage_read)?;
    let image = state
        .preprocessor
        .lock()
        .map_err(|_| ApiError::Internal("native preprocessor lock is poisoned".into()))?
        .process(image, &settings)
        .map_err(ApiError::InvalidRequest)?;
    let request_id = parse_header(&request, "x-slouch-request-id")?;
    let response = state
        .inference
        .send_frame(actors::raw_inference_message(image, request_id))?;
    let result = response
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Inference("inference actor returned no response".into()))
        .and_then(actors::inference_result)?;
    if !inference_is_ready(&state) {
        return Err(ApiError::NotReady(
            "inference became unavailable while processing the frame".into(),
        ));
    }
    let person_found = result.person_found;
    let bbox = result.bbox;
    let keypoints = result.keypoints.clone();
    let classification = result.classification;
    let token = state.inference.cache_result(request_id, result)?;
    encode_response(&InferenceUiResult {
        request_id,
        token,
        person_found,
        bbox,
        keypoints,
        classification,
    })
}

#[tauri::command]
#[specta::specta]
pub fn train_models(
    state: State<'_, AppState>,
    do_cv: Option<bool>,
    on_event: tauri::ipc::Channel<TrainingEvent>,
) -> Result<slouch_ml::ported::training_worker::TrainingResultResponse, ApiError> {
    {
        let _lifecycle = state
            .lifecycle
            .lock()
            .map_err(|_| ApiError::Internal("lifecycle lock is poisoned".into()))?;
        if state.importing.load(Ordering::Acquire) {
            let error = ApiError::Busy("training is unavailable during archive import".into());
            log::warn!(target: "training", "train_models rejected: {error}");
            return Err(error);
        }
        if state.training_running.swap(true, Ordering::AcqRel) {
            let error = ApiError::Busy("training already in progress".into());
            log::warn!(target: "training", "train_models rejected: {error}");
            return Err(error);
        }
    }
    let outcome = (|| {
        validate_training_settings_present(&state)?;
        let do_cv = do_cv.unwrap_or(true);
        let snapshot = state
            .storage
            .training_snapshot(do_cv)
            .map_err(map_storage_read)?;
        let job_id = state.next_training_job_id.fetch_add(1, Ordering::AcqRel);
        let events = on_event.clone();
        let response = match state.training.train(job_id, do_cv, Some(on_event)) {
            Ok(response) => response,
            Err(ApiError::Cancelled(message)) => {
                let _ = events.send(TrainingEvent::Cancelled {
                    job_id,
                    sequence: 2,
                });
                return Err(ApiError::Cancelled(message));
            }
            Err(error) => {
                let _ = events.send(TrainingEvent::Failed {
                    job_id,
                    sequence: 2,
                    error: error.to_string(),
                });
                return Err(error);
            }
        };
        let _ = events.send(TrainingEvent::Progress {
            job_id,
            sequence: 3,
            stage: TrainingStage::Deploying,
            progress: 95,
        });
        match deploy_training_models(&state, &response, snapshot, do_cv) {
            Ok(result) => {
                let _ = events.send(TrainingEvent::Completed {
                    job_id,
                    sequence: 4,
                    result: Box::new(result.clone()),
                });
                Ok(result)
            }
            Err(error) => {
                let _ = events.send(TrainingEvent::Failed {
                    job_id,
                    sequence: 4,
                    error: error.to_string(),
                });
                Err(error)
            }
        }
    })();
    state.training_running.store(false, Ordering::Release);
    if let Err(error) = &outcome {
        log::warn!(target: "training", "train_models rejected: {error}");
    }
    outcome
}

#[tauri::command]
#[specta::specta]
pub fn get_training_status(state: State<'_, AppState>) -> Result<TrainingStatus, ApiError> {
    Ok(TrainingStatus {
        running: state.training_running.load(Ordering::Acquire),
    })
}

#[tauri::command]
#[specta::specta]
pub fn cancel_training(state: State<'_, AppState>) -> Result<(), ApiError> {
    state.training.cancel()
}

#[tauri::command]
#[specta::specta]
pub fn get_dataset_page(
    state: State<'_, AppState>,
    offset: Option<u32>,
    limit: Option<u32>,
) -> Result<DatasetPage, ApiError> {
    let offset = offset.unwrap_or(0) as usize;
    let limit = limit.unwrap_or(MAX_PAGE_SIZE as u32) as usize;
    validate_page_limit(limit)?;
    let page = state
        .storage
        .get_frame_metadata_page(offset, limit)
        .map_err(map_storage_read)?;
    ensure_js_safe_usize(page.offset, "dataset page offset")?;
    ensure_js_safe_usize(page.limit, "dataset page limit")?;
    ensure_js_safe_usize(page.total, "dataset frame count")?;
    ensure_js_safe_u64(page.version, "dataset version")?;
    Ok(DatasetPage {
        frames: page
            .frames
            .into_iter()
            .map(|frame| FrameMetadataDto {
                id: frame.id,
                timestamp: frame.timestamp,
                keypoints: frame.keypoints,
                bbox: frame.bbox,
                label: frame.label,
                thumbnail_mime_type: frame.thumbnail_mime_type,
            })
            .collect(),
        offset: page.offset,
        limit: page.limit,
        total: page.total,
        version: page.version,
        last_modified: page.last_modified,
    })
}

#[tauri::command]
pub fn get_thumbnail(state: State<'_, AppState>, id: String) -> Result<Response, ApiError> {
    validate_id(&id)?;
    let thumbnail = state
        .storage
        .get_thumbnail(&id)
        .map_err(map_storage_read)?
        .ok_or_else(|| ApiError::NotFound("thumbnail was not found".into()))?;
    Ok(Response::new(thumbnail.1))
}

#[tauri::command]
#[specta::specta]
pub fn get_dataset_stats(state: State<'_, AppState>) -> Result<DatasetStats, ApiError> {
    state.storage.get_stats().map_err(map_storage_read)
}

#[tauri::command]
#[specta::specta]
pub fn get_needs_retraining(state: State<'_, AppState>) -> Result<bool, ApiError> {
    state.storage.needs_retraining().map_err(map_storage_read)
}

#[tauri::command]
#[specta::specta]
pub fn get_reservoir_metadata(state: State<'_, AppState>) -> Result<ReservoirMetadata, ApiError> {
    let metadata = state.storage.reservoir_meta().map_err(map_storage_read)?;
    Ok(ReservoirMetadata {
        total_seen: u32::try_from(metadata.total_seen)
            .map_err(|_| ApiError::Internal("reservoir total exceeds u32".into()))?,
        count: u32::try_from(metadata.count)
            .map_err(|_| ApiError::Internal("reservoir count exceeds u32".into()))?,
        max_samples: u32::try_from(metadata.max_samples)
            .map_err(|_| ApiError::Internal("reservoir capacity exceeds u32".into()))?,
    })
}

#[tauri::command]
#[specta::specta]
pub fn get_camera_settings(state: State<'_, AppState>) -> Result<CameraSettings, ApiError> {
    state
        .storage
        .get_camera_settings()
        .map_err(map_storage_read)
}

#[tauri::command]
#[specta::specta]
pub fn save_camera_settings(
    state: State<'_, AppState>,
    settings: CameraSettings,
) -> Result<(), ApiError> {
    settings.validate().map_err(ApiError::InvalidRequest)?;
    state
        .storage
        .save_camera_settings(&settings)
        .map_err(map_storage_write)
}

#[tauri::command]
#[specta::specta]
pub fn reset_camera_settings(state: State<'_, AppState>) -> Result<CameraSettings, ApiError> {
    state
        .storage
        .reset_camera_settings()
        .map_err(map_storage_write)
}

#[tauri::command]
#[specta::specta]
pub fn get_ui_settings(state: State<'_, AppState>) -> Result<UiSettings, ApiError> {
    state.storage.get_ui_settings().map_err(map_storage_read)
}

#[tauri::command]
#[specta::specta]
pub fn save_ui_settings(state: State<'_, AppState>, settings: UiSettings) -> Result<(), ApiError> {
    settings.validate().map_err(ApiError::InvalidRequest)?;
    state
        .storage
        .save_ui_settings(&settings)
        .map_err(map_storage_write)
}

#[tauri::command]
#[specta::specta]
pub fn reset_ui_settings(state: State<'_, AppState>) -> Result<UiSettings, ApiError> {
    state.storage.reset_ui_settings().map_err(map_storage_write)
}

#[tauri::command]
#[specta::specta]
pub fn get_training_settings(
    state: State<'_, AppState>,
) -> Result<Option<TrainingSettings>, ApiError> {
    state
        .storage
        .get_training_settings()
        .map_err(map_storage_read)
}

#[tauri::command]
#[specta::specta]
pub fn reset_training_settings(app: AppHandle, state: State<'_, AppState>) -> Result<(), ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_settings_mutation_allowed(&state)?;
    state
        .storage
        .clear_training_settings()
        .map_err(map_storage_write)?;
    publish_dataset_changed(&app, &state.storage, "training-settings-reset");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn save_training_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: TrainingSettings,
) -> Result<(), ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_settings_mutation_allowed(&state)?;
    validate_training_settings(&settings)?;
    state
        .storage
        .save_training_settings(&settings)
        .map_err(map_storage_write)?;
    publish_dataset_changed(&app, &state.storage, "training-settings");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn update_frame_label(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
    label: FrameLabel,
) -> Result<(), ApiError> {
    validate_id(&id)?;
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_dataset_mutation_allowed(&state)?;
    let previous = state
        .storage
        .get_frame_by_id(&id)
        .map_err(map_storage_read)?
        .ok_or_else(|| ApiError::NotFound(format!("frame with id {id} not found")))?;
    state
        .storage
        .update_frame_label(&id, label)
        .map_err(map_storage_write)?;
    push_undo(&state, UndoAction::RestoreFrame(previous));
    publish_undo_status(&app, &state);
    publish_dataset_changed(&app, &state.storage, "label-updated");
    Ok(())
}

#[tauri::command]
pub fn save_capture(
    app: AppHandle,
    state: State<'_, AppState>,
    request: Request<'_>,
) -> Result<(), ApiError> {
    let outcome = (|| -> Result<(), ApiError> {
        require_ipc_version(&request)?;
        let bytes = raw_body(&request, "save_capture requires a raw thumbnail body")?;
        validate_thumbnail_size(bytes.len())?;
        let mime_type = header_string(&request, "x-slouch-mime-type")?;
        if !matches!(
            mime_type.as_str(),
            "image/jpeg" | "image/png" | "image/webp"
        ) {
            return Err(ApiError::InvalidRequest(
                "thumbnail MIME type must be image/jpeg, image/png, or image/webp".into(),
            ));
        }
        let request_id = parse_header(&request, "x-slouch-request-id")?;
        let token = parse_header(&request, "x-slouch-token")?;
        if token == request_id {
            return Err(ApiError::InvalidRequest(
                "inference token must be distinct from request ID".into(),
            ));
        }
        let frame_id = header_string(&request, "x-slouch-frame-id")?;
        validate_id(&frame_id)?;
        let undo_frame_id = frame_id.clone();
        let timestamp = header_string(&request, "x-slouch-timestamp")?
            .parse::<f64>()
            .map_err(|_| ApiError::InvalidRequest("timestamp must be a finite number".into()))?;
        if !timestamp.is_finite()
            || timestamp <= 0.0
            || timestamp.fract() != 0.0
            || timestamp > MAX_SAFE_JS_INTEGER as f64
        {
            return Err(ApiError::InvalidRequest(
                "timestamp must be a positive JavaScript-safe integer".into(),
            ));
        }
        let label = parse_frame_label(&header_string(&request, "x-slouch-label")?)?;
        let _lifecycle = lock_mutation_lifecycle(&state)?;
        ensure_dataset_mutation_allowed(&state)?;
        if !inference_is_ready(&state) {
            return Err(ApiError::NotReady("inference is unavailable".into()));
        }
        let result = state.inference.checkout_result(token, request_id)?;
        let save_result = capture_frame(
            &state, bytes, mime_type, frame_id, timestamp, label, &result,
        );
        match save_result {
            Ok(()) => {
                if let Err(error) = state.inference.commit_result(token, request_id) {
                    state.inference_ready.store(false, Ordering::Release);
                    log::error!(target: "inference", "capture committed but token finalization failed: {error}");
                }
                push_undo(&state, UndoAction::DeleteFrame(undo_frame_id));
                publish_undo_status(&app, &state);
                publish_dataset_changed(&app, &state.storage, "capture-saved");
                Ok(())
            }
            Err(error) => {
                if let Err(restore_error) =
                    state.inference.restore_result(token, request_id, result)
                {
                    state.inference_ready.store(false, Ordering::Release);
                    log::error!(target: "inference", "capture failed and token restoration also failed: {restore_error}");
                }
                Err(error)
            }
        }
    })();
    if let Err(error) = &outcome {
        log::warn!(target: "storage", "save_capture rejected: {error}");
    }
    outcome
}

fn capture_frame(
    state: &AppState,
    bytes: &[u8],
    mime_type: String,
    frame_id: String,
    timestamp: f64,
    label: FrameLabel,
    result: &NativeInferenceResult,
) -> Result<(), ApiError> {
    if state
        .storage
        .get_frame_by_id(&frame_id)
        .map_err(map_storage_read)?
        .is_some()
    {
        return Err(ApiError::InvalidRequest("frame ID already exists".into()));
    }
    let bbox = result
        .bbox
        .as_ref()
        .map(|expanded| expanded.original)
        .ok_or_else(|| ApiError::InvalidRequest("inference token has no detected person".into()))?;
    let keypoints = result
        .keypoints
        .clone()
        .ok_or_else(|| ApiError::InvalidRequest("inference token has no keypoints".into()))?;
    let frame = slouch_domain::PostureFrame {
        id: frame_id,
        timestamp,
        features: result.features.clone(),
        thumbnail: Thumbnail {
            mime_type,
            bytes: bytes.to_vec(),
        },
        keypoints,
        bbox,
        label,
    };
    state.storage.save_frame(&frame).map_err(map_storage_write)
}

#[tauri::command]
#[specta::specta]
pub fn cleanup_unused_frames(app: AppHandle, state: State<'_, AppState>) -> Result<u32, ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_destructive_mutation_allowed(&state)?;
    let removed = state
        .storage
        .remove_frames_by_label(FrameLabel::Unused)
        .map_err(map_storage_write)?;
    if removed > 0 {
        state
            .undo_history
            .lock()
            .map_err(|_| ApiError::Internal("undo history lock is poisoned".into()))?
            .clear();
        publish_undo_status(&app, &state);
        publish_dataset_changed(&app, &state.storage, "unused-frames-cleaned");
    }
    u32::try_from(removed).map_err(|_| ApiError::Internal("cleanup count exceeds u32".into()))
}

#[tauri::command]
#[specta::specta]
pub fn delete_frame(
    app: AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<(), ApiError> {
    validate_id(&id)?;
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_dataset_mutation_allowed(&state)?;
    let previous = state
        .storage
        .get_frame_by_id(&id)
        .map_err(map_storage_read)?
        .ok_or_else(|| ApiError::NotFound(format!("frame with id {id} not found")))?;
    state.storage.delete_frame(&id).map_err(map_storage_write)?;
    push_undo(&state, UndoAction::RestoreFrame(previous));
    publish_undo_status(&app, &state);
    publish_dataset_changed(&app, &state.storage, "frame-deleted");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn undo_last_dataset_change(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_dataset_mutation_allowed(&state)?;
    let action = state
        .undo_history
        .lock()
        .map_err(|_| ApiError::Internal("undo history lock is poisoned".into()))?
        .pop_back()
        .ok_or_else(|| ApiError::NotFound("there is no dataset change to undo".into()))?;
    let result = match &action {
        UndoAction::DeleteFrame(id) => state.storage.delete_frame(id).map_err(map_storage_write),
        UndoAction::RestoreFrame(frame) => {
            state.storage.save_frame(frame).map_err(map_storage_write)
        }
    };
    if let Err(error) = result {
        push_undo(&state, action);
        return Err(error);
    }
    publish_undo_status(&app, &state);
    publish_dataset_changed(&app, &state.storage, "dataset-undo");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn reset_dataset(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<NativeStateSnapshot, ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_destructive_mutation_allowed(&state)?;
    state.storage.clear_dataset().map_err(map_storage_write)?;
    clear_undo_history(&state);
    state
        .preprocessor
        .lock()
        .map_err(|_| ApiError::Internal("native preprocessor lock is poisoned".into()))?
        .reset();
    if let Err(error) = state
        .inference
        .send(slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier)
        .and_then(|_| state.inference.clear_cache())
    {
        state.inference_ready.store(false, Ordering::Release);
        log::error!(target: "inference", "dataset reset committed but runtime cleanup failed: {error}");
    }
    publish_undo_status(&app, &state);
    publish_dataset_changed(&app, &state.storage, "dataset-reset");
    let snapshot = native_state_snapshot(&state)?;
    publish_native_state(&app, "dataset-reset", snapshot.clone());
    Ok(snapshot)
}

#[tauri::command]
#[specta::specta]
pub fn reset_all_data(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<NativeStateSnapshot, ApiError> {
    let _lifecycle = lock_mutation_lifecycle(&state)?;
    ensure_destructive_mutation_allowed(&state)?;
    state.storage.reset_all().map_err(map_storage_write)?;
    clear_undo_history(&state);
    state
        .preprocessor
        .lock()
        .map_err(|_| ApiError::Internal("native preprocessor lock is poisoned".into()))?
        .reset();
    if let Err(error) = state
        .inference
        .send(slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier)
        .and_then(|_| state.inference.clear_cache())
    {
        state.inference_ready.store(false, Ordering::Release);
        log::error!(target: "inference", "application reset committed but runtime cleanup failed: {error}");
    }
    publish_undo_status(&app, &state);
    publish_dataset_changed(&app, &state.storage, "all-data-reset");
    let snapshot = native_state_snapshot(&state)?;
    publish_native_state(&app, "all-data-reset", snapshot.clone());
    Ok(snapshot)
}

#[tauri::command]
#[specta::specta]
pub fn get_undo_status(state: State<'_, AppState>) -> Result<UndoStatus, ApiError> {
    undo_status(&state)
}

#[tauri::command]
#[specta::specta]
pub fn get_classifier_registry() -> Result<Vec<slouch_domain::ClassifierMetadata>, ApiError> {
    Ok(classifier_registry())
}

#[tauri::command]
#[specta::specta]
pub fn get_feature_registry() -> Result<Vec<slouch_domain::FeatureMetadata>, ApiError> {
    Ok(feature_registry().to_vec())
}

fn active_model_metadata_state(state: &AppState) -> Result<ActiveModelMetadata, ApiError> {
    match state
        .storage
        .load_active_model_pair()
        .map_err(map_storage_read)?
    {
        Some((posture, presence)) => Ok(ActiveModelMetadata {
            posture: posture.map(|model| model_metadata(&model)).transpose()?,
            presence: presence.map(|model| model_metadata(&model)).transpose()?,
        }),
        None => Ok(ActiveModelMetadata {
            posture: None,
            presence: None,
        }),
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_active_model_metadata(
    state: State<'_, AppState>,
) -> Result<ActiveModelMetadata, ApiError> {
    active_model_metadata_state(&state)
}

#[tauri::command]
#[specta::specta]
pub fn export_dataset(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<ArchiveSummaryDto>, ApiError> {
    let selected = app
        .dialog()
        .file()
        .add_filter("Slouch Tracker archive", &["slouchpack"])
        .set_file_name("slouch-tracker.slouchpack")
        .blocking_save_file();
    let Some(selected) = selected else {
        return Ok(None);
    };
    let path = selected
        .into_path()
        .map_err(|error| ApiError::InvalidRequest(error.to_string()))?;
    let summary = state
        .storage
        .export_archive(path, env!("CARGO_PKG_VERSION"))
        .map_err(map_storage_read)?;
    Ok(Some(ArchiveSummaryDto::try_from(summary)?))
}

#[tauri::command]
#[specta::specta]
pub fn import_dataset(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<ArchiveImportResult>, ApiError> {
    {
        let _lifecycle = lock_mutation_lifecycle(&state)?;
        if state.import_reserved.swap(true, Ordering::AcqRel) {
            return Err(ApiError::Busy(
                "archive import is already in progress".into(),
            ));
        }
    }

    let selected = app
        .dialog()
        .file()
        .add_filter("Slouch Tracker archive", &["slouchpack"])
        .blocking_pick_file();
    let Some(selected) = selected else {
        state.import_reserved.store(false, Ordering::Release);
        return Ok(None);
    };
    let path = match selected.into_path() {
        Ok(path) => path,
        Err(error) => {
            state.import_reserved.store(false, Ordering::Release);
            return Err(ApiError::InvalidRequest(error.to_string()));
        }
    };

    {
        let _lifecycle = match lock_mutation_lifecycle(&state) {
            Ok(lifecycle) => lifecycle,
            Err(error) => {
                state.import_reserved.store(false, Ordering::Release);
                return Err(error);
            }
        };
        state.importing.store(true, Ordering::Release);
    }
    if state.training_running.load(Ordering::Acquire) {
        if let Err(error) = state.training.cancel() {
            if !matches!(error, ApiError::NotReady(_)) {
                state.importing.store(false, Ordering::Release);
                state.import_reserved.store(false, Ordering::Release);
                return Err(error);
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while state.training_running.load(Ordering::Acquire) {
            if std::time::Instant::now() >= deadline {
                state.importing.store(false, Ordering::Release);
                state.import_reserved.store(false, Ordering::Release);
                return Err(ApiError::Busy(
                    "training did not quiesce for archive import".into(),
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    state.inference_ready.store(false, Ordering::Release);
    let outcome = (|| {
        state.inference.clear_cache()?;
        let summary = state
            .storage
            .import_archive(path)
            .map_err(map_archive_import)?;
        clear_undo_history(&state);
        state
            .preprocessor
            .lock()
            .map_err(|_| ApiError::Internal("native preprocessor lock is poisoned".into()))?
            .reset();
        state
            .inference
            .send(
                slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier,
            )
            .and_then(|_| state.inference.clear_cache())
            .and_then(|_| initialize_inference_state(&state))?;
        Ok(summary)
    })();
    state.importing.store(false, Ordering::Release);
    state.import_reserved.store(false, Ordering::Release);
    match outcome {
        Ok(summary) => {
            publish_undo_status(&app, &state);
            publish_dataset_changed(&app, &state.storage, "dataset-imported");
            let state_snapshot = native_state_snapshot(&state)?;
            publish_native_state(&app, "dataset-imported", state_snapshot.clone());
            let summary = ArchiveSummaryDto::try_from(summary)?;
            Ok(Some(ArchiveImportResult {
                frame_count: summary.frame_count,
                dataset_version: summary.dataset_version,
                state: state_snapshot,
            }))
        }
        Err(error) => {
            state.inference_ready.store(false, Ordering::Release);
            Err(error)
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn get_shortcut_status() -> Result<ShortcutStatus, ApiError> {
    Ok(ShortcutStatus { registered: true })
}

#[tauri::command]
#[specta::specta]
pub fn start_camera(
    state: State<'_, AppState>,
    on_result: tauri::ipc::Channel<InferenceUiResult>,
) -> Result<(), ApiError> {
    state.camera.set_result_sink(move |result| {
        let _ = on_result.send(result);
    });
    state.camera.set_mode(CameraMode::Foreground)?;
    state.camera.start_capture()
}

#[tauri::command]
#[specta::specta]
pub fn stop_camera(state: State<'_, AppState>) -> Result<(), ApiError> {
    state.camera.stop_capture()
}

#[tauri::command]
#[specta::specta]
pub fn list_cameras(state: State<'_, AppState>) -> Result<Vec<CameraDeviceInfo>, ApiError> {
    state.camera.list_devices()
}

pub fn initialize_state(data_dir: PathBuf, resource_dir: PathBuf) -> Result<AppState, ApiError> {
    initialize_onnx_runtime(&resource_dir)?;
    std::fs::create_dir_all(&data_dir).map_err(|error| ApiError::Storage(error.to_string()))?;
    let storage = Arc::new(
        DatasetStorage::open(data_dir.join("slouch-tracker.sqlite3")).map_err(map_storage_read)?,
    );
    let model_paths = ModelPaths {
        rtmdet: find_resource(&resource_dir, "rtmdet-nano.onnx"),
        rtmpose: find_resource(&resource_dir, "rtmpose-m.onnx"),
    };
    let training = TrainingActor::start(storage.clone())?;
    let inference = match InferenceActor::start(storage.clone()) {
        Ok(actor) => Arc::new(actor),
        Err(error) => {
            training.shutdown();
            return Err(error);
        }
    };
    let camera = match CameraActor::start(inference.clone(), storage.clone()) {
        Ok(actor) => actor,
        Err(error) => {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            inference.shutdown_until(deadline);
            training.shutdown();
            return Err(error);
        }
    };
    Ok(AppState {
        training,
        camera,
        inference,
        storage,
        model_paths,
        inference_ready: AtomicBool::new(false),
        shutdown_started: AtomicBool::new(false),
        training_running: AtomicBool::new(false),
        next_training_job_id: AtomicU64::new(1),
        import_reserved: AtomicBool::new(false),
        importing: AtomicBool::new(false),
        undo_history: Mutex::new(VecDeque::new()),
        undo_revision: AtomicU64::new(0),
        preprocessor: Mutex::new(NativePreprocessor::default()),
        lifecycle: Mutex::new(()),
    })
}

fn initialize_onnx_runtime(resource_dir: &Path) -> Result<(), ApiError> {
    let runtime = [
        resource_dir.join("resources/onnxruntime/windows-x86_64/onnxruntime.dll"),
        resource_dir.join("onnxruntime/windows-x86_64/onnxruntime.dll"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources/onnxruntime/windows-x86_64/onnxruntime.dll"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| ApiError::NotReady("packaged ONNX Runtime DLL is unavailable".into()))?;
    let environment = ort::init_from(runtime).map_err(|error| {
        ApiError::NotReady(format!("failed to load packaged ONNX Runtime: {error}"))
    })?;
    environment.with_name("slouch-tracker").commit();
    Ok(())
}

fn find_resource(resource_dir: &Path, filename: &str) -> Option<PathBuf> {
    [
        resource_dir.join("resources/models").join(filename),
        resource_dir.join("models").join(filename),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources/models")
            .join(filename),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

enum DeploymentFailure {
    Persistence(ApiError),
    Publication(ApiError),
}

fn persist_then_publish<P, U>(persist: P, publish: U) -> Result<(), DeploymentFailure>
where
    P: FnOnce() -> Result<(), ApiError>,
    U: FnOnce() -> Result<(), ApiError>,
{
    persist().map_err(DeploymentFailure::Persistence)?;
    publish().map_err(DeploymentFailure::Publication)
}

fn deploy_training_models(
    state: &AppState,
    response: &TrainingWorkerResponse,
    expected_snapshot: slouch_store::ported::storage::TrainingSnapshot,
    do_cv: bool,
) -> Result<slouch_ml::ported::training_worker::TrainingResultResponse, ApiError> {
    let TrainingWorkerResponse::Result { result, models } = response else {
        return Err(ApiError::Training(
            "training returned an error response".into(),
        ));
    };
    if !result.success {
        return Err(ApiError::Training(result.errors.join("; ")));
    }
    // Oracle semantics: a run with GOOD+BAD but no AWAY frames skips presence and
    // succeeds posture-only (and vice versa). Deploy the whole generation as long
    // as at least one role was trained; an absent role means "no model for that
    // role in this generation" and runtime falls back accordingly (presence ->
    // RTMDet confidence, posture -> no good-probability). Zero roles cannot occur
    // here because the worker returns success=false with an "Insufficient data"
    // error, which the guard above already rejects.
    let posture = models.posture.as_ref();
    let presence = models.presence.as_ref();
    if posture.is_none() && presence.is_none() {
        return Err(ApiError::Training(
            "training reported success but produced no model".into(),
        ));
    }
    if let Some(model) = posture {
        validate_runtime_model(model)?;
    }
    if let Some(model) = presence {
        validate_runtime_model(model)?;
    }
    let previous_generation = state
        .storage
        .active_model_generation_id()
        .map_err(map_storage_read)?;
    let previous = state
        .storage
        .load_active_model_pair()
        .map_err(map_storage_read)?;
    match persist_then_publish(
        || {
            state
                .storage
                .save_model_pair_if_snapshot(
                    posture,
                    presence,
                    expected_snapshot,
                    do_cv,
                    if do_cv {
                        result.posture_result.as_ref().map(|value| &value.metrics)
                    } else {
                        None
                    },
                    if do_cv {
                        result.presence_result.as_ref().map(|value| &value.metrics)
                    } else {
                        None
                    },
                )
                .map_err(map_storage_write)
        },
        || load_runtime_pair(state, posture, presence),
    ) {
        Ok(()) => {}
        Err(DeploymentFailure::Persistence(error)) => return Err(error),
        Err(DeploymentFailure::Publication(error)) => {
            if let (Some(generation), Some((old_posture, old_presence))) =
                (previous_generation, previous)
            {
                let _ = state.storage.restore_active_model_generation(generation);
                let _ = load_runtime_pair(state, old_posture.as_ref(), old_presence.as_ref());
            } else {
                let _ = state.inference.send(
                    slouch_domain::ported::messages::schemas::InferenceWorkerMessage::UnloadClassifier,
                );
            }
            return Err(error);
        }
    }
    Ok((**result).clone())
}

fn validate_runtime_model(
    model: &slouch_ml::ported::types::SerializedModel,
) -> Result<(), ApiError> {
    Model::<slouch_domain::InferenceResult>::from_json(model.clone())
        .map(|_| ())
        .map_err(|error| ApiError::InvalidRequest(format!("trained model is invalid: {error}")))
}

fn load_runtime_pair(
    state: &AppState,
    posture: Option<&slouch_ml::ported::types::SerializedModel>,
    presence: Option<&slouch_ml::ported::types::SerializedModel>,
) -> Result<(), ApiError> {
    let posture_wire = posture.map(storage_model_to_wire).transpose()?;
    let presence_wire = presence.map(storage_model_to_wire).transpose()?;
    state
        .inference
        .publish_model_pair(posture_wire, presence_wire)
}

fn storage_model_to_wire(
    model: &slouch_ml::ported::types::SerializedModel,
) -> Result<slouch_domain::ported::messages::schemas::SerializedModel, ApiError> {
    serde_json::from_value(
        serde_json::to_value(model).map_err(|error| ApiError::Internal(error.to_string()))?,
    )
    .map_err(|error| ApiError::Internal(error.to_string()))
}

fn model_metadata(
    model: &slouch_ml::ported::types::SerializedModel,
) -> Result<ModelMetadata, ApiError> {
    ensure_js_safe_timestamp(model.trained_at, "model training timestamp")?;
    Ok(ModelMetadata {
        classifier_id: model.classifier.classifier_id.clone(),
        trained_at: model.trained_at,
        feature_types: model.feature_extractor.feature_types.clone(),
    })
}

fn validate_training_settings_present(state: &AppState) -> Result<(), ApiError> {
    let settings = state
        .storage
        .get_training_settings()
        .map_err(map_storage_read)?
        .ok_or_else(|| ApiError::InvalidRequest("training settings have not been saved".into()))?;
    validate_training_settings(&settings)
}

fn encode_response<T: Serialize>(value: &T) -> Result<Response, ApiError> {
    rmp_serde::to_vec_named(value)
        .map(Response::new)
        .map_err(|error| ApiError::Internal(format!("binary response encoding failed: {error}")))
}

fn parse_raw_image(request: &Request<'_>) -> Result<ImageData, ApiError> {
    require_ipc_version(request)?;
    let bytes = raw_body(request, "infer_frame requires a raw RGBA body")?;
    ipc_validation::parse_raw_image_from(request.headers(), bytes)
}

fn raw_body<'a>(request: &'a Request<'_>, message: &str) -> Result<&'a [u8], ApiError> {
    match request.body() {
        InvokeBody::Raw(bytes) => Ok(bytes),
        InvokeBody::Json(_) => Err(ApiError::Ipc(message.into())),
    }
}

fn require_ipc_version(request: &Request<'_>) -> Result<(), ApiError> {
    ipc_validation::require_ipc_version_header(request.headers())
}

fn lock_mutation_lifecycle(state: &AppState) -> Result<MutexGuard<'_, ()>, ApiError> {
    state
        .lifecycle
        .lock()
        .map_err(|_| ApiError::Internal("lifecycle lock is poisoned".into()))
}

fn ensure_quiescent_mutation(
    training_running: bool,
    importing: bool,
    operation: &str,
) -> Result<(), ApiError> {
    if training_running {
        return Err(ApiError::Busy(format!(
            "{operation} cannot run while training is running"
        )));
    }
    if importing {
        return Err(ApiError::Busy(format!(
            "{operation} cannot run during archive import"
        )));
    }
    Ok(())
}

fn ensure_settings_mutation_allowed(state: &AppState) -> Result<(), ApiError> {
    ensure_quiescent_mutation(
        state.training_running.load(Ordering::Acquire),
        state.importing.load(Ordering::Acquire),
        "training settings mutation",
    )
}

fn ensure_destructive_mutation_allowed(state: &AppState) -> Result<(), ApiError> {
    ensure_quiescent_mutation(
        state.training_running.load(Ordering::Acquire),
        state.importing.load(Ordering::Acquire),
        "dataset reset",
    )
}

fn ensure_dataset_mutation_allowed(state: &AppState) -> Result<(), ApiError> {
    if state.importing.load(Ordering::Acquire) {
        return Err(ApiError::Busy(
            "dataset mutation cannot run during archive import".into(),
        ));
    }
    Ok(())
}

fn push_undo(state: &AppState, action: UndoAction) {
    match state.undo_history.lock() {
        Ok(mut history) => {
            history.push_back(action);
            while history.len() > 100 {
                history.pop_front();
            }
        }
        Err(_) => log::error!(target: "storage", "undo history lock is poisoned"),
    }
}

fn clear_undo_history(state: &AppState) {
    match state.undo_history.lock() {
        Ok(mut history) => history.clear(),
        Err(_) => log::error!(target: "storage", "undo history lock is poisoned"),
    }
}

fn undo_status(state: &AppState) -> Result<UndoStatus, ApiError> {
    let history = state
        .undo_history
        .lock()
        .map_err(|_| ApiError::Internal("undo history lock is poisoned".into()))?;
    let next_action = history.back().map(|action| match action {
        UndoAction::DeleteFrame(_) => UndoActionKind::RemoveCapture,
        UndoAction::RestoreFrame(_) => UndoActionKind::RestoreFrame,
    });
    Ok(UndoStatus {
        available: !history.is_empty(),
        depth: history.len(),
        next_action,
        revision: state.undo_revision.load(Ordering::Acquire),
    })
}

fn publish_undo_status(app: &AppHandle, state: &AppState) {
    state.undo_revision.fetch_add(1, Ordering::AcqRel);
    match undo_status(state) {
        Ok(status) => {
            if let Err(error) = app.emit("undo-status-changed", UndoStatusChangedEvent { status }) {
                log::error!(target: "storage", "undo status changed but event delivery failed: {error}");
            }
        }
        Err(error) => {
            log::error!(target: "storage", "undo status changed but reload failed: {error}")
        }
    }
}

fn native_state_snapshot(state: &AppState) -> Result<NativeStateSnapshot, ApiError> {
    Ok(NativeStateSnapshot {
        app: app_status_state(state)?,
        camera_settings: state
            .storage
            .get_camera_settings()
            .map_err(map_storage_read)?,
        ui_settings: state.storage.get_ui_settings().map_err(map_storage_read)?,
        training_settings: state
            .storage
            .get_training_settings()
            .map_err(map_storage_read)?,
        active_models: active_model_metadata_state(state)?,
        undo: undo_status(state)?,
    })
}

fn publish_native_state(app: &AppHandle, reason: &str, state: NativeStateSnapshot) {
    if let Err(error) = app.emit(
        "native-state-changed",
        NativeStateChangedEvent {
            reason: reason.to_owned(),
            state,
        },
    ) {
        log::error!(target: "storage", "native state changed but event delivery failed: {error}");
    }
}

fn parse_header(request: &Request<'_>, name: &str) -> Result<u64, ApiError> {
    ipc_validation::parse_header_value(request.headers(), name)
}

fn header_string(request: &Request<'_>, name: &str) -> Result<String, ApiError> {
    ipc_validation::header_string_value(request.headers(), name)
}

fn map_storage_read(error: StorageError) -> ApiError {
    ApiError::Storage(error.to_string())
}

fn map_storage_write(error: StorageError) -> ApiError {
    match error {
        StorageError::Validation(message) => ApiError::InvalidRequest(message),
        StorageError::DatasetChanged { expected, actual } => ApiError::DatasetChanged(format!(
            "dataset changed from version {expected} to {actual}"
        )),
        StorageError::SnapshotChanged(message) => ApiError::DatasetChanged(message),
        StorageError::InvalidData(message) if message.contains("not found") => {
            ApiError::NotFound(message)
        }
        other => ApiError::Storage(other.to_string()),
    }
}

fn map_archive_import(error: StorageError) -> ApiError {
    match error {
        StorageError::Validation(message) | StorageError::InvalidData(message) => {
            ApiError::InvalidRequest(message)
        }
        StorageError::Sqlite(error) => {
            ApiError::InvalidRequest(format!("invalid archive database: {error}"))
        }
        StorageError::Json(error) => {
            ApiError::InvalidRequest(format!("invalid archive JSON: {error}"))
        }
        StorageError::DatasetChanged { expected, actual } => ApiError::DatasetChanged(format!(
            "dataset changed from version {expected} to {actual}"
        )),
        StorageError::SnapshotChanged(message) => ApiError::DatasetChanged(message),
        StorageError::LockPoisoned => {
            ApiError::Storage("storage connection lock is poisoned".into())
        }
        StorageError::Clock => ApiError::Storage("system clock is before the Unix epoch".into()),
        StorageError::Io(message) => ApiError::Storage(format!("I/O error: {message}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_js_safe_timestamp, ensure_js_safe_u64, ensure_quiescent_mutation,
        persist_then_publish, validate_image_layout, validate_training_settings, DeploymentFailure,
        NativePreprocessor, TrainingSettings, UndoActionKind, UndoStatus, MAX_SAFE_JS_INTEGER,
    };
    use slouch_domain::ported::messages::schemas::ImageData;
    use slouch_domain::{
        CameraSettings, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
        DimensionalityReductionMethod, FeatureId, UiSettings,
    };
    use std::{cell::Cell, collections::BTreeMap};

    fn settings() -> TrainingSettings {
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
            cv_folds: 5,
            last_updated: 1.0,
        }
    }

    #[test]
    fn training_settings_reject_invalid_folds_and_noncanonical_features() {
        let mut invalid = settings();
        invalid.cv_folds = 101;
        assert!(validate_training_settings(&invalid).is_err());
        invalid = settings();
        invalid.posture_feature_types.push(FeatureId::GauFeatures);
        assert!(validate_training_settings(&invalid).is_err());
        invalid = settings();
        invalid.posture_feature_types = vec![FeatureId::GauFeatures, FeatureId::BackboneFeatures];
        assert!(validate_training_settings(&invalid).is_err());
    }

    #[test]
    fn training_settings_accept_valid_native_contract() {
        assert!(validate_training_settings(&settings()).is_ok());
    }

    #[test]
    fn camera_and_ui_settings_enforce_native_validation() {
        assert!(CameraSettings::default().validate().is_ok());
        assert!(UiSettings::default().validate().is_ok());

        let camera = CameraSettings {
            gaussian_blur_kernel: 2,
            ..CameraSettings::default()
        };
        assert!(camera.validate().is_err());

        let ui = UiSettings {
            alert_volume: f64::NAN,
            ..UiSettings::default()
        };
        assert!(ui.validate().is_err());
    }

    #[test]
    fn deployment_store_failure_never_calls_inference_publisher() {
        let publisher_calls = Cell::new(0);
        let result = persist_then_publish(
            || {
                Err(crate::errors::ApiError::Storage(
                    "injected write failure".into(),
                ))
            },
            || {
                publisher_calls.set(publisher_calls.get() + 1);
                Ok(())
            },
        );
        assert!(matches!(result, Err(DeploymentFailure::Persistence(_))));
        assert_eq!(publisher_calls.get(), 0);
    }

    #[test]
    fn raw_image_layout_rejects_stride_length_and_size_mismatches() {
        assert!(validate_image_layout(2, 2, 7, 16).is_err());
        assert!(validate_image_layout(2, 2, 8, 15).is_err());
        assert!(validate_image_layout(1920, 1080, 7680, 8_294_401).is_err());
        assert!(validate_image_layout(2, 2, 8, 16).is_ok());
    }

    #[test]
    fn json_integer_contract_rejects_values_above_number_max_safe_integer() {
        assert!(ensure_js_safe_u64(MAX_SAFE_JS_INTEGER, "version").is_ok());
        assert!(ensure_js_safe_u64(MAX_SAFE_JS_INTEGER + 1, "version").is_err());
        assert!(ensure_js_safe_timestamp(MAX_SAFE_JS_INTEGER as f64, "trainedAt").is_ok());
        assert!(ensure_js_safe_timestamp((MAX_SAFE_JS_INTEGER + 1) as f64, "trainedAt").is_err());
        assert!(ensure_js_safe_timestamp(1.5, "trainedAt").is_err());
    }

    #[test]
    fn native_preprocessor_consumes_smoothing_blur_and_contrast_settings() {
        let frame = |values: [u8; 4]| ImageData {
            data: values
                .into_iter()
                .flat_map(|value| [value, value, value, 255])
                .collect(),
            width: 2,
            height: 2,
        };
        let mut settings = CameraSettings {
            smoothing_frames: 2,
            gaussian_blur_kernel: 0,
            clahe_strength: 0.0,
            ..CameraSettings::default()
        };
        let mut preprocessor = NativePreprocessor::default();
        let _ = preprocessor
            .process(frame([0, 0, 0, 0]), &settings)
            .unwrap();
        let smoothed = preprocessor
            .process(frame([100, 100, 100, 100]), &settings)
            .unwrap();
        assert_eq!(smoothed.data[0], 50);

        settings.smoothing_frames = 1;
        settings.gaussian_blur_kernel = 3;
        let blurred = preprocessor
            .process(frame([0, 255, 0, 255]), &settings)
            .unwrap();
        assert_ne!(blurred.data, frame([0, 255, 0, 255]).data);

        settings.gaussian_blur_kernel = 0;
        settings.clahe_strength = 3.5;
        let contrasted = preprocessor
            .process(frame([32, 64, 128, 192]), &settings)
            .unwrap();
        assert_ne!(contrasted.data, frame([32, 64, 128, 192]).data);
    }

    #[test]
    fn undo_status_is_an_authoritative_typed_serialization_contract() {
        let status = UndoStatus {
            available: true,
            depth: 2,
            next_action: Some(UndoActionKind::RestoreFrame),
            revision: 7,
        };
        assert_eq!(
            serde_json::to_value(status).unwrap(),
            serde_json::json!({
                "available": true,
                "depth": 2,
                "nextAction": "restoreFrame",
                "revision": 7,
            })
        );
    }

    #[test]
    fn reset_and_settings_mutations_require_training_and_import_quiescence() {
        assert!(ensure_quiescent_mutation(false, false, "mutation").is_ok());
        assert!(ensure_quiescent_mutation(true, false, "mutation").is_err());
        assert!(ensure_quiescent_mutation(false, true, "mutation").is_err());
    }

    #[test]
    fn storage_model_to_wire_accepts_integral_random_projection_seed() {
        use slouch_ml::ported::random_projection::RandomProjectionState;
        use slouch_ml::ported::types::{
            DimReductionTransformer, DimensionalityReductionConfig, DimensionalityReductionMethod,
            NormalizationMode, SerializedClassifier, SerializedClassifierState,
            SerializedFeatureExtractor, SerializedModel, SerializedSvm,
        };

        // Reproduces the crash-loop bridge: a stored model whose random-projection
        // seed reconstructs as `42.0` (f64) must cross into the wire schema, whose
        // seed field is a `u64`. Before the fix this returned an Internal error
        // ("invalid type: floating point `42.0`, expected u64").
        let model = SerializedModel {
            feature_extractor: SerializedFeatureExtractor {
                feature_types: vec!["gau_features".into()],
                normalization_mode: NormalizationMode::None,
                dim_reduction_config: DimensionalityReductionConfig {
                    method: DimensionalityReductionMethod::RandomProjection,
                    components: 2,
                },
                concatenated_dimensions: 3,
                normalization_mean: None,
                normalization_std: None,
                dim_reduction_transformer: Some(DimReductionTransformer::RandomProjection(
                    RandomProjectionState {
                        projection_matrix: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
                        n_components: 2,
                        n_features: 3,
                        seed: 42.0,
                    },
                )),
            },
            classifier: SerializedClassifier {
                classifier_id: "svm".into(),
                state: SerializedClassifierState::Svm(SerializedSvm {
                    weights: vec![0.1, 0.2],
                    bias: 0.0,
                    class_weights: [1.0, 1.0],
                }),
            },
            trained_at: 1_700_000_000_000.0,
            version: 1.0,
        };

        let wire = super::storage_model_to_wire(&model)
            .expect("bridge must accept an integral random-projection seed");
        match wire.feature_extractor.dim_reduction_transformer {
            Some(slouch_domain::ported::messages::schemas::DimensionalityReductionTransformer::RandomProjection(state)) => {
                assert_eq!(state.seed, 42);
            }
            other => panic!("expected a random-projection transformer, got {other:?}"),
        }
    }
}
