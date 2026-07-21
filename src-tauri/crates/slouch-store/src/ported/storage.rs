//! SQLite-backed dataset, model, and training-settings storage.
//!
//! This is the native counterpart of `src/services/dataset/storage.ts`.
//! IndexedDB keys become normalized SQLite rows: frames own keypoints,
//! features, and thumbnails through foreign keys, while model payloads and
//! settings remain opaque validated records.  The connection is protected by
//! one mutex so all writes are serialized by the store boundary.

use std::{
    borrow::Borrow,
    collections::BTreeMap,
    convert::TryFrom,
    fmt,
    path::Path,
    str::FromStr,
    sync::{Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use slouch_domain::{
    validate_posture_frame, BoundingBox, CameraSettings, DatasetStats,
    DimensionalityReductionConfig, DimensionalityReductionMethod, FeatureId, FrameLabel,
    PostureDataset, PostureFrame, Thumbnail, TrainingSettings, UiSettings,
};
use slouch_ml::ported::types::SerializedModel;

use super::{
    feature_reservoir::{ReservoirMeta, ReservoirSample},
    model_format::{self, ModelRole as ContainerModelRole},
};

const MAX_THUMBNAIL_BYTES: usize = 2 * 1024 * 1024;
const MAX_DATASET_PAGE_SIZE: usize = 100;
const MAX_SAFE_JS_INTEGER: u64 = 9_007_199_254_740_991;
const SCHEMA: &str = include_str!("../../../../schema/live-v1.sql");

const POSTURE_MODEL_ROLE: &str = "posture";
const PRESENCE_MODEL_ROLE: &str = "presence";

/// The roles of an active model generation. Each role is independently optional:
/// a posture-only generation populates only `.0`, a presence-only generation only
/// `.1`, and a full pair populates both.
pub type ActiveModelPair = (Option<SerializedModel>, Option<SerializedModel>);
const TRAINING_SETTINGS_KEY: &str = "training:settings";
const CAMERA_SETTINGS_KEY: &str = "camera:settings";
const UI_SETTINGS_KEY: &str = "ui:settings";
const PCA_CONFIG_KEY: &str = "pca-config";

#[derive(Debug)]
pub enum StorageError {
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
    Validation(String),
    InvalidData(String),
    DatasetChanged { expected: u64, actual: u64 },
    SnapshotChanged(String),
    LockPoisoned,
    Clock,
    Io(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(error) => write!(formatter, "SQLite error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Validation(message) => write!(formatter, "validation failed: {message}"),
            Self::InvalidData(message) => write!(formatter, "invalid stored data: {message}"),
            Self::DatasetChanged { expected, actual } => write!(
                formatter,
                "dataset changed from version {expected} to {actual}"
            ),
            Self::SnapshotChanged(message) => formatter.write_str(message),
            Self::LockPoisoned => formatter.write_str("storage connection lock is poisoned"),
            Self::Clock => formatter.write_str("system clock is before the Unix epoch"),
            Self::Io(message) => write!(formatter, "I/O error: {message}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<rusqlite::Error> for StorageError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Posture,
    Presence,
}

impl ModelRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Posture => POSTURE_MODEL_ROLE,
            Self::Presence => PRESENCE_MODEL_ROLE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageInfo {
    pub used: u64,
    pub available: u64,
    pub quota: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetMetadata {
    pub version: u64,
    pub last_modified: f64,
    pub frame_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameMetadata {
    pub id: String,
    pub timestamp: f64,
    pub keypoints: Vec<slouch_domain::Keypoint>,
    pub bbox: BoundingBox,
    pub label: FrameLabel,
    pub thumbnail_mime_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetMetadataPage {
    pub frames: Vec<FrameMetadata>,
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
    pub version: u64,
    pub last_modified: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingSnapshot {
    pub dataset_version: u64,
    pub dataset_identity_sha256: [u8; 32],
    pub training_config_sha256: [u8; 32],
}

const RESERVOIR_CAPACITY: usize = 1_000;
const RESERVOIR_SAMPLE_INTERVAL_MS: i64 = 1_000;
const RESERVOIR_INITIAL_RNG_STATE: u64 = 0x4d59_5df4_d0f3_3173;

/// The native replacement for the browser `DatasetStorage` singleton.
///
/// Construct one instance per application database.  `open_in_memory` is
/// useful for deterministic tests and command-contract checks.
pub struct DatasetStorage {
    connection: Mutex<Connection>,
}

impl fmt::Debug for DatasetStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DatasetStorage")
            .field("connection", &"SQLite")
            .finish_non_exhaustive()
    }
}

impl DatasetStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let connection = Connection::open(path)?;
        Self::from_connection(connection)
    }

    pub fn open_in_memory() -> Result<Self, StorageError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    pub fn new_in_memory() -> Result<Self, StorageError> {
        Self::open_in_memory()
    }

    fn from_connection(mut connection: Connection) -> Result<Self, StorageError> {
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        verify_connection_policy(&connection)?;
        let application_id =
            connection.query_row("PRAGMA application_id", [], |row| row.get::<_, i64>(0))?;
        let version =
            connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
        if application_id != 1_397_506_888 && !(application_id == 0 && version == 0) {
            return Err(StorageError::InvalidData(
                "database application_id does not belong to Slouch Tracker".into(),
            ));
        }
        if version > 2 {
            return Err(StorageError::InvalidData(format!(
                "database schema version {version} is newer than this application"
            )));
        }
        if version == 0 {
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            transaction.execute_batch(SCHEMA)?;
            transaction.execute_batch("PRAGMA user_version = 2;")?;
            transaction.commit()?;
        }
        ensure_bbox_score_column(&connection)?;
        migrate_reservoir_format_v2(&mut connection, version)?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, StorageError> {
        self.connection
            .lock()
            .map_err(|_| StorageError::LockPoisoned)
    }

    /// Saves one complete frame and increments the dataset version atomically.
    pub fn save_frame<F>(&self, frame: F) -> Result<(), StorageError>
    where
        F: Borrow<PostureFrame>,
    {
        let frame = frame.borrow();
        validate_frame(frame)?;
        let timestamp = timestamp_ms(frame.timestamp)?;
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        insert_frame(&transaction, frame, timestamp)?;
        increment_version(&transaction)?;
        transaction.commit()?;
        Ok(())
    }

    /// Loads all valid frames in deterministic captured-time/id order.
    pub fn load_dataset(&self) -> Result<PostureDataset, StorageError> {
        let connection = self.lock()?;
        load_dataset_from_connection(&connection)
    }

    /// Returns only bounded scalar metadata. Feature and thumbnail BLOBs are
    /// intentionally absent from this query path.
    pub fn get_dataset_metadata(&self) -> Result<DatasetMetadata, StorageError> {
        let connection = self.lock()?;
        dataset_metadata(&connection)
    }

    /// Returns at most 100 frame metadata rows and their 17 keypoints without
    /// selecting feature or thumbnail payload columns.
    pub fn get_frame_metadata_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetMetadataPage, StorageError> {
        if limit == 0 || limit > MAX_DATASET_PAGE_SIZE {
            return Err(StorageError::Validation(
                "dataset page limit must be between 1 and 100".into(),
            ));
        }
        let connection = self.lock()?;
        let metadata = dataset_metadata(&connection)?;
        let sql_offset = i64::try_from(offset)
            .map_err(|_| StorageError::Validation("dataset page offset is too large".into()))?;
        let sql_limit = i64::try_from(limit)
            .map_err(|_| StorageError::Validation("dataset page limit is too large".into()))?;
        let mut statement = connection.prepare(
            "SELECT f.id, f.captured_at_ms, f.label,
                    f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2, f.bbox_score, t.mime_type
             FROM frames f JOIN thumbnails t ON t.frame_id = f.id
             ORDER BY f.captured_at_ms ASC, f.id ASC LIMIT ? OFFSET ?",
        )?;
        let rows = statement.query_map(params![sql_limit, sql_offset], |row| {
            let x1 = row.get::<_, f64>(3)?;
            let y1 = row.get::<_, f64>(4)?;
            let x2 = row.get::<_, f64>(5)?;
            let y2 = row.get::<_, f64>(6)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                BoundingBox {
                    x1,
                    y1,
                    x2,
                    y2,
                    score: row.get::<_, f64>(7)?,
                    width: x2 - x1,
                    height: y2 - y1,
                },
                row.get::<_, String>(8)?,
            ))
        })?;
        let mut frames = Vec::new();
        for row in rows {
            let (id, timestamp, label, bbox, thumbnail_mime_type) = row?;
            if !(1..=MAX_SAFE_JS_INTEGER as i64).contains(&timestamp) {
                return Err(StorageError::InvalidData(format!(
                    "frame {id} timestamp is outside JavaScript's safe integer range"
                )));
            }
            let keypoints = load_keypoints(&connection, &id)?;
            frames.push(FrameMetadata {
                id,
                timestamp: timestamp as f64,
                keypoints,
                bbox,
                label: parse_label(&label)?,
                thumbnail_mime_type,
            });
        }
        Ok(DatasetMetadataPage {
            frames,
            offset,
            limit,
            total: metadata.frame_count,
            version: metadata.version,
            last_modified: metadata.last_modified,
        })
    }

    pub fn update_frame_label(&self, id: &str, label: FrameLabel) -> Result<(), StorageError> {
        if label == FrameLabel::Unused {
            // Mirrors the TS oracle: updateFrameLabel(id, UNUSED) -> deleteFrame, which
            // removeItem's the key (a no-op for an absent id) then increments the version.
            // Relabeling a missing frame to unused therefore SUCCEEDS and still bumps the
            // version, rather than erroring like the public delete_frame does.
            let mut connection = self.lock()?;
            let transaction = connection.transaction()?;
            transaction.execute("DELETE FROM frames WHERE id = ?", [id])?;
            increment_version(&transaction)?;
            transaction.commit()?;
            return Ok(());
        }
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        let changed = transaction.execute(
            "UPDATE frames SET label = ?, captured_at_ms = ? WHERE id = ?",
            params![label_string(label), now_ms()?, id],
        )?;
        if changed == 0 {
            return Err(StorageError::InvalidData(format!(
                "frame with id {id} not found"
            )));
        }
        increment_version(&transaction)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn delete_frame(&self, id: &str) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        let changed = transaction.execute("DELETE FROM frames WHERE id = ?", [id])?;
        if changed == 0 {
            return Err(StorageError::InvalidData(format!(
                "frame with id {id} not found"
            )));
        }
        increment_version(&transaction)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn remove_frames_by_label(&self, label: FrameLabel) -> Result<usize, StorageError> {
        if label != FrameLabel::Unused {
            return Err(StorageError::InvalidData(
                "remove_frames_by_label only accepts the unused label".into(),
            ));
        }
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        let removed =
            transaction.execute("DELETE FROM frames WHERE label = ?", [label_string(label)])?;
        if removed > 0 {
            increment_version_by(&transaction, removed)?;
        }
        transaction.commit()?;
        Ok(removed)
    }

    pub fn clear_dataset(&self) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute("DELETE FROM frames", [])?;
        transaction.execute("DELETE FROM reservoir_samples", [])?;
        transaction.execute("DELETE FROM reservoir_state", [])?;
        transaction.execute("UPDATE model_generations SET active = 0", [])?;
        transaction.execute(
            "UPDATE app_meta SET dataset_version = 0, last_modified_ms = ? WHERE singleton = 1",
            [now_ms()?],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn reset_all(&self) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute("DELETE FROM frames", [])?;
        transaction.execute("DELETE FROM models", [])?;
        transaction.execute("DELETE FROM model_generations", [])?;
        transaction.execute("DELETE FROM reservoir_samples", [])?;
        transaction.execute("DELETE FROM reservoir_state", [])?;
        transaction.execute("DELETE FROM settings", [])?;
        transaction.execute(
            "UPDATE app_meta SET dataset_version = 0, last_modified_ms = ? WHERE singleton = 1",
            [now_ms()?],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn export_archive(
        &self,
        destination: impl AsRef<Path>,
        exporter_version: &str,
    ) -> Result<super::archive::ArchiveSummary, StorageError> {
        let connection = self.lock()?;
        super::archive::export_connection(&connection, destination.as_ref(), exporter_version)
    }

    pub fn import_archive(
        &self,
        source: impl AsRef<Path>,
    ) -> Result<super::archive::ArchiveSummary, StorageError> {
        let mut connection = self.lock()?;
        super::archive::import_connection(&mut connection, source.as_ref())
    }

    pub fn get_all_frames(&self) -> Result<Vec<PostureFrame>, StorageError> {
        Ok(self.load_dataset()?.frames)
    }

    pub fn get_frames_by_label(
        &self,
        label: FrameLabel,
    ) -> Result<Vec<PostureFrame>, StorageError> {
        Ok(self
            .get_all_frames()?
            .into_iter()
            .filter(|frame| frame.label == label)
            .collect())
    }

    pub fn get_frame_by_id(&self, id: &str) -> Result<Option<PostureFrame>, StorageError> {
        let connection = self.lock()?;
        let row = connection
            .query_row(
                "SELECT id, captured_at_ms, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_score
                 FROM frames WHERE id = ?",
                [id],
                |row| {
                    Ok(FrameRow {
                        id: row.get(0)?,
                        timestamp: row.get::<_, i64>(1)?,
                        label: row.get(2)?,
                        bbox: BoundingBox {
                            x1: row.get(3)?,
                            y1: row.get(4)?,
                            x2: row.get(5)?,
                            y2: row.get(6)?,
                            score: row.get(7)?,
                            width: row.get::<_, f64>(5)? - row.get::<_, f64>(3)?,
                            height: row.get::<_, f64>(6)? - row.get::<_, f64>(4)?,
                        },
                    })
                },
            )
            .optional()?;
        row.map(|row| load_frame(&connection, row)).transpose()
    }

    pub fn get_thumbnail(&self, id: &str) -> Result<Option<(String, Vec<u8>)>, StorageError> {
        let connection = self.lock()?;
        connection
            .query_row(
                "SELECT mime_type, bytes FROM thumbnails WHERE frame_id = ?",
                [id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_stats(&self) -> Result<DatasetStats, StorageError> {
        let connection = self.lock()?;
        let (total, good, bad, away, unused) = connection.query_row(
            "SELECT COUNT(*),
                    COALESCE(SUM(label = 'good'), 0),
                    COALESCE(SUM(label = 'bad'), 0),
                    COALESCE(SUM(label = 'away'), 0),
                    COALESCE(SUM(label = 'unused'), 0)
             FROM frames",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )?;
        let total = js_safe_count(total, "dataset frame count")?;
        let good = js_safe_count(good, "good frame count")?;
        let bad = js_safe_count(bad, "bad frame count")?;
        let away = js_safe_count(away, "away frame count")?;
        let unused = js_safe_count(unused, "unused frame count")?;
        let classified = good + bad;
        Ok(DatasetStats {
            total,
            good,
            bad,
            away,
            unused,
            imbalance_ratio: if classified == 0 {
                0.0
            } else {
                good.abs_diff(bad) as f64 / classified as f64
            },
            has_minimum_frames: good > 0 && bad > 0,
            has_away_frames: away > 0,
        })
    }

    pub fn load_posture_model(&self) -> Result<Option<SerializedModel>, StorageError> {
        self.load_model(ModelRole::Posture)
    }

    pub fn clear_posture_model(&self) -> Result<(), StorageError> {
        self.clear_model(ModelRole::Posture)
    }

    pub fn load_presence_model(&self) -> Result<Option<SerializedModel>, StorageError> {
        self.load_model(ModelRole::Presence)
    }

    pub fn clear_presence_model(&self) -> Result<(), StorageError> {
        self.clear_model(ModelRole::Presence)
    }

    pub fn training_snapshot(&self, do_cv: bool) -> Result<TrainingSnapshot, StorageError> {
        let connection = self.lock()?;
        training_snapshot_from_connection(&connection, do_cv)
    }

    /// Persists a new active model generation containing the trained roles. Each
    /// role is optional: a posture-only run passes `presence = None`, matching the
    /// oracle where a run with no AWAY frames trains posture alone. At least one
    /// role must be present. The generation is written atomically and only the
    /// provided roles receive a `models` row.
    pub fn save_model_pair_if_snapshot(
        &self,
        posture: Option<&SerializedModel>,
        presence: Option<&SerializedModel>,
        expected: TrainingSnapshot,
        do_cv: bool,
        posture_metrics: Option<&slouch_domain::TrainingMetrics>,
        presence_metrics: Option<&slouch_domain::TrainingMetrics>,
    ) -> Result<(), StorageError> {
        if posture.is_none() && presence.is_none() {
            return Err(StorageError::Validation(
                "a model generation must contain at least one trained role".into(),
            ));
        }
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let actual = training_snapshot_from_connection(&transaction, do_cv)?;
        if expected.dataset_version != actual.dataset_version {
            return Err(StorageError::DatasetChanged {
                expected: expected.dataset_version,
                actual: actual.dataset_version,
            });
        }
        if expected.dataset_identity_sha256 != actual.dataset_identity_sha256 {
            return Err(StorageError::SnapshotChanged(
                "dataset identity changed while training; models were not activated".into(),
            ));
        }
        if expected.training_config_sha256 != actual.training_config_sha256 {
            return Err(StorageError::SnapshotChanged(
                "training settings changed while training; models were not activated".into(),
            ));
        }
        if do_cv
            && ((posture.is_some() && posture_metrics.is_none())
                || (presence.is_some() && presence_metrics.is_none()))
        {
            return Err(StorageError::Validation(
                "cross-validation metrics are required for each trained model role".into(),
            ));
        }
        transaction.execute(
            "INSERT INTO model_generations(created_at_ms, dataset_version, dataset_identity_sha256, training_config_sha256, active)
             VALUES (?, ?, ?, ?, 0)",
            params![
                now_ms()?,
                i64::try_from(actual.dataset_version).map_err(|_| StorageError::InvalidData("dataset version exceeds SQLite limits".into()))?,
                actual.dataset_identity_sha256.to_vec(),
                actual.training_config_sha256.to_vec(),
            ],
        )?;
        let generation = transaction.last_insert_rowid();
        if let Some(posture) = posture {
            let posture_payload = model_format::encode_model(
                posture,
                ContainerModelRole::Posture,
                actual.dataset_version,
                actual.training_config_sha256,
                posture_metrics,
            )
            .map_err(StorageError::Validation)?;
            transaction.execute(
                "INSERT INTO models(generation_id, role, envelope_version, payload, payload_sha256)
                 VALUES (?, ?, 1, ?, ?)",
                params![
                    generation,
                    POSTURE_MODEL_ROLE,
                    posture_payload,
                    model_format::sha256(&posture_payload).to_vec()
                ],
            )?;
        }
        if let Some(presence) = presence {
            let presence_payload = model_format::encode_model(
                presence,
                ContainerModelRole::Presence,
                actual.dataset_version,
                actual.training_config_sha256,
                presence_metrics,
            )
            .map_err(StorageError::Validation)?;
            transaction.execute(
                "INSERT INTO models(generation_id, role, envelope_version, payload, payload_sha256)
                 VALUES (?, ?, 1, ?, ?)",
                params![
                    generation,
                    PRESENCE_MODEL_ROLE,
                    presence_payload,
                    model_format::sha256(&presence_payload).to_vec()
                ],
            )?;
        }
        transaction.execute("UPDATE model_generations SET active = 0", [])?;
        transaction.execute(
            "UPDATE model_generations SET active = 1 WHERE id = ?",
            [generation],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn active_model_generation_id(&self) -> Result<Option<i64>, StorageError> {
        self.lock()?
            .query_row(
                "SELECT id FROM model_generations WHERE active = 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn restore_active_model_generation(&self, generation_id: i64) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = transaction
            .query_row(
                "SELECT dataset_version, dataset_identity_sha256,
                    (SELECT COUNT(*) FROM models WHERE generation_id = model_generations.id),
                    training_config_sha256
             FROM model_generations WHERE id = ?",
                [generation_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| StorageError::InvalidData("model generation was not found".into()))?;
        let current_version = dataset_meta(&transaction)?.version;
        let distinct_roles: i64 = transaction.query_row(
            "SELECT COUNT(DISTINCT role) FROM models WHERE generation_id = ? AND role IN ('posture', 'presence')",
            [generation_id],
            |row| row.get(0),
        )?;
        let settings_json = transaction
            .query_row(
                "SELECT json FROM settings WHERE key = ? AND schema_version = 1",
                [TRAINING_SETTINGS_KEY],
                |settings_row| settings_row.get::<_, String>(0),
            )
            .optional()?;
        let config_matches = settings_json
            .map(|json| -> Result<bool, StorageError> {
                let settings: TrainingSettings = serde_json::from_str(&json)?;
                validate_training_settings_value(&settings)?;
                let without_cv = model_format::training_config_fingerprint(&settings, false)
                    .map_err(StorageError::Validation)?;
                let with_cv = model_format::training_config_fingerprint(&settings, true)
                    .map_err(StorageError::Validation)?;
                Ok(row.3.as_slice() == without_cv.as_slice()
                    || row.3.as_slice() == with_cv.as_slice())
            })
            .transpose()?
            .unwrap_or(false);
        // A generation is valid with one or two roles (posture-only,
        // presence-only, or both). `row.2 == distinct_roles` rejects rows with
        // unexpected/duplicate roles, and `1..=2` bounds the role count.
        if row.0 != current_version as i64
            || row.1.as_slice() != dataset_identity(&transaction)?.as_slice()
            || row.2 != distinct_roles
            || !(1..=2).contains(&distinct_roles)
            || !config_matches
        {
            return Err(StorageError::InvalidData(
                "model generation is stale or incomplete".into(),
            ));
        }
        transaction.execute("UPDATE model_generations SET active = 0", [])?;
        transaction.execute(
            "UPDATE model_generations SET active = 1 WHERE id = ?",
            [generation_id],
        )?;
        transaction.commit()?;
        Ok(())
    }

    /// Loads the active generation's roles. Returns `None` when there is no active
    /// generation, otherwise `Some((posture, presence))` where each role is
    /// independently optional. A single-role generation (posture-only or
    /// presence-only) is valid and yields exactly one populated role.
    pub fn load_active_model_pair(&self) -> Result<Option<ActiveModelPair>, StorageError> {
        let posture = self.load_model(ModelRole::Posture)?;
        let presence = self.load_model(ModelRole::Presence)?;
        if posture.is_none() && presence.is_none() {
            return Ok(None);
        }
        Ok(Some((posture, presence)))
    }

    pub fn sample_reservoir(
        &self,
        sample: &ReservoirSample,
        observed_at_ms: i64,
    ) -> Result<bool, StorageError> {
        validate_reservoir_sample(sample)?;
        if observed_at_ms <= 0 {
            return Err(StorageError::Validation(
                "reservoir sample time must be positive".into(),
            ));
        }
        let payload = rmp_serde::to_vec_named(sample).map_err(|error| {
            StorageError::InvalidData(format!("reservoir encoding failed: {error}"))
        })?;
        if payload.is_empty() || payload.len() > 8 * 1024 * 1024 {
            return Err(StorageError::Validation(
                "reservoir sample exceeds the 8 MiB limit".into(),
            ));
        }
        let mut connection = self.lock()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let state = transaction
            .query_row(
                "SELECT capacity, seen_count, sample_count, rng_state, last_sampled_ms FROM reservoir_state WHERE singleton = 1",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, i64>(4)?)),
            )
            .optional()?;
        let (capacity, seen, count, rng, last_sampled) = state.unwrap_or((
            RESERVOIR_CAPACITY as i64,
            0,
            0,
            (RESERVOIR_INITIAL_RNG_STATE & i64::MAX as u64) as i64,
            0,
        ));
        if observed_at_ms.saturating_sub(last_sampled) < RESERVOIR_SAMPLE_INTERVAL_MS {
            return Ok(false);
        }
        let next_seen = seen
            .checked_add(1)
            .ok_or_else(|| StorageError::InvalidData("reservoir seen count overflow".into()))?;
        let next_rng = next_reservoir_rng(rng as u64) & i64::MAX as u64;
        let slot = if count < capacity {
            Some(count)
        } else {
            let candidate = (next_rng
                % u64::try_from(next_seen)
                    .map_err(|_| StorageError::InvalidData("reservoir count is invalid".into()))?)
                as i64;
            (candidate < capacity).then_some(candidate)
        };
        if let Some(slot) = slot {
            transaction.execute(
                "INSERT INTO reservoir_samples(slot, payload) VALUES (?, ?)
                 ON CONFLICT(slot) DO UPDATE SET payload = excluded.payload",
                params![slot, payload],
            )?;
        }
        let next_count = count.saturating_add(1).min(capacity);
        transaction.execute(
            "INSERT INTO reservoir_state(singleton, capacity, seen_count, sample_count, rng_state, last_sampled_ms)
             VALUES (1, ?, ?, ?, ?, ?)
             ON CONFLICT(singleton) DO UPDATE SET seen_count = excluded.seen_count,
               sample_count = excluded.sample_count, rng_state = excluded.rng_state,
               last_sampled_ms = excluded.last_sampled_ms",
            params![capacity, next_seen, next_count, next_rng as i64, observed_at_ms],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn load_reservoir_samples(&self) -> Result<Vec<ReservoirSample>, StorageError> {
        let connection = self.lock()?;
        load_reservoir_samples_from_connection(&connection)
    }

    pub fn reservoir_meta(&self) -> Result<ReservoirMeta, StorageError> {
        let connection = self.lock()?;
        let state = connection.query_row(
            "SELECT capacity, seen_count, sample_count FROM reservoir_state WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?)),
        ).optional()?;
        let (capacity, seen, count) = state.unwrap_or((RESERVOIR_CAPACITY as i64, 0, 0));
        Ok(ReservoirMeta {
            total_seen: seen as usize,
            count: count as usize,
            max_samples: capacity as usize,
        })
    }

    pub fn save_dim_reduction_config(
        &self,
        config: &DimensionalityReductionConfig,
    ) -> Result<(), StorageError> {
        self.save_json_setting(PCA_CONFIG_KEY, config)
    }

    pub fn load_dim_reduction_config(&self) -> Result<DimensionalityReductionConfig, StorageError> {
        Ok(self
            .load_json_setting(PCA_CONFIG_KEY)?
            .unwrap_or(DimensionalityReductionConfig {
                method: DimensionalityReductionMethod::None,
                components: 64,
            }))
    }

    pub fn save_training_settings<S>(&self, settings: S) -> Result<(), StorageError>
    where
        S: Borrow<TrainingSettings>,
    {
        validate_training_settings_value(settings.borrow())?;
        let json = serde_json::to_string(settings.borrow())?;
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO settings(key, schema_version, json) VALUES (?, 1, ?)
             ON CONFLICT(key) DO UPDATE SET schema_version = 1, json = excluded.json",
            params![TRAINING_SETTINGS_KEY, json],
        )?;
        transaction.execute("UPDATE model_generations SET active = 0", [])?;
        transaction.commit()?;
        Ok(())
    }

    pub fn get_training_settings(&self) -> Result<Option<TrainingSettings>, StorageError> {
        self.load_json_setting(TRAINING_SETTINGS_KEY)
    }

    pub fn clear_training_settings(&self) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "DELETE FROM settings WHERE key = ?",
            [TRAINING_SETTINGS_KEY],
        )?;
        transaction.execute("UPDATE model_generations SET active = 0", [])?;
        transaction.commit()?;
        Ok(())
    }

    pub fn save_camera_settings(&self, settings: &CameraSettings) -> Result<(), StorageError> {
        settings.validate().map_err(StorageError::Validation)?;
        self.save_json_setting(CAMERA_SETTINGS_KEY, settings)
    }

    pub fn get_camera_settings(&self) -> Result<CameraSettings, StorageError> {
        let settings: CameraSettings = self
            .load_json_setting(CAMERA_SETTINGS_KEY)?
            .unwrap_or_default();
        settings.validate().map_err(|message| {
            StorageError::InvalidData(format!("stored camera settings are invalid: {message}"))
        })?;
        Ok(settings)
    }

    pub fn reset_camera_settings(&self) -> Result<CameraSettings, StorageError> {
        self.delete_setting(CAMERA_SETTINGS_KEY)?;
        Ok(CameraSettings::default())
    }

    pub fn save_ui_settings(&self, settings: &UiSettings) -> Result<(), StorageError> {
        settings.validate().map_err(StorageError::Validation)?;
        self.save_json_setting(UI_SETTINGS_KEY, settings)
    }

    pub fn get_ui_settings(&self) -> Result<UiSettings, StorageError> {
        let settings: UiSettings = self.load_json_setting(UI_SETTINGS_KEY)?.unwrap_or_default();
        settings.validate().map_err(|message| {
            StorageError::InvalidData(format!("stored UI settings are invalid: {message}"))
        })?;
        Ok(settings)
    }

    pub fn reset_ui_settings(&self) -> Result<UiSettings, StorageError> {
        self.delete_setting(UI_SETTINGS_KEY)?;
        Ok(UiSettings::default())
    }

    /// Native storage has no browser quota API. File size is the useful bounded
    /// statistic; available/quota remain zero until the app supplies a policy.
    pub fn get_storage_info(&self) -> Result<StorageInfo, StorageError> {
        let connection = self.lock()?;
        let page_count = connection
            .query_row("PRAGMA page_count", [], |row| row.get::<_, i64>(0))?
            .max(0) as u64;
        let page_size = connection
            .query_row("PRAGMA page_size", [], |row| row.get::<_, i64>(0))?
            .max(0) as u64;
        let used = page_count.saturating_mul(page_size);
        Ok(StorageInfo {
            used,
            available: 0,
            quota: 0,
        })
    }

    pub fn needs_retraining(&self) -> Result<bool, StorageError> {
        Ok(self.load_posture_model()?.is_none())
    }

    fn load_model(&self, role: ModelRole) -> Result<Option<SerializedModel>, StorageError> {
        let connection = self.lock()?;
        let row = connection
            .query_row(
                "SELECT m.payload, m.payload_sha256, g.dataset_version,
                    g.dataset_identity_sha256, g.training_config_sha256
             FROM models m JOIN model_generations g ON g.id = m.generation_id
             WHERE m.role = ? AND g.active = 1 ORDER BY g.id DESC LIMIT 1",
                [role.as_str()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((payload, expected_digest, dataset_version, dataset_identity, config_digest)) =
            row
        else {
            return Ok(None);
        };
        if expected_digest != model_format::sha256(&payload) {
            return Err(StorageError::InvalidData(format!(
                "{role:?} model checksum does not match"
            )));
        }
        let (model, envelope) = model_format::decode_model(&payload).map_err(|error| {
            StorageError::InvalidData(format!("model decoding failed: {error}"))
        })?;
        let expected_role = match role {
            ModelRole::Posture => ContainerModelRole::Posture,
            ModelRole::Presence => ContainerModelRole::Presence,
        };
        if envelope.role != expected_role
            || envelope.dataset_version
                != u64::try_from(dataset_version).map_err(|_| {
                    StorageError::InvalidData("model dataset version is negative".into())
                })?
            || envelope.training_config_sha256.as_slice() != config_digest.as_slice()
        {
            return Err(StorageError::InvalidData(
                "model envelope does not match its generation".into(),
            ));
        }
        if dataset_identity.len() != 32 {
            return Err(StorageError::InvalidData(
                "model generation dataset identity is invalid".into(),
            ));
        }
        Ok(Some(model))
    }

    fn clear_model(&self, role: ModelRole) -> Result<(), StorageError> {
        let mut connection = self.lock()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "UPDATE model_generations SET active = 0
             WHERE id IN (SELECT generation_id FROM models WHERE role = ?)",
            [role.as_str()],
        )?;
        transaction.execute("DELETE FROM models WHERE role = ?", [role.as_str()])?;
        transaction.execute(
            "DELETE FROM model_generations
             WHERE id NOT IN (SELECT DISTINCT generation_id FROM models)",
            [],
        )?;
        transaction.commit()?;
        Ok(())
    }

    fn save_json_setting<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError> {
        let json = serde_json::to_string(value)?;
        self.lock()?.execute(
            "INSERT INTO settings(key, schema_version, json) VALUES (?, 1, ?)
             ON CONFLICT(key) DO UPDATE SET schema_version = 1, json = excluded.json",
            params![key, json],
        )?;
        Ok(())
    }

    fn load_json_setting<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, StorageError> {
        let connection = self.lock()?;
        let row = connection
            .query_row(
                "SELECT schema_version, json FROM settings WHERE key = ?",
                [key],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let Some((schema_version, json)) = row else {
            return Ok(None);
        };
        if schema_version != 1 {
            return Err(StorageError::InvalidData(format!(
                "stored setting {key} uses unsupported schema version {schema_version}",
            )));
        }
        serde_json::from_str(&json).map(Some).map_err(Into::into)
    }

    fn delete_setting(&self, key: &str) -> Result<(), StorageError> {
        self.lock()?
            .execute("DELETE FROM settings WHERE key = ?", [key])?;
        Ok(())
    }
}

#[derive(Debug)]
struct FrameRow {
    id: String,
    timestamp: i64,
    label: String,
    bbox: BoundingBox,
}

pub(crate) fn insert_frame(
    transaction: &Transaction<'_>,
    frame: &PostureFrame,
    timestamp: i64,
) -> Result<(), StorageError> {
    transaction.execute(
        "INSERT INTO frames
         (id, captured_at_ms, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_score)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           captured_at_ms = excluded.captured_at_ms,
           label = excluded.label,
           bbox_x1 = excluded.bbox_x1,
           bbox_y1 = excluded.bbox_y1,
           bbox_x2 = excluded.bbox_x2,
           bbox_y2 = excluded.bbox_y2,
           bbox_score = excluded.bbox_score",
        params![
            frame.id,
            timestamp,
            label_string(frame.label),
            frame.bbox.x1,
            frame.bbox.y1,
            frame.bbox.x2,
            frame.bbox.y2,
            frame.bbox.score,
        ],
    )?;
    transaction.execute(
        "DELETE FROM frame_keypoints WHERE frame_id = ?",
        [&frame.id],
    )?;
    for (index, keypoint) in frame.keypoints.iter().enumerate() {
        transaction.execute(
            "INSERT INTO frame_keypoints(frame_id, keypoint_index, x, y, score)
             VALUES (?, ?, ?, ?, ?)",
            params![
                frame.id,
                index as i64,
                keypoint.x,
                keypoint.y,
                keypoint.score
            ],
        )?;
    }
    transaction.execute("DELETE FROM frame_features WHERE frame_id = ?", [&frame.id])?;
    for (feature_id, values) in &frame.features {
        let encoded = encode_f32(values)?;
        transaction.execute(
            "INSERT INTO frame_features(frame_id, feature_type, dimension, values_le_f32)
             VALUES (?, ?, ?, ?)",
            params![frame.id, feature_id.as_str(), values.len() as i64, encoded],
        )?;
    }
    transaction.execute("DELETE FROM thumbnails WHERE frame_id = ?", [&frame.id])?;
    transaction.execute(
        "INSERT INTO thumbnails(frame_id, mime_type, bytes) VALUES (?, ?, ?)",
        params![frame.id, frame.thumbnail.mime_type, frame.thumbnail.bytes],
    )?;
    Ok(())
}

fn load_keypoints(
    connection: &Connection,
    frame_id: &str,
) -> Result<Vec<slouch_domain::Keypoint>, StorageError> {
    let mut keypoints = Vec::with_capacity(17);
    let mut statement = connection.prepare(
        "SELECT x, y, score FROM frame_keypoints
         WHERE frame_id = ? ORDER BY keypoint_index ASC",
    )?;
    for value in statement.query_map([frame_id], |item| {
        Ok(slouch_domain::Keypoint {
            x: item.get(0)?,
            y: item.get(1)?,
            score: item.get(2)?,
        })
    })? {
        keypoints.push(value?);
    }
    if keypoints.len() != 17 {
        return Err(StorageError::InvalidData(format!(
            "frame {frame_id} does not have exactly 17 keypoints"
        )));
    }
    Ok(keypoints)
}

fn load_frame(connection: &Connection, row: FrameRow) -> Result<PostureFrame, StorageError> {
    let keypoints = load_keypoints(connection, &row.id)?;

    let mut features = BTreeMap::new();
    let mut statement = connection.prepare(
        "SELECT feature_type, dimension, values_le_f32 FROM frame_features
         WHERE frame_id = ? ORDER BY feature_type ASC",
    )?;
    for value in statement.query_map([&row.id], |item| {
        Ok((
            item.get::<_, String>(0)?,
            item.get::<_, i64>(1)? as usize,
            item.get::<_, Vec<u8>>(2)?,
        ))
    })? {
        let (name, dimension, bytes) = value?;
        let id = FeatureId::from_str(&name).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
        if dimension != id.metadata().dimensions {
            return Err(StorageError::InvalidData(format!(
                "frame {} feature {} has dimension {}, expected {}",
                row.id,
                id.as_str(),
                dimension,
                id.metadata().dimensions,
            )));
        }
        let values = decode_f32(&bytes, dimension).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Blob,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
            )
        })?;
        features.insert(id, values);
    }

    let (mime_type, bytes) = connection.query_row(
        "SELECT mime_type, bytes FROM thumbnails WHERE frame_id = ?",
        [&row.id],
        |item| Ok((item.get::<_, String>(0)?, item.get::<_, Vec<u8>>(1)?)),
    )?;
    let timestamp = checked_stored_timestamp(row.timestamp, &row.id)?;
    Ok(PostureFrame {
        id: row.id,
        timestamp,
        features,
        thumbnail: Thumbnail { mime_type, bytes },
        keypoints,
        bbox: row.bbox,
        label: parse_label(&row.label)?,
    })
}

pub(crate) fn validate_frame(frame: &PostureFrame) -> Result<(), StorageError> {
    if frame.thumbnail.bytes.is_empty() || frame.thumbnail.bytes.len() > MAX_THUMBNAIL_BYTES {
        return Err(StorageError::Validation(
            "thumbnail must contain between 1 byte and 2 MiB".into(),
        ));
    }
    if !matches!(
        frame.thumbnail.mime_type.as_str(),
        "image/jpeg" | "image/png" | "image/webp"
    ) {
        return Err(StorageError::Validation(
            "unsupported thumbnail MIME type".into(),
        ));
    }
    validate_posture_frame(frame).map_err(|error| StorageError::Validation(error.to_string()))
}

pub(crate) fn load_dataset_from_connection(
    connection: &Connection,
) -> Result<PostureDataset, StorageError> {
    let meta = dataset_meta(connection)?;
    let mut statement = connection.prepare(
        "SELECT id, captured_at_ms, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_score
         FROM frames ORDER BY captured_at_ms ASC, id ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(FrameRow {
            id: row.get(0)?,
            timestamp: row.get::<_, i64>(1)?,
            label: row.get(2)?,
            bbox: BoundingBox {
                x1: row.get(3)?,
                y1: row.get(4)?,
                x2: row.get(5)?,
                y2: row.get(6)?,
                score: row.get(7)?,
                width: row.get::<_, f64>(5)? - row.get::<_, f64>(3)?,
                height: row.get::<_, f64>(6)? - row.get::<_, f64>(4)?,
            },
        })
    })?;
    let mut frames = Vec::new();
    for row in rows {
        frames.push(load_frame(connection, row?)?);
    }
    Ok(PostureDataset {
        frames,
        version: meta.version,
        last_modified: meta.last_modified,
    })
}

fn dataset_meta(connection: &Connection) -> Result<DatasetMeta, StorageError> {
    let (version, last_modified) = connection.query_row(
        "SELECT dataset_version, last_modified_ms FROM app_meta WHERE singleton = 1",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    let version = u64::try_from(version)
        .map_err(|_| StorageError::InvalidData("dataset version is negative".into()))?;
    if version > MAX_SAFE_JS_INTEGER {
        return Err(StorageError::InvalidData(
            "dataset version exceeds JavaScript's safe integer range".into(),
        ));
    }
    if !(1..=MAX_SAFE_JS_INTEGER as i64).contains(&last_modified) {
        return Err(StorageError::InvalidData(
            "dataset last-modified timestamp is outside JavaScript's safe integer range".into(),
        ));
    }
    Ok(DatasetMeta {
        version,
        last_modified: last_modified as f64,
    })
}

fn dataset_metadata(connection: &Connection) -> Result<DatasetMetadata, StorageError> {
    let meta = dataset_meta(connection)?;
    let frame_count = connection.query_row("SELECT COUNT(*) FROM frames", [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(DatasetMetadata {
        version: meta.version,
        last_modified: meta.last_modified,
        frame_count: js_safe_count(frame_count, "dataset frame count")?,
    })
}

fn js_safe_count(value: i64, name: &str) -> Result<usize, StorageError> {
    let value = u64::try_from(value)
        .map_err(|_| StorageError::InvalidData(format!("{name} is negative")))?;
    if value > MAX_SAFE_JS_INTEGER {
        return Err(StorageError::InvalidData(format!(
            "{name} exceeds JavaScript's safe integer range"
        )));
    }
    usize::try_from(value)
        .map_err(|_| StorageError::InvalidData(format!("{name} exceeds platform limits")))
}

#[derive(Debug, Clone, Copy)]
struct DatasetMeta {
    version: u64,
    last_modified: f64,
}

fn increment_version(transaction: &Transaction<'_>) -> Result<(), StorageError> {
    increment_version_by(transaction, 1)
}

fn increment_version_by(transaction: &Transaction<'_>, amount: usize) -> Result<(), StorageError> {
    let amount = i64::try_from(amount)
        .map_err(|_| StorageError::Validation("dataset version increment is too large".into()))?;
    transaction.execute("UPDATE model_generations SET active = 0", [])?;
    let changed = transaction.execute(
        "UPDATE app_meta
         SET dataset_version = dataset_version + ?, last_modified_ms = ?
         WHERE singleton = 1 AND dataset_version <= ? - ?",
        params![amount, now_ms()?, MAX_SAFE_JS_INTEGER as i64, amount],
    )?;
    if changed != 1 {
        return Err(StorageError::Validation(
            "dataset version reached JavaScript's safe integer limit".into(),
        ));
    }
    Ok(())
}

fn parse_label(value: &str) -> Result<FrameLabel, StorageError> {
    match value {
        "good" => Ok(FrameLabel::Good),
        "bad" => Ok(FrameLabel::Bad),
        "away" => Ok(FrameLabel::Away),
        "unused" => Ok(FrameLabel::Unused),
        other => Err(StorageError::InvalidData(format!(
            "unknown frame label {other}"
        ))),
    }
}

fn label_string(label: FrameLabel) -> &'static str {
    match label {
        FrameLabel::Good => "good",
        FrameLabel::Bad => "bad",
        FrameLabel::Away => "away",
        FrameLabel::Unused => "unused",
    }
}

fn ensure_bbox_score_column(connection: &Connection) -> Result<(), StorageError> {
    let has_score = connection
        .prepare("PRAGMA table_info(frames)")?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|name| name == "bbox_score");
    if !has_score {
        connection.execute(
            "ALTER TABLE frames ADD COLUMN bbox_score REAL NOT NULL DEFAULT 1.0
             CHECK (bbox_score = bbox_score AND bbox_score BETWEEN 0 AND 1)",
            [],
        )?;
    }
    Ok(())
}

/// The NLF-EMBED format break. Reservoir payloads written under schema version 1
/// use the retired fixed-field `ReservoirSample` shape and cannot be decoded into
/// the generic `{features, keypoints, bbox}` sample, so a one-time open of a v1
/// database drops every reservoir row and stamps `user_version = 2`. Only the
/// reservoir is affected: frames, keypoints, features, thumbnails, settings, and
/// models are untouched. Idempotent for versions 0 (fresh, already stamped 2) and
/// 2 (already migrated).
fn migrate_reservoir_format_v2(
    connection: &mut Connection,
    version: i64,
) -> Result<(), StorageError> {
    if version != 1 {
        return Ok(());
    }
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("DELETE FROM reservoir_samples", [])?;
    transaction.execute("DELETE FROM reservoir_state", [])?;
    transaction.execute_batch("PRAGMA user_version = 2;")?;
    transaction.commit()?;
    Ok(())
}

fn verify_connection_policy(connection: &Connection) -> Result<(), StorageError> {
    let foreign_keys =
        connection.query_row("PRAGMA foreign_keys", [], |row| row.get::<_, i64>(0))?;
    let journal_mode =
        connection.query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))?;
    let busy_timeout =
        connection.query_row("PRAGMA busy_timeout", [], |row| row.get::<_, i64>(0))?;
    let synchronous = connection.query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))?;
    let main_path = connection.query_row(
        "SELECT file FROM pragma_database_list WHERE name = 'main'",
        [],
        |row| row.get::<_, String>(0),
    )?;
    let journal_matches = if main_path.is_empty() {
        journal_mode.eq_ignore_ascii_case("memory")
    } else {
        journal_mode.eq_ignore_ascii_case("wal")
    };
    if foreign_keys != 1 || !journal_matches || busy_timeout != 5_000 || synchronous != 1 {
        return Err(StorageError::InvalidData(format!(
            "SQLite connection policy mismatch: foreign_keys={foreign_keys}, journal_mode={journal_mode}, busy_timeout={busy_timeout}, synchronous={synchronous}"
        )));
    }
    Ok(())
}

fn now_ms() -> Result<i64, StorageError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| StorageError::Clock)
        .and_then(|duration| i64::try_from(duration.as_millis()).map_err(|_| StorageError::Clock))
}

fn checked_stored_timestamp(timestamp: i64, frame_id: &str) -> Result<f64, StorageError> {
    if !(1..=MAX_SAFE_JS_INTEGER as i64).contains(&timestamp) {
        return Err(StorageError::InvalidData(format!(
            "frame {frame_id} timestamp is outside JavaScript's safe integer range"
        )));
    }
    Ok(timestamp as f64)
}

fn timestamp_ms(timestamp: f64) -> Result<i64, StorageError> {
    if !timestamp.is_finite()
        || timestamp <= 0.0
        || timestamp.fract() != 0.0
        || timestamp > MAX_SAFE_JS_INTEGER as f64
    {
        return Err(StorageError::Validation(
            "frame timestamp must be a positive JavaScript-safe integer millisecond value".into(),
        ));
    }
    Ok(timestamp as i64)
}

fn encode_f32(values: &[f32]) -> Result<Vec<u8>, StorageError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(StorageError::Validation(
            "feature vectors must be non-empty and finite".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_f32(bytes: &[u8], dimension: usize) -> Result<Vec<f32>, String> {
    if dimension == 0 || bytes.len() != dimension.saturating_mul(4) {
        return Err("feature BLOB length does not match its dimension".into());
    }
    let mut values = Vec::with_capacity(dimension);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err("feature BLOB contains a non-finite value".into());
        }
        values.push(value);
    }
    Ok(values)
}

fn training_snapshot_from_connection(
    connection: &Connection,
    do_cv: bool,
) -> Result<TrainingSnapshot, StorageError> {
    let dataset_version = dataset_meta(connection)?.version;
    let settings_json = connection
        .query_row(
            "SELECT json FROM settings WHERE key = ? AND schema_version = 1",
            [TRAINING_SETTINGS_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .ok_or_else(|| StorageError::Validation("training settings have not been saved".into()))?;
    let settings: TrainingSettings = serde_json::from_str(&settings_json)?;
    validate_training_settings_value(&settings)?;
    let training_config_sha256 = model_format::training_config_fingerprint(&settings, do_cv)
        .map_err(StorageError::Validation)?;
    Ok(TrainingSnapshot {
        dataset_version,
        dataset_identity_sha256: dataset_identity(connection)?,
        training_config_sha256,
    })
}

pub(crate) fn dataset_identity(connection: &Connection) -> Result<[u8; 32], StorageError> {
    let dataset = load_dataset_from_connection(connection)?;
    let mut hasher = Sha256::new();
    hasher.update(b"SLDS\x01");
    hasher.update(rmp_serde::to_vec_named(&dataset).map_err(|error| {
        StorageError::InvalidData(format!("dataset identity encoding failed: {error}"))
    })?);
    Ok(hasher.finalize().into())
}

pub(crate) fn validate_training_settings_value(
    settings: &TrainingSettings,
) -> Result<(), StorageError> {
    // Only an upper sanity bound is enforced, matching the training worker: the
    // evaluation layer skips CV for cv_folds <= 1, so 0/1 are valid "no CV" values.
    if settings.cv_folds > 100 {
        return Err(StorageError::Validation(
            "cvFolds must not exceed 100".into(),
        ));
    }
    if !settings.last_updated.is_finite() || settings.last_updated <= 0.0 {
        return Err(StorageError::Validation(
            "lastUpdated must be a positive finite number".into(),
        ));
    }
    if settings.posture_feature_types.is_empty() || settings.presence_feature_types.is_empty() {
        return Err(StorageError::Validation(
            "posture and presence feature selections must not be empty".into(),
        ));
    }
    for features in [
        &settings.posture_feature_types,
        &settings.presence_feature_types,
    ] {
        if features.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(StorageError::Validation(
                "training feature selections must be unique and in registry order".into(),
            ));
        }
    }
    if settings.dim_reduction_config.components == 0
        || settings.dim_reduction_config.components > 1_048_576
    {
        return Err(StorageError::Validation(
            "dimensionality-reduction components are outside native limits".into(),
        ));
    }
    slouch_ml::ported::classifier_registry::create_classifier(&settings.classifier_config)
        .map(|_| ())
        .map_err(|error| {
            StorageError::Validation(format!("invalid classifier configuration: {error}"))
        })
}

pub(crate) fn validate_reservoir_sample(sample: &ReservoirSample) -> Result<(), StorageError> {
    if sample.features.is_empty() {
        return Err(StorageError::Validation(
            "reservoir sample must contain at least one stored feature".into(),
        ));
    }
    for (id, values) in &sample.features {
        let metadata = id.metadata();
        if metadata.computed {
            return Err(StorageError::Validation(format!(
                "reservoir feature {} is computed and cannot be stored",
                id.as_str()
            )));
        }
        if values.len() != metadata.dimensions || values.iter().any(|value| !value.is_finite()) {
            return Err(StorageError::Validation(format!(
                "reservoir {} feature is invalid",
                id.as_str()
            )));
        }
    }
    if sample.keypoints.len() != 17
        || sample
            .keypoints
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.score.is_finite())
    {
        return Err(StorageError::Validation(
            "reservoir keypoints are invalid".into(),
        ));
    }
    slouch_domain::validate_bbox(&sample.bbox).map_err(|error| {
        StorageError::Validation(format!("reservoir bounding box is invalid: {error}"))
    })?;
    Ok(())
}

pub(crate) fn load_reservoir_samples_from_connection(
    connection: &Connection,
) -> Result<Vec<ReservoirSample>, StorageError> {
    let state_count = connection
        .query_row(
            "SELECT sample_count FROM reservoir_state WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .unwrap_or(0);
    let mut statement =
        connection.prepare("SELECT slot, payload FROM reservoir_samples ORDER BY slot")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    let mut samples = Vec::new();
    for (expected_slot, row) in rows.enumerate() {
        let (slot, payload) = row?;
        if slot != expected_slot as i64 {
            return Err(StorageError::InvalidData(
                "reservoir slots are not contiguous".into(),
            ));
        }
        let sample: ReservoirSample = rmp_serde::from_slice(&payload).map_err(|error| {
            StorageError::InvalidData(format!("reservoir sample decoding failed: {error}"))
        })?;
        validate_reservoir_sample(&sample)
            .map_err(|error| StorageError::InvalidData(error.to_string()))?;
        samples.push(sample);
    }
    if samples.len() != state_count as usize {
        return Err(StorageError::InvalidData(
            "reservoir sample count does not match state".into(),
        ));
    }
    Ok(samples)
}

fn next_reservoir_rng(mut state: u64) -> u64 {
    if state == 0 {
        state = RESERVOIR_INITIAL_RNG_STATE;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

impl super::import::DatasetStore for DatasetStorage {
    fn load_dataset(&self) -> Result<PostureDataset, String> {
        DatasetStorage::load_dataset(self).map_err(|error| error.to_string())
    }

    fn save_frame(&self, frame: PostureFrame) -> Result<(), String> {
        DatasetStorage::save_frame(self, &frame).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use rusqlite::params;
    use slouch_domain::{BoundingBox, FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail};

    use super::{decode_f32, encode_f32, DatasetStorage, MAX_SAFE_JS_INTEGER};

    fn frame(id: &str, timestamp: f64, label: FrameLabel) -> PostureFrame {
        PostureFrame {
            id: id.into(),
            timestamp,
            features: BTreeMap::from([(
                FeatureId::GauFeatures,
                vec![0.25; FeatureId::GauFeatures.metadata().dimensions],
            )]),
            thumbnail: Thumbnail {
                mime_type: "image/webp".into(),
                bytes: vec![7; 2 * 1024 * 1024],
            },
            keypoints: (0..17)
                .map(|index| Keypoint::new(index as f64, index as f64, 0.9))
                .collect(),
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 10.0,
                y2: 20.0,
                score: 0.9,
                width: 10.0,
                height: 20.0,
            },
            label,
        }
    }

    #[test]
    fn file_backed_connections_apply_the_frozen_sqlite_policy() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("slouch-policy-{nonce}.sqlite"));
        let storage = DatasetStorage::open(&path).expect("open storage");
        let connection = storage.connection.lock().expect("lock");

        assert_eq!(
            connection
                .query_row("PRAGMA foreign_keys", [], |row| row.get::<_, i64>(0))
                .expect("foreign keys"),
            1
        );
        assert_eq!(
            connection
                .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
                .expect("journal mode")
                .to_ascii_lowercase(),
            "wal"
        );
        assert_eq!(
            connection
                .query_row("PRAGMA busy_timeout", [], |row| row.get::<_, i64>(0))
                .expect("busy timeout"),
            5_000
        );
        assert_eq!(
            connection
                .query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))
                .expect("synchronous"),
            1
        );
        drop(connection);
        drop(storage);
        fs::remove_file(path).expect("remove database");
    }

    #[test]
    fn feature_vectors_round_trip_as_little_endian_f32() {
        let source = [0.0_f32, -1.5, 3.25];
        let encoded = encode_f32(&source).expect("finite values encode");
        assert_eq!(
            decode_f32(&encoded, source.len()).expect("valid BLOB"),
            source
        );
    }

    #[test]
    fn malformed_feature_lengths_are_rejected() {
        assert!(decode_f32(&[0, 0, 0], 1).is_err());
        assert!(encode_f32(&[f32::NAN]).is_err());
    }

    #[test]
    fn metadata_page_status_and_stats_never_decode_feature_or_thumbnail_payloads() {
        let storage = DatasetStorage::open_in_memory().expect("open storage");
        storage
            .save_frame(frame("first", 1_700_000_000_000.0, FrameLabel::Good))
            .expect("save first frame");
        storage
            .save_frame(frame("second", 1_700_000_000_001.0, FrameLabel::Bad))
            .expect("save second frame");

        let nan_blob = vec![0_u8; FeatureId::GauFeatures.metadata().dimensions * 4];
        let mut nan_blob = nan_blob;
        nan_blob[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        storage
            .connection
            .lock()
            .expect("connection")
            .execute(
                "UPDATE frame_features SET values_le_f32 = ? WHERE frame_id = 'second'",
                [nan_blob],
            )
            .expect("corrupt out-of-page feature payload");

        let page = storage
            .get_frame_metadata_page(0, 1)
            .expect("bounded metadata page");
        assert_eq!(page.frames.len(), 1);
        assert_eq!(page.frames[0].id, "first");
        assert_eq!(page.total, 2);
        assert_eq!(
            storage
                .get_dataset_metadata()
                .expect("metadata")
                .frame_count,
            2
        );
        assert_eq!(storage.get_stats().expect("stats").total, 2);
        assert!(storage.load_dataset().is_err());
    }

    #[test]
    fn unsafe_dataset_versions_are_rejected_and_cannot_increment() {
        let storage = DatasetStorage::open_in_memory().expect("open storage");
        let connection = storage.connection.lock().expect("connection");
        connection
            .execute(
                "UPDATE app_meta SET dataset_version = ? WHERE singleton = 1",
                params![MAX_SAFE_JS_INTEGER as i64],
            )
            .expect("set maximum safe version");
        drop(connection);
        assert_eq!(
            storage
                .get_dataset_metadata()
                .expect("safe maximum")
                .version,
            MAX_SAFE_JS_INTEGER
        );
        assert!(storage
            .save_frame(frame("overflow", 1_700_000_000_002.0, FrameLabel::Good))
            .is_err());

        storage
            .connection
            .lock()
            .expect("connection")
            .execute(
                "UPDATE app_meta SET dataset_version = ? WHERE singleton = 1",
                params![MAX_SAFE_JS_INTEGER as i64 + 1],
            )
            .expect("set unsafe version");
        assert!(storage.get_dataset_metadata().is_err());
    }
}
