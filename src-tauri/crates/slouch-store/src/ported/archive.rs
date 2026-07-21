use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};
use slouch_domain::{
    CameraSettings, DimensionalityReductionConfig, InferenceResult, PostureDataset, PostureFrame,
    TrainingSettings, UiSettings,
};
use slouch_ml::ported::model::Model;

use super::{
    model_format::{self, ModelRole},
    storage::{
        insert_frame, load_dataset_from_connection, load_reservoir_samples_from_connection,
        validate_training_settings_value, StorageError,
    },
};

const ARCHIVE_APPLICATION_ID: i64 = 1_397_510_219;
const ARCHIVE_VERSION: i64 = 1;
const MAX_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_ARCHIVE_FRAMES: i64 = 250_000;
const MAX_SAFE_JS_INTEGER: i64 = 9_007_199_254_740_991;
const ARCHIVE_SCHEMA: &str = include_str!("../../../../schema/archive-v1.sql");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchiveSummary {
    pub frame_count: usize,
    pub dataset_version: u64,
}

#[derive(Debug)]
struct SettingRow {
    key: String,
    schema_version: i64,
    json: String,
}

#[derive(Debug)]
struct GenerationRow {
    id: i64,
    created_at_ms: i64,
    dataset_version: i64,
    dataset_identity_sha256: Vec<u8>,
    training_config_sha256: Vec<u8>,
    active: i64,
}

#[derive(Debug)]
struct ModelRow {
    generation_id: i64,
    role: String,
    envelope_version: i64,
    payload: Vec<u8>,
    payload_sha256: Vec<u8>,
}

#[derive(Debug)]
struct ReservoirStateRow {
    singleton: i64,
    capacity: i64,
    seen_count: i64,
    sample_count: i64,
    rng_state: i64,
    last_sampled_ms: i64,
}

#[derive(Debug)]
struct ReservoirSampleRow {
    slot: i64,
    payload: Vec<u8>,
    payload_sha256: Option<Vec<u8>>,
}

pub(crate) fn export_connection(
    source: &Connection,
    destination: &Path,
    exporter_version: &str,
) -> Result<ArchiveSummary, StorageError> {
    if exporter_version.is_empty() || exporter_version.len() > 64 {
        return Err(StorageError::Validation(
            "archive exporter version must contain between 1 and 64 bytes".into(),
        ));
    }
    let dataset = load_dataset_from_connection(source)?;
    if dataset.frames.len() > MAX_ARCHIVE_FRAMES as usize {
        return Err(StorageError::Validation(
            "dataset exceeds the 250000-frame archive limit".into(),
        ));
    }

    let temporary = temporary_archive_path(destination)?;
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|error| StorageError::Io(error.to_string()))?;
    }

    let result = (|| {
        let mut archive = Connection::open(&temporary)?;
        archive.pragma_update(None, "foreign_keys", "ON")?;
        let transaction = archive.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(ARCHIVE_SCHEMA)?;
        transaction.execute(
            "INSERT INTO archive_meta(singleton, format_version, app_schema_version, created_at_ms, exporter_version, source_dataset_version)
             VALUES (1, 1, 1, ?, ?, ?)",
            params![now_ms()?, exporter_version, i64::try_from(dataset.version).map_err(|_| StorageError::InvalidData("dataset version exceeds SQLite limits".into()))?],
        )?;
        transaction.execute(
            "INSERT INTO app_meta(singleton, dataset_version, last_modified_ms) VALUES (1, ?, ?)",
            params![
                i64::try_from(dataset.version).map_err(|_| StorageError::InvalidData(
                    "dataset version exceeds SQLite limits".into()
                ))?,
                dataset.last_modified as i64
            ],
        )?;
        write_frames(&transaction, &dataset)?;
        copy_settings(source, &transaction)?;
        copy_models(source, &transaction)?;
        copy_reservoir(source, &transaction)?;
        transaction.commit()?;
        let quick_check: String = archive.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
        if quick_check != "ok" {
            return Err(StorageError::InvalidData(format!(
                "exported archive failed quick_check: {quick_check}"
            )));
        }
        drop(archive);
        let size = fs::metadata(&temporary)
            .map_err(|error| StorageError::Io(error.to_string()))?
            .len();
        if size > MAX_ARCHIVE_BYTES {
            return Err(StorageError::Validation(
                "exported archive exceeds the 2 GiB limit".into(),
            ));
        }
        replace_file(&temporary, destination)?;
        Ok(ArchiveSummary {
            frame_count: dataset.frames.len(),
            dataset_version: dataset.version,
        })
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn import_connection(
    live: &mut Connection,
    source_path: &Path,
) -> Result<ArchiveSummary, StorageError> {
    let metadata =
        fs::metadata(source_path).map_err(|error| StorageError::Io(error.to_string()))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_ARCHIVE_BYTES {
        return Err(StorageError::Validation(
            "archive must be a non-empty file no larger than 2 GiB".into(),
        ));
    }
    let archive = Connection::open_with_flags(
        source_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    validate_archive_database(&archive)?;
    let dataset = load_dataset_from_connection(&archive)?;
    for frame in &dataset.frames {
        super::storage::validate_frame(frame)?;
    }
    validate_payload_hashes(&archive)?;
    let settings = read_settings(&archive)?;
    let generations = read_generations(&archive)?;
    let models = read_models(&archive)?;
    let reservoir_states = read_reservoir_states(&archive)?;
    let reservoir_samples = read_reservoir_samples(&archive)?;
    validate_reservoir_rows(&reservoir_states, &reservoir_samples)?;
    let archive_identity = super::storage::dataset_identity(&archive)?;
    validate_model_rows(&generations, &models, dataset.version, archive_identity)?;

    let transaction = live.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("DELETE FROM models", [])?;
    transaction.execute("DELETE FROM model_generations", [])?;
    transaction.execute("DELETE FROM settings", [])?;
    transaction.execute("DELETE FROM reservoir_samples", [])?;
    transaction.execute("DELETE FROM reservoir_state", [])?;
    transaction.execute("DELETE FROM frames", [])?;
    for frame in &dataset.frames {
        insert_frame(&transaction, frame, frame.timestamp as i64)?;
    }
    for row in settings {
        transaction.execute(
            "INSERT INTO settings(key, schema_version, json) VALUES (?, ?, ?)",
            params![row.key, row.schema_version, row.json],
        )?;
    }
    for row in generations {
        transaction.execute(
            "INSERT INTO model_generations(id, created_at_ms, dataset_version, dataset_identity_sha256, training_config_sha256, active)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![row.id, row.created_at_ms, row.dataset_version, row.dataset_identity_sha256, row.training_config_sha256, row.active],
        )?;
    }
    for row in models {
        transaction.execute(
            "INSERT INTO models(generation_id, role, envelope_version, payload, payload_sha256)
             VALUES (?, ?, ?, ?, ?)",
            params![
                row.generation_id,
                row.role,
                row.envelope_version,
                row.payload,
                row.payload_sha256
            ],
        )?;
    }
    for row in reservoir_states {
        transaction.execute(
            "INSERT INTO reservoir_state(singleton, capacity, seen_count, sample_count, rng_state, last_sampled_ms)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![row.singleton, row.capacity, row.seen_count, row.sample_count, row.rng_state, row.last_sampled_ms],
        )?;
    }
    for row in reservoir_samples {
        transaction.execute(
            "INSERT INTO reservoir_samples(slot, payload) VALUES (?, ?)",
            params![row.slot, row.payload],
        )?;
    }
    transaction.execute(
        "UPDATE app_meta SET dataset_version = ?, last_modified_ms = ? WHERE singleton = 1",
        params![
            i64::try_from(dataset.version).map_err(|_| StorageError::InvalidData(
                "dataset version exceeds SQLite limits".into()
            ))?,
            dataset.last_modified as i64
        ],
    )?;
    transaction.commit()?;

    Ok(ArchiveSummary {
        frame_count: dataset.frames.len(),
        dataset_version: dataset.version,
    })
}

fn write_frames(
    transaction: &Transaction<'_>,
    dataset: &PostureDataset,
) -> Result<(), StorageError> {
    for frame in &dataset.frames {
        transaction.execute(
            "INSERT INTO frames(id, captured_at_ms, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_score)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                frame.id,
                frame.timestamp as i64,
                label(frame),
                frame.bbox.x1,
                frame.bbox.y1,
                frame.bbox.x2,
                frame.bbox.y2,
                frame.bbox.score
            ],
        )?;
        for (index, keypoint) in frame.keypoints.iter().enumerate() {
            transaction.execute(
                "INSERT INTO frame_keypoints(frame_id, keypoint_index, x, y, score) VALUES (?, ?, ?, ?, ?)",
                params![frame.id, index as i64, keypoint.x, keypoint.y, keypoint.score],
            )?;
        }
        for (feature, values) in &frame.features {
            let bytes = encode_f32(values)?;
            let digest = sha256(&bytes);
            transaction.execute(
                "INSERT INTO frame_features(frame_id, feature_type, dimension, values_le_f32, payload_sha256)
                 VALUES (?, ?, ?, ?, ?)",
                params![frame.id, feature.as_str(), values.len() as i64, bytes, digest.to_vec()],
            )?;
        }
        transaction.execute(
            "INSERT INTO thumbnails(frame_id, mime_type, bytes, payload_sha256) VALUES (?, ?, ?, ?)",
            params![frame.id, frame.thumbnail.mime_type, frame.thumbnail.bytes, sha256(&frame.thumbnail.bytes).to_vec()],
        )?;
    }
    Ok(())
}

fn copy_settings(source: &Connection, target: &Transaction<'_>) -> Result<(), StorageError> {
    for row in read_settings(source)? {
        target.execute(
            "INSERT INTO settings(key, schema_version, json) VALUES (?, ?, ?)",
            params![row.key, row.schema_version, row.json],
        )?;
    }
    Ok(())
}

fn copy_models(source: &Connection, target: &Transaction<'_>) -> Result<(), StorageError> {
    for row in read_generations(source)? {
        target.execute(
            "INSERT INTO model_generations(id, created_at_ms, dataset_version, dataset_identity_sha256, training_config_sha256, active)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![row.id, row.created_at_ms, row.dataset_version, row.dataset_identity_sha256, row.training_config_sha256, row.active],
        )?;
    }
    for row in read_models(source)? {
        if row.payload_sha256 != sha256(&row.payload) {
            return Err(StorageError::InvalidData(
                "stored model checksum does not match".into(),
            ));
        }
        target.execute(
            "INSERT INTO models(generation_id, role, envelope_version, payload, payload_sha256)
             VALUES (?, ?, ?, ?, ?)",
            params![
                row.generation_id,
                row.role,
                row.envelope_version,
                row.payload,
                row.payload_sha256
            ],
        )?;
    }
    Ok(())
}

fn copy_reservoir(source: &Connection, target: &Transaction<'_>) -> Result<(), StorageError> {
    load_reservoir_samples_from_connection(source)?;
    for row in read_reservoir_states(source)? {
        target.execute(
            "INSERT INTO reservoir_state(singleton, capacity, seen_count, sample_count, rng_state, last_sampled_ms)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![row.singleton, row.capacity, row.seen_count, row.sample_count, row.rng_state, row.last_sampled_ms],
        )?;
    }
    for row in read_reservoir_samples(source)? {
        let digest = sha256(&row.payload);
        target.execute(
            "INSERT INTO reservoir_samples(slot, payload, payload_sha256) VALUES (?, ?, ?)",
            params![row.slot, row.payload, digest.to_vec()],
        )?;
    }
    Ok(())
}

fn validate_archive_database(connection: &Connection) -> Result<(), StorageError> {
    let application_id: i64 =
        connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if application_id != ARCHIVE_APPLICATION_ID || version != ARCHIVE_VERSION {
        return Err(StorageError::Validation(
            "file is not a supported Slouch Tracker .slouchpack archive".into(),
        ));
    }
    let quick_check: String = connection.query_row("PRAGMA quick_check", [], |row| row.get(0))?;
    if quick_check != "ok" {
        return Err(StorageError::InvalidData(format!(
            "archive quick_check failed: {quick_check}"
        )));
    }
    let foreign_key_violation: Option<String> = connection
        .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
        .optional()?;
    if foreign_key_violation.is_some() {
        return Err(StorageError::InvalidData(
            "archive contains foreign-key violations".into(),
        ));
    }
    let expected = Connection::open_in_memory()?;
    expected.execute_batch(ARCHIVE_SCHEMA)?;
    if schema_signature(connection)? != schema_signature(&expected)? {
        return Err(StorageError::Validation(
            "archive schema does not match the frozen format".into(),
        ));
    }
    let forbidden: i64 = connection.query_row(
        "SELECT COUNT(*) FROM sqlite_schema WHERE type IN ('trigger', 'view') OR (type = 'table' AND name NOT IN
         ('archive_meta','app_meta','frames','frame_keypoints','frame_features','thumbnails','settings','model_generations','models','reservoir_state','reservoir_samples'))",
        [],
        |row| row.get(0),
    )?;
    if forbidden != 0 {
        return Err(StorageError::Validation(
            "archive schema contains unsupported objects".into(),
        ));
    }
    let required_tables: i64 = connection.query_row(
        "SELECT COUNT(*) FROM sqlite_schema WHERE type = 'table' AND name IN
         ('archive_meta','app_meta','frames','frame_keypoints','frame_features','thumbnails','settings','model_generations','models','reservoir_state','reservoir_samples')",
        [],
        |row| row.get(0),
    )?;
    if required_tables != 11 {
        return Err(StorageError::Validation(
            "archive schema is incomplete".into(),
        ));
    }
    let metadata_rows: i64 = connection.query_row(
        "SELECT COUNT(*) FROM archive_meta WHERE singleton = 1 AND format_version = 1 AND app_schema_version = 1",
        [],
        |row| row.get(0),
    )?;
    let app_meta_rows: i64 = connection.query_row(
        "SELECT COUNT(*) FROM app_meta WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    if metadata_rows != 1 || app_meta_rows != 1 {
        return Err(StorageError::InvalidData(
            "archive metadata is missing or invalid".into(),
        ));
    }
    let source_version: i64 = connection.query_row(
        "SELECT source_dataset_version FROM archive_meta WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    let dataset_version: i64 = connection.query_row(
        "SELECT dataset_version FROM app_meta WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    if source_version != dataset_version {
        return Err(StorageError::InvalidData(
            "archive dataset identity metadata is inconsistent".into(),
        ));
    }
    if !(0..=MAX_SAFE_JS_INTEGER).contains(&dataset_version) {
        return Err(StorageError::Validation(
            "archive dataset version exceeds JavaScript's safe integer range".into(),
        ));
    }
    let frame_count: i64 =
        connection.query_row("SELECT COUNT(*) FROM frames", [], |row| row.get(0))?;
    if frame_count > MAX_ARCHIVE_FRAMES {
        return Err(StorageError::Validation(
            "archive exceeds the 250000-frame limit".into(),
        ));
    }
    let complete_frames: i64 = connection.query_row(
        "SELECT COUNT(*) FROM frames f
         WHERE (SELECT COUNT(*) FROM frame_keypoints k WHERE k.frame_id = f.id) = 17
           AND EXISTS (SELECT 1 FROM thumbnails t WHERE t.frame_id = f.id)",
        [],
        |row| row.get(0),
    )?;
    if complete_frames != frame_count {
        return Err(StorageError::InvalidData(
            "archive contains incomplete frame records".into(),
        ));
    }
    Ok(())
}

fn schema_signature(
    connection: &Connection,
) -> Result<Vec<(String, String, String)>, StorageError> {
    let mut statement = connection.prepare(
        "SELECT type, name, sql FROM sqlite_schema
         WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_autoindex_%'
         ORDER BY type, name",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn validate_payload_hashes(connection: &Connection) -> Result<(), StorageError> {
    let mut statement =
        connection.prepare("SELECT values_le_f32, payload_sha256 FROM frame_features")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for row in rows {
        let (payload, expected) = row?;
        if expected != sha256(&payload) {
            return Err(StorageError::InvalidData(
                "archive feature checksum does not match".into(),
            ));
        }
        validate_f32_blob(&payload)?;
    }
    let mut statement = connection.prepare("SELECT bytes, payload_sha256 FROM thumbnails")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for row in rows {
        let (payload, expected) = row?;
        if expected != sha256(&payload) {
            return Err(StorageError::InvalidData(
                "archive thumbnail checksum does not match".into(),
            ));
        }
    }
    Ok(())
}

fn validate_model_rows(
    generations: &[GenerationRow],
    models: &[ModelRow],
    dataset_version: u64,
    dataset_identity: [u8; 32],
) -> Result<(), StorageError> {
    if generations.len() > 1_024 || models.len() > 2_048 {
        return Err(StorageError::Validation(
            "archive contains too many model generations".into(),
        ));
    }
    for generation in generations {
        if !(0..=MAX_SAFE_JS_INTEGER).contains(&generation.dataset_version) {
            return Err(StorageError::Validation(
                "archive model dataset version exceeds JavaScript's safe integer range".into(),
            ));
        }
        if generation.dataset_identity_sha256.len() != 32
            || generation.training_config_sha256.len() != 32
        {
            return Err(StorageError::InvalidData(
                "archive model generation fingerprints are invalid".into(),
            ));
        }
        // A generation is valid with one or two roles. The archive schema's
        // `PRIMARY KEY (generation_id, role)` and `role IN ('presence','posture')`
        // CHECK already guarantee the rows are distinct and well-known, so bounding
        // the count is sufficient to reject empty or over-full generations while
        // admitting legitimate posture-only / presence-only generations.
        let generation_models = models
            .iter()
            .filter(|model| model.generation_id == generation.id)
            .collect::<Vec<_>>();
        if generation_models.is_empty() || generation_models.len() > 2 {
            return Err(StorageError::InvalidData(
                "archive model generation must contain one or two model roles".into(),
            ));
        }
        for model in generation_models {
            if model.envelope_version != 1 || model.payload_sha256 != sha256(&model.payload) {
                return Err(StorageError::InvalidData(
                    "archive model checksum or version is invalid".into(),
                ));
            }
            let (serialized, envelope) =
                model_format::decode_model(&model.payload).map_err(|error| {
                    StorageError::InvalidData(format!("archive model is invalid: {error}"))
                })?;
            Model::<InferenceResult>::from_json(serialized).map_err(|error| {
                StorageError::InvalidData(format!("archive model cannot be loaded: {error}"))
            })?;
            let expected_role = if model.role == "posture" {
                ModelRole::Posture
            } else {
                ModelRole::Presence
            };
            if envelope.role != expected_role
                || envelope.dataset_version
                    != u64::try_from(generation.dataset_version).map_err(|_| {
                        StorageError::InvalidData("model dataset version is negative".into())
                    })?
                || envelope.training_config_sha256.as_slice()
                    != generation.training_config_sha256.as_slice()
            {
                return Err(StorageError::InvalidData(
                    "archive model envelope does not match its generation".into(),
                ));
            }
        }
        if generation.active == 1
            && (generation.dataset_version != dataset_version as i64
                || generation.dataset_identity_sha256.as_slice() != dataset_identity.as_slice())
        {
            return Err(StorageError::InvalidData(
                "active archive model generation is stale".into(),
            ));
        }
    }
    if models.iter().any(|model| {
        !generations
            .iter()
            .any(|generation| generation.id == model.generation_id)
    }) {
        return Err(StorageError::InvalidData(
            "archive model has no generation".into(),
        ));
    }
    Ok(())
}

fn validate_reservoir_rows(
    states: &[ReservoirStateRow],
    samples: &[ReservoirSampleRow],
) -> Result<(), StorageError> {
    if states.len() > 1 || samples.len() > 1_000 {
        return Err(StorageError::Validation(
            "archive reservoir exceeds native limits".into(),
        ));
    }
    let expected_count = states.first().map_or(0, |state| state.sample_count);
    if let Some(state) = states.first() {
        if state.singleton != 1
            || state.capacity != 1_000
            || state.seen_count < state.sample_count
            || state.sample_count < 0
            || state.rng_state < 0
            || state.last_sampled_ms < 0
        {
            return Err(StorageError::InvalidData(
                "archive reservoir state is invalid".into(),
            ));
        }
    }
    if expected_count != samples.len() as i64 {
        return Err(StorageError::InvalidData(
            "archive reservoir sample count does not match".into(),
        ));
    }
    for (expected_slot, sample) in samples.iter().enumerate() {
        if sample.slot != expected_slot as i64
            || sample.payload.is_empty()
            || sample.payload.len() > 8 * 1024 * 1024
        {
            return Err(StorageError::InvalidData(
                "archive reservoir slots or payload size are invalid".into(),
            ));
        }
        if sample
            .payload_sha256
            .as_ref()
            .is_some_and(|expected| expected.as_slice() != sha256(&sample.payload))
        {
            return Err(StorageError::InvalidData(
                "archive reservoir checksum does not match".into(),
            ));
        }
        let decoded: super::feature_reservoir::ReservoirSample =
            rmp_serde::from_slice(&sample.payload).map_err(|error| {
                StorageError::InvalidData(format!("archive reservoir sample is invalid: {error}"))
            })?;
        super::storage::validate_reservoir_sample(&decoded)
            .map_err(|error| StorageError::InvalidData(error.to_string()))?;
    }
    Ok(())
}

fn read_settings(connection: &Connection) -> Result<Vec<SettingRow>, StorageError> {
    let mut statement =
        connection.prepare("SELECT key, schema_version, json FROM settings ORDER BY key")?;
    let rows = statement.query_map([], |row| {
        Ok(SettingRow {
            key: row.get(0)?,
            schema_version: row.get(1)?,
            json: row.get(2)?,
        })
    })?;
    let settings = rows.collect::<Result<Vec<_>, _>>()?;
    for setting in &settings {
        if setting.schema_version != 1 {
            return Err(StorageError::InvalidData(format!(
                "archive setting {} has an unsupported schema version",
                setting.key
            )));
        }
        match setting.key.as_str() {
            "camera:settings" => {
                let value: CameraSettings =
                    serde_json::from_str(&setting.json).map_err(|error| {
                        StorageError::InvalidData(format!(
                            "archive camera settings are invalid: {error}"
                        ))
                    })?;
                value.validate().map_err(StorageError::InvalidData)?;
            }
            "ui:settings" => {
                let value: UiSettings = serde_json::from_str(&setting.json).map_err(|error| {
                    StorageError::InvalidData(format!("archive UI settings are invalid: {error}"))
                })?;
                value.validate().map_err(StorageError::InvalidData)?;
            }
            "training:settings" => {
                let value: TrainingSettings =
                    serde_json::from_str(&setting.json).map_err(|error| {
                        StorageError::InvalidData(format!(
                            "archive training settings are invalid: {error}"
                        ))
                    })?;
                validate_training_settings_value(&value)
                    .map_err(|error| StorageError::InvalidData(error.to_string()))?;
            }
            "pca-config" => {
                let value: DimensionalityReductionConfig = serde_json::from_str(&setting.json)
                    .map_err(|error| {
                        StorageError::InvalidData(format!(
                            "archive dimensionality-reduction setting is invalid: {error}"
                        ))
                    })?;
                if value.components == 0 || value.components > 1_048_576 {
                    return Err(StorageError::InvalidData(
                        "archive dimensionality-reduction setting exceeds limits".into(),
                    ));
                }
            }
            _ => {
                return Err(StorageError::InvalidData(format!(
                    "archive contains unknown setting {}",
                    setting.key
                )))
            }
        }
    }
    Ok(settings)
}

fn read_generations(connection: &Connection) -> Result<Vec<GenerationRow>, StorageError> {
    let mut statement = connection.prepare("SELECT id, created_at_ms, dataset_version, dataset_identity_sha256, training_config_sha256, active FROM model_generations ORDER BY id")?;
    let rows = statement.query_map([], |row| {
        Ok(GenerationRow {
            id: row.get(0)?,
            created_at_ms: row.get(1)?,
            dataset_version: row.get(2)?,
            dataset_identity_sha256: row.get(3)?,
            training_config_sha256: row.get(4)?,
            active: row.get(5)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn read_models(connection: &Connection) -> Result<Vec<ModelRow>, StorageError> {
    let mut statement = connection.prepare("SELECT generation_id, role, envelope_version, payload, payload_sha256 FROM models ORDER BY generation_id, role")?;
    let rows = statement.query_map([], |row| {
        Ok(ModelRow {
            generation_id: row.get(0)?,
            role: row.get(1)?,
            envelope_version: row.get(2)?,
            payload: row.get(3)?,
            payload_sha256: row.get(4)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn read_reservoir_states(connection: &Connection) -> Result<Vec<ReservoirStateRow>, StorageError> {
    let mut statement = connection.prepare("SELECT singleton, capacity, seen_count, sample_count, rng_state, last_sampled_ms FROM reservoir_state ORDER BY singleton")?;
    let rows = statement.query_map([], |row| {
        Ok(ReservoirStateRow {
            singleton: row.get(0)?,
            capacity: row.get(1)?,
            seen_count: row.get(2)?,
            sample_count: row.get(3)?,
            rng_state: row.get(4)?,
            last_sampled_ms: row.get(5)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn read_reservoir_samples(
    connection: &Connection,
) -> Result<Vec<ReservoirSampleRow>, StorageError> {
    let has_checksum = connection
        .prepare("SELECT payload_sha256 FROM reservoir_samples LIMIT 0")
        .is_ok();
    let sql = if has_checksum {
        "SELECT slot, payload, payload_sha256 FROM reservoir_samples ORDER BY slot"
    } else {
        "SELECT slot, payload, NULL FROM reservoir_samples ORDER BY slot"
    };
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map([], |row| {
        Ok(ReservoirSampleRow {
            slot: row.get(0)?,
            payload: row.get(1)?,
            payload_sha256: row.get(2)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn encode_f32(values: &[f32]) -> Result<Vec<u8>, StorageError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(StorageError::Validation(
            "feature vectors must be non-empty and finite".into(),
        ));
    }
    Ok(values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect())
}

fn validate_f32_blob(bytes: &[u8]) -> Result<(), StorageError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(StorageError::InvalidData(
            "archive feature payload is not packed f32".into(),
        ));
    }
    for chunk in bytes.chunks_exact(4) {
        if !f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).is_finite() {
            return Err(StorageError::InvalidData(
                "archive feature payload contains a non-finite value".into(),
            ));
        }
    }
    Ok(())
}

fn label(frame: &PostureFrame) -> &'static str {
    match frame.label {
        slouch_domain::FrameLabel::Good => "good",
        slouch_domain::FrameLabel::Bad => "bad",
        slouch_domain::FrameLabel::Away => "away",
        slouch_domain::FrameLabel::Unused => "unused",
    }
}

fn temporary_archive_path(destination: &Path) -> Result<PathBuf, StorageError> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            StorageError::Validation("archive destination has no valid filename".into())
        })?;
    Ok(destination.with_file_name(format!(".{file_name}.{}.tmp", now_ms()?)))
}

fn replace_file(temporary: &Path, destination: &Path) -> Result<(), StorageError> {
    if destination.exists() {
        return Err(StorageError::Validation(
            "archive destination already exists; choose a new filename".into(),
        ));
    }
    fs::rename(temporary, destination).map_err(|error| StorageError::Io(error.to_string()))
}

fn now_ms() -> Result<i64, StorageError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| StorageError::Clock)
        .and_then(|duration| i64::try_from(duration.as_millis()).map_err(|_| StorageError::Clock))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}
