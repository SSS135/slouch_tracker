//! Adversarial tests for the SQLite schema trust boundary of slouch-store.
//!
//! Every test hands the storage layer a database file it did not fully author:
//! foreign headers, half-installed schemas, rows injected around the public
//! API, and byte-level file corruption. The assertions pin the protections
//! `DatasetStorage::from_connection` and the read paths actually implement.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, Connection, ErrorCode};
use slouch_domain::{BoundingBox, FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail};
use slouch_store::ported::storage::{DatasetStorage, StorageError};

const SLOUCH_APPLICATION_ID: i64 = 1_397_506_888;
const SUPPORTED_USER_VERSION: i64 = 2;
const GAU: FeatureId = FeatureId::GauFeatures;

fn temp_database(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("slouch-security-{name}-{nonce}.sqlite"))
}

fn remove_database(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn raw_connection(path: &Path) -> Connection {
    Connection::open(path).expect("raw connection")
}

fn query_i64(connection: &Connection, sql: &str) -> i64 {
    connection.query_row(sql, [], |row| row.get(0)).expect(sql)
}

fn table_exists(connection: &Connection, name: &str) -> bool {
    query_i64(
        connection,
        &format!("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = '{name}'"),
    ) > 0
}

fn frame(id: &str, label: FrameLabel) -> PostureFrame {
    PostureFrame {
        id: id.into(),
        timestamp: 1_700_000_000_000.0,
        features: BTreeMap::from([(GAU, vec![0.5; GAU.metadata().dimensions])]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![7; 64],
        },
        keypoints: (0..17)
            .map(|index| Keypoint::new(index as f64, index as f64 + 1.0, 0.9))
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

fn expect_constraint_violation(result: Result<usize, rusqlite::Error>, needle: &str) {
    match result {
        Err(rusqlite::Error::SqliteFailure(error, Some(message))) => {
            assert_eq!(
                error.code,
                ErrorCode::ConstraintViolation,
                "unexpected error code for: {message}"
            );
            assert!(message.contains(needle), "unexpected message: {message}");
        }
        other => panic!("expected a constraint violation, got {other:?}"),
    }
}

#[test]
fn rejects_newer_schema_version_without_downgrading_the_file() {
    let path = temp_database("newer-version");
    let storage = DatasetStorage::open(&path).expect("create v1 database");
    storage
        .save_frame(frame("kept", FrameLabel::Good))
        .expect("save frame");
    drop(storage);
    {
        let connection = raw_connection(&path);
        connection
            .pragma_update(None, "user_version", 7)
            .expect("bump user_version");
    }

    let error = DatasetStorage::open(&path).expect_err("newer schema version must be rejected");
    assert!(
        matches!(&error, StorageError::InvalidData(message) if message.contains("newer")),
        "unexpected error: {error}"
    );

    // Reject-newer must be read-only: no downgrade stamp, no data loss.
    let connection = raw_connection(&path);
    assert_eq!(query_i64(&connection, "PRAGMA user_version"), 7);
    assert_eq!(
        query_i64(&connection, "PRAGMA application_id"),
        SLOUCH_APPLICATION_ID
    );
    assert_eq!(query_i64(&connection, "SELECT COUNT(*) FROM frames"), 1);
    drop(connection);
    remove_database(&path);
}

#[test]
fn rejects_foreign_application_id_without_touching_rows() {
    let path = temp_database("foreign-app-id");
    let storage = DatasetStorage::open(&path).expect("create v1 database");
    storage
        .save_frame(frame("kept", FrameLabel::Good))
        .expect("save frame");
    drop(storage);
    {
        let connection = raw_connection(&path);
        connection
            .pragma_update(None, "application_id", 0x1122_3344)
            .expect("forge application_id");
    }

    let error = DatasetStorage::open(&path).expect_err("foreign application_id must be rejected");
    assert!(
        matches!(&error, StorageError::InvalidData(message) if message.contains("application_id")),
        "unexpected error: {error}"
    );

    let connection = raw_connection(&path);
    assert_eq!(query_i64(&connection, "PRAGMA application_id"), 0x1122_3344);
    assert_eq!(
        query_i64(&connection, "PRAGMA user_version"),
        SUPPORTED_USER_VERSION
    );
    assert_eq!(query_i64(&connection, "SELECT COUNT(*) FROM frames"), 1);
    drop(connection);
    remove_database(&path);
}

#[test]
fn rejects_zero_application_id_once_schema_version_is_nonzero() {
    // A zero application_id is only trusted together with user_version 0; a
    // versioned database that lost (or never had) the SLCH stamp is foreign.
    let path = temp_database("zero-app-id-versioned");
    drop(DatasetStorage::open(&path).expect("create v1 database"));
    {
        let connection = raw_connection(&path);
        connection
            .pragma_update(None, "application_id", 0)
            .expect("clear application_id");
    }

    let error = DatasetStorage::open(&path).expect_err("unstamped versioned database");
    assert!(
        matches!(&error, StorageError::InvalidData(message) if message.contains("application_id")),
        "unexpected error: {error}"
    );
    remove_database(&path);
}

#[test]
fn adopts_zero_id_version_zero_database_and_stamps_identity() {
    // The pristine gate is purely header-based: application_id 0 plus
    // user_version 0 is adopted even when the file already holds unrelated,
    // non-conflicting objects, which then survive the schema install.
    let path = temp_database("adopt-pristine");
    {
        let connection = raw_connection(&path);
        connection
            .execute_batch(
                "CREATE TABLE visitor_notes (note TEXT NOT NULL);
                 INSERT INTO visitor_notes(note) VALUES ('left-behind');",
            )
            .expect("seed foreign content");
    }

    let storage = DatasetStorage::open(&path).expect("adopt pristine database");
    storage
        .save_frame(frame("first", FrameLabel::Good))
        .expect("save frame after adoption");
    drop(storage);

    let connection = raw_connection(&path);
    assert_eq!(
        query_i64(&connection, "PRAGMA application_id"),
        SLOUCH_APPLICATION_ID
    );
    assert_eq!(
        query_i64(&connection, "PRAGMA user_version"),
        SUPPORTED_USER_VERSION
    );
    assert_eq!(query_i64(&connection, "SELECT COUNT(*) FROM frames"), 1);
    let note: String = connection
        .query_row("SELECT note FROM visitor_notes", [], |row| row.get(0))
        .expect("foreign table preserved");
    assert_eq!(note, "left-behind");
    drop(connection);
    remove_database(&path);
}

#[test]
fn failed_schema_install_rolls_back_header_and_partial_objects() {
    // Schema install runs inside one immediate transaction. A mid-batch
    // failure (here: a pre-existing conflicting table created after several
    // legitimate objects) must leave neither header stamps nor partial tables.
    let path = temp_database("atomic-install");
    {
        let connection = raw_connection(&path);
        connection
            .execute("CREATE TABLE thumbnails (x INTEGER)", [])
            .expect("plant conflicting table");
    }

    let error = DatasetStorage::open(&path).expect_err("conflicting object must abort install");
    assert!(
        error.to_string().contains("already exists"),
        "unexpected error: {error}"
    );

    let connection = raw_connection(&path);
    assert_eq!(query_i64(&connection, "PRAGMA user_version"), 0);
    assert_eq!(query_i64(&connection, "PRAGMA application_id"), 0);
    // app_meta and frames are created before thumbnails in live-v1.sql; their
    // absence proves the batch rolled back instead of committing a prefix.
    assert!(!table_exists(&connection, "app_meta"));
    assert!(!table_exists(&connection, "frames"));
    let sql: String = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'thumbnails'",
            [],
            |row| row.get(0),
        )
        .expect("planted table survives");
    assert!(
        sql.contains("x INTEGER"),
        "planted table was replaced: {sql}"
    );
    drop(connection);
    remove_database(&path);
}

#[test]
fn frame_delete_cascades_through_child_tables() {
    // The storage connection promises foreign_keys=ON; delete_frame relies on
    // ON DELETE CASCADE alone to purge keypoints, features, and thumbnails.
    let path = temp_database("fk-cascade");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("keeper", FrameLabel::Good))
        .expect("save keeper");
    storage
        .save_frame(frame("victim", FrameLabel::Bad))
        .expect("save victim");
    storage.delete_frame("victim").expect("delete victim");
    drop(storage);

    let connection = raw_connection(&path);
    assert_eq!(query_i64(&connection, "SELECT COUNT(*) FROM frames"), 1);
    assert_eq!(
        query_i64(&connection, "SELECT COUNT(*) FROM frame_keypoints"),
        17
    );
    assert_eq!(
        query_i64(&connection, "SELECT COUNT(*) FROM frame_features"),
        1
    );
    assert_eq!(query_i64(&connection, "SELECT COUNT(*) FROM thumbnails"), 1);
    let survivor: String = connection
        .query_row("SELECT frame_id FROM thumbnails", [], |row| row.get(0))
        .expect("surviving thumbnail");
    assert_eq!(survivor, "keeper");
    drop(connection);
    remove_database(&path);
}

#[test]
fn orphaned_child_rows_never_resurrect_frames() {
    // Open performs no foreign_key_check sweep; orphans injected offline are
    // tolerated but must stay invisible: reads are driven from frames only.
    let path = temp_database("fk-orphans");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("keeper", FrameLabel::Good))
        .expect("save keeper");
    drop(storage);
    {
        let connection = raw_connection(&path);
        connection
            .pragma_update(None, "foreign_keys", "OFF")
            .expect("disable enforcement");
        connection
            .execute(
                "INSERT INTO frame_keypoints(frame_id, keypoint_index, x, y, score)
                 VALUES ('ghost', 0, 1.0, 2.0, 0.5)",
                [],
            )
            .expect("orphan keypoint");
        connection
            .execute(
                "INSERT INTO frame_features(frame_id, feature_type, dimension, values_le_f32)
                 VALUES ('ghost', 'orphan-feature', 1, ?)",
                params![vec![0_u8; 4]],
            )
            .expect("orphan feature");
        connection
            .execute(
                "INSERT INTO thumbnails(frame_id, mime_type, bytes)
                 VALUES ('ghost', 'image/webp', ?)",
                params![vec![1_u8]],
            )
            .expect("orphan thumbnail");
    }

    let storage = DatasetStorage::open(&path).expect("reopen with orphans present");
    let dataset = storage.load_dataset().expect("dataset still loads");
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].id, "keeper");
    assert!(storage
        .get_frame_by_id("ghost")
        .expect("ghost lookup is a clean miss")
        .is_none());
    drop(storage);
    remove_database(&path);
}

#[test]
fn missing_child_rows_error_instead_of_yielding_partial_frames() {
    let path = temp_database("missing-children");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("no-thumb", FrameLabel::Good))
        .expect("save no-thumb");
    storage
        .save_frame(frame("few-kp", FrameLabel::Bad))
        .expect("save few-kp");
    drop(storage);
    {
        let connection = raw_connection(&path);
        connection
            .execute("DELETE FROM thumbnails WHERE frame_id = 'no-thumb'", [])
            .expect("strip thumbnail");
        connection
            .execute(
                "DELETE FROM frame_keypoints WHERE frame_id = 'few-kp' AND keypoint_index = 16",
                [],
            )
            .expect("strip one keypoint");
    }

    let storage = DatasetStorage::open(&path).expect("reopen");
    let thumb_error = storage
        .get_frame_by_id("no-thumb")
        .expect_err("frame without thumbnail must not load");
    assert!(
        matches!(thumb_error, StorageError::Sqlite(_)),
        "unexpected error: {thumb_error}"
    );
    let keypoint_error = storage
        .get_frame_by_id("few-kp")
        .expect_err("frame with 16 keypoints must not load");
    assert!(
        matches!(&keypoint_error, StorageError::InvalidData(message) if message.contains("17")),
        "unexpected error: {keypoint_error}"
    );
    assert!(storage.load_dataset().is_err());
    drop(storage);
    remove_database(&path);
}

#[test]
fn check_constraints_block_forged_labels_and_blob_lengths() {
    let path = temp_database("check-constraints");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("kept", FrameLabel::Good))
        .expect("save frame");
    drop(storage);

    {
        let connection = raw_connection(&path);
        expect_constraint_violation(
            connection.execute(
                "INSERT INTO frames(id, captured_at_ms, label, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                 VALUES ('forged', 1, 'evil', 0.0, 0.0, 1.0, 1.0)",
                [],
            ),
            "CHECK",
        );
        expect_constraint_violation(
            connection.execute("UPDATE frames SET label = 'evil' WHERE id = 'kept'", []),
            "CHECK",
        );
        // length(values_le_f32) = dimension * 4 guards the feature payloads.
        expect_constraint_violation(
            connection.execute(
                "UPDATE frame_features SET values_le_f32 = ? WHERE frame_id = 'kept'",
                params![vec![0_u8; 8]],
            ),
            "CHECK",
        );
    }

    let storage = DatasetStorage::open(&path).expect("reopen");
    let dataset = storage
        .load_dataset()
        .expect("rejected writes left data intact");
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].label, FrameLabel::Good);
    drop(storage);
    remove_database(&path);
}

#[test]
fn check_consistent_feature_corruption_surfaces_storage_errors() {
    // Corruption that satisfies the SQL CHECKs must still be caught by the
    // Rust read path: dimension mismatch against the feature registry,
    // non-finite payload bytes, and unknown feature identifiers.
    let path = temp_database("feature-corruption");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("victim", FrameLabel::Good))
        .expect("save frame");
    drop(storage);
    let dimensions = GAU.metadata().dimensions;

    {
        let connection = raw_connection(&path);
        connection
            .execute(
                "UPDATE frame_features SET dimension = 3, values_le_f32 = ?
                 WHERE frame_id = 'victim'",
                params![vec![0_u8; 12]],
            )
            .expect("shrink dimension consistently");
    }
    let storage = DatasetStorage::open(&path).expect("reopen");
    let dimension_error = storage
        .get_frame_by_id("victim")
        .expect_err("registry dimension mismatch must fail the read");
    assert!(
        matches!(&dimension_error, StorageError::InvalidData(message) if message.contains("dimension")),
        "unexpected error: {dimension_error}"
    );
    drop(storage);

    {
        let mut payload = vec![0_u8; dimensions * 4];
        payload[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let connection = raw_connection(&path);
        connection
            .execute(
                "UPDATE frame_features SET dimension = ?, values_le_f32 = ?
                 WHERE frame_id = 'victim'",
                params![dimensions as i64, payload],
            )
            .expect("inject NaN payload");
    }
    let storage = DatasetStorage::open(&path).expect("reopen");
    let nan_error = storage
        .load_dataset()
        .expect_err("NaN feature bytes must fail the read");
    assert!(
        nan_error.to_string().contains("non-finite"),
        "unexpected error: {nan_error}"
    );
    drop(storage);

    {
        let connection = raw_connection(&path);
        connection
            .execute(
                "UPDATE frame_features SET feature_type = 'mystery-feature', dimension = 3,
                 values_le_f32 = ? WHERE frame_id = 'victim'",
                params![vec![0_u8; 12]],
            )
            .expect("rename to unknown feature");
    }
    let storage = DatasetStorage::open(&path).expect("reopen");
    assert!(
        storage.get_frame_by_id("victim").is_err(),
        "unknown feature identifiers must not be silently skipped"
    );
    drop(storage);
    remove_database(&path);
}

#[test]
fn strict_tables_reject_mistyped_values() {
    let path = temp_database("strict-typing");
    drop(DatasetStorage::open(&path).expect("create database"));

    let connection = raw_connection(&path);
    let mut statement = connection
        .prepare(
            "SELECT name, sql FROM sqlite_master
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
        )
        .expect("schema listing");
    let tables: Vec<(String, String)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .expect("schema rows")
        .collect::<Result<_, _>>()
        .expect("schema rows decode");
    drop(statement);
    assert!(tables.len() >= 10, "schema lost tables: {tables:?}");
    for (name, sql) in &tables {
        assert!(sql.contains("STRICT"), "table {name} is not STRICT: {sql}");
    }

    match connection.execute(
        "INSERT INTO settings(key, schema_version, json) VALUES ('adversarial', ?, '{}')",
        params!["not-a-number"],
    ) {
        Err(rusqlite::Error::SqliteFailure(_, Some(message))) => assert!(
            message.contains("cannot store TEXT value in INTEGER column"),
            "unexpected message: {message}"
        ),
        other => panic!("expected a strict typing error, got {other:?}"),
    }
    match connection.execute(
        "INSERT INTO reservoir_samples(slot, payload) VALUES (0, ?)",
        params!["ascii-payload"],
    ) {
        Err(rusqlite::Error::SqliteFailure(_, Some(message))) => assert!(
            message.contains("cannot store TEXT value in BLOB column"),
            "unexpected message: {message}"
        ),
        other => panic!("expected a strict typing error, got {other:?}"),
    }
    drop(connection);
    remove_database(&path);
}

#[test]
fn garbage_header_fails_open_with_an_error() {
    let path = temp_database("garbage-header");
    let storage = DatasetStorage::open(&path).expect("create database");
    storage
        .save_frame(frame("doomed", FrameLabel::Good))
        .expect("save frame");
    drop(storage);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));

    let mut bytes = fs::read(&path).expect("read database file");
    for byte in bytes.iter_mut().take(100) {
        *byte = 0xAB;
    }
    fs::write(&path, bytes).expect("write corrupted header");

    let error = DatasetStorage::open(&path).expect_err("corrupted header must fail open");
    assert!(
        matches!(error, StorageError::Sqlite(_)),
        "unexpected error: {error}"
    );
    remove_database(&path);
}

#[test]
fn non_sqlite_file_fails_open_with_an_error() {
    let path = temp_database("not-a-database");
    let junk: Vec<u8> = (1_usize..=4096)
        .map(|index| (index % 251 + 1) as u8)
        .collect();
    fs::write(&path, junk).expect("write junk file");

    let error = DatasetStorage::open(&path).expect_err("junk file must fail open");
    assert!(
        matches!(error, StorageError::Sqlite(_)),
        "unexpected error: {error}"
    );
    remove_database(&path);
}

#[test]
fn truncated_database_errors_on_open_or_first_read() {
    let path = temp_database("truncated");
    let storage = DatasetStorage::open(&path).expect("create database");
    for index in 0..3 {
        let mut value = frame(&format!("bulk-{index}"), FrameLabel::Good);
        value.timestamp += index as f64;
        value.thumbnail.bytes = vec![9; 32 * 1024];
        storage.save_frame(value).expect("save bulky frame");
    }
    drop(storage);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));

    let original_len = fs::metadata(&path).expect("database metadata").len();
    assert!(
        original_len > 40_000,
        "database too small to truncate: {original_len}"
    );
    let file = fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open for truncation");
    file.set_len(8_192).expect("truncate database");
    drop(file);

    // Whether SQLite notices at open or at first table read, the outcome must
    // be a clean error; silent success with amputated data is a regression.
    match DatasetStorage::open(&path) {
        Ok(storage) => {
            let error = storage
                .load_dataset()
                .expect_err("truncated data pages must fail the read");
            assert!(
                matches!(error, StorageError::Sqlite(_)),
                "unexpected error: {error}"
            );
            drop(storage);
        }
        Err(error) => assert!(
            matches!(error, StorageError::Sqlite(_)),
            "unexpected error: {error}"
        ),
    }
    remove_database(&path);
}
