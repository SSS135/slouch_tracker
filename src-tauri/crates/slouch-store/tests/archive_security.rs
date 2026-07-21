//! Adversarial tests for the two dataset-import trust boundaries:
//! `DatasetStorage::import_archive` (.slouchpack SQLite container) and
//! `import_dataset_from_zip` (browser-format ZIP). Every test feeds a hostile
//! or corrupted input and pins the concrete rejection the production code
//! implements, plus the replace-only atomicity guarantee of archive import.

use std::{
    collections::BTreeMap,
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, Connection};
use serde_json::json;
use sha2::{Digest, Sha256};
use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FeatureMap, FrameLabel, Keypoint, NormalizationMode,
    PostureFrame, Thumbnail, TrainingSettings,
};
use slouch_store::ported::{
    feature_reservoir::ReservoirSample,
    import::{import_dataset_from_zip, DatasetImportError},
    storage::{DatasetStorage, StorageError},
};
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

fn temp_path(name: &str, extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "slouch-archive-security-{name}-{}-{nonce}.{extension}",
        std::process::id()
    ))
}

fn frame(id: &str, timestamp: f64) -> PostureFrame {
    let feature = FeatureId::GauFeatures;
    PostureFrame {
        id: id.to_owned(),
        timestamp,
        features: BTreeMap::from([(feature, vec![0.25; feature.metadata().dimensions])]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: b"thumbnail".to_vec(),
        },
        keypoints: (0..17)
            .map(|index| Keypoint::new(index as f64, index as f64 + 1.0, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 20.0,
            score: 1.0,
            width: 10.0,
            height: 20.0,
        },
        label: FrameLabel::Good,
    }
}

fn training_settings() -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id: ClassifierId::Knn,
            params: BTreeMap::new(),
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: 256,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::RtmDetExtracted],
        feature_types: None,
        normalization_mode: Some(NormalizationMode::None),
        cv_folds: 5,
        last_updated: 1_700_000_000_000.0,
    }
}

fn reservoir_sample(seed: f32) -> ReservoirSample {
    ReservoirSample {
        features: FeatureMap::from([
            (
                FeatureId::NlfBackbone,
                vec![seed; FeatureId::NlfBackbone.metadata().dimensions],
            ),
            (
                FeatureId::RtmDetExtracted,
                vec![seed; FeatureId::RtmDetExtracted.metadata().dimensions],
            ),
        ]),
        keypoints: (0..17)
            .map(|index| Keypoint::new(index as f64, index as f64 + 1.0, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 1.0,
            y1: 2.0,
            x2: 11.0,
            y2: 22.0,
            score: 0.9,
            width: 10.0,
            height: 20.0,
        },
    }
}

fn export_valid_archive(name: &str, exported: &PostureFrame) -> PathBuf {
    let path = temp_path(name, "slouchpack");
    let source = DatasetStorage::open_in_memory().expect("open source storage");
    source.save_frame(exported).expect("save source frame");
    source
        .export_archive(&path, "archive-security-test")
        .expect("export valid archive");
    path
}

fn doctored_copy(original: &Path, name: &str, doctor: impl FnOnce(&Connection)) -> PathBuf {
    let copy = temp_path(name, "slouchpack");
    fs::copy(original, &copy).expect("copy archive");
    let connection = Connection::open(&copy).expect("open archive copy");
    doctor(&connection);
    drop(connection);
    copy
}

fn import_into_fresh_storage(path: &Path) -> Result<(), StorageError> {
    DatasetStorage::open_in_memory()
        .expect("open target storage")
        .import_archive(path)
        .map(|_| ())
}

fn sha256(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

fn feature_payload(connection: &Connection) -> Vec<u8> {
    connection
        .query_row("SELECT values_le_f32 FROM frame_features", [], |row| {
            row.get(0)
        })
        .expect("read feature payload")
}

#[test]
fn import_rejects_non_sqlite_bytes_without_panicking() {
    // Deterministic pseudo-random bytes; the file passes the metadata size
    // check, so rejection must come from SQLite refusing the container.
    let mut state = 0x9E37_79B9_u32;
    let garbage: Vec<u8> = (0..4096)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect();
    let garbage_path = temp_path("garbage", "slouchpack");
    fs::write(&garbage_path, &garbage).expect("write garbage file");

    // A correct 16-byte SQLite magic followed by garbage must not get further.
    let mut magic_prefixed = b"SQLite format 3\0".to_vec();
    magic_prefixed.extend_from_slice(&garbage);
    let magic_path = temp_path("magic-garbage", "slouchpack");
    fs::write(&magic_path, &magic_prefixed).expect("write magic-prefixed garbage");

    let target = DatasetStorage::open_in_memory().expect("open target storage");
    assert!(target.import_archive(&garbage_path).is_err());
    assert!(target.import_archive(&magic_path).is_err());
    assert!(target
        .load_dataset()
        .expect("target must stay usable after rejected imports")
        .frames
        .is_empty());

    let _ = fs::remove_file(garbage_path);
    let _ = fs::remove_file(magic_path);
}

#[test]
fn import_rejects_empty_files_and_directories() {
    let empty = temp_path("empty", "slouchpack");
    fs::write(&empty, []).expect("write empty file");
    let error = import_into_fresh_storage(&empty).expect_err("empty file must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("non-empty"), "{error}");

    let directory = temp_path("directory", "slouchpack");
    fs::create_dir(&directory).expect("create directory");
    let error = import_into_fresh_storage(&directory).expect_err("directory must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");

    let _ = fs::remove_file(empty);
    let _ = fs::remove_dir(directory);
}

#[test]
fn import_rejects_truncated_archives_and_accepts_the_intact_original() {
    let exported = frame("truncate-1", 1_700_000_000_000.0);
    let original = export_valid_archive("truncate", &exported);
    let bytes = fs::read(&original).expect("read archive bytes");
    assert!(bytes.len() > 400, "exported archive unexpectedly small");

    // 16 keeps only the SQLite magic; 100 cuts mid-header page; the fractional
    // offsets remove whole referenced b-tree pages.
    let offsets = [
        16,
        100,
        bytes.len() / 4,
        bytes.len() / 2,
        bytes.len() * 3 / 4,
    ];
    for (index, offset) in offsets.into_iter().enumerate() {
        let truncated = temp_path(&format!("truncated-{index}"), "slouchpack");
        fs::write(&truncated, &bytes[..offset]).expect("write truncated archive");
        assert!(
            import_into_fresh_storage(&truncated).is_err(),
            "archive truncated to {offset} bytes must be rejected"
        );
        let _ = fs::remove_file(truncated);
    }

    // Control: the intact export imports, proving the rejections above target
    // the truncation rather than a generally broken fixture.
    let target = DatasetStorage::open_in_memory().expect("open target storage");
    let summary = target
        .import_archive(&original)
        .expect("intact archive imports");
    assert_eq!(summary.frame_count, 1);
    assert_eq!(
        target.load_dataset().expect("load imported dataset").frames,
        vec![exported]
    );
    let _ = fs::remove_file(original);
}

#[test]
fn import_rejects_wrong_application_id() {
    let original = export_valid_archive("wrong-app-id", &frame("app-id-1", 1_700_000_000_000.0));
    let doctored = doctored_copy(&original, "wrong-app-id-doctored", |connection| {
        connection
            .pragma_update(None, "application_id", 0x0BAD_1DEA_i64)
            .expect("rewrite application_id");
    });

    let error =
        import_into_fresh_storage(&doctored).expect_err("foreign application_id must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");
    assert!(
        error.to_string().contains("not a supported Slouch Tracker"),
        "{error}"
    );

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(doctored);
}

#[test]
fn import_rejects_foreign_archive_schema_versions() {
    let original = export_valid_archive("user-version", &frame("version-1", 1_700_000_000_000.0));
    // 0 is pre-format, 2 and 99 are newer than this build understands.
    for version in [0_i64, 2, 99] {
        let doctored = doctored_copy(
            &original,
            &format!("user-version-{version}"),
            |connection| {
                connection
                    .pragma_update(None, "user_version", version)
                    .expect("rewrite user_version");
            },
        );
        let error = import_into_fresh_storage(&doctored)
            .expect_err("foreign archive version must be rejected");
        assert!(
            matches!(error, StorageError::Validation(_)),
            "version {version}: {error:?}"
        );
        assert!(
            error.to_string().contains("not a supported Slouch Tracker"),
            "version {version}: {error}"
        );
        let _ = fs::remove_file(doctored);
    }
    let _ = fs::remove_file(original);
}

#[test]
fn import_rejects_schema_tampering() {
    let original = export_valid_archive("schema", &frame("schema-1", 1_700_000_000_000.0));
    let tampers = [
        ("dropped-table", "DROP TABLE reservoir_samples"),
        ("extra-table", "CREATE TABLE smuggled(x INTEGER)"),
        (
            "trigger",
            "CREATE TRIGGER smuggled_trigger AFTER INSERT ON frames \
             BEGIN DELETE FROM frames; END",
        ),
    ];
    for (name, sql) in tampers {
        let doctored = doctored_copy(&original, name, |connection| {
            connection.execute_batch(sql).expect("apply schema tamper");
        });
        let error = import_into_fresh_storage(&doctored)
            .expect_err("schema-tampered archive must be rejected");
        assert!(
            matches!(error, StorageError::Validation(_)),
            "{name}: {error:?}"
        );
        assert!(error.to_string().contains("schema"), "{name}: {error}");
        let _ = fs::remove_file(doctored);
    }
    let _ = fs::remove_file(original);
}

#[test]
fn import_rejects_inconsistent_or_unsafe_version_metadata() {
    let original = export_valid_archive("metadata", &frame("metadata-1", 1_700_000_000_000.0));

    let inconsistent = doctored_copy(&original, "metadata-inconsistent", |connection| {
        connection
            .execute(
                "UPDATE archive_meta SET source_dataset_version = source_dataset_version + 1",
                [],
            )
            .expect("desynchronize dataset versions");
    });
    let error = import_into_fresh_storage(&inconsistent)
        .expect_err("desynchronized version metadata must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("inconsistent"), "{error}");

    // 2^53, one past JavaScript's largest exactly-representable integer.
    let unsafe_version = doctored_copy(&original, "metadata-unsafe", |connection| {
        connection
            .execute("UPDATE app_meta SET dataset_version = 9007199254740992", [])
            .expect("write unsafe dataset version");
        connection
            .execute(
                "UPDATE archive_meta SET source_dataset_version = 9007199254740992",
                [],
            )
            .expect("write unsafe source version");
    });
    let error = import_into_fresh_storage(&unsafe_version)
        .expect_err("JS-unsafe dataset version must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("safe integer"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(inconsistent);
    let _ = fs::remove_file(unsafe_version);
}

#[test]
fn import_rejects_feature_checksum_mismatches_in_both_directions() {
    let original = export_valid_archive("checksum", &frame("checksum-1", 1_700_000_000_000.0));

    let flipped_payload = doctored_copy(&original, "checksum-flipped-payload", |connection| {
        let mut payload = feature_payload(connection);
        // Low mantissa bit of the first f32: the value stays finite and the
        // blob keeps its length, so only the recorded digest is now stale.
        payload[0] ^= 1;
        connection
            .execute(
                "UPDATE frame_features SET values_le_f32 = ?",
                params![payload],
            )
            .expect("flip payload byte");
    });
    let error = import_into_fresh_storage(&flipped_payload)
        .expect_err("payload not matching its digest must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("feature checksum"), "{error}");

    let zeroed_digest = doctored_copy(&original, "checksum-zeroed-digest", |connection| {
        connection
            .execute(
                "UPDATE frame_features SET payload_sha256 = zeroblob(32)",
                [],
            )
            .expect("zero recorded digest");
    });
    let error = import_into_fresh_storage(&zeroed_digest)
        .expect_err("digest not matching its payload must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("feature checksum"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(flipped_payload);
    let _ = fs::remove_file(zeroed_digest);
}

#[test]
fn import_rejects_non_finite_feature_values_even_with_a_matching_checksum() {
    let original = export_valid_archive("nan", &frame("nan-1", 1_700_000_000_000.0));
    let doctored = doctored_copy(&original, "nan-doctored", |connection| {
        let mut payload = feature_payload(connection);
        payload[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let digest = sha256(&payload);
        // The digest is recomputed, so a checksum-only defense would pass;
        // rejection must come from finiteness validation of the payload.
        connection
            .execute(
                "UPDATE frame_features SET values_le_f32 = ?, payload_sha256 = ?",
                params![payload, digest],
            )
            .expect("write NaN payload with matching digest");
    });

    // The finiteness guard fires while decoding stored features during the
    // archive dataset load, surfacing as a SQLite blob-conversion error.
    let error =
        import_into_fresh_storage(&doctored).expect_err("NaN feature payload must be rejected");
    assert!(
        matches!(
            error,
            StorageError::Sqlite(_) | StorageError::InvalidData(_)
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("non-finite"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(doctored);
}

#[test]
fn import_rejects_files_above_the_two_gib_size_cap() {
    let path = temp_path("oversized", "slouchpack");
    let file = fs::File::create(&path).expect("create oversized file");
    // Logical size only, no bytes written: the cap must fire on file metadata
    // before any SQLite open touches the content.
    file.set_len(2 * 1024 * 1024 * 1024 + 1)
        .expect("extend file past 2 GiB");
    drop(file);

    let result = import_into_fresh_storage(&path);
    let _ = fs::remove_file(&path);
    let error = result.expect_err("archive above 2 GiB must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("2 GiB"), "{error}");
}

#[test]
fn import_rejects_archives_above_the_frame_count_cap() {
    let original = export_valid_archive("frame-cap", &frame("cap-seed", 1_700_000_000_000.0));
    let copy = temp_path("frame-cap-doctored", "slouchpack");
    fs::copy(&original, &copy).expect("copy archive");
    {
        let mut connection = Connection::open(&copy).expect("open archive copy");
        let transaction = connection.transaction().expect("begin frame flood");
        {
            let mut statement = transaction
                .prepare(
                    "INSERT INTO frames(id, captured_at_ms, label, bbox_x1, bbox_y1, \
                     bbox_x2, bbox_y2, bbox_score) \
                     VALUES (?, 1, 'unused', 0.0, 0.0, 0.0, 0.0, 0.0)",
                )
                .expect("prepare frame insert");
            // The exported seed frame makes this 250001 rows, one past the cap.
            for index in 0..250_000_u32 {
                statement
                    .execute(params![format!("cap-{index}")])
                    .expect("insert flood frame");
            }
        }
        transaction.commit().expect("commit frame flood");
    }

    let error = import_into_fresh_storage(&copy).expect_err("frame flood must be rejected");
    assert!(matches!(error, StorageError::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("250000"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(copy);
}

#[test]
fn import_rejects_incomplete_frame_records() {
    let original = export_valid_archive("incomplete", &frame("incomplete-1", 1_700_000_000_000.0));

    let missing_keypoint = doctored_copy(&original, "incomplete-keypoint", |connection| {
        connection
            .execute("DELETE FROM frame_keypoints WHERE keypoint_index = 16", [])
            .expect("delete one keypoint");
    });
    let error = import_into_fresh_storage(&missing_keypoint)
        .expect_err("frame with 16 keypoints must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("incomplete"), "{error}");

    let missing_thumbnail = doctored_copy(&original, "incomplete-thumbnail", |connection| {
        connection
            .execute("DELETE FROM thumbnails", [])
            .expect("delete thumbnail");
    });
    let error = import_into_fresh_storage(&missing_thumbnail)
        .expect_err("frame without thumbnail must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("incomplete"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(missing_keypoint);
    let _ = fs::remove_file(missing_thumbnail);
}

#[test]
fn import_rejects_unknown_setting_keys() {
    let original = export_valid_archive("settings", &frame("settings-1", 1_700_000_000_000.0));
    let doctored = doctored_copy(&original, "settings-doctored", |connection| {
        connection
            .execute(
                "INSERT INTO settings(key, schema_version, json) \
                 VALUES ('smuggled:setting', 1, '{}')",
                [],
            )
            .expect("insert unknown setting");
    });

    let error = import_into_fresh_storage(&doctored)
        .expect_err("setting outside the whitelist must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("unknown setting"), "{error}");

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(doctored);
}

#[test]
fn import_rejects_model_rows_that_fail_decoding_or_have_no_roles() {
    let original = export_valid_archive("models", &frame("models-1", 1_700_000_000_000.0));
    let payload = b"not-a-real-model-envelope".to_vec();
    let digest = sha256(&payload);

    // Valid checksum, undecodable envelope: rejection must come from the
    // model decode step, not the digest comparison.
    let undecodable = doctored_copy(&original, "models-undecodable", |connection| {
        connection
            .execute(
                "INSERT INTO model_generations(id, created_at_ms, dataset_version, \
                 dataset_identity_sha256, training_config_sha256, active) \
                 VALUES (1, 1, 0, zeroblob(32), zeroblob(32), 0)",
                [],
            )
            .expect("insert generation");
        for role in ["posture", "presence"] {
            connection
                .execute(
                    "INSERT INTO models(generation_id, role, envelope_version, payload, \
                     payload_sha256) VALUES (1, ?, 1, ?, ?)",
                    params![role, payload, digest],
                )
                .expect("insert model row");
        }
    });
    let error = import_into_fresh_storage(&undecodable)
        .expect_err("undecodable model payload must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(error.to_string().contains("model is invalid"), "{error}");

    // A single-role generation is legitimate (posture-only / presence-only), but
    // a generation with NO model rows is a dangling record and must be rejected.
    let empty_generation = doctored_copy(&original, "models-empty-generation", |connection| {
        connection
            .execute(
                "INSERT INTO model_generations(id, created_at_ms, dataset_version, \
                 dataset_identity_sha256, training_config_sha256, active) \
                 VALUES (1, 1, 0, zeroblob(32), zeroblob(32), 0)",
                [],
            )
            .expect("insert generation");
    });
    let error = import_into_fresh_storage(&empty_generation)
        .expect_err("generation with no model roles must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(
        error.to_string().contains("one or two model roles"),
        "{error}"
    );

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(undecodable);
    let _ = fs::remove_file(empty_generation);
}

#[test]
fn import_rejects_reservoir_state_violating_native_invariants() {
    let original = export_valid_archive("reservoir", &frame("reservoir-1", 1_700_000_000_000.0));

    // Capacity 500 satisfies the SQL CHECK range, so rejection proves the
    // stricter code-level capacity pin, not the schema constraint.
    let wrong_capacity = doctored_copy(&original, "reservoir-capacity", |connection| {
        connection
            .execute(
                "INSERT INTO reservoir_state(singleton, capacity, seen_count, sample_count, \
                 rng_state, last_sampled_ms) VALUES (1, 500, 0, 0, 0, 0)",
                [],
            )
            .expect("insert non-native capacity");
    });
    let error = import_into_fresh_storage(&wrong_capacity)
        .expect_err("non-native reservoir capacity must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(
        error.to_string().contains("reservoir state is invalid"),
        "{error}"
    );

    let count_mismatch = doctored_copy(&original, "reservoir-count", |connection| {
        connection
            .execute(
                "INSERT INTO reservoir_state(singleton, capacity, seen_count, sample_count, \
                 rng_state, last_sampled_ms) VALUES (1, 1000, 1, 1, 0, 0)",
                [],
            )
            .expect("insert state claiming an absent sample");
    });
    let error = import_into_fresh_storage(&count_mismatch)
        .expect_err("sample-count mismatch must be rejected");
    assert!(matches!(error, StorageError::InvalidData(_)), "{error:?}");
    assert!(
        error.to_string().contains("sample count does not match"),
        "{error}"
    );

    let _ = fs::remove_file(original);
    let _ = fs::remove_file(wrong_capacity);
    let _ = fs::remove_file(count_mismatch);
}

#[test]
fn failed_imports_leave_the_live_dataset_fully_intact() {
    let live = DatasetStorage::open_in_memory().expect("open live storage");
    live.save_frame(frame("live-1", 1_700_000_000_000.0))
        .expect("seed frame 1");
    live.save_frame(frame("live-2", 1_700_000_001_000.0))
        .expect("seed frame 2");
    live.save_training_settings(training_settings())
        .expect("seed settings");
    assert!(live
        .sample_reservoir(&reservoir_sample(0.25), 10_000)
        .expect("seed reservoir"));

    let baseline = live.load_dataset().expect("baseline dataset");
    let baseline_settings = live.get_training_settings().expect("baseline settings");
    let baseline_reservoir = live.load_reservoir_samples().expect("baseline reservoir");
    assert_eq!(baseline.frames.len(), 2);
    assert!(baseline_settings.is_some());
    assert_eq!(baseline_reservoir.len(), 1);

    let archive_frame = frame("archive-1", 1_700_000_002_000.0);
    let clean = export_valid_archive("atomicity", &archive_frame);
    let clean_bytes = fs::read(&clean).expect("read clean archive");

    // Sabotage variants chosen so imports fail at different stages: file
    // metadata, container open, identity validation, payload hashing, and
    // settings whitelisting.
    let empty = temp_path("atomicity-empty", "slouchpack");
    fs::write(&empty, []).expect("write empty sabotage");
    let truncated = temp_path("atomicity-truncated", "slouchpack");
    fs::write(&truncated, &clean_bytes[..clean_bytes.len() / 2]).expect("write truncated sabotage");
    let wrong_identity = doctored_copy(&clean, "atomicity-wrong-id", |connection| {
        connection
            .pragma_update(None, "application_id", 7_i64)
            .expect("rewrite application_id");
    });
    let stale_digest = doctored_copy(&clean, "atomicity-stale-digest", |connection| {
        let mut payload = feature_payload(connection);
        payload[0] ^= 1;
        connection
            .execute(
                "UPDATE frame_features SET values_le_f32 = ?",
                params![payload],
            )
            .expect("flip payload byte");
    });
    let smuggled_setting = doctored_copy(&clean, "atomicity-setting", |connection| {
        connection
            .execute(
                "INSERT INTO settings(key, schema_version, json) \
                 VALUES ('smuggled:setting', 1, '{}')",
                [],
            )
            .expect("insert unknown setting");
    });

    for path in [
        &empty,
        &truncated,
        &wrong_identity,
        &stale_digest,
        &smuggled_setting,
    ] {
        assert!(
            live.import_archive(path).is_err(),
            "sabotaged archive {} must be rejected",
            path.display()
        );
        let after = live.load_dataset().expect("live dataset stays readable");
        assert_eq!(
            after.frames,
            baseline.frames,
            "frames changed after failed import of {}",
            path.display()
        );
        assert_eq!(
            after.version,
            baseline.version,
            "dataset version changed after failed import of {}",
            path.display()
        );
        assert_eq!(
            live.get_training_settings().expect("settings readable"),
            baseline_settings
        );
        assert_eq!(
            live.load_reservoir_samples().expect("reservoir readable"),
            baseline_reservoir
        );
    }

    // Control: the same live database accepts the clean archive, and a
    // successful import replaces every store instead of merging.
    let summary = live.import_archive(&clean).expect("clean archive imports");
    assert_eq!(summary.frame_count, 1);
    assert_eq!(
        live.load_dataset().expect("load replaced dataset").frames,
        vec![archive_frame]
    );
    assert_eq!(
        live.get_training_settings().expect("settings after import"),
        None
    );
    assert!(live
        .load_reservoir_samples()
        .expect("reservoir after import")
        .is_empty());

    for path in [
        empty,
        truncated,
        wrong_identity,
        stale_digest,
        smuggled_setting,
        clean,
    ] {
        let _ = fs::remove_file(path);
    }
}

const ZIP_FRAME_ID: &str = "frame-1";

fn zip_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    let mut writer = ZipWriter::new(&mut bytes);
    let options = SimpleFileOptions::default();
    for (path, contents) in entries {
        writer.start_file(*path, options).expect("start zip entry");
        writer.write_all(contents).expect("write zip entry");
    }
    writer.finish().expect("finish zip archive");
    bytes.into_inner()
}

fn zip_manifest(frame_id: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version": 2,
        "exportedAt": "2026-01-01T00:00:00.000Z",
        "frameCount": 1,
        "frames": [{
            "id": frame_id,
            "label": "good",
            "timestamp": 1_700_000_000_000.0_f64,
            "features": [],
        }],
    }))
    .expect("encode manifest")
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ff_ff;

    if exponent == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        if unbiased < -24 {
            return sign;
        }
        let mantissa = fraction | 0x80_0000;
        let shift = (-unbiased - 14 + 13) as u32;
        let rounded = (mantissa + (1 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }

    let half_exponent = (unbiased + 15) as u16;
    let rounded_fraction = (fraction + 0x1000) >> 13;
    if rounded_fraction == 0x400 {
        return sign | ((half_exponent + 1) << 10);
    }
    sign | (half_exponent << 10) | rounded_fraction as u16
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f32_to_f16(*value).to_le_bytes())
        .collect()
}

fn zip_keypoints() -> Vec<u8> {
    let values = (0..17)
        .flat_map(|index| [10.0 + index as f32, 20.0 + index as f32, 0.9])
        .collect::<Vec<_>>();
    f16_bytes(&values)
}

fn zip_bbox() -> Vec<u8> {
    f16_bytes(&[100.0, 150.0, 400.0, 550.0, 0.95, 300.0, 400.0])
}

fn valid_zip_dataset() -> Vec<u8> {
    let manifest = zip_manifest(ZIP_FRAME_ID);
    let keypoints = zip_keypoints();
    let bbox = zip_bbox();
    zip_bytes(&[
        ("manifest.json", manifest.as_slice()),
        ("frames/frame-frame-1/thumbnail.webp", b"mock-image"),
        ("frames/frame-frame-1/keypoints.bin", keypoints.as_slice()),
        ("frames/frame-frame-1/bbox.bin", bbox.as_slice()),
    ])
}

#[test]
fn zip_import_rejects_non_zip_bytes_and_archives_missing_the_manifest() {
    let storage = DatasetStorage::open_in_memory().expect("open storage");

    let error = import_dataset_from_zip(b"definitely not a zip archive", &storage)
        .expect_err("non-zip bytes must be rejected");
    assert!(
        matches!(error, DatasetImportError::Manifest(_)),
        "{error:?}"
    );
    assert!(
        error.to_string().starts_with("Dataset import failed:"),
        "{error}"
    );
    assert!(error.to_string().contains("invalid ZIP archive"), "{error}");

    let no_manifest = zip_bytes(&[("readme.txt", b"hello".as_slice())]);
    let error = import_dataset_from_zip(&no_manifest, &storage)
        .expect_err("zip without manifest.json must be rejected");
    assert!(
        matches!(error, DatasetImportError::Manifest(_)),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("manifest.json not found"),
        "{error}"
    );

    assert!(storage
        .load_dataset()
        .expect("storage stays usable")
        .frames
        .is_empty());
}

#[test]
fn zip_import_rejects_truncated_zip_archives_and_accepts_the_intact_original() {
    let intact = valid_zip_dataset();

    // Control first: the fixture genuinely imports, so the rejections below
    // can only come from the truncation.
    let control = DatasetStorage::open_in_memory().expect("open control storage");
    let result = import_dataset_from_zip(&intact, &control).expect("intact fixture imports");
    assert_eq!(result.imported, 1, "errors: {:?}", result.errors);

    // The end-of-central-directory record occupies the final 22 bytes; the
    // central directory sits just before it; len/2 lands mid entry payload.
    let cuts = [intact.len() - 10, intact.len() - 30, intact.len() / 2];
    for cut in cuts {
        let storage = DatasetStorage::open_in_memory().expect("open storage");
        let truncated = &intact[..cut];
        let error = import_dataset_from_zip(truncated, &storage)
            .expect_err("truncated zip must be rejected");
        assert!(
            matches!(error, DatasetImportError::Manifest(_)),
            "cut {cut}: {error:?}"
        );
        assert!(
            error.to_string().contains("invalid ZIP archive"),
            "cut {cut}: {error}"
        );
        assert!(storage
            .load_dataset()
            .expect("storage stays usable")
            .frames
            .is_empty());
    }
}

#[test]
fn zip_import_enforces_the_declared_entry_size_cap() {
    let storage = DatasetStorage::open_in_memory().expect("open storage");
    let manifest = zip_manifest(ZIP_FRAME_ID);
    let keypoints = zip_keypoints();
    let bbox = zip_bbox();
    // One byte past the 16 MiB per-entry cap; deflate shrinks it inside the
    // file, so rejection must come from the declared uncompressed size.
    let oversized_thumbnail = vec![0_u8; 16 * 1024 * 1024 + 1];
    let archive = zip_bytes(&[
        ("manifest.json", manifest.as_slice()),
        (
            "frames/frame-frame-1/thumbnail.webp",
            oversized_thumbnail.as_slice(),
        ),
        ("frames/frame-frame-1/keypoints.bin", keypoints.as_slice()),
        ("frames/frame-frame-1/bbox.bin", bbox.as_slice()),
    ]);

    let result =
        import_dataset_from_zip(&archive, &storage).expect("import isolates the oversized frame");
    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 1);
    assert!(
        result.errors[0].contains("16777216-byte limit"),
        "errors: {:?}",
        result.errors
    );
    assert!(storage
        .load_dataset()
        .expect("load dataset")
        .frames
        .is_empty());
}

#[test]
fn zip_import_rejects_manifests_above_the_size_cap() {
    let storage = DatasetStorage::open_in_memory().expect("open storage");
    // One byte past the 8 MiB manifest cap; the size check must fire before
    // any JSON parsing, so the content never matters.
    let oversized_manifest = vec![b' '; 8 * 1024 * 1024 + 1];
    let archive = zip_bytes(&[("manifest.json", oversized_manifest.as_slice())]);

    let error = import_dataset_from_zip(&archive, &storage)
        .expect_err("oversized manifest must be rejected");
    assert!(
        matches!(error, DatasetImportError::Manifest(_)),
        "{error:?}"
    );
    assert!(error.to_string().contains("8388608-byte limit"), "{error}");
}
