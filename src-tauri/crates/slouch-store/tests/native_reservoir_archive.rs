use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FeatureMap, FrameLabel, Keypoint, NormalizationMode,
    PostureFrame, Thumbnail, TrainingSettings,
};
use slouch_store::ported::{feature_reservoir::ReservoirSample, storage::DatasetStorage};

fn path(name: &str, extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("slouch-{name}-{nonce}.{extension}"))
}

fn sample(seed: f32) -> ReservoirSample {
    ReservoirSample {
        features: FeatureMap::from([
            (
                FeatureId::NlfBackbone,
                vec![seed; FeatureId::NlfBackbone.metadata().dimensions],
            ),
            (
                FeatureId::NlfBackboneMax,
                vec![seed + 0.1; FeatureId::NlfBackboneMax.metadata().dimensions],
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

fn settings() -> TrainingSettings {
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

#[test]
fn sqlite_reservoir_is_rate_limited_restart_safe_and_archive_complete() {
    let database = path("reservoir", "sqlite3");
    let archive = path("reservoir", "slouchpack");
    let first = sample(0.25);
    let second = sample(0.5);
    {
        let storage = DatasetStorage::open(&database).expect("open storage");
        assert!(storage
            .sample_reservoir(&first, 10_000)
            .expect("first sample"));
        assert!(!storage
            .sample_reservoir(&second, 10_999)
            .expect("rate limited"));
        assert!(storage
            .sample_reservoir(&second, 11_000)
            .expect("second sample"));
        assert_eq!(storage.reservoir_meta().expect("meta").total_seen, 2);
    }
    let storage = DatasetStorage::open(&database).expect("reopen storage");
    assert_eq!(
        storage.load_reservoir_samples().expect("persisted samples"),
        vec![first.clone(), second.clone()]
    );
    storage
        .export_archive(&archive, "test")
        .expect("export reservoir");
    storage.reset_all().expect("reset");
    assert!(storage
        .load_reservoir_samples()
        .expect("empty after reset")
        .is_empty());
    storage.import_archive(&archive).expect("import reservoir");
    assert_eq!(
        storage.load_reservoir_samples().expect("imported samples"),
        vec![first, second]
    );
    drop(storage);
    let _ = fs::remove_file(database);
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn archive_rejects_invalid_typed_settings_without_live_mutation() {
    let archive = path("settings", "slouchpack");
    let source = DatasetStorage::open_in_memory().expect("source");
    source
        .save_training_settings(settings())
        .expect("save settings");
    source.export_archive(&archive, "test").expect("export");
    let connection = Connection::open(&archive).expect("open archive");
    connection
        .execute(
            "UPDATE settings SET json = '{}' WHERE key = 'training:settings'",
            [],
        )
        .expect("corrupt typed setting");
    drop(connection);

    let target = DatasetStorage::open_in_memory().expect("target");
    let original = settings();
    target
        .save_training_settings(&original)
        .expect("seed target");
    assert!(target.import_archive(&archive).is_err());
    assert_eq!(
        target.get_training_settings().expect("target settings"),
        Some(original)
    );
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn archive_rejects_reservoir_checksum_corruption_without_mutation() {
    let archive = path("reservoir-corrupt", "slouchpack");
    let source = DatasetStorage::open_in_memory().expect("source");
    source
        .sample_reservoir(&sample(0.25), 10_000)
        .expect("sample");
    source.export_archive(&archive, "test").expect("export");
    let connection = Connection::open(&archive).expect("open archive");
    connection
        .execute(
            "UPDATE reservoir_samples SET payload_sha256 = zeroblob(32)",
            [],
        )
        .expect("corrupt checksum");
    drop(connection);

    let target = DatasetStorage::open_in_memory().expect("target");
    target
        .sample_reservoir(&sample(0.5), 20_000)
        .expect("seed target");
    assert!(target.import_archive(&archive).is_err());
    assert_eq!(
        target.load_reservoir_samples().expect("unchanged target"),
        vec![sample(0.5)]
    );
    fs::remove_file(archive).expect("remove archive");
}

fn query_i64(connection: &Connection, sql: &str) -> i64 {
    connection.query_row(sql, [], |row| row.get(0)).expect(sql)
}

fn migration_frame(id: &str) -> PostureFrame {
    PostureFrame {
        id: id.to_owned(),
        timestamp: 1_700_000_000_000.0,
        features: FeatureMap::from([(
            FeatureId::RtmDetExtracted,
            vec![0.25; FeatureId::RtmDetExtracted.metadata().dimensions],
        )]),
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
            score: 0.9,
            width: 10.0,
            height: 20.0,
        },
        label: FrameLabel::Good,
    }
}

#[test]
fn opening_a_v1_database_wipes_only_the_reservoir_and_stamps_v2() {
    let database = path("v1-open", "sqlite3");

    // Populate a database through the public API, then stamp it back to schema
    // version 1 to simulate a pre-NLF-EMBED database. The live tables are byte
    // identical between v1 and v2 (only PRAGMA user_version differs), so a v2
    // database forced to user_version = 1 with reservoir rows present is a
    // faithful v1 fixture for the reservoir-only format break.
    {
        let storage = DatasetStorage::open(&database).expect("create database");
        storage
            .save_frame(migration_frame("kept"))
            .expect("save frame");
        storage
            .save_training_settings(settings())
            .expect("save settings");
        assert!(storage
            .sample_reservoir(&sample(0.25), 10_000)
            .expect("seed reservoir"));
    }
    {
        let connection = Connection::open(&database).expect("raw connection");
        // A model generation must survive the migration untouched.
        connection
            .execute(
                "INSERT INTO model_generations(id, created_at_ms, dataset_version, \
                 dataset_identity_sha256, training_config_sha256, active) \
                 VALUES (1, 1, 0, zeroblob(32), zeroblob(32), 0)",
                [],
            )
            .expect("insert model generation");
        assert!(query_i64(&connection, "SELECT COUNT(*) FROM reservoir_samples") > 0);
        connection
            .pragma_update(None, "user_version", 1_i64)
            .expect("stamp schema version 1");
    }

    // Opening the v1 fixture runs the one-time reservoir wipe; everything else
    // (frames, settings, models) is retained.
    let storage = DatasetStorage::open(&database).expect("open v1 database");
    assert_eq!(
        storage
            .load_dataset()
            .expect("dataset retained")
            .frames
            .len(),
        1
    );
    assert_eq!(
        storage.get_training_settings().expect("settings retained"),
        Some(settings())
    );
    assert!(storage
        .load_reservoir_samples()
        .expect("reservoir emptied")
        .is_empty());
    drop(storage);

    {
        let connection = Connection::open(&database).expect("raw connection");
        assert_eq!(query_i64(&connection, "PRAGMA user_version"), 2);
        assert_eq!(
            query_i64(&connection, "SELECT COUNT(*) FROM model_generations"),
            1
        );
        assert_eq!(
            query_i64(&connection, "SELECT COUNT(*) FROM reservoir_samples"),
            0
        );
    }

    // Second open is a no-op: version holds at 2 and the reservoir refills.
    let storage = DatasetStorage::open(&database).expect("reopen migrated database");
    assert!(storage
        .load_reservoir_samples()
        .expect("reservoir still empty")
        .is_empty());
    assert!(storage
        .sample_reservoir(&sample(0.5), 20_000)
        .expect("reservoir usable"));
    assert_eq!(
        storage.load_reservoir_samples().expect("resampled").len(),
        1
    );
    drop(storage);
    {
        let connection = Connection::open(&database).expect("raw connection");
        assert_eq!(query_i64(&connection, "PRAGMA user_version"), 2);
    }

    let _ = fs::remove_file(database);
}
