use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, Keypoint, NormalizationMode, TrainingSettings,
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
        backbone_avg: vec![seed; FeatureId::BackboneFeatures.metadata().dimensions],
        backbone_max: vec![seed + 0.1; FeatureId::BackboneFeaturesMax.metadata().dimensions],
        backbone_std: vec![seed + 0.2; FeatureId::BackboneFeaturesStd.metadata().dimensions],
        gau_avg: vec![seed; FeatureId::GauFeatures.metadata().dimensions],
        gau_max: vec![seed + 0.1; FeatureId::GauFeaturesMax.metadata().dimensions],
        gau_std: vec![seed + 0.2; FeatureId::GauFeaturesStd.metadata().dimensions],
        rtmdet: vec![seed; FeatureId::RtmDetExtracted.metadata().dimensions],
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
