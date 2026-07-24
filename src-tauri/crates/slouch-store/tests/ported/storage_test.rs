use std::{
    collections::BTreeMap,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, Connection};

use slouch_domain::{
    BoundingBox, ClassifierConfig, ClassifierId, DimensionalityReductionConfig,
    DimensionalityReductionMethod, FeatureId, FrameLabel, Keypoint, ParameterValue, PostureFrame,
    Thumbnail, TrainingSettings,
};
use slouch_store::ported::storage::DatasetStorage;

const GAU: FeatureId = FeatureId::GauFeatures;
const BACKBONE: FeatureId = FeatureId::BackboneFeatures;
const RTMDET: FeatureId = FeatureId::RtmDetExtracted;

fn temp_database(name: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("slouch-{name}-{nonce}.sqlite"))
}

fn remove_database(path: &std::path::Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn storage() -> DatasetStorage {
    DatasetStorage::new_in_memory().expect("in-memory dataset storage")
}

fn bbox() -> BoundingBox {
    BoundingBox {
        x1: 0.2,
        y1: 0.2,
        x2: 0.8,
        y2: 0.8,
        score: 0.95,
        width: 0.6,
        height: 0.6,
    }
}

fn keypoints() -> Vec<Keypoint> {
    (0..17)
        .map(|index| Keypoint::new(10.0 + index as f64, 20.0 + index as f64, 0.9))
        .collect()
}

fn features(gau: bool, backbone: bool, rtmdet: bool) -> BTreeMap<FeatureId, Vec<f32>> {
    let mut values = BTreeMap::new();
    if gau {
        values.insert(GAU, vec![0.0; GAU.metadata().dimensions]);
    }
    if backbone {
        values.insert(BACKBONE, vec![0.0; BACKBONE.metadata().dimensions]);
    }
    if rtmdet {
        values.insert(RTMDET, vec![0.0; RTMDET.metadata().dimensions]);
    }
    values
}

fn frame(id: impl Into<String>, label: FrameLabel) -> PostureFrame {
    PostureFrame {
        id: id.into(),
        timestamp: 1_700_000_000_000.0,
        features: features(true, false, true),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: b"test".to_vec(),
        },
        keypoints: keypoints(),
        bbox: bbox(),
        label,
    }
}

fn frame_with_features(
    id: impl Into<String>,
    label: FrameLabel,
    features: BTreeMap<FeatureId, Vec<f32>>,
) -> PostureFrame {
    PostureFrame {
        features,
        ..frame(id, label)
    }
}

fn add_frames(storage: &DatasetStorage, frames: impl IntoIterator<Item = PostureFrame>) {
    for value in frames {
        storage.save_frame(value).expect("save frame");
    }
}

fn settings(classifier_id: ClassifierId, components: usize, last_updated: f64) -> TrainingSettings {
    settings_full(
        classifier_id,
        components,
        last_updated,
        BTreeMap::new(),
        DimensionalityReductionMethod::RandomProjection,
    )
}

fn settings_full(
    classifier_id: ClassifierId,
    components: usize,
    last_updated: f64,
    params: BTreeMap<String, ParameterValue>,
    method: DimensionalityReductionMethod,
) -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id,
            params,
        },
        dim_reduction_config: DimensionalityReductionConfig { method, components },
        posture_feature_types: vec![GAU],
        presence_feature_types: vec![RTMDET],
        feature_types: Some(vec![GAU]),
        normalization_mode: None,
        cv_folds: 5,
        last_updated,
    }
}

// Mirrors the oracle's preserve-all-config-parameters values, except maxIterations
// (the port validates against the registry range 10..=1000 on save, so we use 500
// instead of the oracle's 1500 while still exercising a non-default lossless round-trip).
fn mlp_params() -> BTreeMap<String, ParameterValue> {
    BTreeMap::from([
        ("hiddenLayers".to_owned(), ParameterValue::Number(2.0)),
        ("hiddenSize".to_owned(), ParameterValue::Number(128.0)),
        ("maxIterations".to_owned(), ParameterValue::Number(500.0)),
        ("learningRate".to_owned(), ParameterValue::Number(0.02)),
    ])
}

#[test]
fn saves_new_frame_to_empty_dataset() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();

    let dataset = storage.load_dataset().unwrap();
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].id, "frame-1");
    assert_eq!(dataset.frames[0].label, FrameLabel::Good);
    assert_eq!(dataset.version, 1);
}

#[test]
fn updates_existing_frame_with_same_id_and_increments_version() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();
    storage
        .save_frame(frame("frame-1", FrameLabel::Bad))
        .unwrap();

    let dataset = storage.load_dataset().unwrap();
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].label, FrameLabel::Bad);
    assert_eq!(dataset.version, 2);
}

#[test]
fn preserves_float32_feature_values_and_dimensions() {
    let storage = storage();
    let mut gau = vec![0.0; GAU.metadata().dimensions];
    gau[0] = 0.5;
    gau[100] = 0.75;
    storage
        .save_frame(frame_with_features(
            "frame-1",
            FrameLabel::Good,
            BTreeMap::from([
                (RTMDET, vec![0.0; RTMDET.metadata().dimensions]),
                (GAU, gau),
            ]),
        ))
        .unwrap();

    let saved = storage.load_dataset().unwrap().frames.remove(0);
    assert_eq!(saved.features[&GAU].len(), GAU.metadata().dimensions);
    assert!((saved.features[&GAU][0] - 0.5).abs() < f32::EPSILON);
    assert!((saved.features[&GAU][100] - 0.75).abs() < f32::EPSILON);
}

#[test]
fn saves_and_loads_keypoints_and_bbox() {
    let storage = storage();
    let mut value = frame("frame-both", FrameLabel::Good);
    value.keypoints = (0..17)
        .map(|index| {
            Keypoint::new(
                10.0 + index as f64,
                20.0 + index as f64,
                0.9 + index as f64 * 0.005,
            )
        })
        .collect();
    value.bbox = BoundingBox {
        x1: 100.0,
        y1: 150.0,
        x2: 400.0,
        y2: 550.0,
        score: 0.95,
        width: 300.0,
        height: 400.0,
    };
    storage.save_frame(value).unwrap();

    let loaded = storage.get_frame_by_id("frame-both").unwrap().unwrap();
    assert_eq!(loaded.keypoints.len(), 17);
    assert_eq!(loaded.keypoints[0], Keypoint::new(10.0, 20.0, 0.9));
    assert_eq!(loaded.keypoints[16].x, 26.0);
    assert_eq!(loaded.bbox.x1, 100.0);
    assert_eq!(loaded.bbox.y2, 550.0);
    assert_eq!(loaded.bbox.score, 0.95);
    assert_eq!(loaded.bbox.width, 300.0);
    assert_eq!(loaded.bbox.height, 400.0);

    let dataset_frame = storage.load_dataset().unwrap().frames.remove(0);
    assert_eq!(dataset_frame.bbox.score, 0.95);
    let metadata = storage.get_frame_metadata_page(0, 10).unwrap();
    assert_eq!(metadata.frames[0].bbox.score, 0.95);
}

#[test]
fn empty_storage_returns_empty_dataset_with_zero_version() {
    let dataset = storage().load_dataset().unwrap();
    assert!(dataset.frames.is_empty());
    assert_eq!(dataset.version, 0);
    assert!(dataset.last_modified > 0.0);
}

#[test]
fn updates_labels_and_removes_unused_frames() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();
    storage
        .update_frame_label("frame-1", FrameLabel::Bad)
        .unwrap();
    assert_eq!(
        storage.get_frame_by_id("frame-1").unwrap().unwrap().label,
        FrameLabel::Bad
    );

    storage
        .update_frame_label("frame-1", FrameLabel::Unused)
        .unwrap();
    assert!(storage.get_frame_by_id("frame-1").unwrap().is_none());
    assert_eq!(storage.load_dataset().unwrap().version, 3);
}

#[test]
fn updating_missing_frame_returns_not_found_error() {
    let error = storage()
        .update_frame_label("nonexistent", FrameLabel::Good)
        .unwrap_err();
    assert!(error.to_string().contains("nonexistent"));
}

#[test]
fn relabeling_missing_frame_to_unused_succeeds_and_bumps_version() {
    // Mirrors the TS oracle updateFrameLabel(id, UNUSED) -> deleteFrame, which
    // removeItem's an absent key (a no-op) then increments the version. Unlike the
    // non-unused branch, a missing id must NOT error here.
    let storage = storage();
    storage
        .update_frame_label("nonexistent", FrameLabel::Unused)
        .unwrap();
    assert_eq!(storage.load_dataset().unwrap().version, 1);
}

#[test]
fn label_update_replaces_timestamp() {
    let storage = storage();
    let mut value = frame("frame-1", FrameLabel::Good);
    value.timestamp = 1_700_000_000_000.0;
    storage.save_frame(value).unwrap();
    storage
        .update_frame_label("frame-1", FrameLabel::Bad)
        .unwrap();

    let loaded = storage.get_frame_by_id("frame-1").unwrap().unwrap();
    assert_eq!(loaded.label, FrameLabel::Bad);
    assert!(loaded.timestamp > 1_700_000_000_000.0);
}

#[test]
fn removes_all_frames_with_a_label_and_returns_count() {
    let storage = storage();
    add_frames(
        &storage,
        [
            frame("frame-1", FrameLabel::Good),
            frame("frame-2", FrameLabel::Unused),
            frame("frame-3", FrameLabel::Unused),
        ],
    );

    assert_eq!(
        storage.remove_frames_by_label(FrameLabel::Unused).unwrap(),
        2
    );
    let dataset = storage.load_dataset().unwrap();
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].label, FrameLabel::Good);
    assert_eq!(dataset.version, 5);
}

#[test]
fn remove_frames_by_label_returns_zero_when_no_frame_matches() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();
    assert_eq!(
        storage.remove_frames_by_label(FrameLabel::Unused).unwrap(),
        0
    );
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
}

#[test]
fn clear_dataset_removes_all_frames_and_resets_version() {
    let storage = storage();
    add_frames(
        &storage,
        [
            frame("frame-1", FrameLabel::Good),
            frame("frame-2", FrameLabel::Bad),
        ],
    );
    storage.clear_dataset().unwrap();
    let dataset = storage.load_dataset().unwrap();
    assert!(dataset.frames.is_empty());
    assert_eq!(dataset.version, 0);
}

#[test]
fn computes_empty_balanced_and_imbalanced_statistics() {
    let storage = storage();
    let empty = storage.get_stats().unwrap();
    assert_eq!(empty.total, 0);
    assert_eq!(empty.good, 0);
    assert_eq!(empty.bad, 0);
    assert_eq!(empty.unused, 0);
    assert_eq!(empty.imbalance_ratio, 0.0);
    assert!(!empty.has_minimum_frames);

    add_frames(
        &storage,
        (0..10).map(|index| {
            frame(
                format!("balanced-{index}"),
                if index < 5 {
                    FrameLabel::Good
                } else {
                    FrameLabel::Bad
                },
            )
        }),
    );
    let balanced = storage.get_stats().unwrap();
    assert_eq!(balanced.total, 10);
    assert_eq!(balanced.good, 5);
    assert_eq!(balanced.bad, 5);
    assert_eq!(balanced.imbalance_ratio, 0.0);
    assert!(balanced.has_minimum_frames);

    storage.clear_dataset().unwrap();
    add_frames(
        &storage,
        (0..8)
            .map(|index| frame(format!("good-{index}"), FrameLabel::Good))
            .chain((0..2).map(|index| frame(format!("bad-{index}"), FrameLabel::Bad))),
    );
    assert!((storage.get_stats().unwrap().imbalance_ratio - 0.6).abs() < 1e-9);
}

#[test]
fn minimum_frames_requires_three_per_class() {
    let storage = storage();
    add_frames(
        &storage,
        (0..2)
            .map(|index| frame(format!("good-{index}"), FrameLabel::Good))
            .chain((0..3).map(|index| frame(format!("bad-{index}"), FrameLabel::Bad))),
    );
    assert!(!storage.get_stats().unwrap().has_minimum_frames);

    storage
        .save_frame(frame("good-2", FrameLabel::Good))
        .unwrap();
    assert!(storage.get_stats().unwrap().has_minimum_frames);
}

#[test]
fn stores_all_feature_types_and_preserves_exact_values() {
    let storage = storage();
    let mut gau = vec![0.0; GAU.metadata().dimensions];
    let mut backbone = vec![0.0; BACKBONE.metadata().dimensions];
    let mut rtmdet = vec![0.0; RTMDET.metadata().dimensions];
    gau[0] = 0.1;
    gau[4] = 0.5;
    gau[100] = 0.75;
    backbone[99] = 2.5;
    rtmdet[10] = -3.25;
    storage
        .save_frame(frame_with_features(
            "frame-all-features",
            FrameLabel::Good,
            BTreeMap::from([(RTMDET, rtmdet), (GAU, gau), (BACKBONE, backbone)]),
        ))
        .unwrap();

    let loaded = storage
        .get_frame_by_id("frame-all-features")
        .unwrap()
        .unwrap();
    assert_eq!(loaded.features.len(), 3);
    assert!((loaded.features[&GAU][0] - 0.1).abs() < 1e-6);
    assert!((loaded.features[&GAU][4] - 0.5).abs() < 1e-6);
    assert!((loaded.features[&GAU][100] - 0.75).abs() < 1e-6);
    assert_eq!(loaded.features[&BACKBONE][99], 2.5);
    assert_eq!(loaded.features[&RTMDET][10], -3.25);
}

#[test]
fn loads_all_features_and_subsets_without_inventing_missing_features() {
    let storage = storage();
    storage
        .save_frame(frame_with_features(
            "all",
            FrameLabel::Good,
            features(true, true, true),
        ))
        .unwrap();
    storage
        .save_frame(frame_with_features(
            "gau-only",
            FrameLabel::Good,
            features(true, false, true),
        ))
        .unwrap();
    storage
        .save_frame(frame_with_features(
            "empty",
            FrameLabel::Good,
            BTreeMap::new(),
        ))
        .unwrap();

    let all = storage.get_frame_by_id("all").unwrap().unwrap();
    assert!(all.features.contains_key(&GAU));
    assert!(all.features.contains_key(&BACKBONE));
    assert!(all.features.contains_key(&RTMDET));
    let gau_only = storage.get_frame_by_id("gau-only").unwrap().unwrap();
    assert!(gau_only.features.contains_key(&GAU));
    assert!(!gau_only.features.contains_key(&BACKBONE));
    assert!(gau_only.features.contains_key(&RTMDET));
    assert!(storage
        .get_frame_by_id("empty")
        .unwrap()
        .unwrap()
        .features
        .is_empty());
}

#[test]
fn deleting_frames_removes_their_binary_features() {
    let storage = storage();
    storage
        .save_frame(frame_with_features(
            "frame-to-delete",
            FrameLabel::Good,
            features(true, true, true),
        ))
        .unwrap();
    storage
        .update_frame_label("frame-to-delete", FrameLabel::Unused)
        .unwrap();
    assert!(storage
        .get_frame_by_id("frame-to-delete")
        .unwrap()
        .is_none());

    storage
        .save_frame(frame_with_features(
            "frame-1",
            FrameLabel::Good,
            features(true, true, true),
        ))
        .unwrap();
    storage
        .save_frame(frame_with_features(
            "frame-2",
            FrameLabel::Bad,
            features(true, true, true),
        ))
        .unwrap();
    storage.clear_dataset().unwrap();
    assert!(storage.load_dataset().unwrap().frames.is_empty());
}

#[test]
fn feature_values_are_binary_native_vectors_not_json_arrays() {
    let storage = storage();
    let mut gau = vec![0.0; GAU.metadata().dimensions];
    gau[7] = 1.25;
    storage
        .save_frame(frame_with_features(
            "binary",
            FrameLabel::Good,
            BTreeMap::from([
                (RTMDET, vec![0.0; RTMDET.metadata().dimensions]),
                (GAU, gau),
            ]),
        ))
        .unwrap();
    let loaded = storage.get_frame_by_id("binary").unwrap().unwrap();
    assert!(!loaded.features[&GAU].is_empty());
    assert_eq!(loaded.features[&GAU][7], 1.25);
}

#[test]
fn handles_large_features_and_sequential_multiple_saves() {
    let storage = storage();
    add_frames(
        &storage,
        (0..5).map(|index| {
            frame_with_features(
                format!("concurrent-{index}"),
                if index % 2 == 0 {
                    FrameLabel::Good
                } else {
                    FrameLabel::Bad
                },
                features(true, true, true),
            )
        }),
    );
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 5);
}

#[test]
fn handles_nan_and_infinite_feature_values_without_changing_the_payload() {
    let storage = storage();
    let mut gau = vec![0.0; GAU.metadata().dimensions];
    gau[0] = f32::NAN;
    gau[1] = 0.5;
    gau[2] = f32::INFINITY;
    gau[3] = f32::NEG_INFINITY;
    gau[4] = 1.0;
    let value = frame_with_features(
        "nonfinite",
        FrameLabel::Good,
        BTreeMap::from([
            (RTMDET, vec![0.0; RTMDET.metadata().dimensions]),
            (GAU, gau),
        ]),
    );

    // The native validation boundary rejects non-finite feature values rather
    // than silently converting them during SQLite/blob serialization.
    assert!(storage.save_frame(value).is_err());
    assert!(storage.get_frame_by_id("nonfinite").unwrap().is_none());
}

#[test]
fn returns_all_frames_and_filters_by_label() {
    let storage = storage();
    add_frames(
        &storage,
        [
            frame("frame-1", FrameLabel::Good),
            frame("frame-2", FrameLabel::Bad),
            frame("frame-3", FrameLabel::Good),
        ],
    );
    assert_eq!(storage.get_all_frames().unwrap().len(), 3);
    let good = storage.get_frames_by_label(FrameLabel::Good).unwrap();
    assert_eq!(good.len(), 2);
    assert!(good.iter().all(|value| value.label == FrameLabel::Good));
}

#[test]
fn gets_frames_by_id_and_returns_none_for_unknown_id() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();
    assert_eq!(
        storage.get_frame_by_id("frame-1").unwrap().unwrap().id,
        "frame-1"
    );
    assert!(storage.get_frame_by_id("nonexistent").unwrap().is_none());
}

#[test]
fn needs_retraining_when_no_posture_model_exists() {
    assert!(storage().needs_retraining().unwrap());
}

#[test]
fn saves_overwrites_loads_and_clears_training_settings() {
    let storage = storage();
    assert!(storage.get_training_settings().unwrap().is_none());

    let first = settings_full(
        ClassifierId::Mlp,
        64,
        1_234_567_890.0,
        mlp_params(),
        DimensionalityReductionMethod::RandomProjection,
    );
    storage.save_training_settings(first).unwrap();
    let loaded = storage.get_training_settings().unwrap().unwrap();
    assert_eq!(loaded.classifier_config.classifier_id, ClassifierId::Mlp);
    assert_eq!(
        loaded.dim_reduction_config.method,
        DimensionalityReductionMethod::RandomProjection
    );
    assert_eq!(loaded.dim_reduction_config.components, 64);
    assert_eq!(loaded.feature_types, Some(vec![GAU]));
    assert_eq!(loaded.last_updated, 1_234_567_890.0);
    // Mirrors the oracle 'should preserve all config parameters': classifier
    // hyperparameters must survive the SQLite params round-trip losslessly.
    let params = &loaded.classifier_config.params;
    assert_eq!(params["hiddenLayers"], ParameterValue::Number(2.0));
    assert_eq!(params["hiddenSize"], ParameterValue::Number(128.0));
    assert_eq!(params["maxIterations"], ParameterValue::Number(500.0));
    assert_eq!(params["learningRate"], ParameterValue::Number(0.02));

    let second = settings(ClassifierId::Knn, 256, 2_000.0);
    storage.save_training_settings(second).unwrap();
    let overwritten = storage.get_training_settings().unwrap().unwrap();
    assert_eq!(
        overwritten.classifier_config.classifier_id,
        ClassifierId::Knn
    );
    assert_eq!(overwritten.dim_reduction_config.components, 256);

    // Mirrors the oracle 'should load saved settings' method:'none' coverage:
    // exercise the non-RandomProjection enum variant round-trip.
    let none_method = settings_full(
        ClassifierId::Knn,
        128,
        3_000.0,
        BTreeMap::new(),
        DimensionalityReductionMethod::None,
    );
    storage.save_training_settings(none_method).unwrap();
    let loaded_none = storage.get_training_settings().unwrap().unwrap();
    assert_eq!(
        loaded_none.dim_reduction_config.method,
        DimensionalityReductionMethod::None
    );
    assert_eq!(loaded_none.dim_reduction_config.components, 128);

    storage.clear_training_settings().unwrap();
    assert!(storage.get_training_settings().unwrap().is_none());
    storage.clear_training_settings().unwrap();
}

#[test]
fn training_settings_do_not_affect_frame_storage() {
    let storage = storage();
    storage
        .save_frame(frame("frame-1", FrameLabel::Good))
        .unwrap();
    storage
        .save_training_settings(settings(ClassifierId::Knn, 64, 1.0))
        .unwrap();
    storage.clear_training_settings().unwrap();
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
    assert!(storage.get_training_settings().unwrap().is_none());
}

#[test]
fn preserves_settings_for_the_lifetime_of_a_storage_instance() {
    // Mirrors the oracle 'should persist defaults across storage instance creation':
    // settings must survive an independent reopen of the same database, not just a
    // reread from the live instance. Use a file-backed store, drop it, reopen.
    let path = temp_database("settings-persistence");
    let storage = DatasetStorage::open(&path).unwrap();
    storage
        .save_training_settings(settings_full(
            ClassifierId::Knn,
            512,
            1.0,
            BTreeMap::from([("k".to_owned(), ParameterValue::Number(7.0))]),
            DimensionalityReductionMethod::RandomProjection,
        ))
        .unwrap();
    drop(storage);

    let storage = DatasetStorage::open(&path).unwrap();
    let loaded = storage.get_training_settings().unwrap().unwrap();
    assert_eq!(loaded.classifier_config.classifier_id, ClassifierId::Knn);
    assert_eq!(loaded.dim_reduction_config.components, 512);
    assert_eq!(loaded.feature_types, Some(vec![GAU]));
    assert_eq!(
        loaded.classifier_config.params["k"],
        ParameterValue::Number(7.0)
    );
    drop(storage);
    remove_database(&path);
}

#[test]
fn performs_repeated_training_settings_save_load_cycles() {
    let storage = storage();
    for index in 0..5 {
        // All-Knn so the per-cycle `k` param is valid for the classifier; mirrors the
        // oracle 'should handle multiple save/load cycles' asserting params.k == i+1.
        let value = settings_full(
            ClassifierId::Knn,
            64 * (index + 1),
            (index + 1) as f64,
            BTreeMap::from([("k".to_owned(), ParameterValue::Number((index + 1) as f64))]),
            DimensionalityReductionMethod::RandomProjection,
        );
        storage.save_training_settings(value).unwrap();
        let loaded = storage.get_training_settings().unwrap().unwrap();
        assert_eq!(loaded.dim_reduction_config.components, 64 * (index + 1));
        // Mirrors the oracle 'should handle multiple save/load cycles':
        // params.k must survive every cycle, proving no cross-cycle leakage.
        assert_eq!(
            loaded.classifier_config.params["k"],
            ParameterValue::Number((index + 1) as f64)
        );
    }
}

#[test]
fn loads_fifty_frames_with_parallel_native_queries() {
    let storage = storage();
    add_frames(
        &storage,
        (0..50).map(|index| {
            frame(
                format!("perf-frame-{index}"),
                if index % 2 == 0 {
                    FrameLabel::Good
                } else {
                    FrameLabel::Bad
                },
            )
        }),
    );
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 50);
}

#[test]
fn rejects_invalid_frames_at_the_storage_boundary() {
    let storage = storage();
    let mut missing = frame("missing", FrameLabel::Good);
    missing.keypoints.clear();
    assert!(storage.save_frame(missing).is_err());

    let mut invalid_bbox = frame("invalid-bbox", FrameLabel::Good);
    invalid_bbox.bbox.x2 = -1.0;
    assert!(storage.save_frame(invalid_bbox).is_err());

    let mut invalid_thumbnail = frame("string-thumbnail", FrameLabel::Good);
    invalid_thumbnail.thumbnail.mime_type = "text/plain".into();
    assert!(storage.save_frame(invalid_thumbnail).is_err());
    assert!(storage.load_dataset().unwrap().frames.is_empty());
}

#[test]
fn rejects_invalid_feature_dimensions_and_nonfinite_timestamps() {
    let storage = storage();
    let invalid_dimensions = frame_with_features(
        "invalid-dimensions",
        FrameLabel::Good,
        BTreeMap::from([(GAU, vec![0.0; 3])]),
    );
    assert!(storage.save_frame(invalid_dimensions).is_err());

    let mut invalid_timestamp = frame("invalid-timestamp", FrameLabel::Good);
    invalid_timestamp.timestamp = f64::NAN;
    assert!(storage.save_frame(invalid_timestamp).is_err());

    assert!(storage
        .get_frame_by_id("invalid-dimensions")
        .unwrap()
        .is_none());
    assert!(storage
        .get_frame_by_id("invalid-timestamp")
        .unwrap()
        .is_none());
}

#[test]
fn storage_info_reports_native_usage_without_a_browser_quota() {
    let info = storage().get_storage_info().unwrap();
    assert!(info.used > 0);
    assert_eq!(info.available, 0);
    assert_eq!(info.quota, 0);
}

#[test]
fn rejects_versioned_database_without_slouch_application_id() {
    let path = temp_database("application-id");
    drop(DatasetStorage::open(&path).unwrap());
    let connection = Connection::open(&path).unwrap();
    connection.pragma_update(None, "application_id", 0).unwrap();
    drop(connection);

    assert!(DatasetStorage::open(&path).is_err());
    remove_database(&path);
}

#[test]
fn accepts_pristine_version_zero_database() {
    let path = temp_database("pristine");
    drop(Connection::open(&path).unwrap());
    let storage = DatasetStorage::open(&path).unwrap();
    assert_eq!(storage.load_dataset().unwrap().version, 0);
    drop(storage);
    remove_database(&path);
}

#[test]
fn rejects_corrupted_feature_dimensions_and_unsafe_timestamps_after_reopen() {
    let path = temp_database("corruption");
    let storage = DatasetStorage::open(&path).unwrap();
    storage
        .save_frame(frame("corrupt", FrameLabel::Good))
        .unwrap();
    drop(storage);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE frame_features SET dimension = 3, values_le_f32 = ? WHERE frame_id = 'corrupt'",
            [vec![0_u8; 12]],
        )
        .unwrap();
    drop(connection);
    let storage = DatasetStorage::open(&path).unwrap();
    assert!(storage.load_dataset().is_err());
    drop(storage);

    let connection = Connection::open(&path).unwrap();
    for feature in [GAU, RTMDET] {
        let dimension = feature.metadata().dimensions;
        connection
            .execute(
                "UPDATE frame_features SET dimension = ?, values_le_f32 = ?
                 WHERE frame_id = 'corrupt' AND feature_type = ?",
                params![
                    dimension as i64,
                    vec![0_u8; dimension * 4],
                    feature.as_str()
                ],
            )
            .unwrap();
    }
    connection
        .execute(
            "UPDATE frames SET captured_at_ms = ? WHERE id = 'corrupt'",
            params![9_007_199_254_740_992_i64],
        )
        .unwrap();
    drop(connection);
    let storage = DatasetStorage::open(&path).unwrap();
    assert!(storage.get_frame_by_id("corrupt").is_err());
    assert!(storage.get_frame_metadata_page(0, 10).is_err());
    drop(storage);
    remove_database(&path);
}

#[test]
fn invalid_rows_are_rejected_without_polluting_the_dataset() {
    let storage = storage();
    storage
        .save_frame(frame("valid-frame", FrameLabel::Good))
        .unwrap();

    let mut invalid_timestamp = frame("corrupted-frame", FrameLabel::Good);
    invalid_timestamp.timestamp = f64::NAN;
    assert!(storage.save_frame(invalid_timestamp).is_err());

    let mut invalid_label_shape = frame("corrupted-label", FrameLabel::Good);
    invalid_label_shape.keypoints.truncate(16);
    assert!(storage.save_frame(invalid_label_shape).is_err());

    let dataset = storage.load_dataset().unwrap();
    assert_eq!(dataset.frames.len(), 1);
    assert_eq!(dataset.frames[0].id, "valid-frame");
}
