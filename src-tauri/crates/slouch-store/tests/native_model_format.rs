use std::{
    collections::BTreeMap,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use slouch_domain::{
    BoundingBox, CameraSettings, ClassifierConfig, ClassifierId, CrossValidationType,
    DimensionalityReductionConfig, DimensionalityReductionMethod, FeatureId, FrameLabel, Keypoint,
    NormalizationMode as DomainNormalizationMode, ParameterValue, PostureFrame, Thumbnail,
    TrainingMetrics, TrainingSettings, UiSettings,
};
use slouch_ml::ported::pca::SerializedPca;
use slouch_ml::ported::types::{
    DimReductionTransformer, DimensionalityReductionConfig as MlReductionConfig,
    DimensionalityReductionMethod as MlReductionMethod, KnnKernel, NormalizationMode,
    SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
    SerializedGaussianNb, SerializedKMeansLogistic, SerializedKMeansPrototype, SerializedKnn,
    SerializedMlp, SerializedModel, SerializedSvm,
};
use slouch_store::ported::{
    model_format::{decode_model, encode_model, sha256, training_config_fingerprint, ModelRole},
    storage::{DatasetStorage, StorageError},
};

const DIMENSION: usize = 256;

fn extractor() -> SerializedFeatureExtractor {
    SerializedFeatureExtractor {
        feature_types: vec![FeatureId::GauFeatures.as_str().into()],
        normalization_mode: NormalizationMode::None,
        dim_reduction_config: MlReductionConfig {
            method: MlReductionMethod::None,
            components: DIMENSION,
        },
        concatenated_dimensions: DIMENSION,
        normalization_mean: None,
        normalization_std: None,
        dim_reduction_transformer: None,
    }
}

fn mlp_for_dimension(dimension: usize) -> SerializedMlp {
    SerializedMlp {
        layer_weights: vec![vec![0.25; dimension * 2]],
        layer_biases: vec![vec![0.0; 2]],
        layer_shapes: vec![dimension, 2],
        hidden_layers: 0,
        hidden_size: 64,
        class_weights: [1.0, 1.0],
    }
}

fn mlp() -> SerializedMlp {
    mlp_for_dimension(DIMENSION)
}

fn model_with_features(feature_types: Vec<FeatureId>) -> SerializedModel {
    let dimension = feature_types
        .iter()
        .map(|feature| feature.metadata().dimensions)
        .sum();
    SerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: feature_types
                .iter()
                .map(|feature| feature.as_str().to_owned())
                .collect(),
            normalization_mode: NormalizationMode::None,
            dim_reduction_config: MlReductionConfig {
                method: MlReductionMethod::None,
                components: dimension,
            },
            concatenated_dimensions: dimension,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: None,
        },
        classifier: SerializedClassifier {
            classifier_id: "mlp".into(),
            state: SerializedClassifierState::Mlp(mlp_for_dimension(dimension)),
        },
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    }
}

fn replace_unique(payload: &mut [u8], from: &[u8], to: &[u8]) {
    assert_eq!(from.len(), to.len());
    let matches = payload
        .windows(from.len())
        .enumerate()
        .filter_map(|(index, window)| (window == from).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "replacement source must occur exactly once"
    );
    payload[matches[0]..matches[0] + to.len()].copy_from_slice(to);
}

fn rewrite_archive_posture_payload(archive: &std::path::Path, rewrite: impl FnOnce(&mut Vec<u8>)) {
    let connection = Connection::open(archive).expect("archive");
    let mut payload = connection
        .query_row(
            "SELECT payload FROM models WHERE role = 'posture'",
            [],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .expect("posture payload");
    rewrite(&mut payload);
    let digest = sha256(&payload).to_vec();
    connection
        .execute(
            "UPDATE models SET payload = ?, payload_sha256 = ? WHERE role = 'posture'",
            rusqlite::params![payload, digest],
        )
        .expect("rewrite posture payload");
}

fn models() -> Vec<SerializedModel> {
    let states = vec![
        ("mlp", SerializedClassifierState::Mlp(mlp())),
        (
            "knn",
            SerializedClassifierState::Knn(SerializedKnn {
                training_data: vec![vec![0.25; DIMENSION], vec![0.75; DIMENSION]],
                training_labels: vec![0.0, 1.0],
                k: 1,
                kernel: Some(KnnKernel::Cosine),
                gamma: Some(1.0),
            }),
        ),
        (
            "svm",
            SerializedClassifierState::Svm(SerializedSvm {
                weights: vec![0.25; DIMENSION],
                bias: 0.0,
                class_weights: [1.0, 1.0],
            }),
        ),
        (
            "kmeans_prototype",
            SerializedClassifierState::KMeansPrototype(SerializedKMeansPrototype {
                clusters: vec![slouch_ml::ported::types::KMeansPrototypeCluster {
                    centroid: vec![0.25; DIMENSION],
                    prototype_good: Some(vec![0.2; DIMENSION]),
                    prototype_bad: None,
                }],
                global_prototype_good: vec![0.2; DIMENSION],
                global_prototype_bad: vec![0.8; DIMENSION],
                temperature: 1.0,
            }),
        ),
        (
            "gaussian_nb",
            SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                class_means: [vec![0.2; DIMENSION], vec![0.8; DIMENSION]],
                class_variances: [vec![0.1; DIMENSION], vec![0.1; DIMENSION]],
                class_priors: [0.5, 0.5],
                epsilon: 1e-6,
            }),
        ),
        (
            "kmeans_logistic",
            SerializedClassifierState::KMeansLogistic(SerializedKMeansLogistic {
                centroids: vec![vec![0.25; DIMENSION]],
                cluster_models: vec![Some(mlp())],
                global_model: mlp(),
                temperature: 1.0,
            }),
        ),
    ];
    states
        .into_iter()
        .map(|(id, state)| SerializedModel {
            feature_extractor: extractor(),
            classifier: SerializedClassifier {
                classifier_id: id.into(),
                state,
            },
            trained_at: 1_700_000_000_000.0,
            version: 1.0,
        })
        .collect()
}

#[test]
fn clamped_pca_model_round_trips_through_container() {
    // A PCA model whose requested components were clamped to the feature rank records
    // the EFFECTIVE count everywhere: reduction.output_dimension, the PCA state's
    // n_components, and the classifier input dimension are all 7 (torso_invariant's
    // width). The container encodes and decodes without the dimension validator
    // rejecting it — the writer being honest is what keeps the format consistent.
    let dims = 7;
    let components: Vec<Vec<f64>> = (0..dims)
        .map(|row| {
            (0..dims)
                .map(|column| if row == column { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let model = SerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: vec![FeatureId::TorsoInvariant.as_str().into()],
            normalization_mode: NormalizationMode::None,
            dim_reduction_config: MlReductionConfig {
                method: MlReductionMethod::Pca,
                components: dims,
            },
            concatenated_dimensions: dims,
            normalization_mean: None,
            normalization_std: None,
            dim_reduction_transformer: Some(DimReductionTransformer::Pca(SerializedPca {
                components,
                mean: vec![0.0; dims],
                n_components: dims,
                n_features: dims,
                explained_variance: Some(vec![0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02]),
            })),
        },
        classifier: SerializedClassifier {
            classifier_id: "gaussian_nb".into(),
            state: SerializedClassifierState::GaussianNb(SerializedGaussianNb {
                class_means: [vec![0.0; dims], vec![1.0; dims]],
                class_variances: [vec![1.0; dims], vec![1.0; dims]],
                class_priors: [0.5, 0.5],
                epsilon: 1e-9,
            }),
        },
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    };

    let bytes =
        encode_model(&model, ModelRole::Posture, 1, [1; 32], None).expect("encode clamped PCA SLMD");
    let (decoded, _envelope) = decode_model(&bytes).expect("decode clamped PCA SLMD");
    assert_eq!(
        decoded.feature_extractor.dim_reduction_config.components,
        dims
    );
    assert_eq!(decoded.feature_extractor.concatenated_dimensions, dims);
    match decoded.feature_extractor.dim_reduction_transformer {
        Some(DimReductionTransformer::Pca(state)) => {
            assert_eq!(state.n_components, dims);
            assert_eq!(state.n_features, dims);
        }
        other => panic!("expected a PCA transformer, got {other:?}"),
    }
}

fn settings(
    classifier_id: ClassifierId,
    params: BTreeMap<String, ParameterValue>,
) -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id,
            params,
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: DIMENSION,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::GauFeatures],
        feature_types: None,
        normalization_mode: Some(DomainNormalizationMode::None),
        cv_folds: 5,
        last_updated: 1_700_000_000_000.0,
    }
}

fn metrics() -> TrainingMetrics {
    TrainingMetrics {
        cv_accuracy: 0.8,
        cv_std: 0.1,
        mcc: 0.6,
        f1_score: 0.75,
        confusion_matrix: vec![vec![8, 2], vec![2, 8]],
        fold_accuracies: vec![0.75, 0.85],
        balanced_accuracy: 0.8,
        accuracy_ci_low: 0.6,
        accuracy_ci_high: 0.9,
        worst_fold_accuracy: 0.75,
        cv_type: Some(CrossValidationType::ShuffledStratified),
    }
}

fn frame(id: &str, value: f32) -> PostureFrame {
    PostureFrame {
        id: id.into(),
        timestamp: 1_700_000_000_000.0,
        features: BTreeMap::from([(FeatureId::GauFeatures, vec![value; DIMENSION])]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![1, 2, 3],
        },
        keypoints: (0..17)
            .map(|index| Keypoint::new(index as f64, index as f64, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
            score: 1.0,
            width: 10.0,
            height: 10.0,
        },
        label: FrameLabel::Good,
    }
}

#[test]
fn slmd_round_trips_all_six_classifiers_and_binds_envelope_metadata() {
    let digest = [0x5a; 32];
    for model in models() {
        let bytes = encode_model(&model, ModelRole::Posture, 42, digest, Some(&metrics()))
            .expect("encode SLMD");
        assert_eq!(&bytes[..4], b"SLMD");
        let (decoded, envelope) = decode_model(&bytes).expect("decode SLMD");
        assert_eq!(
            decoded.classifier.classifier_id,
            model.classifier.classifier_id
        );
        assert_eq!(
            decoded.feature_extractor.feature_types,
            model.feature_extractor.feature_types
        );
        assert_eq!(envelope.role, ModelRole::Posture);
        assert_eq!(envelope.dataset_version, 42);
        assert_eq!(envelope.training_config_sha256, digest);
    }
}

#[test]
fn slmd_round_trips_calibrated_normalization_reusing_z_score_records() {
    // Calibrated normalization stores its mean/std through the exact same
    // `normalization.mean`/`.std` records as z-score — no new record type.
    let mut model = SerializedModel {
        feature_extractor: SerializedFeatureExtractor {
            feature_types: vec![FeatureId::GauFeatures.as_str().into()],
            normalization_mode: NormalizationMode::Calibrated,
            dim_reduction_config: MlReductionConfig {
                method: MlReductionMethod::None,
                components: DIMENSION,
            },
            concatenated_dimensions: DIMENSION,
            normalization_mean: Some(vec![0.25; DIMENSION]),
            normalization_std: Some(vec![0.5; DIMENSION]),
            dim_reduction_transformer: None,
        },
        classifier: SerializedClassifier {
            classifier_id: "mlp".into(),
            state: SerializedClassifierState::Mlp(mlp()),
        },
        trained_at: 1_700_000_000_000.0,
        version: 1.0,
    };
    model.feature_extractor.normalization_mean.as_mut().unwrap()[0] = -1.5;
    model.feature_extractor.normalization_std.as_mut().unwrap()[1] = 2.0;

    let digest = [0x5a; 32];
    let bytes =
        encode_model(&model, ModelRole::Posture, 7, digest, None).expect("encode calibrated SLMD");
    assert!(
        bytes
            .windows(b"calibrated".len())
            .any(|window| window == b"calibrated"),
        "container must tag the calibrated normalization mode"
    );

    let (decoded, envelope) = decode_model(&bytes).expect("decode calibrated SLMD");
    assert_eq!(
        decoded.feature_extractor.normalization_mode,
        NormalizationMode::Calibrated
    );
    assert_eq!(
        decoded.feature_extractor.normalization_mean,
        model.feature_extractor.normalization_mean
    );
    assert_eq!(
        decoded.feature_extractor.normalization_std,
        model.feature_extractor.normalization_std
    );
    assert_eq!(envelope.role, ModelRole::Posture);
    assert_eq!(envelope.dataset_version, 7);
}

#[test]
fn slmd_enforces_canonical_feature_order_on_encode_and_decode() {
    let ordered_features = vec![FeatureId::GauFeaturesMax, FeatureId::GauFeaturesStd];
    let ordered = model_with_features(ordered_features.clone());
    let bytes = encode_model(&ordered, ModelRole::Posture, 1, [1; 32], None)
        .expect("ordered features encode");
    assert_eq!(
        decode_model(&bytes)
            .expect("ordered features decode")
            .0
            .feature_extractor
            .feature_types,
        ordered.feature_extractor.feature_types
    );

    let reversed = model_with_features(vec![FeatureId::GauFeaturesStd, FeatureId::GauFeaturesMax]);
    assert!(encode_model(&reversed, ModelRole::Posture, 1, [1; 32], None).is_err());
    let duplicate = model_with_features(vec![FeatureId::GauFeaturesMax, FeatureId::GauFeaturesMax]);
    assert!(encode_model(&duplicate, ModelRole::Posture, 1, [1; 32], None).is_err());

    for invalid in [
        b"gau_features_std,gau_features_max".as_slice(),
        b"gau_features_max,gau_features_max".as_slice(),
    ] {
        let mut invalid_bytes = bytes.clone();
        replace_unique(
            &mut invalid_bytes,
            b"gau_features_max,gau_features_std",
            invalid,
        );
        assert!(decode_model(&invalid_bytes).is_err());
    }
}

#[test]
fn slmd_requires_javascript_safe_integer_training_timestamps() {
    const MAX_SAFE: i64 = 9_007_199_254_740_991;
    let mut model = models()[0].clone();
    model.trained_at = MAX_SAFE as f64;
    let bytes = encode_model(&model, ModelRole::Posture, 1, [1; 32], None)
        .expect("maximum safe timestamp encodes");
    assert_eq!(
        decode_model(&bytes)
            .expect("maximum safe timestamp decodes")
            .0
            .trained_at,
        MAX_SAFE as f64
    );

    model.trained_at = 1_700_000_000_000.5;
    assert!(encode_model(&model, ModelRole::Posture, 1, [1; 32], None).is_err());
    model.trained_at = (MAX_SAFE + 1) as f64;
    assert!(encode_model(&model, ModelRole::Posture, 1, [1; 32], None).is_err());

    let mut unsafe_bytes = bytes;
    replace_unique(
        &mut unsafe_bytes,
        &MAX_SAFE.to_le_bytes(),
        &(MAX_SAFE + 1).to_le_bytes(),
    );
    assert!(decode_model(&unsafe_bytes).is_err());
}

#[test]
fn slmd_rejects_magic_corruption_trailing_data_and_unknown_or_noncanonical_records() {
    let mut bytes =
        encode_model(&models()[0], ModelRole::Presence, 1, [1; 32], None).expect("encode");
    bytes[0] = b'X';
    assert!(decode_model(&bytes).is_err());

    let mut bytes =
        encode_model(&models()[0], ModelRole::Presence, 1, [1; 32], None).expect("encode");
    bytes.push(0);
    assert!(decode_model(&bytes).is_err());

    let mut bytes =
        encode_model(&models()[0], ModelRole::Presence, 1, [1; 32], None).expect("encode");
    let position = bytes
        .windows(b"classifier.id".len())
        .position(|window| window == b"classifier.id")
        .expect("classifier record");
    bytes[position + 11] = b'z';
    assert!(decode_model(&bytes).is_err());
}

#[test]
fn slcf_materializes_defaults_and_changes_when_effective_configuration_changes() {
    let implicit = settings(ClassifierId::Knn, BTreeMap::new());
    let explicit = settings(
        ClassifierId::Knn,
        BTreeMap::from([
            ("k".into(), ParameterValue::Number(3.0)),
            ("kernel".into(), ParameterValue::String("cosine".into())),
            ("gamma".into(), ParameterValue::Number(1.0)),
        ]),
    );
    assert_eq!(
        training_config_fingerprint(&implicit, true).expect("implicit fingerprint"),
        training_config_fingerprint(&explicit, true).expect("explicit fingerprint")
    );
    assert_ne!(
        training_config_fingerprint(&implicit, true).expect("CV fingerprint"),
        training_config_fingerprint(&implicit, false).expect("non-CV fingerprint")
    );
}

#[test]
fn slcf_rejects_noncanonical_feature_selections() {
    let mut invalid = settings(ClassifierId::Knn, BTreeMap::new());
    invalid.posture_feature_types = vec![FeatureId::GauFeatures, FeatureId::BackboneFeatures];
    assert!(training_config_fingerprint(&invalid, true).is_err());
    invalid.posture_feature_types = vec![FeatureId::GauFeatures, FeatureId::GauFeatures];
    assert!(training_config_fingerprint(&invalid, true).is_err());
}

#[test]
fn activation_rejects_settings_change_without_a_dataset_version_change() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    let initial = settings(ClassifierId::Knn, BTreeMap::new());
    storage.save_training_settings(&initial).expect("settings");
    let snapshot = storage.training_snapshot(true).expect("snapshot");

    let changed = settings(
        ClassifierId::Knn,
        BTreeMap::from([("k".into(), ParameterValue::Number(1.0))]),
    );
    storage
        .save_training_settings(&changed)
        .expect("changed settings");
    let pair = &models()[1];
    assert!(matches!(
        storage.save_model_pair_if_snapshot(Some(pair), Some(pair), snapshot, true, None, None),
        Err(StorageError::SnapshotChanged(_))
    ));
    assert!(storage.load_active_model_pair().expect("models").is_none());
}

#[test]
fn archive_round_trips_complete_canonical_model_pair() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("slouch-models-{nonce}.slouchpack"));
    let storage = DatasetStorage::open_in_memory().expect("storage");
    storage
        .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
        .expect("settings");
    let snapshot = storage.training_snapshot(true).expect("snapshot");
    let pair = &models()[1];
    storage
        .save_model_pair_if_snapshot(
            Some(pair),
            Some(pair),
            snapshot,
            true,
            Some(&metrics()),
            Some(&metrics()),
        )
        .expect("activate pair");
    storage.export_archive(&archive, "test").expect("export");

    let imported = DatasetStorage::open_in_memory().expect("imported storage");
    imported.import_archive(&archive).expect("import");
    let (posture, presence) = imported
        .load_active_model_pair()
        .expect("load pair")
        .expect("active pair");
    assert_eq!(
        posture.expect("posture role").classifier.classifier_id,
        "knn"
    );
    assert_eq!(
        presence.expect("presence role").classifier.classifier_id,
        "knn"
    );
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn archive_round_trips_a_posture_only_generation() {
    // Oracle parity: a run with GOOD+BAD but no AWAY frames persists a
    // posture-only generation. Such a generation must survive export/import.
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("slouch-posture-only-{nonce}.slouchpack"));
    let storage = DatasetStorage::open_in_memory().expect("storage");
    storage
        .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
        .expect("settings");
    let snapshot = storage.training_snapshot(true).expect("snapshot");
    let pair = &models()[1];
    storage
        .save_model_pair_if_snapshot(Some(pair), None, snapshot, true, Some(&metrics()), None)
        .expect("activate posture-only generation");
    storage.export_archive(&archive, "test").expect("export");

    let imported = DatasetStorage::open_in_memory().expect("imported storage");
    imported.import_archive(&archive).expect("import");
    let (posture, presence) = imported
        .load_active_model_pair()
        .expect("load pair")
        .expect("active generation");
    assert_eq!(
        posture.expect("posture role").classifier.classifier_id,
        "knn"
    );
    assert!(
        presence.is_none(),
        "posture-only generation has no presence"
    );
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn archive_rejects_noncanonical_features_and_unsafe_timestamps_without_live_mutation() {
    const MAX_SAFE: i64 = 9_007_199_254_740_991;
    let ordered_ids = b"gau_features_max,gau_features_std";
    for variant in 0..3 {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let archive =
            std::env::temp_dir().join(format!("slouch-invalid-model-{nonce}-{variant}.slouchpack"));
        let source = DatasetStorage::open_in_memory().expect("source");
        source
            .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
            .expect("source settings");
        let snapshot = source.training_snapshot(true).expect("source snapshot");
        let ordered =
            model_with_features(vec![FeatureId::GauFeaturesMax, FeatureId::GauFeaturesStd]);
        source
            .save_model_pair_if_snapshot(
                Some(&ordered),
                Some(&ordered),
                snapshot,
                true,
                Some(&metrics()),
                Some(&metrics()),
            )
            .expect("source model");
        source.export_archive(&archive, "test").expect("export");
        rewrite_archive_posture_payload(&archive, |payload| match variant {
            0 => replace_unique(payload, ordered_ids, b"gau_features_std,gau_features_max"),
            1 => replace_unique(payload, ordered_ids, b"gau_features_max,gau_features_max"),
            _ => replace_unique(
                payload,
                &1_700_000_000_000_i64.to_le_bytes(),
                &(MAX_SAFE + 1).to_le_bytes(),
            ),
        });

        let target = DatasetStorage::open_in_memory().expect("target");
        target
            .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
            .expect("target settings");
        let target_snapshot = target.training_snapshot(true).expect("target snapshot");
        let baseline = &models()[1];
        target
            .save_model_pair_if_snapshot(
                Some(baseline),
                Some(baseline),
                target_snapshot,
                true,
                Some(&metrics()),
                Some(&metrics()),
            )
            .expect("target model");
        let generation = target
            .active_model_generation_id()
            .expect("target generation")
            .expect("active target generation");

        assert!(target.import_archive(&archive).is_err());
        assert_eq!(
            target
                .active_model_generation_id()
                .expect("generation after rejected import"),
            Some(generation)
        );
        assert_eq!(
            target
                .load_active_model_pair()
                .expect("target pair")
                .expect("active target pair")
                .0
                .expect("posture role")
                .classifier
                .classifier_id,
            "knn"
        );
        fs::remove_file(archive).expect("remove archive");
    }
}

#[test]
fn archive_rejects_model_checksum_corruption_without_live_mutation() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("slouch-model-corrupt-{nonce}.slouchpack"));
    let source = DatasetStorage::open_in_memory().expect("source");
    source
        .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
        .expect("settings");
    let snapshot = source.training_snapshot(true).expect("snapshot");
    let pair = &models()[1];
    source
        .save_model_pair_if_snapshot(
            Some(pair),
            Some(pair),
            snapshot,
            true,
            Some(&metrics()),
            Some(&metrics()),
        )
        .expect("activate pair");
    source.export_archive(&archive, "test").expect("export");
    let connection = Connection::open(&archive).expect("archive");
    connection
        .execute(
            "UPDATE models SET payload_sha256 = zeroblob(32) WHERE role = 'posture'",
            [],
        )
        .expect("corrupt model checksum");
    drop(connection);

    let target = DatasetStorage::open_in_memory().expect("target");
    target
        .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
        .expect("target settings");
    assert!(target.import_archive(&archive).is_err());
    assert!(target
        .load_active_model_pair()
        .expect("target pair")
        .is_none());
    assert!(target
        .get_training_settings()
        .expect("target settings")
        .is_some());
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn activation_rejects_same_version_dataset_replacement_with_a_different_identity() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("slouch-identity-{nonce}.slouchpack"));
    let training_settings = settings(ClassifierId::Knn, BTreeMap::new());

    let source = DatasetStorage::open_in_memory().expect("source");
    source
        .save_training_settings(&training_settings)
        .expect("source settings");
    source
        .save_frame(frame("replacement", 0.75))
        .expect("source frame");
    source
        .export_archive(&archive, "test")
        .expect("source archive");

    let target = DatasetStorage::open_in_memory().expect("target");
    target
        .save_training_settings(&training_settings)
        .expect("target settings");
    target
        .save_frame(frame("original", 0.25))
        .expect("target frame");
    let snapshot = target.training_snapshot(true).expect("snapshot");
    assert_eq!(snapshot.dataset_version, 1);
    target
        .import_archive(&archive)
        .expect("replace with same version");
    assert_eq!(
        target.load_dataset().expect("replacement dataset").version,
        1
    );

    let pair = &models()[1];
    assert!(matches!(
        target.save_model_pair_if_snapshot(Some(pair), Some(pair), snapshot, true, None, None),
        Err(StorageError::SnapshotChanged(_))
    ));
    assert!(target.load_active_model_pair().expect("models").is_none());
    fs::remove_file(archive).expect("remove archive");
}

#[test]
fn dataset_only_reset_deactivates_stale_models_and_preserves_settings() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    let training_settings = settings(ClassifierId::Mlp, BTreeMap::new());
    storage
        .save_training_settings(&training_settings)
        .expect("training settings");
    let camera_settings = CameraSettings {
        camera_width: 1280,
        ..CameraSettings::default()
    };
    storage
        .save_camera_settings(&camera_settings)
        .expect("camera settings");
    let ui_settings = UiSettings {
        alert_delay_seconds: 12.0,
        ..UiSettings::default()
    };
    storage.save_ui_settings(&ui_settings).expect("UI settings");
    storage.save_frame(frame("captured", 0.25)).expect("frame");
    let snapshot = storage.training_snapshot(true).expect("snapshot");
    let pair = &models()[0];
    storage
        .save_model_pair_if_snapshot(
            Some(pair),
            Some(pair),
            snapshot,
            true,
            Some(&metrics()),
            Some(&metrics()),
        )
        .expect("active model pair");
    assert!(storage
        .active_model_generation_id()
        .expect("generation")
        .is_some());

    storage.clear_dataset().expect("dataset-only reset");

    assert!(storage.load_dataset().expect("dataset").frames.is_empty());
    assert_eq!(
        storage
            .get_training_settings()
            .expect("settings after reset"),
        Some(training_settings)
    );
    assert_eq!(
        storage.get_camera_settings().expect("camera after reset"),
        camera_settings
    );
    assert_eq!(
        storage.get_ui_settings().expect("UI after reset"),
        ui_settings
    );
    assert_eq!(
        storage
            .active_model_generation_id()
            .expect("generation after reset"),
        None,
    );
    assert!(storage
        .load_active_model_pair()
        .expect("models after reset")
        .is_none());
}

#[test]
fn complete_reset_removes_models_and_all_settings() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    storage
        .save_training_settings(settings(ClassifierId::Mlp, BTreeMap::new()))
        .expect("training settings");
    storage
        .save_camera_settings(&CameraSettings {
            camera_width: 1280,
            ..CameraSettings::default()
        })
        .expect("camera settings");
    storage
        .save_ui_settings(&UiSettings {
            alert_delay_seconds: 12.0,
            ..UiSettings::default()
        })
        .expect("UI settings");
    storage.save_frame(frame("captured", 0.25)).expect("frame");
    let snapshot = storage.training_snapshot(true).expect("snapshot");
    let pair = &models()[0];
    storage
        .save_model_pair_if_snapshot(
            Some(pair),
            Some(pair),
            snapshot,
            true,
            Some(&metrics()),
            Some(&metrics()),
        )
        .expect("active model pair");

    storage.reset_all().expect("complete reset");

    assert!(storage.load_dataset().expect("dataset").frames.is_empty());
    assert!(storage
        .load_active_model_pair()
        .expect("models after reset")
        .is_none());
    assert!(storage
        .get_training_settings()
        .expect("training settings after reset")
        .is_none());
    assert_eq!(
        storage.get_camera_settings().expect("camera after reset"),
        CameraSettings::default()
    );
    assert_eq!(
        storage.get_ui_settings().expect("UI after reset"),
        UiSettings::default()
    );
}

#[test]
fn posture_only_generation_persists_across_restart_and_restores() {
    // Oracle parity: GOOD+BAD without AWAY yields a posture-only generation. It
    // must persist across an app restart (reopen) and be restorable (rollback
    // guard), even though it carries no presence role.
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("slouch-posture-only-db-{nonce}.sqlite"));

    let generation = {
        let storage = DatasetStorage::open(&db_path).expect("storage");
        storage
            .save_training_settings(settings(ClassifierId::Knn, BTreeMap::new()))
            .expect("settings");
        let snapshot = storage.training_snapshot(true).expect("snapshot");
        let posture = &models()[1];
        storage
            .save_model_pair_if_snapshot(
                Some(posture),
                None,
                snapshot,
                true,
                Some(&metrics()),
                None,
            )
            .expect("save posture-only generation");

        // Same-process round-trip: posture present, presence absent.
        let (posture_role, presence_role) = storage
            .load_active_model_pair()
            .expect("load")
            .expect("active generation");
        assert_eq!(
            posture_role.expect("posture role").classifier.classifier_id,
            "knn"
        );
        assert!(presence_role.is_none());
        storage
            .active_model_generation_id()
            .expect("generation id")
            .expect("active generation id")
    };

    // Reopen the database (simulating an app restart).
    let restarted = DatasetStorage::open(&db_path).expect("reopen storage");
    let (posture_role, presence_role) = restarted
        .load_active_model_pair()
        .expect("load after restart")
        .expect("active generation after restart");
    assert_eq!(
        posture_role.expect("posture role").classifier.classifier_id,
        "knn"
    );
    assert!(presence_role.is_none());

    // Supersede with a full pair, then roll back to the posture-only generation.
    let snapshot2 = restarted.training_snapshot(true).expect("snapshot 2");
    let full = &models()[0];
    restarted
        .save_model_pair_if_snapshot(
            Some(full),
            Some(full),
            snapshot2,
            true,
            Some(&metrics()),
            Some(&metrics()),
        )
        .expect("save full pair");
    restarted
        .restore_active_model_generation(generation)
        .expect("restore posture-only generation");
    let (posture_role, presence_role) = restarted
        .load_active_model_pair()
        .expect("load after restore")
        .expect("active generation after restore");
    assert_eq!(
        posture_role.expect("posture role").classifier.classifier_id,
        "knn"
    );
    assert!(
        presence_role.is_none(),
        "restored generation must remain posture-only"
    );

    // Release the connection before deleting the backing file.
    drop(restarted);
    let _ = fs::remove_file(db_path);
}
