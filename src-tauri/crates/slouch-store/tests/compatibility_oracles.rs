use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};
use slouch_domain::{BoundingBox, FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail};
use slouch_ml::ported::{
    classifier_factory::deserialize_classifier,
    feature_extractor::FeatureExtractor,
    types::{SerializedClassifier, SerializedFeatureExtractor, SerializedMlp},
};
use slouch_store::ported::{
    model_format::{decode_model, encode_model, ModelRole},
    operations::{DatasetOperations, DatasetStore, ImportArchive},
    storage::DatasetStorage,
};

fn fixture(relative: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(relative);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert!(value.len().is_multiple_of(2));
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(text, 16).unwrap()
        })
        .collect()
}

fn f32_sha(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    format!("{:x}", Sha256::digest(bytes))
}

fn case_by_operation<'a>(oracle: &'a Value, operation: &str) -> &'a Value {
    oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["operation"] == operation)
        .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance || difference <= tolerance * expected.abs(),
        "actual {actual} differs from expected {expected}"
    );
}

fn assert_stats(actual: &slouch_domain::DatasetStats, expected: &Value) {
    assert_eq!(actual.total, expected["total"].as_u64().unwrap() as usize);
    assert_eq!(actual.good, expected["good"].as_u64().unwrap() as usize);
    assert_eq!(actual.bad, expected["bad"].as_u64().unwrap() as usize);
    assert_eq!(actual.away, expected["away"].as_u64().unwrap() as usize);
    assert_eq!(actual.unused, expected["unused"].as_u64().unwrap() as usize);
    assert_eq!(
        actual.has_minimum_frames,
        expected["hasMinimumFrames"].as_bool().unwrap()
    );
    assert_eq!(
        actual.has_away_frames,
        expected["hasAwayFrames"].as_bool().unwrap()
    );
}

fn frame(id: &str, label: FrameLabel, offset: f32) -> PostureFrame {
    let ids = [
        FeatureId::RtmDetExtracted,
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
    ];
    let features = ids
        .into_iter()
        .map(|feature| {
            let values = (0..feature.metadata().dimensions)
                .map(|index| offset + index as f32 / 1024.0)
                .collect();
            (feature, values)
        })
        .collect::<BTreeMap<_, _>>();
    PostureFrame {
        id: id.to_owned(),
        timestamp: 1_735_689_601_000.0 + f64::from(offset),
        features,
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: vec![82, 73, 70, 70, offset as u8],
        },
        keypoints: (0..17)
            .map(|index| Keypoint::new(index as f64 / 20.0, index as f64 / 25.0, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 0.1,
            y1: 0.2,
            x2: 0.8,
            y2: 0.9,
            score: 0.95,
            width: 0.7,
            height: 0.7,
        },
        label,
    }
}

struct StorageAdapter<'a>(&'a DatasetStorage);

impl DatasetStore for StorageAdapter<'_> {
    type Error = String;

    fn update_frame_label(&self, id: &str, label: FrameLabel) -> Result<(), Self::Error> {
        self.0
            .update_frame_label(id, label)
            .map_err(|error| error.to_string())
    }
    fn remove_frames_by_label(&self, label: FrameLabel) -> Result<usize, Self::Error> {
        self.0
            .remove_frames_by_label(label)
            .map_err(|error| error.to_string())
    }
    fn clear_dataset(&self) -> Result<(), Self::Error> {
        self.0.clear_dataset().map_err(|error| error.to_string())
    }
    fn clear_training_settings(&self) -> Result<(), Self::Error> {
        self.0
            .clear_training_settings()
            .map_err(|error| error.to_string())
    }
    fn clear_posture_model(&self) -> Result<(), Self::Error> {
        self.0
            .clear_posture_model()
            .map_err(|error| error.to_string())
    }
    fn clear_presence_model(&self) -> Result<(), Self::Error> {
        self.0
            .clear_presence_model()
            .map_err(|error| error.to_string())
    }
    fn get_stats(&self) -> Result<slouch_domain::DatasetStats, Self::Error> {
        self.0.get_stats().map_err(|error| error.to_string())
    }
    fn load_dataset(&self) -> Result<slouch_domain::PostureDataset, Self::Error> {
        self.0.load_dataset().map_err(|error| error.to_string())
    }
    fn get_frames_by_label(&self, label: FrameLabel) -> Result<Vec<PostureFrame>, Self::Error> {
        self.0
            .get_frames_by_label(label)
            .map_err(|error| error.to_string())
    }
    fn get_frame_by_id(&self, id: &str) -> Result<Option<PostureFrame>, Self::Error> {
        self.0
            .get_frame_by_id(id)
            .map_err(|error| error.to_string())
    }
    fn needs_retraining(&self) -> Result<bool, Self::Error> {
        self.0.needs_retraining().map_err(|error| error.to_string())
    }
    fn export_dataset(
        &self,
        _dataset: &slouch_domain::PostureDataset,
        _filename: Option<&str>,
    ) -> Result<(), Self::Error> {
        Err("archive path is integration-owned".into())
    }
    fn import_dataset(
        &self,
        _archive: ImportArchive<'_>,
    ) -> Result<slouch_domain::ImportResult, Self::Error> {
        Err("archive path is integration-owned".into())
    }
}

fn assert_bulk(actual: &slouch_store::ported::operations::BulkOperationResult, expected: &Value) {
    assert_eq!(
        actual.deleted,
        expected["deleted"].as_u64().unwrap() as usize
    );
    assert_eq!(actual.success, expected["success"].as_bool().unwrap());
    assert_eq!(
        actual.error.as_deref(),
        expected.get("error").and_then(Value::as_str)
    );
}

fn operation_case<'a>(oracle: &'a Value, id: &str) -> &'a Value {
    oracle["operationCases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == id)
        .unwrap()
}

fn gau_frame(sample: usize) -> PostureFrame {
    let mut value = frame("extractor-probe", FrameLabel::Good, 0.0);
    value.features = BTreeMap::from([(
        FeatureId::GauFeatures,
        (0..256)
            .map(|feature| sample as f32 * 0.1 + feature as f32 / 512.0)
            .collect(),
    )]);
    value
}

#[test]
fn store_operation_fixture_is_replayed_by_native_storage_with_tightening() {
    let oracle = fixture("store/store-operations-v1.json");
    assert_eq!(
        oracle["environment"]["backend"],
        "localforage 1.10.0 over fake-indexeddb 6.2.3"
    );
    assert_eq!(oracle["dimensions"]["rtmdet_extracted"], 384);
    assert_eq!(oracle["dimensions"]["gau_features"], 256);
    assert_eq!(oracle["dimensions"]["backbone_features"], 768);

    let storage = DatasetStorage::open_in_memory().unwrap();
    let empty = storage.load_dataset().unwrap();
    let expected = case_by_operation(&oracle, "load-empty");
    assert_eq!(empty.version, expected["version"].as_u64().unwrap());
    assert!(empty.frames.is_empty());

    storage
        .save_frame(frame("frame-a", FrameLabel::Good, 1.0))
        .unwrap();
    let first = storage.load_dataset().unwrap();
    let expected = case_by_operation(&oracle, "save-frame-a");
    assert_eq!(first.version, expected["version"].as_u64().unwrap());
    assert_eq!(
        first
            .frames
            .iter()
            .map(|frame| &frame.id)
            .collect::<Vec<_>>(),
        vec!["frame-a"]
    );
    for (feature, hash) in expected["featureHashes"].as_object().unwrap() {
        let id = feature.parse::<FeatureId>().unwrap();
        assert_eq!(
            f32_sha(&first.frames[0].features[&id]),
            hash.as_str().unwrap()
        );
    }

    storage
        .save_frame(frame("frame-b", FrameLabel::Bad, 2.0))
        .unwrap();
    let expected = case_by_operation(&oracle, "save-frame-b");
    assert_stats(&storage.get_stats().unwrap(), &expected["stats"]);
    assert_eq!(
        storage.get_frame_by_id("frame-b").unwrap().unwrap().id,
        expected["frameById"].as_str().unwrap()
    );
    assert_eq!(
        storage
            .get_frames_by_label(FrameLabel::Bad)
            .unwrap()
            .iter()
            .map(|frame| &frame.id)
            .collect::<Vec<_>>(),
        vec!["frame-b"]
    );

    storage
        .save_frame(frame("frame-a", FrameLabel::Away, 3.0))
        .unwrap();
    let expected = case_by_operation(&oracle, "overwrite-frame-a");
    let overwritten = storage.load_dataset().unwrap();
    assert_eq!(overwritten.version, expected["version"].as_u64().unwrap());
    assert_eq!(
        storage.get_frame_by_id("frame-a").unwrap().unwrap().label,
        FrameLabel::Away
    );
    assert_stats(&storage.get_stats().unwrap(), &expected["stats"]);

    storage
        .update_frame_label("frame-b", FrameLabel::Good)
        .unwrap();
    let expected = case_by_operation(&oracle, "update-frame-b-good");
    assert_stats(&storage.get_stats().unwrap(), &expected["stats"]);

    storage
        .update_frame_label("frame-b", FrameLabel::Unused)
        .unwrap();
    let expected = case_by_operation(&oracle, "update-unused-deletes");
    let after_delete = storage.load_dataset().unwrap();
    assert_eq!(after_delete.version, expected["version"].as_u64().unwrap());
    assert_eq!(
        after_delete
            .frames
            .iter()
            .map(|frame| &frame.id)
            .collect::<Vec<_>>(),
        vec!["frame-a"]
    );

    storage.clear_dataset().unwrap();
    let expected = case_by_operation(&oracle, "clear-dataset");
    let cleared = storage.load_dataset().unwrap();
    assert_eq!(cleared.version, expected["version"].as_u64().unwrap());
    assert!(cleared.frames.is_empty());

    let mut malformed = frame("wrong-dimension", FrameLabel::Good, 4.0);
    malformed
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 3]);
    assert!(storage.save_frame(malformed).is_err());

    let operation_storage = DatasetStorage::open_in_memory().unwrap();
    let operations = DatasetOperations::new(StorageAdapter(&operation_storage));
    assert!(operations.get_frame_by_id("missing").unwrap().is_none());
    assert!(operation_case(&oracle, "missing-id-get")["result"].is_null());
    let native_missing_error = operations
        .update_frame_label("missing", FrameLabel::Good)
        .unwrap_err()
        .to_string();
    assert!(native_missing_error.starts_with("Failed to update frame label:"));
    assert_eq!(operation_case(&oracle, "missing-id-update")["ok"], false);
    assert!(operation_case(&oracle, "missing-id-update")["error"]
        .as_str()
        .unwrap()
        .starts_with("Failed to update frame label:"));

    operation_storage
        .save_frame(frame("bulk-a", FrameLabel::Good, 4.0))
        .unwrap();
    operation_storage
        .save_frame(frame("bulk-b", FrameLabel::Bad, 5.0))
        .unwrap();
    operation_storage
        .save_frame(frame("bulk-c", FrameLabel::Away, 6.0))
        .unwrap();
    let all_success = operation_case(&oracle, "delete-bulk-all-success");
    assert_eq!(
        all_success["result"],
        serde_json::json!({ "deleted": 2, "success": true })
    );
    assert_bulk(
        &operations.delete_bulk(["bulk-a", "bulk-b"]).unwrap(),
        &all_success["nativeResult"],
    );
    let partial = operation_case(&oracle, "delete-bulk-partial-failure");
    assert_eq!(
        partial["result"],
        serde_json::json!({ "deleted": 2, "success": true })
    );
    assert_bulk(
        &operations.delete_bulk(["bulk-c", "missing"]).unwrap(),
        &partial["nativeResult"],
    );
    let all_failure = operation_case(&oracle, "delete-bulk-all-failure");
    assert_eq!(
        all_failure["result"],
        serde_json::json!({ "deleted": 2, "success": true })
    );
    assert_bulk(
        &operations.delete_bulk(["missing-a", "missing-b"]).unwrap(),
        &all_failure["nativeResult"],
    );
    let empty = operation_case(&oracle, "delete-bulk-empty");
    assert_eq!(
        empty["result"],
        serde_json::json!({ "deleted": 0, "success": false })
    );
    assert_bulk(
        &operations.delete_bulk(Vec::<String>::new()).unwrap(),
        &empty["nativeResult"],
    );
    assert_bulk(
        &operations.cleanup_unused().unwrap(),
        &operation_case(&oracle, "cleanup-unused-zero")["result"],
    );
    assert_bulk(
        &operations.delete_by_label(FrameLabel::Bad).unwrap(),
        &operation_case(&oracle, "delete-by-label-zero")["result"],
    );
    operation_storage
        .save_frame(frame("label-good-a", FrameLabel::Good, 7.0))
        .unwrap();
    operation_storage
        .save_frame(frame("label-good-b", FrameLabel::Good, 8.0))
        .unwrap();
    assert_bulk(
        &operations.delete_by_label(FrameLabel::Good).unwrap(),
        &operation_case(&oracle, "delete-by-label-success")["result"],
    );
    assert_eq!(
        operations.needs_retraining().unwrap(),
        operation_case(&oracle, "needs-retraining-without-model")["result"]
            .as_bool()
            .unwrap()
    );
    operations.reset_dataset().unwrap();
    assert!(operations.load_dataset().unwrap().frames.is_empty());
    assert_eq!(
        operation_case(&oracle, "reset-dataset")["frameIds"],
        serde_json::json!([])
    );

    let expected_operation_ids = oracle["operationCases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    let consumed_operation_ids = BTreeSet::from([
        "missing-id-get",
        "missing-id-update",
        "delete-bulk-all-success",
        "delete-bulk-partial-failure",
        "delete-bulk-all-failure",
        "delete-bulk-empty",
        "cleanup-unused-zero",
        "delete-by-label-zero",
        "delete-by-label-success",
        "needs-retraining-without-model",
        "reset-dataset",
    ]);
    assert_eq!(consumed_operation_ids, expected_operation_ids);
}

#[test]
fn independently_typescript_encoded_model_envelopes_decode_and_hash_exactly() {
    let oracle = fixture("models/model-envelope-v1.json");
    assert_eq!(
        oracle["jsonContract"]["modelToJsonKeys"],
        serde_json::json!(["featureExtractor", "classifier"])
    );
    assert_eq!(
        oracle["jsonContract"]["callerAddedKeys"],
        serde_json::json!(["trainedAt", "version"])
    );

    let mut consumed_classifiers = BTreeSet::new();
    for case in oracle["cases"].as_array().unwrap() {
        let serialized: SerializedClassifier =
            serde_json::from_value(case["modelJson"]["classifier"].clone()).unwrap();
        assert_eq!(serialized.classifier_id, case["classifierId"]);
        let classifier_id = serialized.classifier_id.clone();
        let classifier =
            deserialize_classifier(&classifier_id, serialized.state, &BTreeMap::new()).unwrap();
        let probes = case["probes"].as_array().unwrap();
        assert_eq!(
            probes.len(),
            case["probabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            probes.len(),
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        for ((probe, expected), loaded_expected) in probes
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
            .zip(case["loadedProbabilities"].as_array().unwrap())
        {
            let probe = probe
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let actual = classifier.predict_proba(&probe).unwrap();
            close(actual, expected.as_f64().unwrap(), 2e-4);
            close(actual, loaded_expected.as_f64().unwrap(), 2e-4);
        }
        consumed_classifiers.insert(classifier_id);
    }
    assert_eq!(
        consumed_classifiers,
        BTreeSet::from([
            "gaussian_nb".to_owned(),
            "kmeans_logistic".to_owned(),
            "kmeans_prototype".to_owned(),
            "knn".to_owned(),
            "mlp".to_owned(),
            "svm".to_owned(),
        ])
    );

    let mut consumed_extractors = BTreeSet::new();
    for case in oracle["extractorVariants"].as_array().unwrap() {
        let state: SerializedFeatureExtractor =
            serde_json::from_value(case["state"].clone()).unwrap();
        let extractor = FeatureExtractor::<PostureFrame>::from_json(state).unwrap();
        let actual = extractor.transform(&gau_frame(3)).unwrap();
        assert_eq!(actual.len(), case["transformed"].as_array().unwrap().len());
        assert_eq!(
            actual.len(),
            case["loadedTransform"].as_array().unwrap().len()
        );
        for (index, ((actual, expected), loaded_expected)) in actual
            .iter()
            .zip(case["transformed"].as_array().unwrap())
            .zip(case["loadedTransform"].as_array().unwrap())
            .enumerate()
        {
            let id = case["id"].as_str().unwrap();
            let expected = expected.as_f64().unwrap();
            let loaded_expected = loaded_expected.as_f64().unwrap();
            let actual = f64::from(*actual);
            assert!(
                (actual - expected).abs() <= 2e-6
                    || (actual - expected).abs() <= 2e-6 * expected.abs(),
                "extractor {id} lane {index}: actual {actual} differs from expected {expected}",
            );
            assert!(
                (actual - loaded_expected).abs() <= 2e-6 ||
                    (actual - loaded_expected).abs() <= 2e-6 * loaded_expected.abs(),
                "extractor {id} lane {index}: actual {actual} differs from loaded expected {loaded_expected}",
            );
        }
        consumed_extractors.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(
        consumed_extractors,
        BTreeSet::from([
            "layer-none".to_owned(),
            "none-none".to_owned(),
            "none-pca".to_owned(),
            "none-random-projection".to_owned(),
            "zscore-none".to_owned(),
        ])
    );

    let index = fixture("models/model-envelope-v1-binaries.json");
    let expected_binary_ids = index["binaries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut consumed_binary_ids = BTreeSet::new();
    for item in index["binaries"].as_array().unwrap() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(
                item["path"]
                    .as_str()
                    .unwrap()
                    .strip_prefix("src-tauri/")
                    .unwrap(),
            );
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len(), item["bytes"].as_u64().unwrap() as usize);
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            item["sha256"].as_str().unwrap()
        );
        consumed_binary_ids.insert(item["id"].as_str().unwrap().to_owned());
        if item["id"] == "training-config" {
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                oracle["trainingConfigSha256"]
            );
            continue;
        }
        let (model, metadata) = decode_model(&bytes).unwrap();
        assert_eq!(metadata.dataset_version, 7);
        let reencoded = encode_model(
            &model,
            metadata.role,
            metadata.dataset_version,
            metadata.training_config_sha256,
            None,
        )
        .unwrap();
        assert_eq!(
            reencoded, bytes,
            "{} must re-encode byte-exactly",
            item["id"]
        );
        if item["id"] == "nonsquare-mlp" {
            assert_eq!(metadata.classifier_id, "mlp");
            let state = match model.classifier.state {
                slouch_ml::ported::types::SerializedClassifierState::Mlp(state) => state,
                _ => panic!("nonsquare fixture must decode as MLP"),
            };
            let expected: SerializedMlp = SerializedMlp {
                layer_weights: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                layer_biases: vec![vec![0.1_f32 as f64, 0.2_f32 as f64, 0.3_f32 as f64]],
                layer_shapes: vec![2, 3],
                hidden_layers: 0,
                hidden_size: 64,
                class_weights: [1.0, 1.0],
            };
            assert_eq!(state, expected);
            assert_eq!(
                oracle["nonsquareMlpTranspose"]["persistedOutIn"],
                serde_json::json!([1, 4, 2, 5, 3, 6])
            );
        } else {
            assert_eq!(metadata.classifier_id, "svm");
            let expected_role = if item["id"] == "presence-svm" {
                ModelRole::Presence
            } else {
                ModelRole::Posture
            };
            assert_eq!(metadata.role, expected_role);
        }
    }
    assert_eq!(consumed_binary_ids, expected_binary_ids);

    let expected_invalid_ids = oracle["invalidCases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut consumed_invalid_ids = BTreeSet::new();
    for case in oracle["invalidCases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        match case["nativeOutcome"].as_str().unwrap() {
            "reject" => {
                let bytes = decode_hex(case["bytesHex"].as_str().unwrap());
                assert_eq!(bytes.len(), case["bytes"].as_u64().unwrap() as usize);
                assert_eq!(format!("{:x}", Sha256::digest(&bytes)), case["sha256"]);
                assert!(decode_model(&bytes).is_err(), "{id} must be rejected");
            }
            "sha256-mismatch" => {
                let bytes = decode_hex(case["bytesHex"].as_str().unwrap());
                assert_ne!(format!("{:x}", Sha256::digest(&bytes)), case["sha256"]);
                assert!(
                    decode_model(&bytes).is_ok(),
                    "checksum verification is the storage boundary"
                );
            }
            "reject-same-role-pair" => {
                let first = decode_model(&decode_hex(case["firstHex"].as_str().unwrap()))
                    .unwrap()
                    .1;
                let second = decode_model(&decode_hex(case["secondHex"].as_str().unwrap()))
                    .unwrap()
                    .1;
                assert_eq!(
                    first.role, second.role,
                    "fixture must concretely encode the invalid same-role pair"
                );
                assert_eq!(first.dataset_version, second.dataset_version);
                assert_eq!(first.training_config_sha256, second.training_config_sha256);
            }
            outcome => panic!("unconsumed model corruption outcome {outcome}"),
        }
        consumed_invalid_ids.insert(id.to_owned());
    }
    assert_eq!(consumed_invalid_ids, expected_invalid_ids);
}
