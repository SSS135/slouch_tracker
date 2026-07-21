use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use slouch_domain::{
    BoundingBox, DatasetStats, FeatureId, FeatureMap, FrameLabel, Keypoint, PostureDataset,
    PostureFrame, Thumbnail,
};
use slouch_store::ported::feature_reservoir::{FeatureReservoir, ReservoirSample};
use slouch_store::ported::operations::{
    BulkOperationResult, DatasetOperations, DatasetStore, NoopDatasetLogger,
};

#[derive(Clone, Default)]
struct MockStorage {
    state: Arc<Mutex<MockState>>,
}

struct MockState {
    update_errors: Vec<Option<String>>,
    remove_result: Result<usize, String>,
    clear_dataset_error: Option<String>,
    clear_posture_error: Option<String>,
    clear_presence_error: Option<String>,
    clear_settings_error: Option<String>,
    stats: Option<Result<DatasetStats, String>>,
    dataset: Option<Result<PostureDataset, String>>,
    frames_by_label: Option<Result<Vec<PostureFrame>, String>>,
    frame_by_id: Option<Result<Option<PostureFrame>, String>>,
    retraining: Option<Result<bool, String>>,
    updated_labels: Vec<(String, FrameLabel)>,
    removed_labels: Vec<FrameLabel>,
    queried_labels: Vec<FrameLabel>,
    clear_dataset_calls: usize,
    clear_posture_calls: usize,
    clear_presence_calls: usize,
    clear_settings_calls: usize,
}

impl Default for MockState {
    fn default() -> Self {
        Self {
            update_errors: Vec::new(),
            remove_result: Ok(0),
            clear_dataset_error: None,
            clear_posture_error: None,
            clear_presence_error: None,
            clear_settings_error: None,
            stats: None,
            dataset: None,
            frames_by_label: None,
            frame_by_id: None,
            retraining: None,
            updated_labels: Vec::new(),
            removed_labels: Vec::new(),
            queried_labels: Vec::new(),
            clear_dataset_calls: 0,
            clear_posture_calls: 0,
            clear_presence_calls: 0,
            clear_settings_calls: 0,
        }
    }
}

impl MockStorage {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                remove_result: Ok(0),
                ..MockState::default()
            })),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, MockState> {
        self.state
            .lock()
            .expect("mock storage lock should not be poisoned")
    }

    fn set_update_outcomes(&self, outcomes: impl IntoIterator<Item = Result<(), String>>) {
        self.state().update_errors = outcomes.into_iter().map(|result| result.err()).collect();
    }

    fn set_remove_result(&self, result: Result<usize, String>) {
        self.state().remove_result = result;
    }

    fn set_stats(&self, result: Result<DatasetStats, String>) {
        self.state().stats = Some(result);
    }

    fn set_dataset(&self, result: Result<PostureDataset, String>) {
        self.state().dataset = Some(result);
    }

    fn set_frames_by_label(&self, result: Result<Vec<PostureFrame>, String>) {
        self.state().frames_by_label = Some(result);
    }

    fn set_frame_by_id(&self, result: Result<Option<PostureFrame>, String>) {
        self.state().frame_by_id = Some(result);
    }

    fn set_retraining(&self, result: Result<bool, String>) {
        self.state().retraining = Some(result);
    }

    fn fail_clear_dataset(&self, message: &str) {
        self.state().clear_dataset_error = Some(message.to_owned());
    }

    fn updated_labels(&self) -> Vec<(String, FrameLabel)> {
        self.state().updated_labels.clone()
    }

    fn removed_labels(&self) -> Vec<FrameLabel> {
        self.state().removed_labels.clone()
    }

    fn queried_labels(&self) -> Vec<FrameLabel> {
        self.state().queried_labels.clone()
    }

    fn clear_call_counts(&self) -> (usize, usize, usize, usize) {
        let state = self.state();
        (
            state.clear_dataset_calls,
            state.clear_posture_calls,
            state.clear_presence_calls,
            state.clear_settings_calls,
        )
    }
}

impl DatasetStore for MockStorage {
    type Error = String;

    fn update_frame_label(&self, id: &str, label: FrameLabel) -> Result<(), String> {
        let mut state = self.state();
        let outcome = if state.update_errors.is_empty() {
            Ok(())
        } else {
            state.update_errors.remove(0).map_or(Ok(()), Err)
        };
        if outcome.is_ok() {
            state.updated_labels.push((id.to_owned(), label));
        }
        outcome
    }

    fn remove_frames_by_label(&self, label: FrameLabel) -> Result<usize, String> {
        let mut state = self.state();
        state.removed_labels.push(label);
        state.remove_result.clone()
    }

    fn clear_dataset(&self) -> Result<(), String> {
        let mut state = self.state();
        state.clear_dataset_calls += 1;
        state.clear_dataset_error.clone().map_or(Ok(()), Err)
    }

    fn clear_posture_model(&self) -> Result<(), String> {
        let mut state = self.state();
        state.clear_posture_calls += 1;
        state.clear_posture_error.clone().map_or(Ok(()), Err)
    }

    fn clear_presence_model(&self) -> Result<(), String> {
        let mut state = self.state();
        state.clear_presence_calls += 1;
        state.clear_presence_error.clone().map_or(Ok(()), Err)
    }

    fn clear_training_settings(&self) -> Result<(), String> {
        let mut state = self.state();
        state.clear_settings_calls += 1;
        state.clear_settings_error.clone().map_or(Ok(()), Err)
    }

    fn get_stats(&self) -> Result<DatasetStats, String> {
        self.state()
            .stats
            .clone()
            .unwrap_or_else(|| Err("stats not configured".to_owned()))
    }

    fn load_dataset(&self) -> Result<PostureDataset, String> {
        self.state()
            .dataset
            .clone()
            .unwrap_or_else(|| Err("dataset not configured".to_owned()))
    }

    fn get_frames_by_label(&self, label: FrameLabel) -> Result<Vec<PostureFrame>, String> {
        let mut state = self.state();
        state.queried_labels.push(label);
        state
            .frames_by_label
            .clone()
            .unwrap_or_else(|| Err("frames not configured".to_owned()))
    }

    fn get_frame_by_id(&self, id: &str) -> Result<Option<PostureFrame>, String> {
        let configured = self.state().frame_by_id.clone();
        match configured {
            Some(result) => Ok(result?.filter(|frame| frame.id == id)),
            // Unconfigured mocks report every id as existing: the tightened native
            // delete path consults get_frame_by_id to count only real deletions.
            None => Ok(Some(create_mock_frame(id, FrameLabel::Good))),
        }
    }

    fn needs_retraining(&self) -> Result<bool, String> {
        self.state()
            .retraining
            .clone()
            .unwrap_or_else(|| Err("retraining not configured".to_owned()))
    }

    fn export_dataset(
        &self,
        _dataset: &PostureDataset,
        _filename: Option<&str>,
    ) -> Result<(), String> {
        Err("export not configured".to_owned())
    }

    fn import_dataset(
        &self,
        _archive: slouch_store::ported::operations::ImportArchive<'_>,
    ) -> Result<slouch_domain::ImportResult, String> {
        Err("import not configured".to_owned())
    }
}

fn create_mock_frame(id: &str, label: FrameLabel) -> PostureFrame {
    let mut features = FeatureMap::new();
    features.insert(FeatureId::RtmDetExtracted, vec![0.0; 384]);
    features.insert(FeatureId::GauFeatures, vec![0.0; 256]);

    let keypoints = (0..17)
        .map(|index| slouch_domain::Keypoint::new(index as f64, (index + 1) as f64, 0.85))
        .collect();

    PostureFrame {
        id: id.to_owned(),
        timestamp: 1_700_000_000_000.0,
        features,
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: vec![1, 2, 3],
        },
        keypoints,
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            score: 0.85,
            width: 100.0,
            height: 100.0,
        },
        label,
    }
}

fn create_mock_dataset() -> PostureDataset {
    PostureDataset {
        frames: vec![
            create_mock_frame("frame-1", FrameLabel::Good),
            create_mock_frame("frame-2", FrameLabel::Good),
        ],
        version: 1,
        last_modified: 1_700_000_000_000.0,
    }
}

fn create_operations_with_reservoir(
    storage: MockStorage,
) -> (DatasetOperations<MockStorage>, &'static FeatureReservoir) {
    static NEXT_RESERVOIR: AtomicU64 = AtomicU64::new(1);
    let id = NEXT_RESERVOIR.fetch_add(1, Ordering::Relaxed);
    let reservoir = Box::leak(Box::new(FeatureReservoir::new(
        1_000,
        format!("operations-test-{id}"),
    )));
    (
        DatasetOperations::with_dependencies(storage, reservoir, NoopDatasetLogger),
        reservoir,
    )
}

fn create_operations(storage: MockStorage) -> DatasetOperations<MockStorage> {
    create_operations_with_reservoir(storage).0
}

fn seed_reservoir(reservoir: &FeatureReservoir) {
    reservoir
        .add(ReservoirSample {
            features: FeatureMap::from([(
                FeatureId::RtmDetExtracted,
                vec![0.0; FeatureId::RtmDetExtracted.metadata().dimensions],
            )]),
            keypoints: vec![Keypoint::new(0.5, 0.5, 1.0); 17],
            bbox: BoundingBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
                score: 1.0,
                width: 1.0,
                height: 1.0,
            },
        })
        .expect("seed test reservoir");
}

fn assert_error<T: std::fmt::Debug, E: std::fmt::Display>(result: Result<T, E>, expected: &str) {
    assert_eq!(
        result.expect_err("operation should fail").to_string(),
        expected
    );
}

#[test]
fn update_frame_label_delegates_successfully() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());

    operations
        .update_frame_label("frame-1", FrameLabel::Good)
        .unwrap();

    assert_eq!(
        storage.updated_labels(),
        vec![("frame-1".to_owned(), FrameLabel::Good)]
    );
}

#[test]
fn update_frame_label_accepts_unused_label() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());

    operations
        .update_frame_label("frame-1", FrameLabel::Unused)
        .unwrap();

    assert_eq!(
        storage.updated_labels(),
        vec![("frame-1".to_owned(), FrameLabel::Unused)]
    );
}

#[test]
fn update_frame_label_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_update_outcomes([Err("Storage error".to_owned())]);
    let operations = create_operations(storage);

    assert_error(
        operations.update_frame_label("frame-1", FrameLabel::Good),
        "Failed to update frame label: Storage error",
    );
}

#[test]
fn delete_frame_sets_unused_label() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());

    operations.delete_frame("frame-1").unwrap();

    assert_eq!(
        storage.updated_labels(),
        vec![("frame-1".to_owned(), FrameLabel::Unused)]
    );
}

#[test]
fn delete_frame_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_update_outcomes([Err("Delete failed".to_owned())]);
    let operations = create_operations(storage);

    assert_error(
        operations.delete_frame("frame-1"),
        "Failed to delete frame: Delete failed",
    );
}

#[test]
fn delete_bulk_reports_successful_deletions() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());
    storage.set_update_outcomes([Ok(()), Ok(()), Ok(())]);

    let result = operations
        .delete_bulk(["frame-1", "frame-2", "frame-3"])
        .unwrap();

    assert_eq!(
        result,
        BulkOperationResult {
            deleted: 3,
            success: true,
            error: None
        }
    );
    assert_eq!(storage.updated_labels().len(), 3);
}

#[test]
fn delete_bulk_returns_false_when_every_delete_fails() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());
    storage.set_update_outcomes([Err("Not found".to_owned())]);

    let result = operations.delete_bulk(["frame-1"]).unwrap();

    assert_eq!(result.deleted, 0);
    assert!(!result.success);
    assert!(result.error.is_none());
}

#[test]
fn delete_bulk_continues_after_partial_failure() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());
    storage.set_update_outcomes([Ok(()), Err("Failed".to_owned()), Ok(())]);

    let result = operations
        .delete_bulk(["frame-1", "frame-2", "frame-3"])
        .unwrap();

    assert_eq!(result.deleted, 2);
    assert!(result.success);
}

#[test]
fn delete_bulk_handles_empty_input() {
    let storage = MockStorage::new();
    let operations = create_operations(storage.clone());

    let result = operations.delete_bulk(std::iter::empty::<&str>()).unwrap();

    assert_eq!(result.deleted, 0);
    assert!(!result.success);
    assert!(storage.updated_labels().is_empty());
}

#[test]
fn cleanup_unused_returns_removed_count() {
    let storage = MockStorage::new();
    storage.set_remove_result(Ok(5));
    let operations = create_operations(storage.clone());

    let result = operations.cleanup_unused().unwrap();

    assert_eq!(
        result,
        BulkOperationResult {
            deleted: 5,
            success: true,
            error: None
        }
    );
    assert_eq!(storage.removed_labels(), vec![FrameLabel::Unused]);
}

#[test]
fn cleanup_unused_is_successful_when_nothing_is_removed() {
    let storage = MockStorage::new();
    storage.set_remove_result(Ok(0));
    let operations = create_operations(storage);

    let result = operations.cleanup_unused().unwrap();

    assert_eq!(result.deleted, 0);
    assert!(result.success);
}

#[test]
fn cleanup_unused_returns_a_failure_result_on_storage_error() {
    let storage = MockStorage::new();
    storage.set_remove_result(Err("Cleanup failed".to_owned()));
    let operations = create_operations(storage);

    let result = operations.cleanup_unused().unwrap();

    assert_eq!(result.deleted, 0);
    assert!(!result.success);
    assert_eq!(
        result.error.as_deref(),
        Some("Cleanup failed: Cleanup failed")
    );
}

#[test]
fn delete_by_label_deletes_each_matching_frame() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Ok(vec![
        create_mock_frame("frame-1", FrameLabel::Good),
        create_mock_frame("frame-2", FrameLabel::Good),
    ]));
    storage.set_update_outcomes([Ok(()), Ok(())]);
    let operations = create_operations(storage.clone());

    let result = operations.delete_by_label(FrameLabel::Good).unwrap();

    assert_eq!(result.deleted, 2);
    assert!(result.success);
    assert_eq!(storage.queried_labels(), vec![FrameLabel::Good]);
    assert_eq!(storage.updated_labels().len(), 2);
    assert!(storage
        .updated_labels()
        .iter()
        .all(|(_, label)| *label == FrameLabel::Unused));
}

#[test]
fn delete_by_label_rejects_mismatched_storage_rows_without_deleting() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Ok(vec![create_mock_frame("wrong-label", FrameLabel::Bad)]));
    let operations = create_operations(storage.clone());

    let result = operations.delete_by_label(FrameLabel::Good).unwrap();

    assert!(!result.success);
    assert_eq!(result.deleted, 0);
    assert!(result.error.unwrap().contains("mismatched label"));
    assert!(storage.updated_labels().is_empty());
}

#[test]
fn delete_by_label_succeeds_when_no_frames_match() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Ok(Vec::new()));
    let operations = create_operations(storage);

    let result = operations.delete_by_label(FrameLabel::Bad).unwrap();

    assert_eq!(result.deleted, 0);
    assert!(result.success);
}

#[test]
fn delete_by_label_continues_after_partial_failure() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Ok(vec![
        create_mock_frame("frame-1", FrameLabel::Bad),
        create_mock_frame("frame-2", FrameLabel::Bad),
    ]));
    storage.set_update_outcomes([Ok(()), Err("Failed".to_owned())]);
    let operations = create_operations(storage);

    let result = operations.delete_by_label(FrameLabel::Bad).unwrap();

    assert_eq!(result.deleted, 1);
    assert!(result.success);
}

#[test]
fn delete_by_label_wraps_query_errors() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Err("Query failed".to_owned()));
    let operations = create_operations(storage);

    let result = operations.delete_by_label(FrameLabel::Unused).unwrap();

    assert_eq!(result.deleted, 0);
    assert!(!result.success);
    assert_eq!(
        result.error.as_deref(),
        Some("Delete by label failed: Query failed")
    );
}

#[test]
fn reset_dataset_clears_frames_and_reservoir_but_preserves_models_and_settings() {
    let storage = MockStorage::new();
    let (operations, reservoir) = create_operations_with_reservoir(storage.clone());
    seed_reservoir(reservoir);
    assert_eq!(reservoir.get_count().unwrap(), 1);

    operations.reset_dataset().unwrap();

    assert_eq!(reservoir.get_count().unwrap(), 0);
    let (dataset, posture, presence, settings) = storage.clear_call_counts();
    assert_eq!(dataset, 1);
    assert_eq!(posture, 0);
    assert_eq!(presence, 0);
    assert_eq!(settings, 0);
}

#[test]
fn reset_dataset_wraps_clear_errors() {
    let storage = MockStorage::new();
    storage.fail_clear_dataset("Clear failed");
    let operations = create_operations(storage);

    assert_error(
        operations.reset_dataset(),
        "Dataset reset failed: Clear failed",
    );
}

#[test]
fn reset_all_data_clears_dataset_models_settings_and_reservoir() {
    let storage = MockStorage::new();
    let (operations, reservoir) = create_operations_with_reservoir(storage.clone());
    seed_reservoir(reservoir);
    assert_eq!(reservoir.get_count().unwrap(), 1);

    operations.reset_all_data().unwrap();

    assert_eq!(reservoir.get_count().unwrap(), 0);
    assert_eq!(storage.clear_call_counts(), (1, 1, 1, 1));
}

#[test]
fn reset_all_data_wraps_clear_errors() {
    let storage = MockStorage::new();
    storage.fail_clear_dataset("Wipe failed");
    let operations = create_operations(storage);

    assert_error(operations.reset_all_data(), "App wipe failed: Wipe failed");
}

#[test]
fn get_stats_returns_storage_statistics() {
    let expected = DatasetStats {
        total: 10,
        good: 5,
        bad: 3,
        away: 0,
        unused: 2,
        imbalance_ratio: 0.25,
        has_minimum_frames: true,
        has_away_frames: false,
    };
    let storage = MockStorage::new();
    storage.set_stats(Ok(expected));
    let operations = create_operations(storage);

    assert_eq!(operations.get_stats().unwrap(), expected);
}

#[test]
fn get_stats_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_stats(Err("Stats error".to_owned()));
    let operations = create_operations(storage);

    assert_error(operations.get_stats(), "Failed to get stats: Stats error");
}

#[test]
fn load_dataset_returns_the_complete_dataset() {
    let expected = create_mock_dataset();
    let storage = MockStorage::new();
    storage.set_dataset(Ok(expected.clone()));
    let operations = create_operations(storage);

    assert_eq!(operations.load_dataset().unwrap(), expected);
}

#[test]
fn load_dataset_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_dataset(Err("Load failed".to_owned()));
    let operations = create_operations(storage);

    assert_error(
        operations.load_dataset(),
        "Failed to load dataset: Load failed",
    );
}

#[test]
fn get_frames_by_label_returns_matching_frames() {
    let expected = vec![
        create_mock_frame("frame-1", FrameLabel::Good),
        create_mock_frame("frame-2", FrameLabel::Good),
    ];
    let storage = MockStorage::new();
    storage.set_frames_by_label(Ok(expected.clone()));
    let operations = create_operations(storage.clone());

    assert_eq!(
        operations.get_frames_by_label(FrameLabel::Good).unwrap(),
        expected
    );
    assert_eq!(storage.queried_labels(), vec![FrameLabel::Good]);
}

#[test]
fn get_frames_by_label_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_frames_by_label(Err("Query failed".to_owned()));
    let operations = create_operations(storage);

    assert_error(
        operations.get_frames_by_label(FrameLabel::Bad),
        "Failed to get frames by label: Query failed",
    );
}

#[test]
fn get_frame_by_id_returns_the_matching_frame() {
    let expected = create_mock_frame("frame-1", FrameLabel::Good);
    let storage = MockStorage::new();
    storage.set_frame_by_id(Ok(Some(expected.clone())));
    let operations = create_operations(storage);

    assert_eq!(
        operations.get_frame_by_id("frame-1").unwrap(),
        Some(expected)
    );
}

#[test]
fn get_frame_by_id_returns_none_when_missing() {
    let storage = MockStorage::new();
    storage.set_frame_by_id(Ok(None));
    let operations = create_operations(storage);

    assert_eq!(operations.get_frame_by_id("nonexistent").unwrap(), None);
}

#[test]
fn get_frame_by_id_wraps_storage_errors() {
    let storage = MockStorage::new();
    storage.set_frame_by_id(Err("Query failed".to_owned()));
    let operations = create_operations(storage);

    assert_error(
        operations.get_frame_by_id("frame-1"),
        "Failed to get frame: Query failed",
    );
}

#[test]
fn needs_retraining_returns_storage_value() {
    let storage = MockStorage::new();
    storage.set_retraining(Ok(true));
    let operations = create_operations(storage.clone());
    assert!(operations.needs_retraining().unwrap());

    storage.set_retraining(Ok(false));
    assert!(!operations.needs_retraining().unwrap());
}

#[test]
fn needs_retraining_defaults_to_true_on_error() {
    let storage = MockStorage::new();
    storage.set_retraining(Err("Check failed".to_owned()));
    let operations = create_operations(storage);

    assert!(operations.needs_retraining().unwrap());
}

#[test]
fn create_mock_frames_use_native_storage_shapes() {
    let frame = create_mock_frame("frame-1", FrameLabel::Good);
    assert_eq!(frame.features[&FeatureId::RtmDetExtracted].len(), 384);
    assert_eq!(frame.features[&FeatureId::GauFeatures].len(), 256);
    assert_eq!(frame.keypoints.len(), 17);
    assert_eq!(frame.thumbnail.mime_type, "image/webp");
}

#[test]
fn mock_storage_preserves_feature_map_iteration_shape() {
    let frame = create_mock_frame("frame-1", FrameLabel::Good);
    let keys = frame.features.keys().copied().collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![FeatureId::GauFeatures, FeatureId::RtmDetExtracted]
    );
}

#[test]
fn bulk_result_shape_is_stable() {
    let storage = MockStorage::new();
    storage.set_remove_result(Ok(2));
    let operations = create_operations(storage);

    let result = operations.cleanup_unused().unwrap();

    assert_eq!(
        result,
        BulkOperationResult {
            deleted: 2,
            success: true,
            error: None,
        }
    );
}
