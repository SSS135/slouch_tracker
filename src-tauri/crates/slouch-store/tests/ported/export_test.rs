use std::io::{Cursor, Read};
use std::sync::MutexGuard;

use rmp_serde::from_slice;
use serde::Deserialize;
use slouch_domain::{
    BoundingBox, FeatureId, FeatureMap, FrameLabel, Keypoint, PostureDataset, PostureFrame,
    Thumbnail,
};
use slouch_store::ported::export::{
    build_manifest, export_dataset_to_zip, export_labeled_data, export_unlabeled_data,
};
use slouch_store::ported::feature_reservoir::{feature_reservoir, ReservoirSample};

#[derive(Debug, Deserialize)]
struct MessagePackData {
    version: u32,
    #[serde(rename = "exportedAt")]
    exported_at: String,
    #[serde(rename = "frameCount")]
    frame_count: usize,
    labels: Option<Vec<FrameLabel>>,
    timestamps: Option<Vec<f64>>,
    keypoints: Vec<f32>,
    bboxes: Vec<f32>,
    backbone_avg: Vec<f32>,
    backbone_max: Vec<f32>,
    backbone_std: Vec<f32>,
    gau_avg: Vec<f32>,
    gau_max: Vec<f32>,
    gau_std: Vec<f32>,
    rtmdet: Vec<f32>,
    engineered: Vec<f32>,
    rtmdet_engineered: Vec<f32>,
    shapes: Shapes,
    reservoir: Option<ReservoirData>,
}

#[derive(Debug, Deserialize)]
struct Shapes {
    keypoints: Vec<usize>,
    bboxes: Vec<usize>,
    backbone_avg: Vec<usize>,
    backbone_max: Vec<usize>,
    backbone_std: Vec<usize>,
    gau_avg: Vec<usize>,
    gau_max: Vec<usize>,
    gau_std: Vec<usize>,
    rtmdet: Vec<usize>,
    engineered: Vec<usize>,
    rtmdet_engineered: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ReservoirData {
    count: usize,
    #[serde(rename = "totalSeen")]
    total_seen: usize,
    #[serde(rename = "maxSamples")]
    max_samples: usize,
}

fn create_mock_frame(id: &str, label: FrameLabel, include_all_features: bool) -> PostureFrame {
    let mut features = FeatureMap::from([
        (FeatureId::RtmDetExtracted, vec![60.0; 384]),
        (FeatureId::GauFeatures, vec![1.0; 256]),
    ]);

    if include_all_features {
        features.insert(FeatureId::BackboneFeatures, vec![30.0; 768]);
        features.insert(FeatureId::BackboneFeaturesMax, vec![40.0; 768]);
        features.insert(FeatureId::BackboneFeaturesStd, vec![50.0; 768]);
        features.insert(FeatureId::GauFeaturesMax, vec![11.0; 256]);
        features.insert(FeatureId::GauFeaturesStd, vec![21.0; 256]);
    }

    PostureFrame {
        id: id.to_owned(),
        timestamp: 1_700_000_000_000.0
            + id.rsplit('-').next().unwrap_or("0").parse::<f64>().unwrap(),
        features,
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: b"mock-image".to_vec(),
        },
        label,
        keypoints: (0..17)
            .map(|index| Keypoint::new(10.0 + index as f64, 20.0 + index as f64, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 100.0,
            y1: 150.0,
            x2: 400.0,
            y2: 550.0,
            score: 0.95,
            width: 300.0,
            height: 400.0,
        },
    }
}

fn create_mock_reservoir_sample() -> ReservoirSample {
    ReservoirSample {
        keypoints: (0..17)
            .map(|index| Keypoint::new(10.0 + index as f64, 20.0 + index as f64, 0.9))
            .collect(),
        bbox: BoundingBox {
            x1: 100.0,
            y1: 150.0,
            x2: 400.0,
            y2: 550.0,
            score: 0.95,
            width: 300.0,
            height: 400.0,
        },
        backbone_avg: vec![0.1; 768],
        backbone_max: vec![0.2; 768],
        backbone_std: vec![0.3; 768],
        gau_avg: vec![0.4; 256],
        gau_max: vec![0.5; 256],
        gau_std: vec![0.6; 256],
        rtmdet: vec![0.7; 384],
    }
}

fn dataset(frames: Vec<PostureFrame>) -> PostureDataset {
    PostureDataset {
        frames,
        version: 1,
        last_modified: 1_700_000_000_000.0,
    }
}

fn decode_message_pack(bytes: &[u8]) -> MessagePackData {
    from_slice(bytes).expect("export should contain valid MessagePack")
}

fn reservoir_test_lock() -> MutexGuard<'static, ()> {
    crate::default_reservoir_test_lock()
}

fn clear_reservoir() {
    feature_reservoir().clear().unwrap();
}

struct ReservoirCleanup;

impl Drop for ReservoirCleanup {
    fn drop(&mut self) {
        clear_reservoir();
    }
}

#[test]
fn build_manifest_preserves_frame_metadata_and_current_version() {
    let frame1 = create_mock_frame("frame-1", FrameLabel::Good, false);
    let frame2 = create_mock_frame("frame-2", FrameLabel::Bad, true);
    let first_timestamp = frame1.timestamp;

    let manifest = build_manifest(&dataset(vec![frame1, frame2]));

    assert_eq!(manifest.version, 3);
    assert_eq!(manifest.frame_count, 2);
    assert_eq!(manifest.frames.len(), 2);
    assert!(!manifest.exported_at.is_empty());
    assert!(manifest.exported_at.contains('T'));
    assert!(manifest.exported_at.ends_with('Z'));
    assert_eq!(manifest.frames[0].id, "frame-1");
    assert_eq!(manifest.frames[0].label, FrameLabel::Good);
    assert_eq!(manifest.frames[0].timestamp, first_timestamp);
}

#[test]
fn build_manifest_includes_all_stored_features() {
    let manifest = build_manifest(&dataset(vec![create_mock_frame(
        "frame-1",
        FrameLabel::Good,
        true,
    )]));
    let features = &manifest.frames[0].features;

    assert!(features.contains(&FeatureId::RtmDetExtracted));
    assert!(features.contains(&FeatureId::GauFeatures));
    assert!(features.contains(&FeatureId::BackboneFeatures));
}

#[test]
fn export_labeled_data_rejects_empty_datasets() {
    let error = export_labeled_data(&dataset(Vec::new())).unwrap_err();
    assert!(error.to_string().contains("No labeled frames to export"));
}

#[test]
fn export_labeled_data_rejects_malformed_and_non_finite_feature_vectors() {
    let mut short = create_mock_frame("frame-1", FrameLabel::Good, true);
    short
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 255]);
    let error = export_labeled_data(&dataset(vec![short])).unwrap_err();
    assert!(error.to_string().contains("expected 256"));

    let mut non_finite = create_mock_frame("frame-1", FrameLabel::Good, true);
    non_finite
        .features
        .get_mut(&FeatureId::BackboneFeatures)
        .unwrap()[0] = f32::NAN;
    let error = export_labeled_data(&dataset(vec![non_finite])).unwrap_err();
    assert!(error.to_string().contains("finite"));
}

#[test]
fn export_labeled_data_contains_labels_timestamps_and_shapes() {
    let encoded = export_labeled_data(&dataset(vec![
        create_mock_frame("frame-1", FrameLabel::Good, true),
        create_mock_frame("frame-2", FrameLabel::Bad, true),
    ]))
    .unwrap();
    assert!(!encoded.is_empty());

    let data = decode_message_pack(&encoded);
    assert_eq!(data.version, 3);
    assert!(!data.exported_at.is_empty());
    assert_eq!(data.frame_count, 2);
    assert_eq!(data.labels, Some(vec![FrameLabel::Good, FrameLabel::Bad]));
    assert_eq!(data.timestamps.as_ref().unwrap().len(), 2);
    assert_eq!(data.keypoints.len(), 2 * 17 * 3);
    assert_eq!(data.bboxes.len(), 2 * 7);
    assert_eq!(data.shapes.keypoints, vec![2, 17, 3]);
    assert_eq!(data.shapes.bboxes, vec![2, 7]);
    assert_eq!(data.shapes.backbone_avg, vec![2, 768]);
    assert_eq!(data.shapes.gau_avg, vec![2, 256]);
    assert_eq!(data.shapes.rtmdet, vec![2, 384]);
    assert_eq!(data.shapes.engineered, vec![2, 54]);
    assert_eq!(data.shapes.rtmdet_engineered, vec![2, 135]);
}

#[test]
fn export_labeled_data_flattens_keypoints_in_source_order() {
    let data = decode_message_pack(
        &export_labeled_data(&dataset(vec![create_mock_frame(
            "frame-1",
            FrameLabel::Good,
            false,
        )]))
        .unwrap(),
    );

    assert_eq!(data.keypoints[0], 10.0);
    assert_eq!(data.keypoints[1], 20.0);
    assert!((data.keypoints[2] - 0.9).abs() <= f32::EPSILON);
}

#[test]
fn export_labeled_data_flattens_bbox_in_source_order() {
    let data = decode_message_pack(
        &export_labeled_data(&dataset(vec![create_mock_frame(
            "frame-1",
            FrameLabel::Good,
            false,
        )]))
        .unwrap(),
    );

    assert_eq!(&data.bboxes[..4], &[100.0, 150.0, 400.0, 550.0]);
    assert!((data.bboxes[4] - 0.95).abs() <= f32::EPSILON);
    assert_eq!(&data.bboxes[5..7], &[300.0, 400.0]);
}

#[test]
fn export_unlabeled_data_rejects_an_empty_reservoir() {
    let _guard = reservoir_test_lock();
    let _cleanup = ReservoirCleanup;
    clear_reservoir();

    let error = export_unlabeled_data().unwrap_err();
    assert!(error.to_string().contains("No reservoir samples to export"));
}

#[test]
fn export_unlabeled_data_contains_reservoir_metadata_and_shapes() {
    let _guard = reservoir_test_lock();
    let _cleanup = ReservoirCleanup;
    clear_reservoir();
    let reservoir = feature_reservoir();
    reservoir.add(create_mock_reservoir_sample()).unwrap();
    reservoir.add(create_mock_reservoir_sample()).unwrap();

    let data = decode_message_pack(&export_unlabeled_data().unwrap());
    assert_eq!(data.version, 3);
    assert!(!data.exported_at.is_empty());
    assert_eq!(data.frame_count, 2);
    assert!(data.labels.is_none());
    assert!(data.timestamps.is_none());

    let metadata = data.reservoir.unwrap();
    assert_eq!(metadata.count, 2);
    assert_eq!(metadata.total_seen, 2);
    assert_eq!(metadata.max_samples, 1_000);
    assert_eq!(data.shapes.keypoints, vec![2, 17, 3]);
    assert_eq!(data.shapes.bboxes, vec![2, 7]);
    assert_eq!(data.shapes.backbone_avg, vec![2, 768]);
    assert_eq!(data.shapes.gau_avg, vec![2, 256]);
    assert_eq!(data.shapes.rtmdet, vec![2, 384]);
    assert_eq!(data.shapes.backbone_max, vec![2, 768]);
    assert_eq!(data.shapes.backbone_std, vec![2, 768]);
    assert_eq!(data.shapes.gau_max, vec![2, 256]);
    assert_eq!(data.shapes.gau_std, vec![2, 256]);
    assert_eq!(data.backbone_max.len(), 2 * 768);
    assert_eq!(data.backbone_std.len(), 2 * 768);
    assert_eq!(data.gau_avg.len(), 2 * 256);
    assert_eq!(data.gau_max.len(), 2 * 256);
    assert_eq!(data.gau_std.len(), 2 * 256);
    assert_eq!(data.rtmdet.len(), 2 * 384);
}

#[test]
fn export_unlabeled_data_converts_float_features_to_numeric_values() {
    let _guard = reservoir_test_lock();
    let _cleanup = ReservoirCleanup;
    clear_reservoir();
    feature_reservoir()
        .add(create_mock_reservoir_sample())
        .unwrap();

    let data = decode_message_pack(&export_unlabeled_data().unwrap());
    assert_eq!(data.backbone_avg.len(), 768);
    assert!((data.backbone_avg[0] - 0.1).abs() <= 1e-6);
    assert!(!data.engineered.is_empty());
    assert!(!data.rtmdet_engineered.is_empty());
}

#[test]
fn export_dataset_to_zip_contains_labeled_and_unlabeled_message_pack_files() {
    let _guard = reservoir_test_lock();
    let _cleanup = ReservoirCleanup;
    clear_reservoir();
    feature_reservoir()
        .add(create_mock_reservoir_sample())
        .unwrap();

    let bytes = export_dataset_to_zip(&dataset(vec![create_mock_frame(
        "frame-1",
        FrameLabel::Good,
        true,
    )]))
    .unwrap();
    assert!(!bytes.is_empty());

    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
    let mut labeled = Vec::new();
    archive
        .by_name("labeled.msgpack")
        .unwrap()
        .read_to_end(&mut labeled)
        .unwrap();
    let mut unlabeled = Vec::new();
    archive
        .by_name("unlabeled.msgpack")
        .unwrap()
        .read_to_end(&mut unlabeled)
        .unwrap();

    let labeled_data = decode_message_pack(&labeled);
    assert_eq!(labeled_data.labels, Some(vec![FrameLabel::Good]));
    assert!(labeled_data.timestamps.is_some());
    let unlabeled_data = decode_message_pack(&unlabeled);
    assert!(unlabeled_data.labels.is_none());
    assert!(unlabeled_data.reservoir.is_some());
}

#[test]
fn export_dataset_to_zip_rejects_empty_labeled_data() {
    let error = export_dataset_to_zip(&dataset(Vec::new())).unwrap_err();
    assert!(error.to_string().contains("No labeled frames to export"));
}

#[test]
fn export_dataset_to_zip_rejects_empty_unlabeled_data() {
    let _guard = reservoir_test_lock();
    let _cleanup = ReservoirCleanup;
    clear_reservoir();
    let error = export_dataset_to_zip(&dataset(vec![create_mock_frame(
        "frame-1",
        FrameLabel::Good,
        false,
    )]))
    .unwrap_err();
    assert!(error.to_string().contains("No reservoir samples to export"));
}
