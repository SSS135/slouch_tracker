use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use slouch_domain::{BoundingBox, FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail};
use slouch_store::ported::storage::DatasetStorage;

fn archive_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("slouch-{name}-{nonce}.slouchpack"))
}

fn frame() -> PostureFrame {
    let feature = FeatureId::GauFeatures;
    PostureFrame {
        id: "frame-1".into(),
        timestamp: 1_700_000_000_000.0,
        features: BTreeMap::from([(feature, vec![0.25; feature.metadata().dimensions])]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
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

#[test]
fn slouchpack_export_reset_import_round_trip() {
    let path = archive_path("round-trip");
    let source = DatasetStorage::open_in_memory().expect("source storage");
    source.save_frame(frame()).expect("save frame");
    let summary = source
        .export_archive(&path, "test")
        .expect("export archive");
    assert_eq!(summary.frame_count, 1);

    let target = DatasetStorage::open_in_memory().expect("target storage");
    let imported = target.import_archive(&path).expect("import archive");
    assert_eq!(imported.frame_count, 1);
    let dataset = target.load_dataset().expect("load imported dataset");
    assert_eq!(dataset.frames, vec![frame()]);
    fs::remove_file(path).expect("remove archive");
}

#[test]
fn slouchpack_rejects_payload_checksum_corruption_without_mutation() {
    let path = archive_path("corrupt");
    let source = DatasetStorage::open_in_memory().expect("source storage");
    source.save_frame(frame()).expect("save frame");
    source
        .export_archive(&path, "test")
        .expect("export archive");
    let connection = Connection::open(&path).expect("open archive for corruption");
    connection
        .execute("UPDATE thumbnails SET payload_sha256 = zeroblob(32)", [])
        .expect("corrupt hash");
    drop(connection);

    let target = DatasetStorage::open_in_memory().expect("target storage");
    let original = frame();
    target.save_frame(&original).expect("seed target");
    assert!(target.import_archive(&path).is_err());
    assert_eq!(
        target.load_dataset().expect("load target").frames,
        vec![original]
    );
    fs::remove_file(path).expect("remove archive");
}
