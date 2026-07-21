use std::io::{Cursor, Write};

use serde_json::{json, Value};
use slouch_domain::{BoundingBox, FrameLabel, Keypoint, PostureFrame, Thumbnail};
use slouch_store::ported::feature_reservoir::FeatureReservoir;
use slouch_store::ported::import::{
    import_dataset_from_archive, import_dataset_from_zip, DatasetImportError,
};
use slouch_store::ported::storage::DatasetStorage;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

const TIMESTAMP: f64 = 1_700_000_000_000.0;

#[derive(Clone, Copy)]
struct MockFrame {
    id: &'static str,
    label: &'static str,
    timestamp: f64,
}

fn create_mock_frame(id: &'static str, label: &'static str) -> MockFrame {
    MockFrame {
        id,
        label,
        timestamp: TIMESTAMP,
    }
}

fn manifest(frames: &[MockFrame]) -> Value {
    json!({
        "version": 2,
        "exportedAt": "2026-01-01T00:00:00.000Z",
        "frameCount": frames.len(),
        "frames": frames.iter().map(|frame| json!({
            "id": frame.id,
            "label": frame.label,
            "timestamp": frame.timestamp,
            "features": ["rtmdet_extracted", "gau_features"],
        })).collect::<Vec<_>>(),
    })
}

fn zip_file(manifest: Option<Value>, files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(&mut bytes);
    let options = SimpleFileOptions::default();

    if let Some(manifest) = manifest {
        zip.start_file("manifest.json", options).unwrap();
        zip.write_all(serde_json::to_string(&manifest).unwrap().as_bytes())
            .unwrap();
    }

    for (path, contents) in files {
        zip.start_file(*path, options).unwrap();
        zip.write_all(contents).unwrap();
    }

    zip.finish().unwrap();
    bytes.into_inner()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f32_to_f16(*value).to_le_bytes())
        .collect()
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

fn keypoints_bytes() -> Vec<u8> {
    let values = (0..17)
        .flat_map(|index| [10.0 + index as f32, 20.0 + index as f32, 0.9])
        .collect::<Vec<_>>();
    f16_bytes(&values)
}

fn bbox_bytes() -> Vec<u8> {
    f16_bytes(&[100.0, 150.0, 400.0, 550.0, 0.95, 300.0, 400.0])
}

fn frame_files(id: &str, include_bbox: bool, include_keypoints: bool) -> Vec<(String, Vec<u8>)> {
    let prefix = format!("frames/frame-{id}/");
    let mut files = vec![(format!("{prefix}thumbnail.webp"), b"mock-image".to_vec())];
    if include_keypoints {
        files.push((format!("{prefix}keypoints.bin"), keypoints_bytes()));
    }
    if include_bbox {
        files.push((format!("{prefix}bbox.bin"), bbox_bytes()));
    }
    files
}

fn archive_for_frames(frames: &[MockFrame]) -> Vec<u8> {
    let files = frames
        .iter()
        .flat_map(|frame| frame_files(frame.id, true, true))
        .collect::<Vec<_>>();
    let borrowed = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_slice()))
        .collect::<Vec<_>>();
    zip_file(Some(manifest(frames)), &borrowed)
}

fn storage() -> DatasetStorage {
    DatasetStorage::new_in_memory().expect("create in-memory dataset storage")
}

fn native_frame(id: &str, label: FrameLabel) -> PostureFrame {
    PostureFrame {
        id: id.to_owned(),
        timestamp: TIMESTAMP,
        features: Default::default(),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: b"mock-image".to_vec(),
        },
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
        label,
    }
}

fn import(bytes: &[u8], storage: &DatasetStorage) -> slouch_domain::ImportResult {
    let _guard = crate::default_reservoir_test_lock();
    import_dataset_from_zip(bytes, storage).unwrap()
}

#[test]
fn imports_multiple_frames_successfully() {
    let storage = storage();
    let archive = archive_for_frames(&[
        create_mock_frame("frame-1", "good"),
        create_mock_frame("frame-2", "bad"),
        create_mock_frame("frame-3", "unused"),
    ]);

    let result = import(&archive, &storage);

    assert_eq!(result.imported, 3);
    assert_eq!(result.skipped, 0);
    assert!(result.errors.is_empty());
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 3);
}

#[test]
fn skips_duplicate_frame_ids() {
    let storage = storage();
    storage
        .save_frame(native_frame("frame-1", FrameLabel::Good))
        .unwrap();
    let archive = archive_for_frames(&[
        create_mock_frame("frame-1", "good"),
        create_mock_frame("frame-2", "bad"),
    ]);

    let result = import(&archive, &storage);

    assert_eq!(result.imported, 1);
    assert_eq!(result.skipped, 1);
    assert!(result.errors[0].contains("duplicate"));
}

#[test]
fn rejects_corrupted_zip() {
    let storage = storage();
    let error = import_dataset_from_zip(b"not a zip file", &storage).unwrap_err();
    assert!(error.to_string().contains("import"));
}

#[test]
fn rejects_missing_manifest() {
    let archive = zip_file(None, &[]);
    let error = import_dataset_from_zip(&archive, &storage()).unwrap_err();
    assert!(error.to_string().contains("manifest.json not found"));
}

#[test]
fn rejects_invalid_manifest_version() {
    let archive = zip_file(
        Some(json!({
            "version": 999,
            "exportedAt": "2026-01-01T00:00:00.000Z",
            "frameCount": 0,
            "frames": [],
        })),
        &[],
    );
    let error = import_dataset_from_zip(&archive, &storage()).unwrap_err();
    assert!(error.to_string().contains("Unsupported manifest version"));
}

#[test]
fn skips_frame_with_missing_thumbnail() {
    let frame = create_mock_frame("frame-1", "good");
    let keypoints = keypoints_bytes();
    let bbox = bbox_bytes();
    let archive = zip_file(
        Some(manifest(&[frame])),
        &[
            ("frames/frame-frame-1/keypoints.bin", &keypoints),
            ("frames/frame-frame-1/bbox.bin", &bbox),
        ],
    );

    let result = import(&archive, &storage());

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 1);
    assert!(result.errors[0].contains("missing files"));
}

#[test]
fn skips_frame_with_missing_feature_files() {
    let frame = create_mock_frame("frame-1", "good");
    let thumbnail = b"image";
    let extracted = [7_u8, 8, 9];
    let archive = zip_file(
        Some(manifest(&[frame])),
        &[
            ("frames/frame-frame-1/thumbnail.webp", thumbnail),
            ("frames/frame-frame-1/rtmdet_extracted.bin", &extracted),
        ],
    );

    let result = import(&archive, &storage());

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 1);
    assert!(result.errors[0].contains("missing files"));
}

#[test]
fn preserves_frame_metadata_during_import() {
    let storage = storage();
    let result = import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );
    assert_eq!(result.imported, 1);

    let frame = &storage.load_dataset().unwrap().frames[0];
    assert_eq!(frame.id, "frame-1");
    assert_eq!(frame.label, FrameLabel::Good);
}

#[test]
fn imports_with_empty_features_when_features_are_excluded_from_export() {
    let storage = storage();
    import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );

    assert!(storage.load_dataset().unwrap().frames[0]
        .features
        .is_empty());
}

#[test]
fn handles_empty_dataset() {
    let storage = storage();
    let result = import(&zip_file(Some(manifest(&[])), &[]), &storage);

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 0);
    assert!(result.errors.is_empty());
}

#[test]
fn skips_frame_when_bbox_is_missing() {
    let frame = create_mock_frame("frame-1", "good");
    let keypoints = keypoints_bytes();
    let thumbnail = b"image";
    let archive = zip_file(
        Some(manifest(&[frame])),
        &[
            ("frames/frame-frame-1/thumbnail.webp", thumbnail),
            ("frames/frame-frame-1/keypoints.bin", &keypoints),
        ],
    );

    let result = import(&archive, &storage());

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 1);
    assert!(result.errors[0].contains("bbox"));
}

#[test]
fn completes_export_import_round_trip() {
    let storage = storage();
    let archive = archive_for_frames(&[
        create_mock_frame("frame-1", "good"),
        create_mock_frame("frame-2", "bad"),
    ]);

    let result = import(&archive, &storage);
    let frames = storage.load_dataset().unwrap().frames;

    assert_eq!(result.imported, 2);
    assert_eq!(result.skipped, 0);
    assert!(result.errors.is_empty());
    assert_eq!(
        frames
            .iter()
            .find(|frame| frame.id == "frame-1")
            .unwrap()
            .label,
        FrameLabel::Good
    );
    assert_eq!(
        frames
            .iter()
            .find(|frame| frame.id == "frame-2")
            .unwrap()
            .label,
        FrameLabel::Bad
    );
    for frame in &frames {
        assert_eq!(frame.keypoints.len(), 17);
        assert_eq!(frame.bbox.x1, 100.0);
        assert!((frame.bbox.score - 0.95).abs() < 0.001);
    }
}

#[test]
fn imports_keypoints_from_binary() {
    let storage = storage();
    import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );
    let keypoints = &storage.load_dataset().unwrap().frames[0].keypoints;

    assert_eq!(keypoints.len(), 17);
    assert!((keypoints[0].x - 10.0).abs() < 0.01);
    assert!((keypoints[0].y - 20.0).abs() < 0.01);
    assert!((keypoints[0].score - 0.9).abs() < 0.01);
}

#[test]
fn imports_bbox_from_binary() {
    let storage = storage();
    import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );
    let bbox = storage.load_dataset().unwrap().frames[0].bbox;

    assert_eq!(bbox.x1, 100.0);
    assert_eq!(bbox.y1, 150.0);
    assert_eq!(bbox.x2, 400.0);
    assert_eq!(bbox.y2, 550.0);
    assert!((bbox.score - 0.95).abs() < 0.001);
    assert_eq!(bbox.width, 300.0);
    assert_eq!(bbox.height, 400.0);
}

#[test]
fn rejects_overlong_bbox_payloads() {
    let frame = create_mock_frame("frame-1", "good");
    let keypoints = keypoints_bytes();
    let bbox = f16_bytes(&[100.0, 150.0, 400.0, 550.0, 0.95, 300.0, 400.0, 999.0]);
    let archive = zip_file(
        Some(manifest(&[frame])),
        &[
            ("frames/frame-frame-1/thumbnail.webp", b"image"),
            ("frames/frame-frame-1/keypoints.bin", &keypoints),
            ("frames/frame-frame-1/bbox.bin", &bbox),
        ],
    );

    let result = import(&archive, &storage());
    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 1);
    assert!(result.errors[0].contains("exceeds the 14-byte limit"));
}

#[test]
fn accepts_javascript_integer_numeric_spellings() {
    let archive = zip_file(
        Some(json!({
            "version": 2.0,
            "exportedAt": "2026-01-01T00:00:00.000Z",
            "frameCount": 0.0,
            "frames": [],
            "reservoir": { "count": 0.0 },
        })),
        &[],
    );
    let result = import(&archive, &storage());
    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 0);
}

#[test]
fn imports_frame_with_keypoints_and_bbox() {
    let storage = storage();
    let result = import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );
    let frame = &storage.load_dataset().unwrap().frames[0];

    assert_eq!(result.imported, 1);
    assert!(frame.keypoints.len() == 17);
    assert_eq!(frame.bbox.x1, 100.0);
}

#[test]
fn imports_ok_when_reservoir_samples_have_no_features() {
    let storage = storage();
    let frame = create_mock_frame("frame-1", "good");
    let mut manifest = manifest(&[frame]);
    manifest["reservoir"] = json!({ "count": 1 });

    let mut files = frame_files("frame-1", true, true);
    let keypoints = keypoints_bytes();
    let bbox = bbox_bytes();
    files.push(("reservoir/sample-0/keypoints.bin".to_owned(), keypoints));
    files.push(("reservoir/sample-0/bbox.bin".to_owned(), bbox));
    let borrowed = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_slice()))
        .collect::<Vec<_>>();
    let archive = zip_file(Some(manifest), &borrowed);

    let result = import(&archive, &storage);

    assert_eq!(result.imported, 1);
    assert_eq!(result.skipped, 0);
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
}

#[test]
fn malformed_reservoir_count_does_not_abort_import() {
    for bad_count in [json!(2000), json!("abc"), json!(-3), json!(1.5)] {
        let storage = storage();
        let mut manifest = manifest(&[create_mock_frame("frame-1", "good")]);
        manifest["reservoir"] = json!({ "count": bad_count });

        let files = frame_files("frame-1", true, true);
        let borrowed = files
            .iter()
            .map(|(path, contents)| (path.as_str(), contents.as_slice()))
            .collect::<Vec<_>>();
        let archive = zip_file(Some(manifest), &borrowed);

        let result = import(&archive, &storage);

        assert_eq!(
            result.imported, 1,
            "count {bad_count:?} should still import the valid frame"
        );
        assert_eq!(result.skipped, 0);
        assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
    }
}

#[test]
fn skips_frame_with_invalid_timestamp_without_aborting_import() {
    let storage = storage();
    let mut manifest = manifest(&[
        create_mock_frame("frame-1", "good"),
        create_mock_frame("frame-2", "good"),
    ]);
    manifest["frames"][0]
        .as_object_mut()
        .unwrap()
        .remove("timestamp");
    manifest["frames"][1]["timestamp"] = json!(-5.0);

    let mut files = frame_files("frame-1", true, true);
    files.extend(frame_files("frame-2", true, true));
    let borrowed = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_slice()))
        .collect::<Vec<_>>();
    let archive = zip_file(Some(manifest), &borrowed);

    let result = import(&archive, &storage);

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, 2);
    assert!(result
        .errors
        .iter()
        .all(|error| error.contains("timestamp")));
}

#[test]
fn long_frame_id_is_skipped_without_aborting_the_whole_import() {
    // The TS oracle's validateManifest never rejects a long id, so a long id must
    // not be a manifest-fatal error that discards every other frame. The storage
    // schema still enforces its own 1..128 id length CHECK, so the long-id frame is
    // isolated as a skipped frame while the valid sibling frame imports normally.
    let storage = storage();
    let long_id = "a".repeat(200);
    let manifest = json!({
        "version": 2,
        "exportedAt": "2026-01-01T00:00:00.000Z",
        "frameCount": 2,
        "frames": [
            {
                "id": long_id,
                "label": "good",
                "timestamp": TIMESTAMP,
                "features": ["rtmdet_extracted", "gau_features"],
            },
            {
                "id": "frame-ok",
                "label": "good",
                "timestamp": TIMESTAMP,
                "features": ["rtmdet_extracted", "gau_features"],
            },
        ],
    });
    let mut files = frame_files(&long_id, true, true);
    files.extend(frame_files("frame-ok", true, true));
    let borrowed = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_slice()))
        .collect::<Vec<_>>();
    let archive = zip_file(Some(manifest), &borrowed);

    // Must not abort with a Manifest error over the long id.
    let result = import(&archive, &storage);

    assert_eq!(result.imported, 1, "errors: {:?}", result.errors);
    assert_eq!(result.skipped, 1);
    let frames = storage.load_dataset().unwrap().frames;
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].id, "frame-ok");
}

#[test]
fn frame_count_above_previous_cap_does_not_abort_as_manifest_error() {
    // The TS oracle imposes no upper bound on frameCount; a large but consistent
    // manifest must not fail validate_manifest. Frames here are skipped for
    // missing files, but the import as a whole must succeed (Ok, not a Manifest error).
    let storage = storage();
    let count: usize = 100_001;
    let frames = (0..count)
        .map(|index| {
            json!({
                "id": index.to_string(),
                "label": "good",
                "timestamp": 1,
                "features": [],
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "version": 2,
        "exportedAt": "2026-01-01T00:00:00.000Z",
        "frameCount": count,
        "frames": frames,
    });
    let archive = zip_file(Some(manifest), &[]);

    let result = import_dataset_from_zip(&archive, &storage)
        .expect("large frameCount must not abort with a Manifest error");

    assert_eq!(result.imported, 0);
    assert_eq!(result.skipped, count);
}

/// A reservoir whose every operation, including `clear()`, fails capacity
/// validation — the same zero-capacity injection used by
/// feature_reservoir_test's rejects_invalid_capacities.
fn broken_reservoir(name: &str) -> FeatureReservoir {
    FeatureReservoir::new(0, name)
}

#[test]
fn failing_reservoir_clear_aborts_import_after_frames_were_saved() {
    // Regression for RR5-PARITY-001: in the oracle, `await reservoir.clear()`
    // (import.ts:225) sits OUTSIDE the per-sample try/catch, so a clear
    // rejection propagates to importDatasetFromZip's outer catch and fails the
    // ENTIRE import as "Dataset import failed: ...", even though frames were
    // already saved. The port must not swallow that failure.
    let storage = storage();
    let mut manifest = manifest(&[create_mock_frame("frame-1", "good")]);
    manifest["reservoir"] = json!({ "count": 1 });

    let mut files = frame_files("frame-1", true, true);
    files.push((
        "reservoir/sample-0/keypoints.bin".to_owned(),
        keypoints_bytes(),
    ));
    files.push(("reservoir/sample-0/bbox.bin".to_owned(), bbox_bytes()));
    let borrowed = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_slice()))
        .collect::<Vec<_>>();
    let archive = zip_file(Some(manifest), &borrowed);

    let reservoir = broken_reservoir("import-test-broken-clear");
    let error = import_dataset_from_archive(&archive, &storage, &reservoir)
        .expect_err("a failing reservoir clear must abort the import");

    assert!(matches!(error, DatasetImportError::Storage(_)));
    let message = error.to_string();
    assert!(message.starts_with("Dataset import failed:"), "{message}");
    assert!(message.contains("clear"), "{message}");
    // Frames import before the reservoir phase, so the saved frame survives
    // the late abort exactly as it does in the oracle.
    assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
}

#[test]
fn reservoir_clear_gate_mirrors_oracle_truthiness() {
    // Oracle gate (import.ts:220):
    // `if (!manifest.reservoir || manifest.reservoir.count === 0) return 0;`
    // Any truthy `reservoir` whose `count` is not the number 0 reaches the
    // fatal clear() even when the count is malformed; a falsy reservoir or a
    // count of exactly 0 skips the reservoir phase (and its clear) entirely.
    let reservoir = broken_reservoir("import-test-broken-clear-gate");
    for (reservoir_value, clear_runs) in [
        (json!({ "count": "abc" }), true),
        (json!({ "count": -3 }), true),
        (json!({}), true),
        (json!([]), true),
        (json!({ "count": 0 }), false),
        (json!(null), false),
        (json!(0), false),
        (json!(""), false),
    ] {
        let storage = storage();
        let mut manifest = manifest(&[create_mock_frame("frame-1", "good")]);
        manifest["reservoir"] = reservoir_value.clone();

        let files = frame_files("frame-1", true, true);
        let borrowed = files
            .iter()
            .map(|(path, contents)| (path.as_str(), contents.as_slice()))
            .collect::<Vec<_>>();
        let archive = zip_file(Some(manifest), &borrowed);

        let result = import_dataset_from_archive(&archive, &storage, &reservoir);
        assert_eq!(
            result.is_err(),
            clear_runs,
            "reservoir {reservoir_value:?} should {}reach clear()",
            if clear_runs { "" } else { "not " }
        );
        // The frame phase always completes before the reservoir phase.
        assert_eq!(storage.load_dataset().unwrap().frames.len(), 1);
    }
}

#[test]
fn reconstructs_typed_keypoints_from_binary() {
    let storage = storage();
    import(
        &archive_for_frames(&[create_mock_frame("frame-1", "good")]),
        &storage,
    );
    let keypoint = storage.load_dataset().unwrap().frames[0].keypoints[0];

    assert!((keypoint.x - 10.0).abs() < 0.01);
    assert!((keypoint.y - 20.0).abs() < 0.01);
    assert!((keypoint.score - 0.9).abs() < 0.01);
    assert!(keypoint.x.is_finite());
    assert!(keypoint.y.is_finite());
    assert!(keypoint.score.is_finite());
}
