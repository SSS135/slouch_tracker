//! Dataset import service.
//!
//! The browser implementation consumes a JSZip instance and reconstructs frames
//! from `manifest.json`, thumbnails, and little-endian binary float16 files.
//! Native archive decoding is deliberately kept behind [`ZipArchive`], so the
//! application boundary can choose its ZIP implementation without moving byte
//! decoding or import semantics out of this module.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read},
};

use serde_json::Value;
use slouch_domain::{
    validate_posture_frame, BoundingBox, FrameLabel, ImportResult, PostureDataset, PostureFrame,
    Thumbnail,
};

use super::feature_reservoir::{feature_reservoir, FeatureReservoir, ReservoirSample};

const MANIFEST_VERSION: u64 = 2;
const BATCH_SIZE: usize = 10;
const MAX_ARCHIVE_BYTES: usize = 512 * 1024 * 1024;
const MAX_ENTRY_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_RESERVOIR_SAMPLES: usize = 1_000;

/// Read-only access to the entries of a parsed ZIP archive.
///
/// Implementations should return `Ok(None)` for an absent entry and `Err` only
/// when the archive itself cannot be read. Returned bytes are owned because a
/// ZIP implementation may need to release an internal decompression buffer
/// after the call.
pub trait ZipArchive {
    fn read_entry(&self, path: &str) -> Result<Option<Vec<u8>>, String>;
}

/// Minimal storage boundary used by the importer.
///
/// `DatasetStorage` can implement this trait without coupling archive parsing
/// to its persistence backend.
pub trait DatasetStore {
    fn load_dataset(&self) -> Result<PostureDataset, String>;
    fn save_frame(&self, frame: PostureFrame) -> Result<(), String>;
}

/// A small in-memory archive adapter useful for native command handlers and
/// deterministic tests. ZIP decompression remains the responsibility of the
/// caller that populates it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MemoryZipArchive {
    entries: BTreeMap<String, Vec<u8>>,
}

impl MemoryZipArchive {
    pub fn new(entries: impl IntoIterator<Item = (String, Vec<u8>)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    pub fn insert(&mut self, path: impl Into<String>, bytes: Vec<u8>) {
        self.entries.insert(path.into(), bytes);
    }
}

impl ZipArchive for MemoryZipArchive {
    fn read_entry(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        Ok(self.entries.get(path).cloned())
    }
}

impl ZipArchive for [u8] {
    fn read_entry(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        read_zip_entry(self, path)
    }
}

impl ZipArchive for Vec<u8> {
    fn read_entry(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        read_zip_entry(self, path)
    }
}

impl<const N: usize> ZipArchive for [u8; N] {
    fn read_entry(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        read_zip_entry(self, path)
    }
}

fn read_zip_entry(bytes: &[u8], path: &str) -> Result<Option<Vec<u8>>, String> {
    if bytes.len() > MAX_ARCHIVE_BYTES {
        return Err(format!(
            "ZIP archive exceeds the {MAX_ARCHIVE_BYTES}-byte limit"
        ));
    }
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|error| format!("invalid ZIP archive: {error}"))?;
    let entry = match archive.by_name(path) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(error) => return Err(format!("failed to read ZIP entry {path}: {error}")),
    };
    if entry.size() > MAX_ENTRY_BYTES as u64 {
        return Err(format!(
            "ZIP entry {path} exceeds the {MAX_ENTRY_BYTES}-byte limit"
        ));
    }
    let mut payload = Vec::with_capacity(entry.size() as usize);
    entry
        .take(MAX_ENTRY_BYTES as u64 + 1)
        .read_to_end(&mut payload)
        .map_err(|error| format!("failed to decompress ZIP entry {path}: {error}"))?;
    if payload.len() > MAX_ENTRY_BYTES {
        return Err(format!(
            "ZIP entry {path} exceeds the {MAX_ENTRY_BYTES}-byte limit"
        ));
    }
    Ok(Some(payload))
}

fn read_bounded_entry<A: ZipArchive + ?Sized>(
    archive: &A,
    path: &str,
    limit: usize,
) -> Result<Option<Vec<u8>>, String> {
    let value = archive.read_entry(path)?;
    if value.as_ref().is_some_and(|bytes| bytes.len() > limit) {
        return Err(format!("ZIP entry {path} exceeds the {limit}-byte limit"));
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq)]
struct ImportManifest {
    frame_count: usize,
    frames: Vec<ImportFrameMetadata>,
    reservoir_requires_clear: bool,
    reservoir_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ImportFrameMetadata {
    id: String,
    timestamp: f64,
    label: FrameLabel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetImportError {
    Archive(String),
    Manifest(String),
    Storage(String),
}

impl std::fmt::Display for DatasetImportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Archive(message) | Self::Manifest(message) | Self::Storage(message) => message,
        };
        write!(formatter, "Dataset import failed: {message}")
    }
}

impl std::error::Error for DatasetImportError {}

/// Import a dataset from an already parsed ZIP archive.
///
/// This is the native equivalent of `importDatasetFromZip`. ZIP parsing is
/// supplied by the caller through [`ZipArchive`]; frame failures are isolated
/// and reported in [`ImportResult`] so later frames continue to import.
pub fn import_dataset_from_zip<A: ZipArchive + ?Sized, S: DatasetStore>(
    archive: &A,
    storage: &S,
) -> Result<ImportResult, DatasetImportError> {
    import_dataset_from_archive(archive, storage, feature_reservoir())
}

/// Import using an explicit reservoir, primarily for tests or an application
/// that owns more than one reservoir store.
pub fn import_dataset_from_archive<A: ZipArchive + ?Sized, S: DatasetStore>(
    archive: &A,
    storage: &S,
    reservoir: &FeatureReservoir,
) -> Result<ImportResult, DatasetImportError> {
    let mut result = ImportResult {
        imported: 0,
        skipped: 0,
        errors: Vec::new(),
    };

    let manifest = parse_manifest(archive).map_err(DatasetImportError::Manifest)?;
    if manifest.frame_count == 0 {
        return Ok(result);
    }

    let existing_frames = storage
        .load_dataset()
        .map_err(DatasetImportError::Storage)?;
    let mut existing_ids: BTreeSet<String> = existing_frames
        .frames
        .into_iter()
        .map(|frame| frame.id)
        .collect();

    for batch in manifest.frames.chunks(BATCH_SIZE) {
        for frame_metadata in batch {
            match import_frame(archive, storage, frame_metadata, &mut existing_ids) {
                Ok(()) => result.imported += 1,
                Err(FrameImportError::Skipped(message)) => {
                    result.skipped += 1;
                    result
                        .errors
                        .push(format!("Frame {}: {message}", frame_metadata.id));
                }
                Err(FrameImportError::Archive(message)) => {
                    result.skipped += 1;
                    result
                        .errors
                        .push(format!("Frame {}: {message}", frame_metadata.id));
                }
            }
        }
    }

    // The TS oracle catches and logs each per-sample reservoir failure
    // (import.ts:257-259), but `await reservoir.clear()` sits OUTSIDE that
    // try/catch (import.ts:225): a clear rejection reaches
    // importDatasetFromZip's outer catch and fails the entire import as
    // "Dataset import failed: ..." even though frames were already saved.
    // Mirror that split exactly: only the clear failure is fatal; per-sample
    // decode/add failures stay best-effort inside import_reservoir_from_zip.
    import_reservoir_from_zip(archive, &manifest, reservoir)
        .map_err(DatasetImportError::Storage)?;

    Ok(result)
}

fn import_frame<A: ZipArchive + ?Sized, S: DatasetStore>(
    archive: &A,
    storage: &S,
    frame_metadata: &ImportFrameMetadata,
    existing_ids: &mut BTreeSet<String>,
) -> Result<(), FrameImportError> {
    if existing_ids.contains(&frame_metadata.id) {
        return Err(FrameImportError::Skipped(
            "duplicate ID (skipped)".to_owned(),
        ));
    }

    if !validate_frame_files(archive, frame_metadata).map_err(FrameImportError::Archive)? {
        return Err(FrameImportError::Skipped(
            "missing files (skipped)".to_owned(),
        ));
    }

    let frame = load_frame_from_zip(archive, frame_metadata).map_err(FrameImportError::Archive)?;
    validate_posture_frame(&frame).map_err(|error| FrameImportError::Skipped(error.to_string()))?;
    let frame_id = frame.id.clone();
    storage
        .save_frame(frame)
        .map_err(FrameImportError::Skipped)?;
    existing_ids.insert(frame_id);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FrameImportError {
    Skipped(String),
    Archive(String),
}

fn validate_frame_files<A: ZipArchive + ?Sized>(
    archive: &A,
    frame_metadata: &ImportFrameMetadata,
) -> Result<bool, String> {
    let frame_path = format!("frames/frame-{}", frame_metadata.id);
    let thumbnail = read_bounded_entry(
        archive,
        &format!("{frame_path}/thumbnail.webp"),
        2 * 1024 * 1024,
    )?;
    if thumbnail.is_none() {
        return Ok(false);
    }

    let keypoints =
        read_bounded_entry(archive, &format!("{frame_path}/keypoints.bin"), 17 * 3 * 2)?;
    Ok(keypoints.is_some())
}

fn load_frame_from_zip<A: ZipArchive + ?Sized>(
    archive: &A,
    frame_metadata: &ImportFrameMetadata,
) -> Result<PostureFrame, String> {
    let result = (|| {
        let frame_path = format!("frames/frame-{}", frame_metadata.id);

        let thumbnail = read_bounded_entry(
            archive,
            &format!("{frame_path}/thumbnail.webp"),
            2 * 1024 * 1024,
        )?
        .ok_or_else(|| "Thumbnail file not found".to_owned())?;
        let keypoints_bytes =
            read_bounded_entry(archive, &format!("{frame_path}/keypoints.bin"), 17 * 3 * 2)?
                .ok_or_else(|| "keypoints.bin missing - frames must have keypoints".to_owned())?;
        let bbox_bytes = read_bounded_entry(archive, &format!("{frame_path}/bbox.bin"), 7 * 2)?
            .ok_or_else(|| "bbox.bin missing - frames must have bbox".to_owned())?;

        let keypoints = buffer_to_keypoints(&keypoints_bytes)?;
        let bbox = buffer_to_bbox(&bbox_bytes)?;

        Ok(PostureFrame {
            id: frame_metadata.id.clone(),
            timestamp: frame_metadata.timestamp,
            features: BTreeMap::new(),
            thumbnail: Thumbnail {
                mime_type: "image/webp".to_owned(),
                bytes: thumbnail,
            },
            keypoints,
            bbox,
            label: frame_metadata.label,
        })
    })();

    result.map_err(|error: String| format!("Failed to load frame {}: {error}", frame_metadata.id))
}

fn buffer_to_keypoints(buffer: &[u8]) -> Result<Vec<slouch_domain::Keypoint>, String> {
    let values = decode_float16_values(buffer)?;
    let mut keypoints = Vec::with_capacity(values.len() / 3);
    for chunk in values.chunks(3) {
        if chunk.len() != 3 {
            return Err("keypoints.bin has an incomplete keypoint".to_owned());
        }
        keypoints.push(slouch_domain::Keypoint::new(chunk[0], chunk[1], chunk[2]));
    }
    Ok(keypoints)
}

fn buffer_to_bbox(buffer: &[u8]) -> Result<BoundingBox, String> {
    let values = decode_float16_values(buffer)?;
    if values.len() != 7 {
        return Err(format!(
            "bbox.bin has {} values; expected exactly 7",
            values.len()
        ));
    }

    Ok(BoundingBox {
        x1: values[0],
        y1: values[1],
        x2: values[2],
        y2: values[3],
        score: values[4],
        width: values[5],
        height: values[6],
    })
}

fn decode_float16_values(buffer: &[u8]) -> Result<Vec<f64>, String> {
    if !buffer.len().is_multiple_of(2) {
        return Err("binary float16 data has an odd byte length".to_owned());
    }

    Ok(buffer
        .chunks_exact(2)
        .map(|bytes| float16_to_f64(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

/// Decode an IEEE-754 binary16 value without adding a float16 dependency.
fn float16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;

    match exponent {
        0 => {
            if fraction == 0 {
                sign * 0.0
            } else {
                sign * 2f64.powi(-14) * (fraction as f64 / 1024.0)
            }
        }
        0x1f => {
            if fraction == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * 2f64.powi(exponent - 15) * (1.0 + fraction as f64 / 1024.0),
    }
}

fn parse_manifest<A: ZipArchive + ?Sized>(archive: &A) -> Result<ImportManifest, String> {
    let manifest_bytes = read_bounded_entry(archive, "manifest.json", MAX_MANIFEST_BYTES)
        .map_err(|error| format!("failed to read manifest.json: {error}"))?
        .ok_or_else(|| "manifest.json not found in ZIP file".to_owned())?;

    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("Failed to parse manifest: {error}"))?;
    validate_manifest(&manifest)
}

fn validate_manifest(manifest: &Value) -> Result<ImportManifest, String> {
    let object = manifest
        .as_object()
        .ok_or_else(|| "Invalid manifest: not an object".to_owned())?;

    let version = object.get("version").and_then(json_nonnegative_integer);
    if version != Some(MANIFEST_VERSION) {
        return Err(format!(
            "Unsupported manifest version: {} (expected {})",
            json_value_or_undefined(object.get("version")),
            MANIFEST_VERSION
        ));
    }

    if object.get("exportedAt").and_then(Value::as_str).is_none() {
        return Err("Invalid manifest: exportedAt must be string".to_owned());
    }

    let frame_count_value = object
        .get("frameCount")
        .and_then(Value::as_f64)
        .ok_or_else(|| "Invalid manifest: frameCount must be non-negative number".to_owned())?;
    if !frame_count_value.is_finite() || frame_count_value < 0.0 || frame_count_value.fract() != 0.0
    {
        return Err("Invalid manifest: frameCount must be a non-negative integer".to_owned());
    }

    let frames = object
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "Invalid manifest: frames must be array".to_owned())?;
    if frames.len() as f64 != frame_count_value {
        return Err(format!(
            "Invalid manifest: frameCount mismatch ({} vs {})",
            frames.len(),
            frame_count_value
        ));
    }

    let mut parsed_frames = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let frame_object = frame
            .as_object()
            .ok_or_else(|| format!("Invalid manifest: frame {index} is not an object"))?;
        let id = frame_object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| format!("Invalid manifest: frame {index} has invalid id"))?;
        let label = parse_label(frame_object.get("label")).ok_or_else(|| {
            format!(
                "Invalid manifest: frame {index} has invalid label: {}",
                json_value_or_undefined(frame_object.get("label"))
            )
        })?;
        // The TS oracle's validateManifest checks only id, label, and features;
        // a missing/malformed timestamp is not manifest-fatal. Pass the value
        // through verbatim (NaN when absent/non-numeric) so per-frame
        // validate_posture_frame isolates a bad timestamp as a skipped frame
        // rather than aborting the whole import.
        let timestamp = frame_object
            .get("timestamp")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN);
        if !frame_object.get("features").is_some_and(Value::is_array) {
            return Err(format!(
                "Invalid manifest: frame {index} has invalid featureTypes array"
            ));
        }

        parsed_frames.push(ImportFrameMetadata {
            id: id.to_owned(),
            timestamp,
            label,
        });
    }

    // The TS oracle's validateManifest never inspects `reservoir`; it is
    // consumed by importReservoirFromZip (import.ts:219-264), where every
    // per-sample failure is caught and logged, so a non-numeric, negative, or
    // fractional count simply yields a loop that imports nothing while frame
    // import already succeeded. Mirror that lenient count parse, clamping to
    // MAX_RESERVOIR_SAMPLES as the native sample-count bound. The one fatal
    // step of that phase is the unguarded `await reservoir.clear()`
    // (import.ts:225), reached whenever the oracle's guard
    // `!manifest.reservoir || manifest.reservoir.count === 0` (import.ts:220)
    // passes; `reservoir_requires_clear` reproduces that JavaScript
    // truthiness/strict-equality gate.
    let reservoir_value = object.get("reservoir");
    let reservoir_requires_clear = reservoir_value.is_some_and(|value| {
        json_is_truthy(value)
            && !matches!(
                value
                    .as_object()
                    .and_then(|reservoir| reservoir.get("count"))
                    .and_then(Value::as_f64),
                Some(count) if count == 0.0
            )
    });
    let reservoir_count = reservoir_value
        .and_then(Value::as_object)
        .and_then(|reservoir| reservoir.get("count"))
        .and_then(json_nonnegative_integer)
        .and_then(|count| usize::try_from(count).ok())
        .map(|count| count.min(MAX_RESERVOIR_SAMPLES))
        .unwrap_or(0);

    Ok(ImportManifest {
        frame_count: frame_count_value as usize,
        frames: parsed_frames,
        reservoir_requires_clear,
        reservoir_count,
    })
}

fn json_nonnegative_integer(value: &Value) -> Option<u64> {
    let number = value.as_f64()?;
    if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > u64::MAX as f64 {
        return None;
    }
    Some(number as u64)
}

fn parse_label(value: Option<&Value>) -> Option<FrameLabel> {
    match value.and_then(Value::as_str) {
        Some("good") => Some(FrameLabel::Good),
        Some("bad") => Some(FrameLabel::Bad),
        Some("unused") => Some(FrameLabel::Unused),
        Some("away") => Some(FrameLabel::Away),
        _ => None,
    }
}

fn json_value_or_undefined(value: Option<&Value>) -> String {
    value.map_or_else(|| "undefined".to_owned(), ToString::to_string)
}

/// JavaScript truthiness for a JSON value: `null`, `false`, `0`, and `""` are
/// falsy; every other JSON value (including arrays and objects) is truthy.
fn json_is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn import_reservoir_from_zip<A: ZipArchive + ?Sized>(
    archive: &A,
    manifest: &ImportManifest,
    reservoir: &FeatureReservoir,
) -> Result<usize, String> {
    if !manifest.reservoir_requires_clear {
        return Ok(0);
    }

    // The oracle awaits this clear OUTSIDE its per-sample try/catch
    // (import.ts:225), so a failure here propagates and aborts the whole
    // import. Everything below is per-sample best-effort, matching the
    // oracle's catch-log-continue loop (import.ts:231-259).
    reservoir
        .clear()
        .map_err(|error| format!("failed to clear reservoir: {error}"))?;

    let mut imported = 0;
    for index in 0..manifest.reservoir_count {
        let sample_path = format!("reservoir/sample-{index}");
        let keypoints = match read_bounded_entry(
            archive,
            &format!("{sample_path}/keypoints.bin"),
            17 * 3 * 2,
        ) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(_) => continue,
        };
        let bbox = match read_bounded_entry(archive, &format!("{sample_path}/bbox.bin"), 7 * 2) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(_) => continue,
        };

        let sample = match (buffer_to_keypoints(&keypoints), buffer_to_bbox(&bbox)) {
            (Ok(keypoints), Ok(bbox)) => ReservoirSample {
                keypoints,
                bbox,
                backbone_avg: Vec::new(),
                backbone_max: Vec::new(),
                backbone_std: Vec::new(),
                gau_avg: Vec::new(),
                gau_max: Vec::new(),
                gau_std: Vec::new(),
                rtmdet: Vec::new(),
            },
            _ => continue,
        };

        // Browser-exported reservoir samples intentionally carry empty feature
        // vectors (features are excluded from export), which the native
        // registry-sized validator always rejects. The TS oracle performs no
        // dimension validation and skips per-sample failures, so treat an
        // invalid sample as a skip rather than failing the whole import.
        if super::storage::validate_reservoir_sample(&sample).is_err() {
            continue;
        }
        // `reservoir.add` sits inside the oracle's per-sample try/catch
        // (import.ts:255-259), so an add failure skips the sample rather than
        // aborting the import.
        if reservoir.add(sample).is_ok() {
            imported += 1;
        }
    }

    Ok(imported)
}
