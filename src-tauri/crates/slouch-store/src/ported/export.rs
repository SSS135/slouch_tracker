//! Dataset export service.
//!
//! The browser implementation emits two named MessagePack payloads inside a ZIP
//! archive.  Native callers receive the same bytes and can hand them to the
//! platform download boundary.

use std::io::{Cursor, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use rmp_serde::encode::Error as MessagePackError;
use serde::Serialize;
use slouch_domain::{
    validate_posture_frame, DatasetManifest, FeatureId, FrameLabel, FrameMetadata, PostureDataset,
    PostureFrame,
};
use slouch_ml::ported::constants::{ENGINEERED_1D_DIMS, RTMDET_ENGINEERED_DIMS};
use slouch_ml::ported::engineered_features::extract_engineered_features;
use slouch_ml::ported::rtmdet_engineered_features::extract_rtm_det_engineered_features;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::feature_reservoir::{
    feature_reservoir, FeatureReservoir, ReservoirError, ReservoirSample,
};

const EXPECTED_KEYPOINTS: usize = 17;
const EXPECTED_BBOX_VALUES: usize = 7;
const MAX_EXPORT_FRAMES: usize = 100_000;
const MAX_ENCODED_EXPORT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug)]
pub enum ExportError {
    NoLabeledFrames,
    InvalidKeypoints {
        kind: &'static str,
        index: usize,
        actual: usize,
    },
    InvalidLabel {
        index: usize,
        label: FrameLabel,
    },
    NoReservoirSamples,
    InvalidFrame(String),
    DatasetExport(String),
    Reservoir(ReservoirError),
    EngineeredFeatures(String),
    RtmDetEngineeredFeatures(String),
    MessagePack(MessagePackError),
    Archive(zip::result::ZipError),
    Io(std::io::Error),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLabeledFrames => formatter.write_str("No labeled frames to export"),
            Self::InvalidKeypoints {
                kind,
                index,
                actual,
            } => write!(
                formatter,
                "{kind} {index} has invalid keypoints: expected {EXPECTED_KEYPOINTS}, got {actual}"
            ),
            Self::InvalidLabel { index, label } => write!(
                formatter,
                "Frame {index} has invalid label: {label:?}. Only good/bad/away frames can be exported"
            ),
            Self::NoReservoirSamples => formatter.write_str("No reservoir samples to export"),
            Self::InvalidFrame(message) => formatter.write_str(message),
            Self::DatasetExport(message) => write!(formatter, "Dataset export failed: {message}"),
            Self::Reservoir(error) => error.fmt(formatter),
            Self::EngineeredFeatures(error) => error.fmt(formatter),
            Self::RtmDetEngineeredFeatures(error) => error.fmt(formatter),
            Self::MessagePack(error) => error.fmt(formatter),
            Self::Archive(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<ReservoirError> for ExportError {
    fn from(error: ReservoirError) -> Self {
        Self::Reservoir(error)
    }
}

impl From<MessagePackError> for ExportError {
    fn from(error: MessagePackError) -> Self {
        Self::MessagePack(error)
    }
}

impl From<zip::result::ZipError> for ExportError {
    fn from(error: zip::result::ZipError) -> Self {
        Self::Archive(error)
    }
}

impl From<std::io::Error> for ExportError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, Serialize)]
struct MessagePackData {
    version: u32,
    #[serde(rename = "exportedAt")]
    exported_at: String,
    #[serde(rename = "frameCount")]
    frame_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Vec<FrameLabel>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamps: Option<Vec<f64>>,
    keypoints: Vec<f64>,
    bboxes: Vec<f64>,
    backbone_avg: Vec<f64>,
    backbone_max: Vec<f64>,
    backbone_std: Vec<f64>,
    gau_avg: Vec<f64>,
    gau_max: Vec<f64>,
    gau_std: Vec<f64>,
    rtmdet: Vec<f64>,
    engineered: Vec<f64>,
    rtmdet_engineered: Vec<f64>,
    shapes: MessagePackShapes,
    #[serde(skip_serializing_if = "Option::is_none")]
    reservoir: Option<MessagePackReservoir>,
}

#[derive(Debug, Clone, Serialize)]
struct MessagePackShapes {
    keypoints: [usize; 3],
    bboxes: [usize; 2],
    backbone_avg: [usize; 2],
    backbone_max: [usize; 2],
    backbone_std: [usize; 2],
    gau_avg: [usize; 2],
    gau_max: [usize; 2],
    gau_std: [usize; 2],
    rtmdet: [usize; 2],
    engineered: [usize; 2],
    rtmdet_engineered: [usize; 2],
}

#[derive(Debug, Clone, Serialize)]
struct MessagePackReservoir {
    count: usize,
    #[serde(rename = "totalSeen")]
    total_seen: usize,
    #[serde(rename = "maxSamples")]
    max_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadPayload {
    pub filename: String,
    pub bytes: Vec<u8>,
}

/// Build the metadata manifest used by the dataset import boundary.
pub fn build_manifest(dataset: &PostureDataset) -> DatasetManifest {
    let frames = dataset
        .frames
        .iter()
        .map(|frame| FrameMetadata {
            id: frame.id.clone(),
            label: frame.label,
            timestamp: frame.timestamp,
            features: frame.features.keys().copied().collect(),
        })
        .collect();

    DatasetManifest {
        version: 3,
        exported_at: now_iso8601(),
        frame_count: dataset.frames.len(),
        frames,
        reservoir: None,
    }
}

/// Encode all labeled frames as the source-compatible MessagePack object.
pub fn export_labeled_data(dataset: &PostureDataset) -> Result<Vec<u8>, ExportError> {
    let frames = &dataset.frames;
    let frame_count = frames.len();
    if frame_count == 0 {
        return Err(ExportError::NoLabeledFrames);
    }

    validate_labeled_frames(frames)?;
    validate_labeled_feature_shapes(frames)?;

    let backbone_dim = feature_dimension(frames, FeatureId::BackboneFeatures);
    let gau_dim = feature_dimension(frames, FeatureId::GauFeatures);
    let rtmdet_dim = feature_dimension(frames, FeatureId::RtmDetExtracted);

    let engineered = frames
        .iter()
        .map(engineered_for_frame)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .map(f64::from)
        .collect();
    let rtmdet_engineered = frames
        .iter()
        .map(rtm_det_for_frame)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .map(f64::from)
        .collect();

    let data = MessagePackData {
        version: 3,
        exported_at: now_iso8601(),
        frame_count,
        labels: Some(frames.iter().map(|frame| frame.label).collect()),
        timestamps: Some(frames.iter().map(|frame| frame.timestamp).collect()),
        keypoints: flatten_labeled_keypoints(frames),
        bboxes: flatten_labeled_bboxes(frames),
        backbone_avg: flatten_labeled_feature(frames, FeatureId::BackboneFeatures),
        backbone_max: flatten_labeled_feature(frames, FeatureId::BackboneFeaturesMax),
        backbone_std: flatten_labeled_feature(frames, FeatureId::BackboneFeaturesStd),
        gau_avg: flatten_labeled_feature(frames, FeatureId::GauFeatures),
        gau_max: flatten_labeled_feature(frames, FeatureId::GauFeaturesMax),
        gau_std: flatten_labeled_feature(frames, FeatureId::GauFeaturesStd),
        rtmdet: flatten_labeled_feature(frames, FeatureId::RtmDetExtracted),
        engineered,
        rtmdet_engineered,
        shapes: shapes(frame_count, backbone_dim, gau_dim, rtmdet_dim),
        reservoir: None,
    };

    let encoded = rmp_serde::to_vec_named(&data)?;
    if encoded.len() > MAX_ENCODED_EXPORT_BYTES {
        return Err(ExportError::InvalidFrame(format!(
            "encoded labeled export exceeds the {MAX_ENCODED_EXPORT_BYTES}-byte limit"
        )));
    }
    Ok(encoded)
}

/// Encode the process-wide unlabeled feature reservoir as MessagePack.
pub fn export_unlabeled_data() -> Result<Vec<u8>, ExportError> {
    export_unlabeled_data_from(feature_reservoir())
}

/// Explicit-reservoir variant used by native storage and deterministic tests.
pub fn export_unlabeled_data_from(reservoir: &FeatureReservoir) -> Result<Vec<u8>, ExportError> {
    let samples = reservoir.get_all_samples()?;
    let metadata = reservoir.get_meta()?;
    if samples.is_empty() {
        return Err(ExportError::NoReservoirSamples);
    }

    validate_reservoir_samples(&samples)?;

    let frame_count = samples.len();
    let first = &samples[0];
    let data = MessagePackData {
        version: 3,
        exported_at: now_iso8601(),
        frame_count,
        labels: None,
        timestamps: None,
        keypoints: flatten_unlabeled_keypoints(&samples),
        bboxes: flatten_unlabeled_bboxes(&samples),
        backbone_avg: flatten_unlabeled_feature(&samples, backbone_avg),
        backbone_max: flatten_unlabeled_feature(&samples, backbone_max),
        backbone_std: flatten_unlabeled_feature(&samples, backbone_std),
        gau_avg: flatten_unlabeled_feature(&samples, gau_avg),
        gau_max: flatten_unlabeled_feature(&samples, gau_max),
        gau_std: flatten_unlabeled_feature(&samples, gau_std),
        rtmdet: flatten_unlabeled_feature(&samples, rtmdet),
        engineered: samples
            .iter()
            .map(|sample| engineered_for_keypoints(&sample.keypoints))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .map(f64::from)
            .collect(),
        rtmdet_engineered: samples
            .iter()
            .map(|sample| {
                extract_rtm_det_engineered_features(Some(&sample.keypoints), Some(&sample.bbox))
                    .map_err(|error| ExportError::RtmDetEngineeredFeatures(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .map(f64::from)
            .collect(),
        shapes: shapes(
            frame_count,
            first.backbone_avg.len(),
            first.gau_avg.len(),
            first.rtmdet.len(),
        ),
        reservoir: Some(MessagePackReservoir {
            count: metadata.count,
            total_seen: metadata.total_seen,
            max_samples: metadata.max_samples,
        }),
    };

    let encoded = rmp_serde::to_vec_named(&data)?;
    if encoded.len() > MAX_ENCODED_EXPORT_BYTES {
        return Err(ExportError::InvalidFrame(format!(
            "encoded reservoir export exceeds the {MAX_ENCODED_EXPORT_BYTES}-byte limit"
        )));
    }
    Ok(encoded)
}

/// Create a legacy oracle ZIP containing `labeled.msgpack` and `unlabeled.msgpack`.
/// Production native export uses `DatasetStorage::export_archive` and never
/// publishes these browser-compatibility bytes.
pub fn export_dataset_to_zip(dataset: &PostureDataset) -> Result<Vec<u8>, ExportError> {
    (|| {
        let labeled = export_labeled_data(dataset)?;
        let unlabeled = export_unlabeled_data()?;
        let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

        archive.start_file("labeled.msgpack", options)?;
        archive.write_all(&labeled)?;
        archive.start_file("unlabeled.msgpack", options)?;
        archive.write_all(&unlabeled)?;

        let bytes = archive.finish()?.into_inner();
        if bytes.len() > MAX_ENCODED_EXPORT_BYTES {
            return Err(ExportError::InvalidFrame(format!(
                "ZIP export exceeds the {MAX_ENCODED_EXPORT_BYTES}-byte limit"
            )));
        }
        Ok(bytes)
    })()
    .map_err(|error: ExportError| ExportError::DatasetExport(error.to_string()))
}

/// Replace the browser-only `file-saver` side effect with a native download payload.
pub fn download_dataset_export(
    zip_bytes: Vec<u8>,
    filename: Option<&str>,
) -> Result<DownloadPayload, ExportError> {
    let filename = filename
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(default_filename);
    Ok(DownloadPayload {
        filename,
        bytes: zip_bytes,
    })
}

pub fn export_and_download_dataset(
    dataset: &PostureDataset,
    filename: Option<&str>,
) -> Result<DownloadPayload, ExportError> {
    download_dataset_export(export_dataset_to_zip(dataset)?, filename)
}

fn validate_labeled_frames(frames: &[PostureFrame]) -> Result<(), ExportError> {
    if frames.len() > MAX_EXPORT_FRAMES {
        return Err(ExportError::InvalidFrame(format!(
            "dataset contains more than {MAX_EXPORT_FRAMES} frames"
        )));
    }
    for (index, frame) in frames.iter().enumerate() {
        if frame.keypoints.len() != EXPECTED_KEYPOINTS {
            return Err(ExportError::InvalidFrame(format!(
                "Frame {index} (id: {}) has invalid keypoints: expected {EXPECTED_KEYPOINTS}, got {}",
                frame.id,
                frame.keypoints.len(),
            )));
        }
        if !matches!(
            frame.label,
            FrameLabel::Good | FrameLabel::Bad | FrameLabel::Away
        ) {
            return Err(ExportError::InvalidFrame(format!(
                "Frame {index} (id: {}) has invalid label: {}. Only good/bad/away frames can be exported",
                frame.id,
                frame_label_wire(frame.label),
            )));
        }
        validate_posture_frame(frame).map_err(|error| {
            ExportError::InvalidFrame(format!(
                "Frame {index} (id: {}) is invalid: {error}",
                frame.id
            ))
        })?;
    }
    Ok(())
}

fn validate_labeled_feature_shapes(frames: &[PostureFrame]) -> Result<(), ExportError> {
    const STORED_FEATURES: [FeatureId; 7] = [
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
        FeatureId::RtmDetExtracted,
    ];
    for feature in STORED_FEATURES {
        let any_present = frames
            .iter()
            .any(|frame| frame.features.contains_key(&feature));
        if !any_present {
            continue;
        }
        for frame in frames {
            let actual = frame.features.get(&feature).map_or(0, Vec::len);
            let expected = feature.metadata().dimensions;
            if actual != expected {
                return Err(ExportError::InvalidFrame(format!(
                    "Frame {} feature {} has dimension {actual}; expected {expected}",
                    frame.id,
                    feature.as_str(),
                )));
            }
        }
    }
    Ok(())
}

fn validate_reservoir_samples(samples: &[ReservoirSample]) -> Result<(), ExportError> {
    if samples.len() > MAX_EXPORT_FRAMES {
        return Err(ExportError::InvalidFrame(format!(
            "reservoir contains more than {MAX_EXPORT_FRAMES} samples"
        )));
    }
    for (index, sample) in samples.iter().enumerate() {
        super::storage::validate_reservoir_sample(sample).map_err(|error| {
            ExportError::InvalidFrame(format!("Reservoir sample {index} is invalid: {error}"))
        })?;
    }
    Ok(())
}

fn frame_label_wire(label: FrameLabel) -> &'static str {
    match label {
        FrameLabel::Good => "good",
        FrameLabel::Bad => "bad",
        FrameLabel::Away => "away",
        FrameLabel::Unused => "unused",
    }
}

fn flatten_labeled_keypoints(frames: &[PostureFrame]) -> Vec<f64> {
    frames
        .iter()
        .flat_map(|frame| {
            frame
                .keypoints
                .iter()
                .flat_map(|keypoint| [keypoint.x, keypoint.y, keypoint.score])
        })
        .collect()
}

fn flatten_labeled_bboxes(frames: &[PostureFrame]) -> Vec<f64> {
    frames
        .iter()
        .flat_map(|frame| bbox_values(&frame.bbox))
        .collect()
}

fn flatten_labeled_feature(frames: &[PostureFrame], feature: FeatureId) -> Vec<f64> {
    frames
        .iter()
        .filter_map(|frame| frame.features.get(&feature))
        .flatten()
        .map(|value| f64::from(*value))
        .collect()
}

fn flatten_unlabeled_keypoints(samples: &[ReservoirSample]) -> Vec<f64> {
    samples
        .iter()
        .flat_map(|sample| {
            sample
                .keypoints
                .iter()
                .flat_map(|keypoint| [keypoint.x, keypoint.y, keypoint.score])
        })
        .collect()
}

fn flatten_unlabeled_bboxes(samples: &[ReservoirSample]) -> Vec<f64> {
    samples
        .iter()
        .flat_map(|sample| bbox_values(&sample.bbox))
        .collect()
}

fn flatten_unlabeled_feature(
    samples: &[ReservoirSample],
    select: fn(&ReservoirSample) -> &[f32],
) -> Vec<f64> {
    samples
        .iter()
        .flat_map(select)
        .copied()
        .map(f64::from)
        .collect()
}

fn backbone_avg(sample: &ReservoirSample) -> &[f32] {
    &sample.backbone_avg
}

fn backbone_max(sample: &ReservoirSample) -> &[f32] {
    &sample.backbone_max
}

fn backbone_std(sample: &ReservoirSample) -> &[f32] {
    &sample.backbone_std
}

fn gau_avg(sample: &ReservoirSample) -> &[f32] {
    &sample.gau_avg
}

fn gau_max(sample: &ReservoirSample) -> &[f32] {
    &sample.gau_max
}

fn gau_std(sample: &ReservoirSample) -> &[f32] {
    &sample.gau_std
}

fn rtmdet(sample: &ReservoirSample) -> &[f32] {
    &sample.rtmdet
}

fn bbox_values(bbox: &slouch_domain::BoundingBox) -> [f64; EXPECTED_BBOX_VALUES] {
    [
        bbox.x1,
        bbox.y1,
        bbox.x2,
        bbox.y2,
        bbox.score,
        bbox.width,
        bbox.height,
    ]
}

fn feature_dimension(frames: &[PostureFrame], feature: FeatureId) -> usize {
    frames
        .iter()
        .find_map(|frame| frame.features.get(&feature).map(Vec::len))
        .unwrap_or(0)
}

fn engineered_for_frame(frame: &PostureFrame) -> Result<Vec<f32>, ExportError> {
    engineered_for_keypoints(&frame.keypoints)
}

fn engineered_for_keypoints(
    keypoints: &[slouch_domain::Keypoint],
) -> Result<Vec<f32>, ExportError> {
    extract_engineered_features(keypoints)
        .map_err(|error| ExportError::EngineeredFeatures(error.to_string()))
        .map(|features| features.unwrap_or_default())
}

fn rtm_det_for_frame(frame: &PostureFrame) -> Result<Vec<f32>, ExportError> {
    extract_rtm_det_engineered_features(Some(&frame.keypoints), Some(&frame.bbox))
        .map_err(|error| ExportError::RtmDetEngineeredFeatures(error.to_string()))
}

fn shapes(
    frame_count: usize,
    backbone_dim: usize,
    gau_dim: usize,
    rtmdet_dim: usize,
) -> MessagePackShapes {
    MessagePackShapes {
        keypoints: [frame_count, EXPECTED_KEYPOINTS, 3],
        bboxes: [frame_count, EXPECTED_BBOX_VALUES],
        backbone_avg: [frame_count, backbone_dim],
        backbone_max: [frame_count, backbone_dim],
        backbone_std: [frame_count, backbone_dim],
        gau_avg: [frame_count, gau_dim],
        gau_max: [frame_count, gau_dim],
        gau_std: [frame_count, gau_dim],
        rtmdet: [frame_count, rtmdet_dim],
        engineered: [frame_count, ENGINEERED_1D_DIMS],
        rtmdet_engineered: [frame_count, RTMDET_ENGINEERED_DIMS],
    }
}

fn default_filename() -> String {
    let timestamp = now_iso8601();
    let timestamp = timestamp
        .split('.')
        .next()
        .unwrap_or(&timestamp)
        .replace([':', 'T'], "-");
    format!("slouch-tracker-dataset-{timestamp}.zip")
}

fn now_iso8601() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = duration.as_secs() as i64;
    let milliseconds = duration.subsec_millis();
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day / 60 % 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{milliseconds:03}Z")
}

// Howard Hinnant's Gregorian civil-date conversion, kept local to avoid a
// datetime dependency in the storage crate.
fn civil_from_days(days_since_1970: i64) -> (i64, i64, i64) {
    let shifted = days_since_1970 + 719_468;
    let era = if shifted >= 0 {
        shifted / 146_097
    } else {
        (shifted - 146_096) / 146_097
    };
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::{civil_from_days, now_iso8601};

    #[test]
    fn formats_current_time_as_iso8601() {
        assert!(now_iso8601().ends_with('Z'));
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(18_262), (2020, 1, 1));
    }
}
