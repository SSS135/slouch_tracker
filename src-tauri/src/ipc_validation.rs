//! Pure validation and parsing helpers for the raw IPC trust boundary.
//!
//! This file is compiled twice on purpose: `api.rs` includes it as a private
//! submodule for the production commands, and `tests/ipc_security.rs` includes
//! it at the test-crate root (the app lib keeps its modules private, so source
//! inclusion is the only integration seam). Everything here must stay pure:
//! inputs are plain values or `HeaderMap`s, never `tauri::ipc::Request`.

use tauri::http::HeaderMap;

use crate::errors::ApiError;
use slouch_domain::ported::messages::schemas::ImageData;
use slouch_domain::{FrameLabel, TrainingSettings};

pub(crate) const IPC_VERSION: &str = "1";
pub(crate) const MAX_IMAGE_WIDTH: u32 = 1920;
pub(crate) const MAX_IMAGE_HEIGHT: u32 = 1080;
pub(crate) const MAX_IMAGE_BYTES: usize = 8_294_400;
pub(crate) const MAX_THUMBNAIL_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const MAX_PAGE_SIZE: usize = 100;
pub(crate) const MAX_SAFE_JS_INTEGER: u64 = 9_007_199_254_740_991;

pub(crate) fn validate_page_limit(limit: usize) -> Result<(), ApiError> {
    if limit == 0 || limit > MAX_PAGE_SIZE {
        return Err(ApiError::InvalidRequest(
            "dataset page limit must be between 1 and 100".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_thumbnail_size(byte_len: usize) -> Result<(), ApiError> {
    if byte_len == 0 || byte_len > MAX_THUMBNAIL_BYTES {
        return Err(ApiError::InvalidRequest(
            "thumbnail must contain between 1 byte and 2 MiB".into(),
        ));
    }
    Ok(())
}

/// Header-driven core of `infer_frame` raw-image parsing. The IPC version and
/// raw-body checks stay in the `Request`-facing wrapper in `api.rs`.
pub(crate) fn parse_raw_image_from(
    headers: &HeaderMap,
    bytes: &[u8],
) -> Result<ImageData, ApiError> {
    if header_string_value(headers, "x-slouch-pixel-format")? != "rgba8" {
        return Err(ApiError::InvalidRequest(
            "only rgba8 raw frames are supported".into(),
        ));
    }
    let width = u32::try_from(parse_header_value(headers, "x-slouch-width")?)
        .map_err(|_| ApiError::InvalidRequest("image width overflows u32".into()))?;
    let height = u32::try_from(parse_header_value(headers, "x-slouch-height")?)
        .map_err(|_| ApiError::InvalidRequest("image height overflows u32".into()))?;
    let stride = usize::try_from(parse_header_value(headers, "x-slouch-stride")?)
        .map_err(|_| ApiError::InvalidRequest("image stride overflows platform limits".into()))?;
    validate_image_dimensions(width, height)?;
    validate_image_layout(width, height, stride, bytes.len())?;
    Ok(ImageData {
        data: bytes.to_vec(),
        width,
        height,
    })
}

pub(crate) fn validate_image_dimensions(width: u32, height: u32) -> Result<(), ApiError> {
    if width == 0 || height == 0 || width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_HEIGHT {
        return Err(ApiError::InvalidRequest(
            "image dimensions are outside the native limits".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_image_layout(
    width: u32,
    height: u32,
    stride: usize,
    byte_len: usize,
) -> Result<(), ApiError> {
    let row_bytes = usize::try_from(width)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ApiError::InvalidRequest("image width overflows".into()))?;
    if stride != row_bytes {
        return Err(ApiError::InvalidRequest(
            "image stride must equal the tightly packed RGBA row".into(),
        ));
    }
    let expected = stride
        .checked_mul(height as usize)
        .ok_or_else(|| ApiError::InvalidRequest("image dimensions overflow".into()))?;
    if expected != byte_len {
        return Err(ApiError::InvalidRequest(format!(
            "raw frame has {byte_len} bytes, expected {expected}"
        )));
    }
    if expected > MAX_IMAGE_BYTES {
        return Err(ApiError::InvalidRequest(
            "raw frame exceeds the 8 MiB limit".into(),
        ));
    }
    Ok(())
}

pub(crate) fn require_ipc_version_header(headers: &HeaderMap) -> Result<(), ApiError> {
    if header_string_value(headers, "x-slouch-ipc-version")? != IPC_VERSION {
        return Err(ApiError::InvalidRequest(
            "unsupported raw IPC version".into(),
        ));
    }
    Ok(())
}

pub(crate) fn ensure_js_safe_u64(value: u64, name: &str) -> Result<(), ApiError> {
    if value > MAX_SAFE_JS_INTEGER {
        return Err(ApiError::Storage(format!(
            "{name} exceeds JavaScript's safe integer range"
        )));
    }
    Ok(())
}

pub(crate) fn ensure_js_safe_usize(value: usize, name: &str) -> Result<(), ApiError> {
    let value = u64::try_from(value)
        .map_err(|_| ApiError::Storage(format!("{name} exceeds native integer limits")))?;
    ensure_js_safe_u64(value, name)
}

pub(crate) fn ensure_js_safe_timestamp(value: f64, name: &str) -> Result<(), ApiError> {
    if !value.is_finite()
        || value <= 0.0
        || value.fract() != 0.0
        || value > MAX_SAFE_JS_INTEGER as f64
    {
        return Err(ApiError::Storage(format!(
            "{name} is not a positive JavaScript-safe integer"
        )));
    }
    Ok(())
}

pub(crate) fn validate_id(id: &str) -> Result<(), ApiError> {
    if id.trim().is_empty() || id.len() > 128 {
        return Err(ApiError::InvalidRequest(
            "ID must contain between 1 and 128 bytes".into(),
        ));
    }
    Ok(())
}

pub(crate) fn parse_frame_label(value: &str) -> Result<FrameLabel, ApiError> {
    match value {
        "good" => Ok(FrameLabel::Good),
        "bad" => Ok(FrameLabel::Bad),
        "away" => Ok(FrameLabel::Away),
        "unused" => Ok(FrameLabel::Unused),
        _ => Err(ApiError::InvalidRequest("invalid frame label".into())),
    }
}

pub(crate) fn validate_training_settings(settings: &TrainingSettings) -> Result<(), ApiError> {
    // Only an upper sanity bound is enforced, matching the training worker: the
    // evaluation layer skips CV for cv_folds <= 1, so 0/1 are valid "no CV" values.
    if settings.cv_folds > 100 {
        return Err(ApiError::InvalidRequest(
            "cvFolds must not exceed 100".into(),
        ));
    }
    if !settings.last_updated.is_finite() || settings.last_updated <= 0.0 {
        return Err(ApiError::InvalidRequest(
            "lastUpdated must be a positive finite number".into(),
        ));
    }
    if settings.posture_feature_types.is_empty() || settings.presence_feature_types.is_empty() {
        return Err(ApiError::InvalidRequest(
            "posture and presence feature selections must not be empty".into(),
        ));
    }
    for (name, values) in [
        ("postureFeatureTypes", &settings.posture_feature_types),
        ("presenceFeatureTypes", &settings.presence_feature_types),
    ] {
        if values.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ApiError::InvalidRequest(format!(
                "{name} must contain unique feature IDs in registry order"
            )));
        }
    }
    if settings.dim_reduction_config.components == 0
        || settings.dim_reduction_config.components > 1_048_576
    {
        return Err(ApiError::InvalidRequest(
            "dimensionality-reduction components are outside native limits".into(),
        ));
    }
    slouch_ml::ported::classifier_registry::create_classifier(&settings.classifier_config)
        .map(|_| ())
        .map_err(|error| {
            ApiError::InvalidRequest(format!("invalid classifier configuration: {error}"))
        })
}

pub(crate) fn parse_header_value(headers: &HeaderMap, name: &str) -> Result<u64, ApiError> {
    let value = header_string_value(headers, name)?.parse().map_err(|_| {
        ApiError::InvalidRequest(format!("header {name} must be an unsigned integer"))
    })?;
    if value > MAX_SAFE_JS_INTEGER {
        return Err(ApiError::InvalidRequest(format!(
            "header {name} exceeds JavaScript's safe integer range"
        )));
    }
    Ok(value)
}

pub(crate) fn header_string_value(headers: &HeaderMap, name: &str) -> Result<String, ApiError> {
    let values = headers.get_all(name);
    if values.iter().count() != 1 {
        return Err(ApiError::InvalidRequest(format!(
            "header {name} must occur exactly once"
        )));
    }
    values
        .iter()
        .next()
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .ok_or_else(|| ApiError::InvalidRequest(format!("header {name} is not valid UTF-8")))
}
