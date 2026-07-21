//! Fast runtime guards for persisted posture data and feature identifiers.
//!
//! TypeScript receives `unknown` values at the IndexedDB boundary. Rust
//! deserialization gives these guards typed domain values instead, so the
//! structural checks are enforced by `PostureFrame` and `FeatureMap`; the
//! native boundary additionally applies the domain validation contract.

use std::str::FromStr;

use crate::{validate_posture_frame, FeatureId, PostureFrame};

/// Returns whether a typed posture frame satisfies the native frame contract.
///
/// The TypeScript fast guard accepts arbitrary feature-vector lengths, while
/// the native contract requires registry-sized, finite feature vectors and
/// valid thumbnail metadata. `validate_posture_frame` is the single source of
/// truth for those stricter boundary checks.
pub fn is_posture_frame(frame: &PostureFrame) -> bool {
    validate_posture_frame(frame).is_ok()
}

/// Returns whether `value` names one of the registered feature types.
///
/// `FeatureId` is backed by the same complete registry used by the domain
/// metadata, so parsing preserves the TypeScript `FEATURE_TYPES.includes`
/// check without duplicating the list here.
pub fn is_feature_type(value: &str) -> bool {
    FeatureId::from_str(value).is_ok()
}
