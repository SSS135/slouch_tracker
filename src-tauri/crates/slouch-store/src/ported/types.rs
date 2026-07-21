//! Rust port of `src/services/dataset/types.ts`.
//!
//! The canonical DTOs live in `slouch-domain`, which owns labels, frames,
//! classifier/training configuration, and archive metadata.  This module is
//! the storage-layer export surface for the same contracts rather than a
//! second, incompatible set of wire types.

pub use slouch_domain::{
    CaptureAction, ClassificationResult, ClassifierConfig, CrossValidationType, DatasetManifest,
    DatasetStats, DimensionalityReductionConfig, DimensionalityReductionMethod, FeatureId,
    FeatureSource as FeatureContainer, FrameLabel, FrameMetadata, ImportResult, InferenceResult,
    PostureDataset, PostureFrame, PostureFrameMetadata, ReservoirManifest, TrainingMetrics,
    TrainingResult, TrainingSettings,
};

/// TypeScript's `FeatureType` is the string-backed feature registry identifier.
/// `FeatureId` is its canonical Rust representation and retains the same wire
/// names through serde.
pub type FeatureType = FeatureId;
