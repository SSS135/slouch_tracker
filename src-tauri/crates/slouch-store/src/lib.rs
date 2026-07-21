//! Native SQLite persistence boundary.
//!
//! Storage ports intentionally begin after the approved trial; this crate
//! establishes ownership of `rusqlite` without changing frontend persistence.

pub mod ported;

pub use ported::{
    archive, constants, export, feature_registry, feature_reservoir, import, model_format,
    operations, storage, types,
};
