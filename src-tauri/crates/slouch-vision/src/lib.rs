//! Native vision boundary.
//!
//! ONNX session and image-pipeline ports intentionally begin after the approved
//! trial; this crate establishes their dependency boundary without frontend use.

pub mod ported;
pub mod preprocessing;

#[cfg(test)]
pub use ported::mocks;
pub use ported::{create_inference_worker, inference_worker};
pub use preprocessing::NativePreprocessor;
