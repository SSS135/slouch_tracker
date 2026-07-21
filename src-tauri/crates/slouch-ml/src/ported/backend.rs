//! Native backend capability contract.
//!
//! TensorFlow.js backend initialization and tensor-memory telemetry have no
//! implementation inside `slouch-ml`. ONNX Runtime is initialized by the
//! application/vision resource boundary, which owns the packaged runtime DLL
//! and model sessions. This module therefore reports the unsupported contract
//! honestly instead of publishing false readiness or reliable zero counters.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendError {
    InitializationOwnedByApplication,
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitializationOwnedByApplication => formatter.write_str(
                "native ONNX Runtime readiness is owned by the application resource boundary",
            ),
        }
    }
}

impl std::error::Error for BackendError {}

/// Tensor-style memory counters are unavailable from the native ML crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    pub num_tensors: Option<usize>,
    pub num_data_buffers: Option<usize>,
    pub num_bytes: Option<usize>,
    pub unreliable: bool,
}

/// Structured logging boundary for the unsupported telemetry diagnostic.
pub trait BackendLogger {
    fn is_info_enabled(&self, category: &str) -> bool;
    fn info(&self, category: &str, message: &str);
}

/// The legacy TensorFlow initializer cannot establish native readiness.
/// Callers must use the application-owned ONNX Runtime initialization path.
pub fn init_tensorflow_backend() -> Result<(), BackendError> {
    Err(BackendError::InitializationOwnedByApplication)
}

/// No backend name is available until the owning application commits ONNX
/// Runtime. `slouch-ml` deliberately does not maintain a second readiness flag.
pub fn get_current_backend() -> Option<&'static str> {
    None
}

/// Returns explicitly unavailable native tensor telemetry.
pub fn get_memory_info() -> MemoryInfo {
    MemoryInfo {
        num_tensors: None,
        num_data_buffers: None,
        num_bytes: None,
        unreliable: true,
    }
}

/// Routes the unsupported-telemetry diagnostic through the supplied logger.
pub fn log_memory_usage<L: BackendLogger>(logger: &L) {
    const CATEGORY: &str = "training";
    if logger.is_info_enabled(CATEGORY) {
        logger.info(
            CATEGORY,
            "[TF_BACKEND] Tensor memory telemetry is unavailable in the native ML crate",
        );
    }
}
