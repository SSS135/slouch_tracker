//! Native counterpart of `src/workers/createInferenceWorker.ts`.
//!
//! The browser implementation constructs a module `Worker`. Native inference
//! runs inside the `slouch-vision` boundary, so these factories return the same
//! operational worker used for model initialization, frame processing, model
//! lifecycle management, typed errors, and drop-based cleanup.

use super::inference_worker::{
    InferenceRuntime, InferenceWorker, ModelFactory, NoopLogger, OrtRuntime, WorkerLogger,
};

/// Creates an operational native inference worker with the production ONNX
/// runtime and no-op logger.
pub fn create_inference_worker<F>(model_factory: F) -> InferenceWorker<F, NoopLogger, OrtRuntime>
where
    F: ModelFactory,
{
    InferenceWorker::new(model_factory)
}

/// Creates an operational native inference worker with explicit dependencies.
/// This is the deterministic construction seam used by actors and tests.
pub fn create_inference_worker_with_runtime<F, L, R>(
    model_factory: F,
    logger: L,
    runtime: R,
) -> InferenceWorker<F, L, R>
where
    F: ModelFactory,
    L: WorkerLogger,
    R: InferenceRuntime,
{
    InferenceWorker::with_runtime(model_factory, logger, runtime)
}
