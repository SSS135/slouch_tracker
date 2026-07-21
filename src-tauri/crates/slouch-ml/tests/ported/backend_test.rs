use std::cell::RefCell;

use slouch_ml::backend::{
    get_current_backend, get_memory_info, init_tensorflow_backend, log_memory_usage, BackendError,
    BackendLogger,
};

#[derive(Default)]
struct RecordingLogger {
    enabled: bool,
    messages: RefCell<Vec<(String, String)>>,
}

impl BackendLogger for RecordingLogger {
    fn is_info_enabled(&self, category: &str) -> bool {
        self.enabled && category == "training"
    }

    fn info(&self, category: &str, message: &str) {
        self.messages
            .borrow_mut()
            .push((category.to_owned(), message.to_owned()));
    }
}

#[test]
fn does_not_claim_readiness_owned_by_the_application() {
    assert_eq!(
        init_tensorflow_backend(),
        Err(BackendError::InitializationOwnedByApplication)
    );
    assert_eq!(get_current_backend(), None);
}

#[test]
fn repeated_legacy_initialization_remains_a_typed_failure() {
    assert_eq!(init_tensorflow_backend(), init_tensorflow_backend());
    assert_eq!(get_current_backend(), None);
}

#[test]
fn reports_tensor_memory_telemetry_as_unavailable() {
    let memory = get_memory_info();
    assert_eq!(memory.num_tensors, None);
    assert_eq!(memory.num_data_buffers, None);
    assert_eq!(memory.num_bytes, None);
    assert!(memory.unreliable);
}

#[test]
fn routes_diagnostics_through_the_native_logging_boundary() {
    let disabled = RecordingLogger::default();
    log_memory_usage(&disabled);
    assert!(disabled.messages.borrow().is_empty());

    let enabled = RecordingLogger {
        enabled: true,
        ..RecordingLogger::default()
    };
    log_memory_usage(&enabled);
    assert_eq!(enabled.messages.borrow().len(), 1);
    assert_eq!(enabled.messages.borrow()[0].0, "training");
    assert!(enabled.messages.borrow()[0]
        .1
        .contains("telemetry is unavailable"));
}
