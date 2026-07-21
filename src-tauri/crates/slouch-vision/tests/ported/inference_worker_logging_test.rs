//! Production-entry replacement for
//! `src/workers/__tests__/inference-worker-logging.test.ts`
//! (frozen SHA-256 `1e0acc065a6ba7b8f2cfa98ff7de898690bccddc1f0a46c0d4d26067bd73d955`).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, LogLevelPayload, ProcessPayload,
};
use slouch_vision::ported::inference_worker::{
    InferenceWorker, WorkerError, WorkerLogger, WorkerResponse,
};

use super::support::{image, CreateOutcome, TestFactory, TestLogger, TestRuntime};

#[derive(Clone, Default)]
struct ConfigurableLogger {
    parameter: Arc<Mutex<String>>,
    records: Arc<Mutex<Vec<(String, String)>>>,
}

impl WorkerLogger for ConfigurableLogger {
    fn set_from_url_param(&self, log_param: &str) {
        *self.parameter.lock().expect("parameter") = log_param.to_owned();
    }

    fn is_debug_enabled(&self) -> bool {
        matches!(
            self.parameter.lock().expect("parameter").as_str(),
            "debug" | "worker:debug"
        )
    }

    fn debug(&self, message: &str) {
        if self.is_debug_enabled() {
            self.records
                .lock()
                .expect("records")
                .push(("debug".into(), message.into()));
        }
    }

    fn info(&self, message: &str) {
        self.records
            .lock()
            .expect("records")
            .push(("info".into(), message.into()));
    }

    fn warn(&self, message: &str) {
        self.records
            .lock()
            .expect("records")
            .push(("warn".into(), message.into()));
    }

    fn error(&self, message: &str) {
        self.records
            .lock()
            .expect("records")
            .push(("error".into(), message.into()));
    }
}

#[test]
fn actual_worker_routes_configuration_initialization_and_processing_logs() {
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Err(WorkerError::Inference(
            "run failed".into(),
        ))])),
        CreateOutcome::Session(VecDeque::new()),
    ]);
    let mut worker = InferenceWorker::with_runtime(TestFactory::default(), logger, runtime);

    assert!(worker
        .handle_message(InferenceWorkerMessage::SetLogLevel {
            payload: Some(LogLevelPayload {
                log_param: Some("worker:debug".into())
            }),
        })
        .is_empty());
    assert!(matches!(
        &worker.handle_message(InferenceWorkerMessage::Initialize {
            payload: InitializePayload {
                rtmdet_path: "det".into(),
                rtmw3d_path: "pose".into(),
                nlf_path: None,
            },
        })[..],
        [WorkerResponse::Initialized { .. }]
    ));
    assert!(matches!(
        &worker.handle_message(InferenceWorkerMessage::Process {
            payload: ProcessPayload {
                image_data: image(4, 4),
                request_id: 7
            },
        })[..],
        [WorkerResponse::Error {
            request_id: Some(7),
            ..
        }]
    ));

    let logs = logs.lock().expect("logs");
    assert!(logs
        .iter()
        .any(|(level, message)| level == "info"
            && message.contains("Log level updated: worker:debug")));
    assert!(logs
        .iter()
        .any(|(level, message)| level == "info" && message.contains("Initializing ONNX Runtime")));
    assert!(logs
        .iter()
        .any(|(level, message)| level == "info"
            && message.contains("Both models loaded successfully")));
    assert!(logs.iter().any(
        |(level, message)| level == "error" && message.contains("Processing error: run failed")
    ));
}

#[test]
fn actual_worker_logs_initialization_failure_without_publishing_initialized() {
    let logger = TestLogger::default();
    let logs = logger.0.clone();
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Error("one".into()),
        CreateOutcome::Error("two".into()),
        CreateOutcome::Error("three".into()),
    ]);
    let mut worker = InferenceWorker::with_runtime(TestFactory::default(), logger, runtime);
    let response = worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "bad".into(),
            rtmw3d_path: "unused".into(),
            nlf_path: None,
        },
    });
    assert!(matches!(&response[..], [WorkerResponse::Error { .. }]));
    let logs = logs.lock().expect("logs");
    assert_eq!(
        logs.iter()
            .filter(|(level, message)| level == "info" && message.contains("Loading RTMDet"))
            .count(),
        3
    );
    assert_eq!(
        logs.iter()
            .filter(|(level, message)| level == "warn" && message.contains("Retrying RTMDet"))
            .count(),
        2
    );
    assert!(logs.iter().any(
        |(level, message)| level == "error" && message.contains("Initialization error: three")
    ));
}

#[test]
fn set_log_level_mutates_configuration_before_logging_and_normalizes_empty_values() {
    let logger = ConfigurableLogger::default();
    let parameter = Arc::clone(&logger.parameter);
    let records = Arc::clone(&logger.records);
    let (runtime, _) = TestRuntime::new([]);
    let mut worker = InferenceWorker::with_runtime(TestFactory::default(), logger, runtime);

    for (value, debug_enabled, displayed) in [
        ("worker:debug", true, "worker:debug"),
        ("debug", true, "debug"),
        ("", false, "none"),
    ] {
        assert!(worker
            .handle_message(InferenceWorkerMessage::SetLogLevel {
                payload: Some(LogLevelPayload {
                    log_param: Some(value.into()),
                }),
            })
            .is_empty());
        assert_eq!(parameter.lock().expect("parameter").as_str(), value);
        let record_guard = records.lock().expect("records");
        assert!(record_guard.last().is_some_and(|(level, message)| {
            level == "info" && message.contains(&format!("Log level updated: {displayed}"))
        }));
        drop(record_guard);

        let logger = ConfigurableLogger {
            parameter: Arc::clone(&parameter),
            records: Arc::clone(&records),
        };
        let before = logger.records.lock().expect("records").len();
        logger.debug("expensive debug payload");
        let after = logger.records.lock().expect("records").len();
        assert_eq!(after > before, debug_enabled);
    }
}

#[test]
fn logger_contract_accepts_objects_as_preformatted_messages_without_wall_clock_assumptions() {
    let logger = ConfigurableLogger::default();
    logger.set_from_url_param("worker:debug");
    logger.debug("Model I/O: inputs=[input], outputs=[simcc_x,simcc_y]");
    logger.info("Loading RTMDet attempt=1 maxRetries=3");
    logger.warn("RTMPose backbone_features missing");
    logger.error("ONNX inference failed");

    let records = logger.records.lock().expect("records");
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].0, "debug");
    assert_eq!(records[1].0, "info");
    assert_eq!(records[2].0, "warn");
    assert_eq!(records[3].0, "error");
}
