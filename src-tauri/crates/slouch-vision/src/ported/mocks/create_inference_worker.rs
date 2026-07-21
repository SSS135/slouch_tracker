//! Shared native mock for the inference worker factory.

use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerMessage {
    pub kind: String,
    pub payload: Vec<u8>,
}

impl WorkerMessage {
    pub fn new(kind: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            kind: kind.into(),
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferRecord {
    pub identity: u64,
    pub byte_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandlerToken(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerEventKind {
    Message,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerCall {
    PostMessage {
        message: WorkerMessage,
        transfer: Vec<TransferRecord>,
    },
    Terminate,
    SetOnMessage(Option<HandlerToken>),
    SetOnError(Option<HandlerToken>),
    Dispatch {
        kind: WorkerEventKind,
        handlers: Vec<HandlerToken>,
    },
    AddEventListener {
        kind: WorkerEventKind,
        handler: HandlerToken,
    },
    RemoveEventListener {
        kind: WorkerEventKind,
        handler: HandlerToken,
        removed: bool,
    },
}

/// State recorded by the worker methods used by browser-side tests.
#[derive(Debug, Default)]
pub struct MockWorker {
    pub calls: Vec<WorkerCall>,
    pub on_message: Option<HandlerToken>,
    pub on_error: Option<HandlerToken>,
    message_listeners: Vec<HandlerToken>,
    error_listeners: Vec<HandlerToken>,
}

impl MockWorker {
    pub fn post_message(&mut self, message: WorkerMessage, transfer: Vec<TransferRecord>) {
        self.calls
            .push(WorkerCall::PostMessage { message, transfer });
    }

    pub fn terminate(&mut self) {
        self.calls.push(WorkerCall::Terminate);
    }

    pub fn set_on_message(&mut self, handler: Option<HandlerToken>) {
        self.on_message = handler;
        self.calls.push(WorkerCall::SetOnMessage(handler));
    }

    pub fn set_on_error(&mut self, handler: Option<HandlerToken>) {
        self.on_error = handler;
        self.calls.push(WorkerCall::SetOnError(handler));
    }

    pub fn add_event_listener(&mut self, kind: WorkerEventKind, handler: HandlerToken) {
        self.listeners_mut(kind).push(handler);
        self.calls
            .push(WorkerCall::AddEventListener { kind, handler });
    }

    pub fn remove_event_listener(&mut self, kind: WorkerEventKind, handler: HandlerToken) -> bool {
        let listeners = self.listeners_mut(kind);
        let removed =
            if let Some(index) = listeners.iter().position(|candidate| *candidate == handler) {
                listeners.remove(index);
                true
            } else {
                false
            };
        self.calls.push(WorkerCall::RemoveEventListener {
            kind,
            handler,
            removed,
        });
        removed
    }

    /// Returns the exact handler identities that a deterministic dispatch
    /// would invoke: the property handler first, followed by listeners in
    /// registration order.
    pub fn dispatch(&mut self, kind: WorkerEventKind) -> Vec<HandlerToken> {
        let mut handlers = Vec::new();
        match kind {
            WorkerEventKind::Message => {
                handlers.extend(self.on_message);
                handlers.extend(self.message_listeners.iter().copied());
            }
            WorkerEventKind::Error => {
                handlers.extend(self.on_error);
                handlers.extend(self.error_listeners.iter().copied());
            }
        }
        self.calls.push(WorkerCall::Dispatch {
            kind,
            handlers: handlers.clone(),
        });
        handlers
    }

    fn listeners_mut(&mut self, kind: WorkerEventKind) -> &mut Vec<HandlerToken> {
        match kind {
            WorkerEventKind::Message => &mut self.message_listeners,
            WorkerEventKind::Error => &mut self.error_listeners,
        }
    }

    /// Clears recorded calls, listeners, and property handlers.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub type MockWorkerHandle = Arc<Mutex<MockWorker>>;

static MOCK_WORKER: OnceLock<MockWorkerHandle> = OnceLock::new();
static FACTORY_SPY: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();

fn factory_spy() -> &'static Mutex<Vec<usize>> {
    FACTORY_SPY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Returns the module-scoped worker mock shared by the factory.
pub fn mock_worker() -> MockWorkerHandle {
    MOCK_WORKER
        .get_or_init(|| Arc::new(Mutex::new(MockWorker::default())))
        .clone()
}

/// Creates the shared inference worker mock and records the ordered invocation.
pub fn create_inference_worker() -> MockWorkerHandle {
    let mut calls = factory_spy()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let call_index = calls.len();
    calls.push(call_index);
    mock_worker()
}

pub fn factory_calls() -> Vec<usize> {
    factory_spy()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

pub fn reset_factory_calls() {
    factory_spy()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_worker(handle: &MockWorkerHandle) -> std::sync::MutexGuard<'_, MockWorker> {
        handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn records_payload_transfer_listener_identity_and_call_order() {
        let handle = mock_worker();
        let mut worker = lock_worker(&handle);
        worker.reset();
        worker.post_message(
            WorkerMessage::new("process", vec![1, 2, 3]),
            vec![TransferRecord {
                identity: 7,
                byte_length: 3,
            }],
        );
        worker.set_on_message(Some(HandlerToken(10)));
        worker.add_event_listener(WorkerEventKind::Message, HandlerToken(11));
        assert_eq!(
            worker.dispatch(WorkerEventKind::Message),
            vec![HandlerToken(10), HandlerToken(11)],
        );
        assert!(!worker.remove_event_listener(WorkerEventKind::Message, HandlerToken(99)));
        assert!(worker.remove_event_listener(WorkerEventKind::Message, HandlerToken(11)));
        worker.set_on_message(None);
        assert!(worker.dispatch(WorkerEventKind::Message).is_empty());

        assert!(matches!(
            &worker.calls[0],
            WorkerCall::PostMessage { message, transfer }
                if message == &WorkerMessage::new("process", vec![1, 2, 3])
                    && transfer.as_slice() == [TransferRecord { identity: 7, byte_length: 3 }]
        ));
        assert!(matches!(
            &worker.calls[4],
            WorkerCall::RemoveEventListener { removed: false, .. }
        ));
        assert!(matches!(
            &worker.calls[5],
            WorkerCall::RemoveEventListener { removed: true, .. }
        ));
    }

    #[test]
    fn factory_spy_is_resettable_and_returns_shared_state() {
        reset_factory_calls();
        let first = create_inference_worker();
        let second = create_inference_worker();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(factory_calls(), vec![0, 1]);
        reset_factory_calls();
        assert!(factory_calls().is_empty());
    }
}
