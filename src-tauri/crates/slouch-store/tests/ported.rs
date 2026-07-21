use std::sync::{Mutex, MutexGuard, OnceLock};

fn default_reservoir_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[path = "ported/export_test.rs"]
mod export;
#[path = "ported/feature_registry_test.rs"]
mod feature_registry;
#[path = "ported/feature_reservoir_test.rs"]
mod feature_reservoir;
#[path = "ported/import_test.rs"]
mod import;
#[path = "ported/operations_test.rs"]
mod operations;
#[path = "ported/storage_test.rs"]
mod storage;
