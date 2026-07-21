//! Cooperative batching helpers for ML work.

use std::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

/// Configuration for [`batch_process_async`].
pub struct BatchProcessOptions<'a> {
    /// Number of items to process before yielding.
    pub batch_size: usize,
    /// Receives `(processed, total)` after each completed batch.
    pub on_progress: Option<&'a mut dyn FnMut(usize, usize)>,
}

impl<'a> Default for BatchProcessOptions<'a> {
    fn default() -> Self {
        Self {
            batch_size: 10,
            on_progress: None,
        }
    }
}

impl<'a> BatchProcessOptions<'a> {
    /// Creates options using the source utility's default batch size.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Errors produced by the asynchronous batching helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncUtilsError {
    /// A zero-sized batch cannot make progress.
    InvalidBatchSize,
}

impl fmt::Display for AsyncUtilsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBatchSize => formatter.write_str("batch size must be positive"),
        }
    }
}

impl std::error::Error for AsyncUtilsError {}

/// Future that cooperatively yields once to the async executor.
#[derive(Debug, Default)]
struct YieldOnce {
    yielded: bool,
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yields once to the async executor before continuing ML work.
pub async fn yield_to_main_thread() {
    YieldOnce::default().await;
}

/// Processes items in order, yielding after every batch.
///
/// Progress is reported after processing each batch, including a final partial
/// batch. Empty input produces an empty result without invoking the callback.
pub async fn batch_process_async<T, R, I, F>(
    items: I,
    mut process_fn: F,
    mut options: BatchProcessOptions<'_>,
) -> Result<Vec<R>, AsyncUtilsError>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T) -> R,
{
    if options.batch_size == 0 {
        return Err(AsyncUtilsError::InvalidBatchSize);
    }

    let items: Vec<T> = items.into_iter().collect();
    let total = items.len();
    let mut results = Vec::with_capacity(total);

    for (index, item) in items.into_iter().enumerate() {
        results.push(process_fn(item));
        let processed = index.saturating_add(1);

        if processed % options.batch_size == 0 || processed == total {
            if let Some(callback) = options.on_progress.as_mut() {
                (*callback)(processed, total);
            }
            yield_to_main_thread().await;
        }
    }

    Ok(results)
}
