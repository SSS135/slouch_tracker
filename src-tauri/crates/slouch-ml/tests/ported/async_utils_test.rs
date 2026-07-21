use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
};

use slouch_ml::ported::async_utils::{
    batch_process_async, yield_to_main_thread, AsyncUtilsError, BatchProcessOptions,
};

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(future);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[test]
fn yield_to_main_thread_is_pending_once_before_resolving() {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(yield_to_main_thread());

    assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
    assert_eq!(future.as_mut().poll(&mut context), Poll::Ready(()));
}

#[test]
fn yield_to_main_thread_allows_multiple_yields_in_sequence() {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = pin!(async {
        yield_to_main_thread().await;
        yield_to_main_thread().await;
        yield_to_main_thread().await;
    });

    assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
    assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
    assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
    assert_eq!(future.as_mut().poll(&mut context), Poll::Ready(()));
}

#[test]
fn batch_process_async_rejects_zero_batch_size() {
    let result = block_on(batch_process_async(
        vec![1],
        |value| value,
        BatchProcessOptions {
            batch_size: 0,
            on_progress: None,
        },
    ));

    assert_eq!(result, Err(AsyncUtilsError::InvalidBatchSize));
}

#[test]
fn batch_process_async_processes_all_items() {
    let items = vec![1, 2, 3, 4, 5];
    let results = block_on(batch_process_async(
        items,
        |value| value * 2,
        BatchProcessOptions::default(),
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, vec![2, 4, 6, 8, 10]);
}

#[test]
fn batch_process_async_uses_specified_batch_size() {
    let items: Vec<_> = (0..25).collect();
    let results = block_on(batch_process_async(
        items.clone(),
        |value| value,
        BatchProcessOptions {
            batch_size: 10,
            on_progress: None,
        },
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, items);
}

#[test]
fn batch_process_async_calls_progress_callback() {
    let items = vec![1, 2, 3, 4, 5];
    let mut progress_calls = Vec::new();
    let mut on_progress = |processed, total| progress_calls.push((processed, total));

    let results = block_on(batch_process_async(
        items,
        |value| value,
        BatchProcessOptions {
            batch_size: 2,
            on_progress: Some(&mut on_progress),
        },
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, vec![1, 2, 3, 4, 5]);
    assert_eq!(progress_calls, vec![(2, 5), (4, 5), (5, 5)]);
}

#[test]
fn batch_process_async_handles_empty_array() {
    let results = block_on(batch_process_async(
        Vec::<i32>::new(),
        |value| value,
        BatchProcessOptions::default(),
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, Vec::<i32>::new());
}

#[test]
fn batch_process_async_handles_single_item() {
    let results = block_on(batch_process_async(
        vec![42],
        |value| value,
        BatchProcessOptions::default(),
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, vec![42]);
}

#[test]
fn batch_process_async_works_with_different_data_types() {
    let items = vec!["a", "b", "c"];
    let results = block_on(batch_process_async(
        items,
        |value: &str| value.to_uppercase(),
        BatchProcessOptions::default(),
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, vec!["A", "B", "C"]);
}

#[test]
fn batch_process_async_preserves_order_of_items() {
    let items: Vec<_> = (0..100).collect();
    let results = block_on(batch_process_async(
        items.clone(),
        |value| value,
        BatchProcessOptions {
            batch_size: 10,
            on_progress: None,
        },
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, items);
}

#[test]
fn batch_process_async_works_with_float32_array_transformation() {
    let arrays = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![4.0_f32, 5.0, 6.0],
        vec![7.0_f32, 8.0, 9.0],
    ];
    let results = block_on(batch_process_async(
        arrays,
        |array: Vec<f32>| array.into_iter().map(f64::from).sum::<f64>(),
        BatchProcessOptions {
            batch_size: 2,
            on_progress: None,
        },
    ))
    .expect("batch processing should succeed");

    assert_eq!(results, vec![6.0, 15.0, 24.0]);
}
