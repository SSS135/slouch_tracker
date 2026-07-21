use slouch_ml::ported::layer_norm::{apply_l2_norm_batch, apply_layer_norm_batch, layer_norm_tf};

fn close(actual: f32, expected: f32, tolerance: f32) -> bool {
    (actual - expected).abs() < tolerance
}

#[test]
fn layer_norm_normalizes_tensor_to_mean_zero_and_std_one() {
    let result = layer_norm_tf(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let mean = result.iter().sum::<f32>() / result.len() as f32;
    let variance = result
        .iter()
        .map(|value| {
            let difference = *value - mean;
            difference * difference
        })
        .sum::<f32>()
        / result.len() as f32;
    let std = variance.sqrt();

    assert!(close(mean, 0.0, 1e-6));
    assert!(close(std, 1.0, 1e-3));
}

#[test]
fn layer_norm_handles_zero_variance_with_epsilon() {
    let result = layer_norm_tf(&[5.0, 5.0, 5.0, 5.0]).unwrap();

    assert!(result.iter().all(|value| value.is_finite()));
    assert!(result.iter().all(|value| !value.is_nan()));
    assert!(result.iter().all(|value| close(*value, 0.0, 1e-6)));
}

#[test]
fn apply_layer_norm_batch_normalizes_each_sample_to_mean_zero_and_std_one() {
    let batch = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![-5.0, -10.0, -15.0, -20.0, -25.0],
    ];

    let result = apply_layer_norm_batch(&batch).unwrap();

    assert_eq!(result.len(), 3);
    for normalized in result {
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;
        let variance = normalized
            .iter()
            .map(|value| {
                let difference = *value - mean;
                difference * difference
            })
            .sum::<f32>()
            / normalized.len() as f32;
        let std = variance.sqrt();

        assert!(close(mean, 0.0, 1e-6));
        assert!(close(std, 1.0, 1e-3));
    }
}

#[test]
fn apply_layer_norm_batch_handles_single_sample() {
    let result = apply_layer_norm_batch(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 5);
    assert!(result[0].iter().all(|value| value.is_finite()));
}

#[test]
fn apply_layer_norm_batch_handles_empty_batch() {
    let result = apply_layer_norm_batch(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn apply_layer_norm_batch_handles_zero_variance_samples_with_epsilon() {
    let batch = vec![vec![3.5, 3.5, 3.5, 3.5], vec![1.0, 2.0, 3.0, 4.0]];
    let result = apply_layer_norm_batch(&batch).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result[0].iter().all(|value| !value.is_nan()));
    assert!(result[0].iter().all(|value| value.is_finite()));
    assert!(result[0].iter().all(|value| close(*value, 0.0, 1e-6)));

    let mean = result[1].iter().sum::<f32>() / result[1].len() as f32;
    assert!(close(mean, 0.0, 1e-6));
}

#[test]
fn apply_l2_norm_batch_normalizes_each_row_to_unit_length() {
    let batch = vec![
        vec![3.0, 4.0, 0.0],
        vec![5.0, 12.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ];
    let result = apply_l2_norm_batch(&batch).unwrap();

    for row in &result {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(close(norm, 1.0, 1e-6));
    }

    assert!(close(result[0][0], 0.6, 1e-6));
    assert!(close(result[0][1], 0.8, 1e-6));

    let expected_5_13 = 5.0 / 13.0;
    let expected_12_13 = 12.0 / 13.0;
    assert!(close(result[1][0], expected_5_13, 1e-6));
    assert!(close(result[1][1], expected_12_13, 1e-6));
}

#[test]
fn apply_l2_norm_batch_rejects_ragged_input() {
    let input = vec![vec![3.0, 4.0], vec![5.0, 12.0], vec![1.0, 0.0, 0.0]];

    assert_eq!(
        apply_l2_norm_batch(&input),
        Err(slouch_ml::ported::layer_norm::LayerNormError::RaggedBatch {
            index: 2,
            expected: 2,
            actual: 3,
        })
    );
}

#[test]
fn apply_l2_norm_batch_handles_zero_vectors_without_division_by_zero() {
    let batch = vec![vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0]];
    let result = apply_l2_norm_batch(&batch).unwrap();

    assert!(result[0].iter().all(|value| !value.is_nan()));
    assert!(result[0].iter().all(|value| value.is_finite()));
    assert!(result[0].iter().all(|value| *value == 0.0));

    let norm = result[1]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(close(norm, 1.0, 1e-6));
}

#[test]
fn apply_l2_norm_batch_handles_near_zero_vectors_with_epsilon() {
    let batch = vec![vec![1e-5, 1e-5, 1e-5], vec![1.0, 2.0, 3.0]];
    let result = apply_l2_norm_batch(&batch).unwrap();

    assert!(result[0].iter().all(|value| !value.is_nan()));
    assert!(result[0].iter().all(|value| value.is_finite()));

    let norm = result[1]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(close(norm, 1.0, 1e-6));
}

#[test]
fn apply_l2_norm_batch_returns_sub_epsilon_vectors_unchanged() {
    let input = vec![vec![1e-7, 1e-7, 1e-7], vec![1.0, 2.0, 3.0]];
    let result = apply_l2_norm_batch(&input).unwrap();

    assert_eq!(result[0], input[0]);
    let norm = result[1]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(close(norm, 1.0, 1e-6));
}
