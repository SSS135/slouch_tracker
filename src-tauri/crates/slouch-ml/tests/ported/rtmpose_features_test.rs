use slouch_ml::ported::{
    constants::{
        EPSILON_STABLE, RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_BACKBONE_RAW_DIMS,
        RTMPOSE_BACKBONE_SHAPE, RTMPOSE_GAU_POOLED_DIMS, RTMPOSE_GAU_RAW_DIMS, RTMPOSE_GAU_SHAPE,
    },
    rtmpose_features::{
        pool_backbone_features_max, pool_backbone_features_std, pool_gau_features_max,
        pool_gau_features_std,
    },
};

fn assert_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
}

fn assert_constant(values: &[f32], expected: f32) {
    assert!(values.iter().all(|value| *value == expected));
}

#[test]
fn backbone_max_returns_the_expected_dimension() {
    let raw = vec![1.0_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let result = pool_backbone_features_max(&raw).unwrap();

    assert_eq!(result.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn backbone_max_returns_finite_raw_unnormalized_features() {
    let raw: Vec<f32> = (0..RTMPOSE_BACKBONE_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.1).sin() + (index * 0.05).cos()) as f32
        })
        .collect();
    for (index, actual) in raw.iter().enumerate() {
        let value = index as f64;
        assert_eq!(
            actual.to_bits(),
            (((value * 0.1).sin() + (value * 0.05).cos()) as f32).to_bits(),
        );
    }
    let result = pool_backbone_features_max(&raw).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn backbone_max_captures_distinct_spatial_maxima() {
    let mut raw = vec![0.5_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let channel_size = RTMPOSE_BACKBONE_SHAPE[2] * RTMPOSE_BACKBONE_SHAPE[3];
    raw[10] = 5.0;
    raw[channel_size + 20] = 8.0;

    let result = pool_backbone_features_max(&raw).unwrap();
    let first_values = result
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();

    assert!(first_values.windows(2).any(|window| window[0] != window[1]));
}

#[test]
fn backbone_max_keeps_constant_inputs_unnormalized() {
    let raw = vec![2.5_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let result = pool_backbone_features_max(&raw).unwrap();

    assert_finite(&result);
    assert_constant(&result, 2.5);
}

#[test]
fn backbone_max_repeated_calls_keep_output_valid() {
    let raw = vec![1.0_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let before = pool_backbone_features_max(&raw).unwrap();
    let after = pool_backbone_features_max(&raw).unwrap();

    assert_eq!(before, after);
    assert_eq!(after.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn backbone_std_returns_the_expected_dimension() {
    let raw = vec![1.0_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let result = pool_backbone_features_std(&raw).unwrap();

    assert_eq!(result.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn backbone_std_returns_finite_raw_unnormalized_features() {
    let raw: Vec<f32> = (0..RTMPOSE_BACKBONE_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.1).sin() * 2.0 + (index * 0.2).cos() * 3.0) as f32
        })
        .collect();
    let result = pool_backbone_features_std(&raw).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn backbone_std_captures_distinct_spatial_standard_deviations() {
    let mut raw = vec![0.0_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let channel_size = RTMPOSE_BACKBONE_SHAPE[2] * RTMPOSE_BACKBONE_SHAPE[3];

    for (index, value) in raw[..channel_size].iter_mut().enumerate() {
        *value = if index % 2 == 0 { 10.0 } else { 0.0 };
    }
    for (index, value) in raw[channel_size..2 * channel_size].iter_mut().enumerate() {
        *value = 5.0 + (index % 2) as f32 * 0.1;
    }

    let result = pool_backbone_features_std(&raw).unwrap();
    assert!(result[0].to_bits() != result[1].to_bits());
}

#[test]
fn backbone_std_uses_epsilon_for_constant_inputs() {
    let raw = vec![3.7_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let result = pool_backbone_features_std(&raw).unwrap();
    let expected = EPSILON_STABLE.sqrt();

    assert_finite(&result);
    assert!(result.iter().all(|value| (*value - expected).abs() < 1e-6));
}

#[test]
fn backbone_std_repeated_calls_keep_output_valid() {
    let raw = vec![1.0_f32; RTMPOSE_BACKBONE_RAW_DIMS];
    let before = pool_backbone_features_std(&raw).unwrap();
    let after = pool_backbone_features_std(&raw).unwrap();

    assert_eq!(before, after);
    assert_eq!(after.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn gau_max_returns_the_expected_dimension() {
    let raw = vec![1.0_f32; RTMPOSE_GAU_RAW_DIMS];
    let result = pool_gau_features_max(&raw).unwrap();

    assert_eq!(result.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn gau_max_returns_finite_raw_unnormalized_features() {
    let raw: Vec<f32> = (0..RTMPOSE_GAU_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.15).sin() + (index * 0.08).cos()) as f32
        })
        .collect();
    for (index, actual) in raw.iter().enumerate() {
        let value = index as f64;
        assert_eq!(
            actual.to_bits(),
            (((value * 0.15).sin() + (value * 0.08).cos()) as f32).to_bits(),
        );
    }
    let result = pool_gau_features_max(&raw).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn gau_max_captures_distinct_keypoint_maxima() {
    let mut raw = vec![0.3_f32; RTMPOSE_GAU_RAW_DIMS];
    let feature_dim = RTMPOSE_GAU_SHAPE[2];
    raw[5 * feature_dim] = 7.0;
    raw[10 * feature_dim + 1] = 9.0;

    let result = pool_gau_features_max(&raw).unwrap();
    let first_values = result
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();

    assert!(first_values.windows(2).any(|window| window[0] != window[1]));
}

#[test]
fn gau_max_keeps_constant_inputs_unnormalized() {
    let raw = vec![1.5_f32; RTMPOSE_GAU_RAW_DIMS];
    let result = pool_gau_features_max(&raw).unwrap();

    assert_finite(&result);
    assert_constant(&result, 1.5);
}

#[test]
fn gau_max_repeated_calls_keep_output_valid() {
    let raw = vec![1.0_f32; RTMPOSE_GAU_RAW_DIMS];
    let before = pool_gau_features_max(&raw).unwrap();
    let after = pool_gau_features_max(&raw).unwrap();

    assert_eq!(before, after);
    assert_eq!(after.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn gau_std_returns_the_expected_dimension() {
    let raw = vec![1.0_f32; RTMPOSE_GAU_RAW_DIMS];
    let result = pool_gau_features_std(&raw).unwrap();

    assert_eq!(result.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn gau_std_returns_finite_raw_unnormalized_features() {
    let raw: Vec<f32> = (0..RTMPOSE_GAU_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.12).sin() * 1.5 + (index * 0.25).cos() * 2.5) as f32
        })
        .collect();
    let result = pool_gau_features_std(&raw).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn gau_std_captures_distinct_keypoint_standard_deviations() {
    let mut raw = vec![0.0_f32; RTMPOSE_GAU_RAW_DIMS];
    let feature_dim = RTMPOSE_GAU_SHAPE[2];
    let keypoints = RTMPOSE_GAU_SHAPE[1];

    for keypoint in 0..keypoints {
        raw[keypoint * feature_dim] = if keypoint % 2 == 0 { 10.0 } else { 0.0 };
    }
    for keypoint in 0..keypoints {
        raw[keypoint * feature_dim + 1] = 5.0 + (keypoint % 2) as f32 * 0.1;
    }

    let result = pool_gau_features_std(&raw).unwrap();
    assert!(result[0].to_bits() != result[1].to_bits());
}

#[test]
fn gau_std_uses_epsilon_for_constant_inputs() {
    let raw = vec![4.2_f32; RTMPOSE_GAU_RAW_DIMS];
    let result = pool_gau_features_std(&raw).unwrap();
    let expected = EPSILON_STABLE.sqrt();

    assert_finite(&result);
    assert!(result.iter().all(|value| (*value - expected).abs() < 1e-6));
}

#[test]
fn gau_std_repeated_calls_keep_output_valid() {
    let raw = vec![1.0_f32; RTMPOSE_GAU_RAW_DIMS];
    let before = pool_gau_features_std(&raw).unwrap();
    let after = pool_gau_features_std(&raw).unwrap();

    assert_eq!(before, after);
    assert_eq!(after.len(), RTMPOSE_GAU_POOLED_DIMS);
}
