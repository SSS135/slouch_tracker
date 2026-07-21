use std::collections::HashSet;

use slouch_ml::ported::{
    constants::{EPSILON_STABLE, RTMDET_EXTRACTED_DIMS, RTMDET_RAW_DIMS, RTMDET_SHAPE},
    rtmdet_features::{extract_rtm_det_features, RtmDetFeaturesError},
    rtmpose_features::PoolingError,
};

fn assert_close(actual: f32, expected: f32) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= 1e-6 || difference <= 1e-6 * expected.abs(),
        "actual {actual}, expected {expected}, difference {difference}",
    );
}

fn assert_finite(values: &[f32]) {
    assert!(values.iter().all(|value| value.is_finite()));
}

#[test]
fn returns_correct_dimensional_float32_vector() {
    let raw_cls_p5 = vec![1.0_f32; RTMDET_RAW_DIMS];
    let raw_reg_p5 = vec![1.0_f32; RTMDET_RAW_DIMS];

    let result = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_eq!(result.len(), RTMDET_EXTRACTED_DIMS);
}

#[test]
fn extracts_features_without_l2_normalization() {
    let raw_cls_p5: Vec<f32> = (0..RTMDET_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.1).sin() + (index * 0.05).cos()) as f32
        })
        .collect();
    let raw_reg_p5: Vec<f32> = (0..RTMDET_RAW_DIMS)
        .map(|index| {
            let index = index as f64;
            ((index * 0.15).cos() + (index * 0.08).sin()) as f32
        })
        .collect();
    for index in 0..RTMDET_RAW_DIMS {
        let value = index as f64;
        assert_eq!(
            raw_cls_p5[index].to_bits(),
            (((value * 0.1).sin() + (value * 0.05).cos()) as f32).to_bits(),
        );
        assert_eq!(
            raw_reg_p5[index].to_bits(),
            (((value * 0.15).cos() + (value * 0.08).sin()) as f32).to_bits(),
        );
    }

    let result = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMDET_EXTRACTED_DIMS);

    let per_vector_dims = RTMDET_SHAPE.channels;
    let any_norm_not_one = result.chunks_exact(per_vector_dims).take(6).any(|vector| {
        let norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        (norm - 1.0).abs() > 0.01
    });

    assert!(any_norm_not_one);
}

#[test]
fn computes_average_standard_deviation_and_maximum_pooling() {
    let mut raw_cls_p5 = vec![0.0_f32; RTMDET_RAW_DIMS];
    let mut raw_reg_p5 = vec![0.0_f32; RTMDET_RAW_DIMS];
    let channel_size = RTMDET_SHAPE.height * RTMDET_SHAPE.width;

    for channel in 0..RTMDET_SHAPE.channels {
        for index in 0..channel_size {
            raw_cls_p5[channel * channel_size + index] =
                (channel + 1) as f32 * if index % 2 == 0 { 10.0 } else { 0.0 };
            raw_reg_p5[channel * channel_size + index] =
                (channel + 1) as f32 * if index % 3 == 0 { 5.0 } else { 0.0 };
        }
    }

    let result = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_eq!(result.len(), RTMDET_EXTRACTED_DIMS);

    let channels = RTMDET_SHAPE.channels;
    let sections = result.chunks_exact(channels).collect::<Vec<_>>();
    assert_eq!(sections.len(), 6);
    for (channel, cls_mean) in sections[0].iter().copied().enumerate() {
        let scale = (channel + 1) as f32;
        assert_close(cls_mean, 5.0 * scale);
        assert_close(
            sections[1][channel],
            (25.0 * scale * scale + EPSILON_STABLE).sqrt(),
        );
        assert_close(sections[2][channel], 10.0 * scale);
        assert_close(sections[3][channel], 1.7 * scale);
        assert_close(
            sections[4][channel],
            (5.61 * scale * scale + EPSILON_STABLE).sqrt(),
        );
        assert_close(sections[5][channel], 5.0 * scale);
    }

    for vector in sections.iter().take(3) {
        let unique_values = vector
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<HashSet<_>>();
        assert!(unique_values.len() > 1);
    }
}

#[test]
fn six_decimal_uniqueness_matches_the_typescript_to_fixed_assertion() {
    let bit_distinct = [1.0000001_f32, 1.0000002_f32];
    assert_ne!(bit_distinct[0].to_bits(), bit_distinct[1].to_bits());
    assert_eq!(
        bit_distinct
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<HashSet<_>>()
            .len(),
        1,
    );
}

#[test]
fn handles_constant_inputs_without_nan_or_infinity() {
    let raw_cls_p5 = vec![2.5_f32; RTMDET_RAW_DIMS];
    let raw_reg_p5 = vec![3.7_f32; RTMDET_RAW_DIMS];

    let result = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_finite(&result);
    assert_eq!(result.len(), RTMDET_EXTRACTED_DIMS);

    let per_vector_dims = RTMDET_SHAPE.channels;
    let cls_avg = &result[..per_vector_dims];
    let cls_max = &result[2 * per_vector_dims..3 * per_vector_dims];

    assert!((cls_avg[0] - 2.5).abs() < 5e-5);
    assert!((cls_max[0] - 2.5).abs() < 5e-5);
    assert!((2.500049_f32 - 2.5).abs() < 5e-5);
    assert!((2.500051_f32 - 2.5).abs() >= 5e-5);
}

#[test]
fn handles_zero_variance_in_standard_deviation_pooling() {
    let raw_cls_p5 = vec![4.2_f32; RTMDET_RAW_DIMS];
    let raw_reg_p5 = vec![1.8_f32; RTMDET_RAW_DIMS];

    let result = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_finite(&result);

    let per_vector_dims = RTMDET_SHAPE.channels;
    let cls_std = &result[per_vector_dims..2 * per_vector_dims];
    let reg_std = &result[4 * per_vector_dims..5 * per_vector_dims];

    assert!(cls_std
        .iter()
        .all(|value| (*value - cls_std[0]).abs() < 1e-6));
    assert!(reg_std
        .iter()
        .all(|value| (*value - reg_std[0]).abs() < 1e-6));
}

#[test]
fn repeated_extraction_is_deterministic() {
    let raw_cls_p5 = vec![1.0_f32; RTMDET_RAW_DIMS];
    let raw_reg_p5 = vec![1.0_f32; RTMDET_RAW_DIMS];

    let before = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();
    let after = extract_rtm_det_features(&raw_cls_p5, &raw_reg_p5).unwrap();

    assert_eq!(before, after);
    assert_eq!(after.len(), RTMDET_EXTRACTED_DIMS);
}

#[test]
fn rejects_short_and_oversized_cls_and_reg_inputs() {
    let valid = vec![0.0_f32; RTMDET_RAW_DIMS];
    for actual in [RTMDET_RAW_DIMS - 1, RTMDET_RAW_DIMS + 1] {
        let malformed = vec![0.0_f32; actual];
        assert_eq!(
            extract_rtm_det_features(&malformed, &valid),
            Err(RtmDetFeaturesError::Pooling(PoolingError::Shape {
                tensor: "features",
                expected: RTMDET_RAW_DIMS,
                actual,
            })),
        );
        assert_eq!(
            extract_rtm_det_features(&valid, &malformed),
            Err(RtmDetFeaturesError::Pooling(PoolingError::Shape {
                tensor: "features",
                expected: RTMDET_RAW_DIMS,
                actual,
            })),
        );
    }
}

#[test]
fn rejects_non_finite_cls_and_reg_inputs() {
    let valid = vec![0.0_f32; RTMDET_RAW_DIMS];
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut malformed = valid.clone();
        malformed[17] = non_finite;
        assert_eq!(
            extract_rtm_det_features(&malformed, &valid),
            Err(RtmDetFeaturesError::Pooling(PoolingError::NonFiniteInput {
                tensor: "features",
                index: 17,
            })),
        );
        assert_eq!(
            extract_rtm_det_features(&valid, &malformed),
            Err(RtmDetFeaturesError::Pooling(PoolingError::NonFiniteInput {
                tensor: "features",
                index: 17,
            })),
        );
    }
}
