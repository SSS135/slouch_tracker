use serde_json::Value;
use slouch_ml::ported::{
    constants::{
        RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_BACKBONE_RAW_DIMS, RTMPOSE_GAU_POOLED_DIMS,
        RTMPOSE_GAU_RAW_DIMS,
    },
    rtmpose_features::{
        extract_backbone_raw, extract_gau_raw, pool_backbone, pool_backbone_features,
        pool_backbone_features_max, pool_backbone_features_std, pool_gau, pool_gau_features,
        pool_gau_features_max, pool_gau_features_std, PoolingError, PoolingKind,
    },
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/math/rtmpose-pooling-v1.json"
    ))
    .unwrap()
}

fn f32_array(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

fn usize_array(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_u64().unwrap() as usize)
        .collect()
}

fn close(actual: f32, expected: f32) -> bool {
    let difference = (actual - expected).abs();
    difference <= 1e-6 || difference <= 1e-6 * expected.abs()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            close(*actual, *expected),
            "index {index}: {actual} != {expected}"
        );
    }
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
    }
}

#[test]
fn fixture_pins_sources_shapes_lengths_and_axes() {
    let fixture = fixture();
    assert_eq!(
        fixture["sources"]["constants.ts"],
        "614849727adaf32c07fbb6a6722c39d66c7659587d7b6ec8420524c36fb6519e"
    );
    assert_eq!(
        fixture["sources"]["rtmposeFeatures.ts"],
        "c15b0b4e91f67176d95b348fd447b0cb9339ac12d6e326cf72e3dec161592ca5"
    );
    assert_eq!(
        fixture["sources"]["rtmposeFeatures.test.ts"],
        "a16a7c4e4352c56ff8a36283f9e9e49f9bfe539209de8db630d9865691b90077"
    );
    assert_eq!(usize_array(&fixture["backbone"]["shape"]), [1, 768, 8, 6]);
    assert_eq!(usize_array(&fixture["gau"]["shape"]), [1, 17, 256]);
    assert_eq!(usize_array(&fixture["backbone"]["poolingAxes"]), [2, 3]);
    assert_eq!(usize_array(&fixture["gau"]["poolingAxes"]), [1]);
    assert_eq!(
        fixture["backbone"]["rawLength"].as_u64().unwrap() as usize,
        RTMPOSE_BACKBONE_RAW_DIMS
    );
    assert_eq!(
        fixture["backbone"]["pooledLength"].as_u64().unwrap() as usize,
        RTMPOSE_BACKBONE_POOLED_DIMS
    );
    assert_eq!(
        fixture["gau"]["rawLength"].as_u64().unwrap() as usize,
        RTMPOSE_GAU_RAW_DIMS
    );
    assert_eq!(
        fixture["gau"]["pooledLength"].as_u64().unwrap() as usize,
        RTMPOSE_GAU_POOLED_DIMS
    );
}

#[test]
fn small_backbone_golden_covers_mean_max_and_population_std() {
    let fixture = fixture();
    let small = &fixture["backbone"]["small"];
    let input = f32_array(&small["input"]);
    assert_close(
        &pool_backbone(&input, 2, 2, 3, PoolingKind::Mean).unwrap(),
        &f32_array(&small["mean"]),
    );
    assert_bits(
        &pool_backbone(&input, 2, 2, 3, PoolingKind::Max).unwrap(),
        &f32_array(&small["max"]),
    );
    assert_close(
        &pool_backbone(&input, 2, 2, 3, PoolingKind::PopulationStd).unwrap(),
        &f32_array(&small["std"]),
    );
}

#[test]
fn cancellation_sensitive_backbone_mean_matches_typescript_reduction() {
    let fixture = fixture();
    let case = &fixture["backbone"]["cancellationSensitive"];
    let length = case["length"].as_u64().unwrap() as usize;
    let mut input = vec![case["middleValue"].as_f64().unwrap() as f32; length];
    input[0] = case["first"].as_f64().unwrap() as f32;
    input[length - 1] = case["last"].as_f64().unwrap() as f32;
    assert_eq!(length - 2, case["middleCount"].as_u64().unwrap() as usize);

    let mean = pool_backbone(&input, 1, 1, length, PoolingKind::Mean).unwrap();
    let std = pool_backbone(&input, 1, 1, length, PoolingKind::PopulationStd).unwrap();
    assert_close(&mean, &[case["mean"].as_f64().unwrap() as f32]);
    assert_close(&std, &[case["std"].as_f64().unwrap() as f32]);
}

#[test]
fn small_gau_golden_covers_mean_max_and_population_std() {
    let fixture = fixture();
    let small = &fixture["gau"]["small"];
    let input = f32_array(&small["input"]);
    assert_close(
        &pool_gau(&input, 3, 2, PoolingKind::Mean).unwrap(),
        &f32_array(&small["mean"]),
    );
    assert_bits(
        &pool_gau(&input, 3, 2, PoolingKind::Max).unwrap(),
        &f32_array(&small["max"]),
    );
    assert_close(
        &pool_gau(&input, 3, 2, PoolingKind::PopulationStd).unwrap(),
        &f32_array(&small["std"]),
    );
}

#[test]
fn fixed_backbone_sparse_spikes_match_golden_outputs() {
    let fixture = fixture();
    let sparse = &fixture["backbone"]["sparse"];
    let fill = sparse["fill"].as_f64().unwrap() as f32;
    let mut input = vec![fill; RTMPOSE_BACKBONE_RAW_DIMS];
    for pair in sparse["overrides"].as_array().unwrap() {
        input[pair[0].as_u64().unwrap() as usize] = pair[1].as_f64().unwrap() as f32;
    }
    let mean = pool_backbone_features(&input).unwrap();
    let max = pool_backbone_features_max(&input).unwrap();
    let std = pool_backbone_features_std(&input).unwrap();
    assert_close(&mean[..2], &f32_array(&sparse["firstMean"]));
    assert_bits(&max[..2], &f32_array(&sparse["firstMax"]));
    assert_close(&std[..2], &f32_array(&sparse["firstStd"]));
    assert!(mean[2..].iter().all(|value| close(*value, fill)));
    assert!(max[2..]
        .iter()
        .all(|value| value.to_bits() == fill.to_bits()));
    assert!(std[2..].iter().all(|value| close(*value, 0.001)));
}

#[test]
fn fixed_gau_sparse_spikes_match_golden_outputs() {
    let fixture = fixture();
    let sparse = &fixture["gau"]["sparse"];
    let fill = sparse["fill"].as_f64().unwrap() as f32;
    let mut input = vec![fill; RTMPOSE_GAU_RAW_DIMS];
    for pair in sparse["overrides"].as_array().unwrap() {
        input[pair[0].as_u64().unwrap() as usize] = pair[1].as_f64().unwrap() as f32;
    }
    let mean = pool_gau_features(&input).unwrap();
    let max = pool_gau_features_max(&input).unwrap();
    let std = pool_gau_features_std(&input).unwrap();
    assert_close(&mean[..2], &f32_array(&sparse["firstMean"]));
    assert_bits(&max[..2], &f32_array(&sparse["firstMax"]));
    assert_close(&std[..2], &f32_array(&sparse["firstStd"]));
    assert!(mean[2..].iter().all(|value| close(*value, fill)));
    assert!(max[2..]
        .iter()
        .all(|value| value.to_bits() == fill.to_bits()));
    assert!(std[2..].iter().all(|value| close(*value, 0.001)));
}

#[test]
fn raw_extraction_preserves_full_flattening_order_and_bits() {
    let backbone: Vec<f32> = (0..RTMPOSE_BACKBONE_RAW_DIMS)
        .map(|index| index as f32)
        .collect();
    let gau: Vec<f32> = (0..RTMPOSE_GAU_RAW_DIMS)
        .map(|index| -(index as f32))
        .collect();
    assert_bits(&extract_backbone_raw(&backbone).unwrap(), &backbone);
    assert_bits(&extract_gau_raw(&gau).unwrap(), &gau);
}

#[test]
fn sequential_full_tensors_preserve_layout_and_expected_statistics() {
    let fixture = fixture();
    let backbone: Vec<f32> = (0..RTMPOSE_BACKBONE_RAW_DIMS)
        .map(|index| index as f32)
        .collect();
    let mean = pool_backbone_features(&backbone).unwrap();
    let max = pool_backbone_features_max(&backbone).unwrap();
    let std = pool_backbone_features_std(&backbone).unwrap();
    let expected = &fixture["backbone"]["sequential"];
    assert!(close(
        mean[0],
        expected["firstMean"].as_f64().unwrap() as f32
    ));
    assert!(close(
        mean[767],
        expected["lastMean"].as_f64().unwrap() as f32
    ));
    assert_eq!(
        max[0].to_bits(),
        (expected["firstMax"].as_f64().unwrap() as f32).to_bits()
    );
    assert_eq!(
        max[767].to_bits(),
        (expected["lastMax"].as_f64().unwrap() as f32).to_bits()
    );
    assert!(std
        .iter()
        .all(|value| close(*value, expected["std"].as_f64().unwrap() as f32)));

    let gau: Vec<f32> = (0..RTMPOSE_GAU_RAW_DIMS)
        .map(|index| index as f32)
        .collect();
    let mean = pool_gau_features(&gau).unwrap();
    let max = pool_gau_features_max(&gau).unwrap();
    let std = pool_gau_features_std(&gau).unwrap();
    let expected = &fixture["gau"]["sequential"];
    assert!(close(
        mean[0],
        expected["firstMean"].as_f64().unwrap() as f32
    ));
    assert!(close(
        mean[255],
        expected["lastMean"].as_f64().unwrap() as f32
    ));
    assert_eq!(
        max[0].to_bits(),
        (expected["firstMax"].as_f64().unwrap() as f32).to_bits()
    );
    assert_eq!(
        max[255].to_bits(),
        (expected["lastMax"].as_f64().unwrap() as f32).to_bits()
    );
    assert!(std
        .iter()
        .all(|value| close(*value, expected["std"].as_f64().unwrap() as f32)));
}

#[test]
fn constant_negative_inputs_keep_negative_max_and_sqrt_epsilon_std() {
    let fixture = fixture();
    let backbone_value = fixture["backbone"]["negativeConstant"].as_f64().unwrap() as f32;
    let backbone = vec![backbone_value; RTMPOSE_BACKBONE_RAW_DIMS];
    assert!(pool_backbone_features_max(&backbone)
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == backbone_value.to_bits()));
    assert!(pool_backbone_features_std(&backbone)
        .unwrap()
        .iter()
        .all(|value| close(*value, 0.001)));

    let gau_value = fixture["gau"]["negativeConstant"].as_f64().unwrap() as f32;
    let gau = vec![gau_value; RTMPOSE_GAU_RAW_DIMS];
    assert!(pool_gau_features_max(&gau)
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == gau_value.to_bits()));
    assert!(pool_gau_features_std(&gau)
        .unwrap()
        .iter()
        .all(|value| close(*value, 0.001)));
}

#[test]
fn zero_dimensions_return_typed_errors() {
    assert_eq!(
        pool_backbone(&[], 1, 0, 1, PoolingKind::Mean),
        Err(PoolingError::ZeroDimension {
            tensor: "backbone",
            dimension: "height",
        })
    );
    assert_eq!(
        pool_gau(&[], 0, 1, PoolingKind::Mean),
        Err(PoolingError::ZeroDimension {
            tensor: "gau",
            dimension: "keypoints",
        })
    );
}

#[test]
fn dimension_overflow_returns_typed_errors() {
    assert_eq!(
        pool_backbone(&[], 1, usize::MAX, 2, PoolingKind::Mean),
        Err(PoolingError::DimensionOverflow { tensor: "backbone" })
    );
    assert_eq!(
        pool_gau(&[], usize::MAX, 2, PoolingKind::Mean),
        Err(PoolingError::DimensionOverflow { tensor: "gau" })
    );
}

#[test]
fn nonfinite_inputs_return_typed_errors() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            pool_backbone(&[0.0, value], 1, 1, 2, PoolingKind::Mean),
            Err(PoolingError::NonFiniteInput {
                tensor: "backbone",
                index: 1,
            })
        );
        assert_eq!(
            pool_gau(&[value], 1, 1, PoolingKind::Max),
            Err(PoolingError::NonFiniteInput {
                tensor: "gau",
                index: 0,
            })
        );
    }

    let mut backbone = vec![0.0; RTMPOSE_BACKBONE_RAW_DIMS];
    backbone[7] = f32::NAN;
    assert!(matches!(
        extract_backbone_raw(&backbone),
        Err(PoolingError::NonFiniteInput {
            tensor: "backbone",
            index: 7
        })
    ));
}

#[test]
fn malformed_lengths_return_typed_errors_without_panics() {
    let fixture = fixture();
    for length in usize_array(&fixture["backbone"]["malformedLengths"]) {
        let error = pool_backbone_features(&vec![0.0; length]).unwrap_err();
        assert_eq!(
            error,
            PoolingError::Shape {
                tensor: "backbone",
                expected: RTMPOSE_BACKBONE_RAW_DIMS,
                actual: length,
            }
        );
        assert!(extract_backbone_raw(&vec![0.0; length]).is_err());
    }
    for length in usize_array(&fixture["gau"]["malformedLengths"]) {
        let error = pool_gau_features(&vec![0.0; length]).unwrap_err();
        assert_eq!(
            error,
            PoolingError::Shape {
                tensor: "gau",
                expected: RTMPOSE_GAU_RAW_DIMS,
                actual: length,
            }
        );
        assert!(extract_gau_raw(&vec![0.0; length]).is_err());
    }
}
