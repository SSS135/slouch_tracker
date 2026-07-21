use slouch_ml::ported::random_projection::RandomProjectionTransformer;

fn transformer(n_components: usize) -> RandomProjectionTransformer {
    RandomProjectionTransformer::new(n_components, 42.0).expect("valid transformer")
}

fn assert_close(actual: f32, expected: f32) {
    let absolute_error = (actual - expected).abs();
    let relative_error = 2e-6_f32 * expected.abs();
    assert!(
        absolute_error <= 2e-6 || absolute_error <= relative_error,
        "{actual} != {expected}"
    );
}

#[test]
fn fits_and_transforms_data() {
    let mut transformer = transformer(2);
    let samples = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    transformer.fit(&samples).unwrap();
    assert!(transformer.is_fitted());
    assert_eq!(transformer.n_components(), 2);

    let result = transformer.transform(&samples[0]).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn same_seed_produces_deterministic_results() {
    let mut transformer1 = transformer(2);
    let mut transformer2 = transformer(2);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    transformer1.fit(&samples).unwrap();
    transformer2.fit(&samples).unwrap();

    assert_eq!(
        transformer1.transform(&samples[0]).unwrap(),
        transformer2.transform(&samples[0]).unwrap()
    );
}

#[test]
fn transform_before_fit_returns_error() {
    let transformer = transformer(2);
    let error = transformer.transform(&[1.0, 2.0, 3.0]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Random projection must be fitted before transform"
    );
}

#[test]
fn transform_dimension_mismatch_returns_error() {
    let mut transformer = transformer(2);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();

    assert!(matches!(
        transformer.transform(&[1.0, 2.0]),
        Err(
            slouch_ml::ported::random_projection::RandomProjectionError::DimensionMismatch {
                expected: 3,
                actual: 2,
            }
        )
    ));
}

#[test]
fn serialization_round_trip_preserves_transform() {
    let mut transformer = transformer(2);
    let samples = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    transformer.fit(&samples).unwrap();
    let original = transformer.transform(&samples[0]).unwrap();

    let state = transformer.to_json().unwrap();
    let restored = RandomProjectionTransformer::from_json(state).unwrap();
    let result = restored.transform(&samples[0]).unwrap();

    assert_eq!(result.len(), original.len());
    for (actual, expected) in result.iter().zip(original.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn serializing_unfitted_transformer_returns_error() {
    let transformer = transformer(2);
    assert_eq!(
        transformer.to_json().unwrap_err().to_string(),
        "Cannot serialize unfitted random projection"
    );
}

#[test]
fn matrix_multiplication_uses_consistent_indexing() {
    let mut transformer = transformer(2);
    let samples = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        vec![6.0, 7.0, 8.0, 9.0, 10.0],
    ];
    transformer.fit(&samples).unwrap();

    let result = transformer.transform(&samples[0]).unwrap();
    let state = transformer.to_json().unwrap();
    let mut expected = [0.0_f32; 2];
    for (row_index, row) in state.projection_matrix.iter().enumerate() {
        let sum = row
            .iter()
            .zip(samples[0].iter())
            .map(|(weight, feature)| f64::from(*weight as f32) * f64::from(*feature))
            .sum::<f64>();
        expected[row_index] = sum as f32;
    }

    assert_eq!(result.len(), expected.len());
    for (actual, expected) in result.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn serialization_preserves_matrix_multiplication() {
    let mut transformer = transformer(3);
    let samples = vec![vec![1.0_f32, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    transformer.fit(&samples).unwrap();
    let original = transformer.transform(&samples[1]).unwrap();

    let state = transformer.to_json().unwrap();
    let restored = RandomProjectionTransformer::from_json(state).unwrap();
    let result = restored.transform(&samples[1]).unwrap();

    assert_eq!(result, original);
}

#[test]
fn supports_one_component() {
    let mut transformer = transformer(1);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();

    let result = transformer.transform(&samples[0]).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn supports_more_components_than_features() {
    let mut transformer = transformer(5);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();

    assert!(transformer.is_fitted());
    assert_eq!(transformer.n_components(), 5);

    let result = transformer.transform(&samples[0]).unwrap();
    assert_eq!(result.len(), 5);
    assert!(result.iter().all(|value| value.is_finite()));
}

#[test]
fn rejects_zero_components() {
    assert_eq!(
        RandomProjectionTransformer::new(0, 42.0)
            .unwrap_err()
            .to_string(),
        "nComponents must be positive"
    );
}

#[test]
fn fit_creates_projection_cache() {
    let mut transformer = transformer(2);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();

    let result = transformer.transform(&samples[0]).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn dispose_releases_projection_cache() {
    let mut transformer = transformer(2);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();
    transformer.dispose();

    let error = transformer.transform(&samples[0]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Random projection must be fitted before transform"
    );
}

#[test]
fn repeated_fit_replaces_projection_cache() {
    let mut transformer = transformer(2);
    let first_samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let second_samples = vec![vec![7.0_f32, 8.0, 9.0], vec![10.0, 11.0, 12.0]];

    transformer.fit(&first_samples).unwrap();
    assert_eq!(transformer.transform(&first_samples[0]).unwrap().len(), 2);

    transformer.fit(&second_samples).unwrap();
    assert_eq!(transformer.transform(&second_samples[0]).unwrap().len(), 2);
}

#[test]
fn deserialization_recreates_projection_cache() {
    let mut transformer = transformer(2);
    let samples = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    transformer.fit(&samples).unwrap();
    let state = transformer.to_json().unwrap();

    let restored = RandomProjectionTransformer::from_json(state).unwrap();
    let result = restored.transform(&samples[0]).unwrap();
    assert_eq!(result.len(), 2);
}
