use slouch_ml::ported::pca::{PcaError, PcaTransformer};

fn within_frozen_tolerance(actual: f64, expected: f64) -> bool {
    let difference = (actual - expected).abs();
    difference <= 2e-6 || difference <= 2e-6 * expected.abs()
}

#[test]
fn basic_functionality_fits_and_transforms_data() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![10.0, 11.0, 12.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let result = transformer.transform(&data).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result[0].len(), 2);

    transformer.dispose();
}

#[test]
fn basic_functionality_reduces_dimensionality() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![2.0, 1.0, 0.5, 0.0, 0.0],
        vec![1.0, 2.0, 0.5, 0.0, 0.0],
    ];

    transformer.fit(&data, 3).unwrap();
    assert_eq!(transformer.get_output_dimension(), 3);

    let result = transformer.transform(&data).unwrap();
    assert_eq!(result.len(), 5);
    assert_eq!(result[0].len(), 3);

    transformer.dispose();
}

#[test]
fn basic_functionality_rejects_transform_before_fit() {
    let transformer = PcaTransformer::new();
    let data = vec![vec![1.0, 2.0, 3.0]];

    assert_eq!(
        transformer.transform(&data).unwrap_err().to_string(),
        "PCA must be fitted before transform"
    );
}

#[test]
fn basic_functionality_centers_data_before_transformation() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![101.0, 202.0, 303.0],
        vec![104.0, 205.0, 306.0],
        vec![107.0, 208.0, 309.0],
        vec![110.0, 211.0, 312.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let result = transformer.transform(&data).unwrap();

    let mut means = [0.0; 2];
    for row in &result {
        means[0] += row[0];
        means[1] += row[1];
    }
    means[0] /= result.len() as f64;
    means[1] /= result.len() as f64;

    assert!(means[0].abs() < 1e-4);
    assert!(means[1].abs() < 1e-4);

    transformer.dispose();
}

#[test]
fn output_dimensions_returns_requested_component_count() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6.0, 5.0, 4.0, 3.0, 2.0],
    ];

    transformer.fit(&data, 3).unwrap();
    assert_eq!(transformer.get_output_dimension(), 3);

    let result = transformer.transform(&data).unwrap();
    assert_eq!(result[0].len(), 3);

    transformer.dispose();
}

#[test]
fn output_dimensions_clamps_components_to_centered_data_rank() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
    ];

    transformer.fit(&data, 10).unwrap();
    assert_eq!(transformer.get_output_dimension(), 2);

    transformer.dispose();
}

#[test]
fn output_dimensions_handles_a_single_component() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    transformer.fit(&data, 1).unwrap();
    assert_eq!(transformer.get_output_dimension(), 1);

    let result = transformer.transform(&data).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].len(), 1);

    transformer.dispose();
}

#[test]
fn serialization_round_trips_transform_results() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![7.0, 8.0, 9.0, 10.0],
        vec![10.0, 11.0, 12.0, 13.0],
        vec![2.0, 3.0, 4.0, 5.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let original_result = transformer.transform(&data).unwrap();

    let state = transformer.to_json().unwrap();
    let restored = PcaTransformer::from_json(state).unwrap();
    let restored_result = restored.transform(&data).unwrap();

    assert_eq!(restored_result.len(), original_result.len());
    assert_eq!(restored_result[0].len(), original_result[0].len());

    for (original_row, restored_row) in original_result.iter().zip(&restored_result) {
        for (original, restored) in original_row.iter().zip(restored_row) {
            assert!(within_frozen_tolerance(*restored, *original));
        }
    }

    transformer.dispose();
}

#[test]
fn serialization_rejects_unfitted_transformer() {
    let transformer = PcaTransformer::new();

    assert_eq!(
        transformer.to_json().unwrap_err().to_string(),
        "Cannot serialize unfitted PCA"
    );
}

#[test]
fn serialization_includes_explained_variance() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![2.0, 4.0, 6.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let state = transformer.to_json().unwrap();
    let explained_variance = state.explained_variance.as_ref().unwrap();

    assert_eq!(explained_variance.len(), 2);
    assert!(explained_variance[0] >= explained_variance[1]);

    transformer.dispose();
}

#[test]
fn memory_management_allows_multiple_dispose_calls() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    transformer.fit(&data, 2).unwrap();
    transformer.dispose();
    transformer.dispose();

    assert_eq!(
        transformer.transform(&data).unwrap_err().to_string(),
        "PCA must be fitted before transform"
    );
}

#[test]
fn memory_management_handles_multiple_fits() {
    let mut transformer = PcaTransformer::new();
    let first_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let second_data = vec![
        vec![10.0, 11.0, 12.0, 13.0],
        vec![14.0, 15.0, 16.0, 17.0],
        vec![18.0, 19.0, 20.0, 21.0],
        vec![22.0, 23.0, 24.0, 25.0],
    ];

    transformer.fit(&first_data, 2).unwrap();
    let first_result = transformer.transform(&first_data).unwrap();
    assert_eq!(first_result.len(), 3);
    assert_eq!(first_result[0].len(), 2);

    transformer.fit(&second_data, 3).unwrap();
    let second_result = transformer.transform(&second_data).unwrap();
    assert_eq!(second_result.len(), 4);
    assert_eq!(second_result[0].len(), 3);

    transformer.dispose();
}

#[test]
fn memory_management_loads_from_json() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![7.0, 8.0, 9.0, 10.0],
        vec![2.0, 3.0, 4.0, 5.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let state = transformer.to_json().unwrap();

    let restored = PcaTransformer::from_json(state).unwrap();
    let result = restored.transform(&data).unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].len(), 2);

    transformer.dispose();
}

#[test]
fn edge_cases_reject_zero_or_negative_component_counts() {
    let mut transformer = PcaTransformer::new();
    let data = vec![vec![1.0, 2.0, 3.0]];

    assert_eq!(
        transformer.fit(&data, 0).unwrap_err().to_string(),
        "nComponents must be positive"
    );
    assert_eq!(
        transformer.fit(&data, -1).unwrap_err().to_string(),
        "nComponents must be positive"
    );
}

#[test]
fn edge_cases_handle_highly_correlated_features() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
        vec![4.0, 8.0, 12.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let result = transformer.transform(&data).unwrap();

    assert_eq!(result.len(), 4);
    assert_eq!(result[0].len(), 2);

    transformer.dispose();
}

#[test]
fn edge_cases_handle_square_data_matrix() {
    let mut transformer = PcaTransformer::new();
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    transformer.fit(&data, 2).unwrap();
    let result = transformer.transform(&data).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].len(), 2);

    transformer.dispose();
}

#[test]
fn edge_cases_reject_empty_data() {
    let mut transformer = PcaTransformer::new();

    assert_eq!(
        transformer.fit(&[], 2).unwrap_err().to_string(),
        "Cannot fit PCA on empty data"
    );
}

#[test]
fn loaded_state_omits_training_only_explained_variance() {
    let mut transformer = PcaTransformer::new();
    let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 8.0]];
    transformer.fit(&data, 1).unwrap();
    let fitted = transformer.to_json().unwrap();
    assert!(fitted.explained_variance.is_some());

    let restored = PcaTransformer::from_json(fitted).unwrap();
    assert!(restored.to_json().unwrap().explained_variance.is_none());
}

#[test]
fn constant_and_finite_extreme_inputs_do_not_commit_non_finite_state() {
    for data in [
        vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
        vec![vec![f64::MAX, f64::MAX], vec![f64::MAX, -f64::MAX]],
        vec![vec![0.0, 0.0], vec![f64::MIN_POSITIVE, 0.0]],
    ] {
        let mut transformer = PcaTransformer::new();
        assert!(matches!(
            transformer.fit(&data, 1),
            Err(PcaError::InvalidState(_))
        ));
        assert!(transformer.to_json().is_err());
    }
}

#[test]
fn transform_rejects_finite_inputs_that_overflow_projection() {
    let state = slouch_ml::ported::pca::SerializedPca {
        components: vec![vec![f64::MAX, f64::MAX]],
        mean: vec![0.0, 0.0],
        n_components: 1,
        n_features: 2,
        explained_variance: None,
    };
    let transformer = PcaTransformer::from_json(state).unwrap();
    assert!(matches!(
        transformer.transform(&[vec![2.0, 2.0]]),
        Err(PcaError::InvalidState(_))
    ));
}

#[test]
fn frozen_tolerance_rejects_values_between_two_and_ten_e_minus_six() {
    assert!(within_frozen_tolerance(1.0 + 1.9e-6, 1.0));
    assert!(!within_frozen_tolerance(1.0 + 5e-6, 1.0));
}
