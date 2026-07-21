use slouch_ml::ported::binning::soft_bin_with_fixed_edges;

#[test]
fn rejects_edges_with_fewer_than_two_elements() {
    assert_eq!(
        soft_bin_with_fixed_edges(0.0, 1.0, &[])
            .unwrap_err()
            .to_string(),
        "edges must have at least 2 elements"
    );
    assert_eq!(
        soft_bin_with_fixed_edges(0.0, 1.0, &[1.0])
            .unwrap_err()
            .to_string(),
        "edges must have at least 2 elements"
    );
}

#[test]
fn rejects_nonfinite_values() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            soft_bin_with_fixed_edges(value, 1.0, &[0.0, 1.0, 2.0])
                .unwrap_err()
                .to_string(),
            "value must be a finite number"
        );
    }
}

#[test]
fn rejects_nonfinite_or_out_of_range_confidence() {
    for confidence in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert_eq!(
            soft_bin_with_fixed_edges(0.0, confidence, &[0.0, 1.0, 2.0])
                .unwrap_err()
                .to_string(),
            "confidence must be a finite number between 0 and 1"
        );
    }
}

#[test]
fn duplicate_edges_use_the_uniform_fallback() {
    let probabilities = soft_bin_with_fixed_edges(0.0, 1.0, &[0.0, 0.0, 1.0]).unwrap();
    let uniform = 1.0 / probabilities.len() as f64;

    for probability in probabilities {
        assert!((f64::from(probability) - uniform).abs() < 0.5e-5);
    }
}

#[test]
fn nonfinite_edges_use_the_uniform_fallback() {
    let probabilities = soft_bin_with_fixed_edges(0.0, 1.0, &[f64::NAN, 1.0, 2.0]).unwrap();
    let uniform = 1.0 / probabilities.len() as f64;

    for probability in probabilities {
        assert!((f64::from(probability) - uniform).abs() < 0.5e-5);
    }
}

#[test]
fn nonuniform_bins_favor_the_exact_edge() {
    let edges = [-1000.0, -1.0, 0.0, 10.0, 1000.0];
    let probabilities = soft_bin_with_fixed_edges(0.0, 1.0, &edges).unwrap();
    let zero_index = edges.iter().position(|edge| *edge == 0.0).unwrap();
    assert!(probabilities[zero_index] > 0.5);
}

#[test]
fn nonuniform_bins_favor_adjacent_edges_for_intermediate_values() {
    let edges = [-1000.0, -1.0, 0.0, 10.0, 1000.0];
    let probabilities = soft_bin_with_fixed_edges(5.0, 1.0, &edges).unwrap();
    let minus_one_index = edges.iter().position(|edge| *edge == -1.0).unwrap();
    let zero_index = edges.iter().position(|edge| *edge == 0.0).unwrap();
    let ten_index = edges.iter().position(|edge| *edge == 10.0).unwrap();

    assert!(probabilities[zero_index] > probabilities[minus_one_index]);
    assert!(probabilities[ten_index] > probabilities[minus_one_index]);
}

#[test]
fn uniform_bins_favor_the_exact_edge() {
    let probabilities = soft_bin_with_fixed_edges(2.0, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(probabilities[2] > 0.5);
}

#[test]
fn uniform_bins_split_intermediate_values_between_adjacent_edges() {
    let probabilities = soft_bin_with_fixed_edges(1.5, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(probabilities[1] > probabilities[0]);
    assert!(probabilities[2] > probabilities[0]);
    assert!((f64::from(probabilities[1]) - f64::from(probabilities[2])).abs() < 0.5e-5);
}

#[test]
fn values_outside_the_range_still_produce_valid_probabilities() {
    let probabilities = soft_bin_with_fixed_edges(100.0, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let sum: f64 = probabilities
        .iter()
        .map(|probability| f64::from(*probability))
        .sum();
    assert!((sum - 1.0).abs() < 0.5e-5);
    assert!(probabilities
        .iter()
        .all(|probability| (0.0..=1.0).contains(&f64::from(*probability))));
}

#[test]
fn confidence_one_uses_pure_gaussian_soft_binning() {
    let probabilities = soft_bin_with_fixed_edges(2.0, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(probabilities[2] > 0.5);
}

#[test]
fn confidence_zero_returns_a_uniform_distribution() {
    let probabilities = soft_bin_with_fixed_edges(2.0, 0.0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let uniform = 1.0 / probabilities.len() as f64;
    for probability in probabilities {
        assert!((f64::from(probability) - uniform).abs() < 0.5e-5);
    }
}

#[test]
fn confidence_half_blends_gaussian_and_uniform_distributions() {
    let edges = [0.0, 1.0, 2.0, 3.0, 4.0];
    let gaussian_probabilities = soft_bin_with_fixed_edges(2.0, 1.0, &edges).unwrap();
    let blended_probabilities = soft_bin_with_fixed_edges(2.0, 0.5, &edges).unwrap();
    let uniform = 1.0 / edges.len() as f64;

    for (gaussian, blended) in gaussian_probabilities
        .iter()
        .zip(blended_probabilities.iter())
    {
        let expected = 0.5 * f64::from(*gaussian) + 0.5 * uniform;
        assert!((f64::from(*blended) - expected).abs() < 0.5e-5);
    }
}

#[test]
fn outputs_sum_to_one() {
    let test_cases = [
        (0.0, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0][..]),
        (5.0, 0.5, &[-1000.0, -1.0, 0.0, 10.0, 1000.0][..]),
        (-100.0, 0.8, &[0.0, 1.0, 2.0][..]),
        (1.5, 0.0, &[0.0, 1.0, 2.0, 3.0][..]),
    ];

    for (value, confidence, edges) in test_cases {
        let probabilities = soft_bin_with_fixed_edges(value, confidence, edges).unwrap();
        let sum: f64 = probabilities
            .iter()
            .map(|probability| f64::from(*probability))
            .sum();
        assert!((sum - 1.0).abs() < 0.5e-5);
    }
}

#[test]
fn outputs_are_between_zero_and_one() {
    let test_cases = [
        (0.0, 1.0, &[0.0, 1.0, 2.0, 3.0, 4.0][..]),
        (1000.0, 0.5, &[-1000.0, -1.0, 0.0, 10.0, 1000.0][..]),
        (-1000.0, 0.2, &[0.0, 1.0, 2.0][..]),
    ];

    for (value, confidence, edges) in test_cases {
        let probabilities = soft_bin_with_fixed_edges(value, confidence, edges).unwrap();
        assert!(probabilities
            .iter()
            .all(|probability| (0.0..=1.0).contains(&f64::from(*probability))));
    }
}
