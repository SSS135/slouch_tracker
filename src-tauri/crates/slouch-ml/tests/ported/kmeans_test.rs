use slouch_ml::ported::kmeans::{
    kmeans, kmeans_default, select_best_k, select_best_k_default, silhouette_score,
};

fn create_feature(values: &[f32]) -> Vec<f32> {
    values.to_vec()
}

fn create_clustered_data(
    cluster_centers: &[&[f32]],
    points_per_cluster: usize,
    noise: f32,
) -> Vec<Vec<f32>> {
    let mut features = Vec::new();
    for (cluster_index, center) in cluster_centers.iter().enumerate() {
        for point_index in 0..points_per_cluster {
            let mut point = Vec::with_capacity(center.len());
            for (dimension, value) in center.iter().enumerate() {
                let pattern =
                    ((cluster_index * 31 + point_index * 17 + dimension * 13) % 11) as f32 / 10.0
                        - 0.5;
                point.push(*value + pattern * 2.0 * noise);
            }
            features.push(point);
        }
    }
    features
}

#[test]
fn basic_clustering_handles_empty_input() {
    let result = kmeans_default(&[], 3).unwrap();
    assert_eq!(result.k, 0);
    assert!(result.centroids.is_empty());
    assert!(result.assignments.is_empty());
}

#[test]
fn basic_clustering_handles_k_one() {
    let features = vec![
        create_feature(&[1.0, 0.0]),
        create_feature(&[0.0, 1.0]),
        create_feature(&[1.0, 1.0]),
    ];
    let result = kmeans_default(&features, 1).unwrap();

    assert_eq!(result.k, 1);
    assert_eq!(result.centroids.len(), 1);
    assert_eq!(result.assignments, vec![0, 0, 0]);
    assert_eq!(result.silhouette_score, 0.0);
    assert!((f64::from(result.centroids[0][0]) - 2.0 / 3.0).abs() < 5e-6);
    assert!((f64::from(result.centroids[0][1]) - 2.0 / 3.0).abs() < 5e-6);
}

#[test]
fn basic_clustering_clamps_k_to_number_of_samples() {
    let features = vec![create_feature(&[1.0, 0.0]), create_feature(&[0.0, 1.0])];
    let result = kmeans_default(&features, 5).unwrap();

    assert_eq!(result.k, 2);
    assert_eq!(result.centroids.len(), 2);
}

#[test]
fn basic_clustering_separates_well_clustered_data() {
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.1]),
        create_feature(&[0.0, 0.1]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.1]),
        create_feature(&[10.0, 10.1]),
    ];

    let result = kmeans_default(&features, 2).unwrap();
    assert_eq!(result.k, 2);
    assert_eq!(result.centroids.len(), 2);

    let cluster1 = &result.assignments[..3];
    let cluster2 = &result.assignments[3..6];
    assert_eq!(
        cluster1
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        1
    );
    assert_eq!(
        cluster2
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        1
    );
    assert_ne!(cluster1[0], cluster2[0]);
    assert!(result.silhouette_score > 0.8);
}

#[test]
fn basic_clustering_converges_within_max_iterations() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[10.0, 10.0]], 10, 0.1);
    let result = kmeans(&features, 2, 5, 42.0).unwrap();

    assert_eq!(result.k, 2);
    assert_eq!(result.assignments.len(), 20);
}

#[test]
fn kmeans_plus_plus_default_seed_is_valid_and_repeatable() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[5.0, 5.0], &[10.0, 0.0]], 5, 0.1);
    let expected = kmeans_default(&features, 3).unwrap();

    for _ in 0..5 {
        let result = kmeans_default(&features, 3).unwrap();
        assert_eq!(result, expected);
        assert_eq!(result.k, 3);
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 15);
    }
}

#[test]
fn kmeans_plus_plus_explicit_seed_variation_remains_supported() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[5.0, 5.0], &[10.0, 0.0]], 5, 0.1);

    for seed in 42..47 {
        let result = kmeans(&features, 3, 100, seed as f64).unwrap();
        assert_eq!(result.k, 3);
        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 15);
    }
}

#[test]
fn edge_cases_handle_single_sample() {
    let features = vec![create_feature(&[1.0, 2.0, 3.0])];
    let result = kmeans_default(&features, 3).unwrap();

    assert_eq!(result.k, 1);
    assert_eq!(result.centroids.len(), 1);
    assert_eq!(result.assignments, vec![0]);
}

#[test]
fn edge_cases_handle_k_zero() {
    let features = vec![create_feature(&[1.0, 2.0])];
    let result = kmeans_default(&features, 0).unwrap();

    assert_eq!(result.k, 0);
    assert!(result.centroids.is_empty());
}

#[test]
fn edge_cases_handle_identical_points() {
    let features = vec![
        create_feature(&[1.0, 1.0]),
        create_feature(&[1.0, 1.0]),
        create_feature(&[1.0, 1.0]),
    ];
    let result = kmeans_default(&features, 2).unwrap();

    assert!(result.k <= 2);
    assert_eq!(result.assignments.len(), 3);
}

#[test]
fn silhouette_score_returns_zero_for_k_one() {
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[1.0, 1.0]),
        create_feature(&[2.0, 2.0]),
    ];
    let assignments = [0, 0, 0];
    let centroids = vec![create_feature(&[1.0, 1.0])];

    let score = silhouette_score(&features, &assignments, &centroids).unwrap();
    assert_eq!(score, 0.0);
}

#[test]
fn silhouette_score_returns_zero_for_single_sample() {
    let features = vec![create_feature(&[0.0, 0.0])];
    let assignments = [0];
    let centroids = vec![create_feature(&[0.0, 0.0])];

    let score = silhouette_score(&features, &assignments, &centroids).unwrap();
    assert_eq!(score, 0.0);
}

#[test]
fn silhouette_score_is_high_for_well_separated_clusters() {
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[0.1, 0.0]),
        create_feature(&[0.0, 0.1]),
        create_feature(&[10.0, 10.0]),
        create_feature(&[10.1, 10.0]),
        create_feature(&[10.0, 10.1]),
    ];
    let assignments = [0, 0, 0, 1, 1, 1];
    let centroids = vec![
        create_feature(&[0.033, 0.033]),
        create_feature(&[10.033, 10.033]),
    ];

    let score = silhouette_score(&features, &assignments, &centroids).unwrap();
    assert!(score > 0.9);
}

#[test]
fn silhouette_score_is_low_for_poorly_separated_clusters() {
    let features = vec![
        create_feature(&[0.0, 0.0]),
        create_feature(&[1.0, 1.0]),
        create_feature(&[2.0, 2.0]),
        create_feature(&[1.5, 1.5]),
        create_feature(&[2.5, 2.5]),
        create_feature(&[3.0, 3.0]),
    ];
    let assignments = [0, 0, 0, 1, 1, 1];
    let centroids = vec![create_feature(&[1.0, 1.0]), create_feature(&[2.33, 2.33])];

    let score = silhouette_score(&features, &assignments, &centroids).unwrap();
    assert!(score < 0.5);
}

#[test]
fn select_best_k_handles_empty_input() {
    let result = select_best_k_default(&[]).unwrap();
    assert_eq!(result.k, 0);
}

#[test]
fn select_best_k_selects_two_for_two_well_separated_clusters() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[20.0, 20.0]], 10, 0.5);
    let result = select_best_k(&features, &[1, 2, 3, 4], 42.0).unwrap();

    assert_eq!(result.k, 2);
    assert!(result.silhouette_score > 0.7);
}

#[test]
fn select_best_k_selects_three_for_three_well_separated_clusters() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[20.0, 0.0], &[10.0, 20.0]], 10, 0.5);
    let result = select_best_k(&features, &[1, 2, 3, 4, 5], 42.0).unwrap();

    assert_eq!(result.k, 3);
    assert!(result.silhouette_score > 0.7);
}

#[test]
fn select_best_k_uses_default_k_values_when_none_are_provided() {
    let features = create_clustered_data(&[&[0.0, 0.0], &[10.0, 10.0]], 5, 0.2);
    let result = select_best_k_default(&features).unwrap();

    assert!((1..=7).contains(&result.k));
}

#[test]
fn select_best_k_filters_out_invalid_k_values() {
    let features = vec![create_feature(&[0.0, 0.0]), create_feature(&[1.0, 1.0])];
    let result = select_best_k(&features, &[1, 2, 5, 7], 42.0).unwrap();

    assert!(result.k <= 2);
}

#[test]
fn select_best_k_handles_single_sample() {
    let features = vec![create_feature(&[1.0, 2.0, 3.0])];
    let result = select_best_k(&features, &[1, 2, 3], 42.0).unwrap();

    assert_eq!(result.k, 1);
    assert_eq!(result.centroids.len(), 1);
}
