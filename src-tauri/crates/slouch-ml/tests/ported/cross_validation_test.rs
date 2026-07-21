use slouch_ml::ported::cross_validation::{
    create_stratified_k_fold, create_temporal_block_k_fold, create_time_ordered_folds,
    detect_temporal_blocks, get_class_distribution, validate_folds, CvFold,
};

fn max_time(fold: &CvFold, timestamps: &[f64]) -> f64 {
    fold.train_indices
        .iter()
        .map(|&index| timestamps[index])
        .fold(f64::MIN, f64::max)
}

fn min_time(fold: &CvFold, timestamps: &[f64]) -> f64 {
    fold.test_indices
        .iter()
        .map(|&index| timestamps[index])
        .fold(f64::MAX, f64::min)
}

#[test]
fn time_ordered_folds_keep_test_strictly_after_purged_train() {
    let timestamps: Vec<f64> = (0..12).map(|index| index as f64 * 2_000.0).collect();
    let labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let gap = 3_000.0;

    let folds = create_time_ordered_folds(&timestamps, &labels, 3, gap);
    assert_eq!(folds.len(), 3);

    for fold in &folds {
        assert!(!fold.train_indices.is_empty());
        assert!(!fold.test_indices.is_empty());
        let latest_train = max_time(fold, &timestamps);
        let earliest_test = min_time(fold, &timestamps);
        // Every test frame is captured strictly later than all train frames, and the
        // nearest train frame sits at least one purge gap before the split boundary.
        assert!(earliest_test > latest_train);
        assert!(earliest_test - latest_train >= gap);
    }
}

#[test]
fn time_ordered_folds_fall_back_to_single_holdout() {
    // Four frames cannot fill six segments for a 5-fold split, so a single
    // earliest-train / latest-test holdout is produced instead.
    let timestamps = [0.0, 10_000.0, 20_000.0, 30_000.0];
    let labels = [0, 0, 1, 1];

    let folds = create_time_ordered_folds(&timestamps, &labels, 5, 15_000.0);
    assert_eq!(folds.len(), 1);

    let fold = &folds[0];
    assert_ne!(fold.train_indices, fold.test_indices);
    assert!(!fold.train_indices.is_empty());
    assert!(!fold.test_indices.is_empty());
    // The frame one step before the boundary is purged by the 15s embargo.
    assert!(min_time(fold, &timestamps) - max_time(fold, &timestamps) >= 15_000.0);
}

#[test]
fn time_ordered_folds_return_identity_sentinel_for_single_burst() {
    // All frames inside one sub-gap burst: no purged holdout is possible, so the
    // single all-data identity fold ("insufficient data") is returned.
    let timestamps = [0.0, 1_000.0, 2_000.0, 3_000.0];
    let labels = [0, 0, 1, 1];

    let folds = create_time_ordered_folds(&timestamps, &labels, 2, 15_000.0);
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].train_indices, folds[0].test_indices);
}

#[test]
fn time_ordered_folds_reject_malformed_input() {
    assert!(create_time_ordered_folds(&[], &[], 3, 15_000.0).is_empty());
    assert!(create_time_ordered_folds(&[1.0, f64::INFINITY], &[0, 1], 3, 15_000.0).is_empty());
    assert!(create_time_ordered_folds(&[1.0, 2.0], &[0, -1], 3, 15_000.0).is_empty());
}

#[test]
fn creates_folds_with_stratified_distribution() {
    let labels = [0, 0, 0, 1, 1, 1];

    let folds = create_stratified_k_fold(&labels, 3, 42.0);

    // With 3 samples per class and requesting 3 folds, we get 3 folds.
    assert_eq!(folds.len(), 3);
    assert!(validate_folds(&folds, labels.len()));
}

#[test]
fn ensures_no_overlap_between_train_and_test() {
    // Use more samples to get multiple folds: 6 per class -> floor(6 / 2) = 3 folds.
    let labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    let folds = create_stratified_k_fold(&labels, 3, 42.0);

    for fold in folds {
        for index in fold.test_indices {
            assert!(!fold.train_indices.contains(&index));
        }
    }
}

#[test]
fn maintains_class_balance_across_folds() {
    let labels = [0, 0, 0, 1, 1, 1];

    let folds = create_stratified_k_fold(&labels, 3, 42.0);

    for fold in folds {
        let distribution = get_class_distribution(&labels, &fold.test_indices).unwrap();
        // Each fold should have 1 good (0) and 1 bad (1).
        assert!(distribution[&0] >= 1);
        assert!(distribution[&1] >= 1);
    }
}

#[test]
fn calculates_class_distribution() {
    let labels = [0, 1, 0];
    let indices = [0, 1, 2];

    let distribution = get_class_distribution(&labels, &indices).unwrap();

    assert_eq!(distribution[&0], 2);
    assert_eq!(distribution[&1], 1);
}

#[test]
fn validates_valid_folds() {
    let folds = [
        CvFold {
            train_indices: vec![1, 2],
            test_indices: vec![0],
        },
        CvFold {
            train_indices: vec![0, 2],
            test_indices: vec![1],
        },
        CvFold {
            train_indices: vec![0, 1],
            test_indices: vec![2],
        },
    ];

    assert!(validate_folds(&folds, 3));
}

#[test]
fn rejects_duplicate_test_indices() {
    let folds = [
        CvFold {
            train_indices: vec![1, 2],
            test_indices: vec![0],
        },
        CvFold {
            train_indices: vec![0, 2],
            test_indices: vec![0],
        },
    ];

    assert!(!validate_folds(&folds, 3));
}

#[test]
fn returns_empty_temporal_blocks_for_empty_input() {
    let blocks = detect_temporal_blocks(&[], &[], 15_000.0);
    assert!(blocks.is_empty());
}

#[test]
fn groups_consecutive_frames_with_same_label_into_one_block() {
    let timestamps = [1_000.0, 2_000.0, 3_000.0, 4_000.0];
    let labels = [0, 0, 0, 0];

    let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);

    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].indices, vec![0, 1, 2, 3]);
    assert_eq!(blocks[0].label, 0);
    assert_eq!(blocks[0].start_time, 1_000.0);
    assert_eq!(blocks[0].end_time, 4_000.0);
}

#[test]
fn splits_temporal_blocks_when_label_changes() {
    let timestamps = [1_000.0, 2_000.0, 3_000.0, 4_000.0];
    let labels = [0, 0, 1, 1];

    let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].indices, vec![0, 1]);
    assert_eq!(blocks[0].label, 0);
    assert_eq!(blocks[1].indices, vec![2, 3]);
    assert_eq!(blocks[1].label, 1);
}

#[test]
fn splits_temporal_blocks_when_gap_exceeds_threshold() {
    let timestamps = [1_000.0, 2_000.0, 20_000.0, 21_000.0];
    let labels = [0, 0, 0, 0];

    let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].indices, vec![0, 1]);
    assert_eq!(blocks[1].indices, vec![2, 3]);
}

#[test]
fn sorts_unsorted_timestamps_in_temporal_blocks() {
    let timestamps = [3_000.0, 1_000.0, 4_000.0, 2_000.0];
    let labels = [0, 0, 0, 0];

    let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);

    assert_eq!(blocks.len(), 1);
    // Indices are sorted by timestamp: [1, 3, 0, 2].
    assert_eq!(blocks[0].indices, vec![1, 3, 0, 2]);
}

#[test]
fn returns_empty_temporal_block_folds_for_empty_input() {
    let folds = create_temporal_block_k_fold(&[], &[], 5, 15_000.0, 42.0);
    assert!(folds.is_empty());
}

#[test]
fn keeps_temporal_blocks_together_in_folds() {
    // Four blocks of class 0 followed by four blocks of class 1.
    let mut timestamps = Vec::new();
    let mut labels = Vec::new();

    for block in 0..8 {
        let label = if block < 4 { 0 } else { 1 };
        let base_time = block as f64 * 20_000.0;
        for offset in 0..3 {
            timestamps.push(base_time + offset as f64 * 1_000.0);
            labels.push(label);
        }
    }

    let folds = create_temporal_block_k_fold(&timestamps, &labels, 4, 15_000.0, 42.0);

    assert_eq!(folds.len(), 4);
    assert!(validate_folds(&folds, labels.len()));

    for fold in folds {
        let test_distribution = get_class_distribution(&labels, &fold.test_indices).unwrap();
        assert!(test_distribution[&0] > 0);
        assert!(test_distribution[&1] > 0);
    }
}

#[test]
fn does_not_split_frames_from_one_block_across_train_and_test() {
    let timestamps = [
        1_000.0, 2_000.0, 3_000.0, // Block 0: class 0
        20_000.0, 21_000.0, 22_000.0, // Block 1: class 0
        40_000.0, 41_000.0, 42_000.0, // Block 2: class 1
        60_000.0, 61_000.0, 62_000.0, // Block 3: class 1
    ];
    let labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    let folds = create_temporal_block_k_fold(&timestamps, &labels, 2, 15_000.0, 42.0);

    assert_eq!(folds.len(), 2);
    let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);

    for fold in folds {
        for block in &blocks {
            let indices_in_test = block
                .indices
                .iter()
                .filter(|&&index| fold.test_indices.contains(&index))
                .count();
            assert!(indices_in_test == 0 || indices_in_test == block.indices.len());
        }
    }
}

#[test]
fn falls_back_to_single_fold_when_temporal_blocks_are_insufficient() {
    let timestamps = [1_000.0, 2_000.0, 50_000.0, 51_000.0];
    let labels = [0, 0, 1, 1];

    let folds = create_temporal_block_k_fold(&timestamps, &labels, 5, 15_000.0, 42.0);

    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].train_indices, vec![0, 1, 2, 3]);
    assert_eq!(folds[0].test_indices, vec![0, 1, 2, 3]);
}

#[test]
fn empty_labels_preserve_source_shaped_folds() {
    assert_eq!(
        create_stratified_k_fold(&[], 1, 42.0),
        vec![CvFold::default()]
    );
    assert_eq!(create_stratified_k_fold(&[], 5, 42.0).len(), 5);
    assert!(create_stratified_k_fold(&[], 5, 42.0)
        .iter()
        .all(|fold| fold.train_indices.is_empty() && fold.test_indices.is_empty()));
}

#[test]
fn sparse_labels_fall_back_to_single_all_data_fold() {
    // A single sample labeled 1 has max(label) = 1 >= sample_count = 1. The source
    // computes numClasses = 2 -> class 0 is empty -> minClassSize = 0 -> single fold.
    assert_eq!(
        create_stratified_k_fold(&[1], 5, 42.0),
        vec![CvFold {
            train_indices: vec![0],
            test_indices: vec![0],
        }]
    );

    // Sparse non-dense labeling: class 1..=8 are empty buckets -> single fold.
    assert_eq!(
        create_stratified_k_fold(&[0, 0, 0, 0, 0, 9], 5, 42.0),
        vec![CvFold {
            train_indices: vec![0, 1, 2, 3, 4, 5],
            test_indices: vec![0, 1, 2, 3, 4, 5],
        }]
    );

    // get_class_distribution initializes every bucket 0..=max(label) even when sparse.
    let distribution = get_class_distribution(&[0, 0, 0, 0, 0, 9], &[0, 1, 2, 3, 4, 5]).unwrap();
    assert_eq!(distribution[&0], 5);
    assert_eq!(distribution[&9], 1);
    assert_eq!(distribution[&5], 0);
}

#[test]
fn zero_folds_matches_source_single_all_data_fallback() {
    // Source: actualFolds = min(0, minClassSize) = 0 < 2 -> single all-data fold.
    // The n_folds == 0 case is NOT an early empty return; the `< 2` fallback owns it.
    assert_eq!(
        create_stratified_k_fold(&[0, 0, 1, 1], 0, 42.0),
        vec![CvFold {
            train_indices: vec![0, 1, 2, 3],
            test_indices: vec![0, 1, 2, 3],
        }]
    );
    assert_eq!(
        create_stratified_k_fold(&[0, 1], 0, 42.0),
        vec![CvFold {
            train_indices: vec![0, 1],
            test_indices: vec![0, 1],
        }]
    );

    // Temporal variant: when the smallest class has exactly one block, the source
    // takes the `minBlockCount < 2` fallback (no throw), so n_folds == 0 must also
    // yield the single all-data fold rather than an empty vector.
    assert_eq!(
        create_temporal_block_k_fold(
            &[1_000.0, 2_000.0, 50_000.0, 51_000.0],
            &[0, 0, 1, 1],
            0,
            15_000.0,
            42.0
        ),
        vec![CvFold {
            train_indices: vec![0, 1, 2, 3],
            test_indices: vec![0, 1, 2, 3],
        }]
    );
}

#[test]
fn validate_folds_matches_source_duplicate_only_check() {
    // Source validateFolds only rejects duplicate test indices, then checks
    // distinct-count == nSamples. It never range-checks index values. Five distinct
    // test indices where one is out of range still satisfies size === nSamples.
    assert!(validate_folds(
        &[CvFold {
            train_indices: Vec::new(),
            test_indices: vec![0, 1, 2, 3, 5],
        }],
        5,
    ));
}

#[test]
fn malformed_cross_validation_inputs_fail_safely() {
    assert!(create_stratified_k_fold(&[0, -1], 2, 42.0).is_empty());
    assert!(create_stratified_k_fold(&[0, 1], 2, f64::NAN).is_empty());
    assert!(detect_temporal_blocks(&[1.0, f64::INFINITY], &[0, 1], 15_000.0).is_empty());
    assert!(create_temporal_block_k_fold(&[1.0], &[0], 1, 15_000.0, f64::NAN).is_empty());
    assert!(!validate_folds(
        &[CvFold {
            train_indices: Vec::new(),
            test_indices: vec![2],
        }],
        2,
    ));
    assert!(get_class_distribution(&[0, 1], &[2]).is_err());
}
