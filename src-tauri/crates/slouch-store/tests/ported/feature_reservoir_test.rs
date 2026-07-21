use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use slouch_domain::{BoundingBox, FeatureId, FeatureMap, Keypoint};
use slouch_store::ported::feature_reservoir::{FeatureReservoir, ReservoirSample};

static NEXT_DATABASE_ID: AtomicUsize = AtomicUsize::new(0);

/// Builds a database name that is unique both within a process (atomic counter)
/// and across separate `cargo test` invocations (process id + a nanosecond
/// timestamp). Without the cross-run components a file left behind by a run that
/// panicked before its trailing `clear()` could be reused by a later run with
/// the same in-process counter, feeding it stale rows.
fn unique_database_name(prefix: &str) -> String {
    let counter = NEXT_DATABASE_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!(
        "feature-reservoir-test-{prefix}-{}-{}-{counter}",
        std::process::id(),
        nanos
    )
}

fn create_mock_sample(id: usize) -> ReservoirSample {
    let keypoints = (0..17)
        .map(|index| Keypoint::new((10 + index + id) as f64, (20 + index + id) as f64, 0.9))
        .collect();

    ReservoirSample {
        features: FeatureMap::from([
            (
                FeatureId::BackboneFeatures,
                vec![0.1 + id as f32 * 0.01; FeatureId::BackboneFeatures.metadata().dimensions],
            ),
            (
                FeatureId::BackboneFeaturesMax,
                vec![0.2 + id as f32 * 0.01; FeatureId::BackboneFeaturesMax.metadata().dimensions],
            ),
            (
                FeatureId::BackboneFeaturesStd,
                vec![
                    0.05 + id as f32 * 0.001;
                    FeatureId::BackboneFeaturesStd.metadata().dimensions
                ],
            ),
            (
                FeatureId::GauFeatures,
                vec![0.3 + id as f32 * 0.01; FeatureId::GauFeatures.metadata().dimensions],
            ),
            (
                FeatureId::GauFeaturesMax,
                vec![0.4 + id as f32 * 0.01; FeatureId::GauFeaturesMax.metadata().dimensions],
            ),
            (
                FeatureId::GauFeaturesStd,
                vec![0.08 + id as f32 * 0.001; FeatureId::GauFeaturesStd.metadata().dimensions],
            ),
            (
                FeatureId::RtmDetExtracted,
                vec![0.5 + id as f32 * 0.01; FeatureId::RtmDetExtracted.metadata().dimensions],
            ),
        ]),
        keypoints,
        bbox: BoundingBox {
            x1: (100 + id) as f64,
            y1: (150 + id) as f64,
            x2: (400 + id) as f64,
            y2: (550 + id) as f64,
            score: 0.95,
            width: 300.0,
            height: 400.0,
        },
    }
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual} to be within {tolerance} of {expected}"
    );
}

#[test]
fn add_samples_until_max_samples() {
    let database_name = unique_database_name("fill");
    let reservoir = FeatureReservoir::new(10, &database_name);

    for id in 0..5 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    assert_eq!(reservoir.get_count().unwrap(), 5);
    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 5);
    assert_eq!(meta.count, 5);

    reservoir.clear().unwrap();
}

#[test]
fn fill_reservoir_to_max_samples() {
    let database_name = unique_database_name("full");
    let reservoir = FeatureReservoir::new(10, &database_name);

    for id in 0..10 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    assert_eq!(reservoir.get_count().unwrap(), 10);
    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 10);
    assert_eq!(meta.count, 10);

    reservoir.clear().unwrap();
}

#[test]
fn reservoir_sampling_keeps_maximum_count_after_full() {
    let database_name = unique_database_name("sampling");
    let reservoir = FeatureReservoir::new(10, &database_name);

    for id in 0..20 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 20);
    assert_eq!(meta.count, 10);
    assert_eq!(reservoir.get_all_samples().unwrap().len(), 10);

    reservoir.clear().unwrap();
}

#[test]
fn add_maintains_total_seen_count() {
    let database_name = unique_database_name("total-seen");
    let reservoir = FeatureReservoir::new(10, &database_name);

    for id in 0..100 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 100);
    assert_eq!(meta.count, 10);

    reservoir.clear().unwrap();
}

#[test]
fn get_all_samples_returns_empty_for_empty_reservoir() {
    let database_name = unique_database_name("empty");
    let reservoir = FeatureReservoir::new(10, &database_name);

    assert!(reservoir.get_all_samples().unwrap().is_empty());

    reservoir.clear().unwrap();
}

#[test]
fn get_all_samples_returns_samples_with_correct_data() {
    let database_name = unique_database_name("data");
    let reservoir = FeatureReservoir::new(10, &database_name);
    reservoir.add(create_mock_sample(0)).unwrap();
    reservoir.add(create_mock_sample(1)).unwrap();

    let samples = reservoir.get_all_samples().unwrap();
    assert_eq!(samples.len(), 2);

    assert_eq!(
        samples[0].features[&FeatureId::BackboneFeatures].len(),
        FeatureId::BackboneFeatures.metadata().dimensions
    );
    assert_close(
        samples[0].features[&FeatureId::BackboneFeatures][0],
        0.1,
        1e-6,
    );
    assert_eq!(
        samples[0].features[&FeatureId::GauFeatures].len(),
        FeatureId::GauFeatures.metadata().dimensions
    );
    assert_close(samples[0].features[&FeatureId::GauFeatures][0], 0.3, 1e-6);
    assert_eq!(
        samples[0].features[&FeatureId::RtmDetExtracted].len(),
        FeatureId::RtmDetExtracted.metadata().dimensions
    );
    assert_eq!(samples[0].keypoints.len(), 17);
    assert_eq!(samples[0].bbox.x1, 100.0);

    assert_close(
        samples[1].features[&FeatureId::BackboneFeatures][0],
        0.11,
        1e-6,
    );
    assert_eq!(samples[1].bbox.x1, 101.0);

    reservoir.clear().unwrap();
}

#[test]
fn get_all_samples_preserves_native_feature_vectors() {
    let database_name = unique_database_name("vectors");
    let reservoir = FeatureReservoir::new(10, &database_name);
    reservoir.add(create_mock_sample(0)).unwrap();

    let samples = reservoir.get_all_samples().unwrap();
    assert_eq!(samples.len(), 1);
    for id in [
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
        FeatureId::RtmDetExtracted,
    ] {
        assert_eq!(
            samples[0].features[&id].len(),
            id.metadata().dimensions,
            "{}",
            id.as_str()
        );
    }

    reservoir.clear().unwrap();
}

#[test]
fn get_count_returns_zero_for_empty_reservoir() {
    let database_name = unique_database_name("count-empty");
    let reservoir = FeatureReservoir::new(10, &database_name);

    assert_eq!(reservoir.get_count().unwrap(), 0);

    reservoir.clear().unwrap();
}

#[test]
fn get_count_returns_count_after_adding_samples() {
    let database_name = unique_database_name("count");
    let reservoir = FeatureReservoir::new(10, &database_name);
    for id in 0..3 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    assert_eq!(reservoir.get_count().unwrap(), 3);

    reservoir.clear().unwrap();
}

#[test]
fn get_meta_returns_default_values_for_empty_reservoir() {
    let database_name = unique_database_name("meta-empty");
    let reservoir = FeatureReservoir::new(10, &database_name);

    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 0);
    assert_eq!(meta.count, 0);
    assert_eq!(meta.max_samples, 10);

    reservoir.clear().unwrap();
}

#[test]
fn get_meta_reflects_add_operations() {
    let database_name = unique_database_name("meta");
    let reservoir = FeatureReservoir::new(10, &database_name);
    for id in 0..15 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 15);
    assert_eq!(meta.count, 10);
    assert_eq!(meta.max_samples, 10);

    reservoir.clear().unwrap();
}

#[test]
fn clear_removes_all_data() {
    let database_name = unique_database_name("clear-data");
    let reservoir = FeatureReservoir::new(10, &database_name);
    for id in 0..5 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    reservoir.clear().unwrap();

    assert_eq!(reservoir.get_count().unwrap(), 0);
    assert!(reservoir.get_all_samples().unwrap().is_empty());
}

#[test]
fn clear_resets_meta() {
    let database_name = unique_database_name("clear-meta");
    let reservoir = FeatureReservoir::new(10, &database_name);
    for id in 0..5 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    reservoir.clear().unwrap();

    let meta = reservoir.get_meta().unwrap();
    assert_eq!(meta.total_seen, 0);
    assert_eq!(meta.count, 0);
}

#[test]
fn reservoirs_with_same_database_share_samples() {
    let database_name = unique_database_name("shared");
    let reservoir = FeatureReservoir::new(10, &database_name);
    reservoir.add(create_mock_sample(0)).unwrap();

    let reservoir2 = FeatureReservoir::new(10, &database_name);
    assert_eq!(reservoir2.get_count().unwrap(), 1);

    reservoir.clear().unwrap();
}

#[test]
fn shared_database_keeps_max_samples_instance_specific() {
    let database_name = unique_database_name("shared-max");
    let reservoir1 = FeatureReservoir::new(10, &database_name);
    let reservoir2 = FeatureReservoir::new(100, &database_name);

    reservoir1.add(create_mock_sample(0)).unwrap();

    let meta = reservoir2.get_meta().unwrap();
    assert_eq!(meta.count, 1);
    assert_eq!(meta.max_samples, 100);

    reservoir1.clear().unwrap();
}

#[test]
fn reservoir_sampling_represents_each_generation() {
    let total_elements = 1_000;
    let run_count = 5;
    let mut early = 0;
    let mut middle = 0;
    let mut late = 0;

    // Each iteration must be an INDEPENDENT trial. `clear()` deletes the state
    // row, so re-adding to the same reservoir restarts the RNG from the fixed
    // seed derived from the database name and reproduces byte-identical
    // contents; reusing one reservoir would collapse all runs into a single
    // deterministic trial. A distinct database name per run gives a distinct
    // seed, so the five runs are 500 genuinely-independent samples.
    for run in 0..run_count {
        let database_name = unique_database_name(&format!("statistics-{run}"));
        let reservoir = FeatureReservoir::new(100, &database_name);

        for id in 0..total_elements {
            let mut sample = create_mock_sample(id);
            sample
                .features
                .get_mut(&FeatureId::BackboneFeatures)
                .unwrap()[0] = id as f32;
            reservoir.add(sample).unwrap();
        }

        for sample in reservoir.get_all_samples().unwrap() {
            let id = sample.features[&FeatureId::BackboneFeatures][0];
            if id < 100.0 {
                early += 1;
            } else if id < 500.0 {
                middle += 1;
            } else {
                late += 1;
            }
        }

        reservoir.clear().unwrap();
    }

    let total_samples = (run_count * 100) as f32;
    assert!(early > 0);
    assert!(middle > 0);
    assert!(late > 0);
    assert!(early as f32 / total_samples > 0.02);
    assert!(middle as f32 / total_samples > 0.15);
    assert!(late as f32 / total_samples > 0.25);
}

#[test]
fn max_samples_one_keeps_one_sample_and_tracks_total_seen() {
    let database_name = unique_database_name("one");
    let reservoir = FeatureReservoir::new(1, &database_name);
    for id in 0..10 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    assert_eq!(reservoir.get_count().unwrap(), 1);
    assert_eq!(reservoir.get_meta().unwrap().total_seen, 10);

    reservoir.clear().unwrap();
}

#[test]
fn large_max_samples_handles_small_input() {
    let database_name = unique_database_name("large");
    let reservoir = FeatureReservoir::new(10_000, &database_name);
    for id in 0..5 {
        reservoir.add(create_mock_sample(id)).unwrap();
    }

    assert_eq!(reservoir.get_count().unwrap(), 5);

    reservoir.clear().unwrap();
}

#[test]
fn feature_values_preserve_float_precision() {
    let database_name = unique_database_name("precision");
    let reservoir = FeatureReservoir::new(10, &database_name);
    let mut sample = create_mock_sample(0);
    {
        let backbone = sample
            .features
            .get_mut(&FeatureId::BackboneFeatures)
            .unwrap();
        backbone[0] = 0.123_456_79;
        backbone[767] = -0.987_654_3;
    }
    reservoir.add(sample).unwrap();

    let samples = reservoir.get_all_samples().unwrap();
    assert_close(
        samples[0].features[&FeatureId::BackboneFeatures][0],
        0.123_456_79,
        1e-6,
    );
    assert_close(
        samples[0].features[&FeatureId::BackboneFeatures][767],
        -0.987_654_3,
        1e-6,
    );

    reservoir.clear().unwrap();
}

#[test]
fn persistence_survives_reopen_and_clear_is_atomic() {
    let database_name = unique_database_name("reopen");
    let path;
    {
        let reservoir = FeatureReservoir::new(10, &database_name);
        path = reservoir.database_path().to_path_buf();
        reservoir.add(create_mock_sample(0)).unwrap();
        reservoir.add(create_mock_sample(1)).unwrap();
        assert_eq!(reservoir.get_meta().unwrap().total_seen, 2);
    }

    {
        let reopened = FeatureReservoir::new(10, &database_name);
        assert_eq!(reopened.get_count().unwrap(), 2);
        assert_eq!(reopened.get_meta().unwrap().total_seen, 2);
        assert_eq!(reopened.get_all_samples().unwrap().len(), 2);
        reopened.clear().unwrap();
    }

    let reopened = FeatureReservoir::new(10, &database_name);
    assert_eq!(reopened.get_meta().unwrap().count, 0);
    assert_eq!(reopened.get_meta().unwrap().total_seen, 0);
    assert!(reopened.get_all_samples().unwrap().is_empty());
    drop(reopened);
    let _ = std::fs::remove_file(path);
}

#[test]
fn malformed_samples_and_capacities_are_rejected_without_mutation() {
    let database_name = unique_database_name("invalid");
    let reservoir = FeatureReservoir::new(10, &database_name);
    let baseline = reservoir.get_meta().unwrap();

    let mut invalid_cases = Vec::new();
    let mut stale_gau = create_mock_sample(0);
    stale_gau
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 384]);
    invalid_cases.push(stale_gau);
    let mut stale_rtmdet = create_mock_sample(0);
    stale_rtmdet
        .features
        .insert(FeatureId::RtmDetExtracted, vec![0.0; 768]);
    invalid_cases.push(stale_rtmdet);
    let mut non_finite = create_mock_sample(0);
    non_finite
        .features
        .get_mut(&FeatureId::BackboneFeatures)
        .unwrap()[0] = f32::NAN;
    invalid_cases.push(non_finite);
    let mut empty_features = create_mock_sample(0);
    empty_features.features.clear();
    invalid_cases.push(empty_features);
    let mut computed_feature = create_mock_sample(0);
    computed_feature
        .features
        .insert(FeatureId::EngineeredFeatures, vec![0.0; 54]);
    invalid_cases.push(computed_feature);
    let mut wrong_keypoints = create_mock_sample(0);
    wrong_keypoints.keypoints.pop();
    invalid_cases.push(wrong_keypoints);
    let mut nonfinite_score = create_mock_sample(0);
    nonfinite_score.keypoints[0].score = f64::NAN;
    invalid_cases.push(nonfinite_score);
    let mut bad_bbox = create_mock_sample(0);
    bad_bbox.bbox.x2 = bad_bbox.bbox.x1 - 1.0;
    invalid_cases.push(bad_bbox);
    let mut negative_extent_bbox = create_mock_sample(0);
    negative_extent_bbox.bbox.width = -1.0;
    invalid_cases.push(negative_extent_bbox);

    for sample in invalid_cases {
        assert!(reservoir.add(sample).is_err());
        assert_eq!(reservoir.get_meta().unwrap(), baseline);
    }

    // Keypoint scores are SimCC activation means, not probabilities: a score > 1 is
    // legitimate on real frames and must be accepted, not rejected.
    let mut high_score = create_mock_sample(0);
    high_score.keypoints[0].score = 3.2;
    assert!(reservoir.add(high_score).is_ok());

    // Inference clamps bbox coords to the frame but keeps the UNclamped detector
    // extent in width/height, so width != x2 - x1 is legitimate near frame edges
    // and must be accepted. Requiring that span identity was the recurring bug.
    let mut edge_extent = create_mock_sample(0);
    edge_extent.bbox.width += 50.0;
    edge_extent.bbox.height += 50.0;
    assert!(reservoir.add(edge_extent).is_ok());

    let zero = FeatureReservoir::new(0, unique_database_name("zero-capacity"));
    assert!(zero.add(create_mock_sample(0)).is_err());
    let excessive = FeatureReservoir::new(usize::MAX, unique_database_name("huge-capacity"));
    assert!(excessive.add(create_mock_sample(0)).is_err());

    reservoir.clear().unwrap();
}
