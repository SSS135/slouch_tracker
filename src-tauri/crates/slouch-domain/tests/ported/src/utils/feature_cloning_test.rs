use slouch_domain::{
    ported::src::utils::feature_cloning::clone_inference_features, FeatureId, FeatureMap,
};

fn feature_map() -> FeatureMap {
    FeatureMap::from([
        (FeatureId::RtmDetExtracted, vec![1.0_f32, 2.0]),
        (FeatureId::BackboneFeatures, vec![3.0_f32, 4.0]),
    ])
}

#[test]
fn clone_inference_features_clones_unified_features_dictionary() {
    let features = feature_map();
    let cloned = clone_inference_features(&features);

    assert_ne!(
        cloned[&FeatureId::RtmDetExtracted].as_ptr(),
        features[&FeatureId::RtmDetExtracted].as_ptr()
    );
    assert_ne!(
        cloned[&FeatureId::BackboneFeatures].as_ptr(),
        features[&FeatureId::BackboneFeatures].as_ptr()
    );
}

#[test]
fn clone_inference_features_does_not_share_mutations_in_features() {
    let features = feature_map();
    let mut cloned = clone_inference_features(&features);

    cloned.get_mut(&FeatureId::RtmDetExtracted).unwrap()[0] = 999.0;
    assert_eq!(features[&FeatureId::RtmDetExtracted][0], 1.0);
}

#[test]
fn clone_inference_features_does_not_share_mutations_in_other_features() {
    let features = feature_map();
    let mut cloned = clone_inference_features(&features);

    cloned.get_mut(&FeatureId::BackboneFeatures).unwrap()[0] = 999.0;
    assert_eq!(features[&FeatureId::BackboneFeatures][0], 3.0);
}

#[test]
fn clone_inference_features_handles_empty_features_dictionary() {
    let features = FeatureMap::new();
    let cloned = clone_inference_features(&features);

    assert!(cloned.is_empty());
}
