use std::collections::BTreeSet;

use serde_json::json;
use slouch_domain::{feature_registry, FeatureId, ModelCategory};

#[test]
fn registry_contains_all_twenty_five_unique_features_with_hidden_features_hidden() {
    let registry = feature_registry();
    assert_eq!(registry.len(), 25);
    assert_eq!(
        registry
            .iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>()
            .len(),
        25
    );

    // The 6 RTMPose backbone/GAU pooled features are retired: their variants and
    // dimensions persist for deserialization, but they are hidden from the selector.
    let retired = BTreeSet::from([
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
    ]);
    // Distinct category from retired: RawKeypoints3d is a present-by-design substrate,
    // produced every frame and consumed as a dependency by the computed 3D posture
    // features, so it is hidden from direct selection rather than retired.
    let hidden_substrate = BTreeSet::from([FeatureId::RawKeypoints3d]);
    for item in &registry {
        let hidden = retired.contains(&item.id) || hidden_substrate.contains(&item.id);
        assert_eq!(
            item.user_selectable, !hidden,
            "unexpected user_selectable for {}",
            item.id
        );
    }
}

#[test]
fn dimensions_storage_and_model_categories_match_current_registry() {
    let expected = [
        (
            FeatureId::BackboneFeatures,
            768,
            3072,
            ModelCategory::Posture,
        ),
        (
            FeatureId::BackboneFeaturesMax,
            768,
            3072,
            ModelCategory::Posture,
        ),
        (
            FeatureId::BackboneFeaturesStd,
            768,
            3072,
            ModelCategory::Posture,
        ),
        (FeatureId::GauFeatures, 256, 1024, ModelCategory::Posture),
        (FeatureId::GauFeaturesMax, 256, 1024, ModelCategory::Posture),
        (FeatureId::GauFeaturesStd, 256, 1024, ModelCategory::Posture),
        (
            FeatureId::RtmDetExtracted,
            384,
            1536,
            ModelCategory::Presence,
        ),
        (FeatureId::RtmDetEngineered, 135, 0, ModelCategory::Presence),
        (FeatureId::EngineeredFeatures, 54, 0, ModelCategory::Posture),
        (FeatureId::Joint2d, 81, 0, ModelCategory::Posture),
        (FeatureId::Joint3d, 125, 0, ModelCategory::Posture),
        (FeatureId::Joint4d, 625, 0, ModelCategory::Posture),
        (FeatureId::PostureRaw, 5, 0, ModelCategory::Posture),
        (FeatureId::RawKeypoints, 34, 0, ModelCategory::Posture),
        (FeatureId::PostureGeometry, 10, 0, ModelCategory::Posture),
        (FeatureId::TorsoInvariant, 7, 0, ModelCategory::Posture),
        (FeatureId::NlfDepth, 14, 56, ModelCategory::Posture),
        (FeatureId::NlfBackbone, 512, 2048, ModelCategory::Posture),
        (FeatureId::NlfBackboneMax, 512, 2048, ModelCategory::Posture),
        (FeatureId::NlfBackboneStd, 512, 2048, ModelCategory::Posture),
        (FeatureId::RawKeypoints3d, 51, 204, ModelCategory::Posture),
        (FeatureId::PostureRaw3d, 6, 0, ModelCategory::Posture),
        (FeatureId::PostureGeometry3d, 10, 0, ModelCategory::Posture),
        (FeatureId::TorsoInvariant3d, 9, 0, ModelCategory::Posture),
    ];
    for (id, dimensions, storage, model) in expected {
        let metadata = id.metadata();
        assert_eq!(
            (
                metadata.dimensions,
                metadata.storage_cost,
                metadata.model_type
            ),
            (dimensions, storage, Some(model))
        );
        assert_eq!(metadata.computed, storage == 0);
    }

    let keypoint_scores = FeatureId::KeypointScores.metadata();
    assert_eq!(
        (keypoint_scores.dimensions, keypoint_scores.storage_cost),
        (17, 0)
    );
    assert_eq!(keypoint_scores.model_type, None);
}

#[test]
fn optional_metadata_fields_match_typescript_omission_semantics() {
    let stored = serde_json::to_value(FeatureId::BackboneFeatures.metadata()).unwrap();
    assert_eq!(stored["modelType"], "posture");
    assert!(stored.get("requiresFitting").is_none());

    let detection = serde_json::to_value(FeatureId::RtmDetEngineered.metadata()).unwrap();
    assert_eq!(detection["modelType"], "presence");
    assert!(detection.get("requiresFitting").is_none());

    let keypoint_scores = serde_json::to_value(FeatureId::KeypointScores.metadata()).unwrap();
    assert!(keypoint_scores.get("modelType").is_none());
    assert_eq!(keypoint_scores["requiresFitting"], false);

    let nlf_depth = serde_json::to_value(FeatureId::NlfDepth.metadata()).unwrap();
    assert_eq!(nlf_depth["modelType"], "posture");
    assert_eq!(nlf_depth["dimensions"], 14);
    assert_eq!(nlf_depth["storageCost"], 56);
    assert_eq!(nlf_depth["computed"], false);
    assert!(nlf_depth.get("requiresFitting").is_none());

    let nlf_backbone_max = serde_json::to_value(FeatureId::NlfBackboneMax.metadata()).unwrap();
    assert_eq!(nlf_backbone_max["id"], "nlf_backbone_max");
    assert_eq!(nlf_backbone_max["modelType"], "posture");
    assert_eq!(nlf_backbone_max["dimensions"], 512);
    assert_eq!(nlf_backbone_max["storageCost"], 2048);
    assert_eq!(nlf_backbone_max["computed"], false);
    assert_eq!(nlf_backbone_max["userSelectable"], true);
    assert!(nlf_backbone_max.get("requiresFitting").is_none());

    // Hidden substrate: stored (not computed), storage cost 204, hidden from selection.
    let raw_keypoints_3d = serde_json::to_value(FeatureId::RawKeypoints3d.metadata()).unwrap();
    assert_eq!(raw_keypoints_3d["id"], "raw_keypoints_3d");
    assert_eq!(raw_keypoints_3d["modelType"], "posture");
    assert_eq!(raw_keypoints_3d["dimensions"], 51);
    assert_eq!(raw_keypoints_3d["storageCost"], 204);
    assert_eq!(raw_keypoints_3d["computed"], false);
    assert_eq!(raw_keypoints_3d["userSelectable"], false);
    assert!(raw_keypoints_3d.get("requiresFitting").is_none());

    // Computed 3D posture features: no storage cost, selectable, no fitting required.
    for id in [
        FeatureId::PostureRaw3d,
        FeatureId::PostureGeometry3d,
        FeatureId::TorsoInvariant3d,
    ] {
        let value = serde_json::to_value(id.metadata()).unwrap();
        assert_eq!(value["modelType"], "posture", "modelType for {id}");
        assert_eq!(value["storageCost"], 0, "storageCost for {id}");
        assert_eq!(value["computed"], true, "computed for {id}");
        assert_eq!(value["userSelectable"], true, "userSelectable for {id}");
        assert_eq!(value["requiresFitting"], false, "requiresFitting for {id}");
    }

    let engineered = serde_json::to_value(FeatureId::EngineeredFeatures.metadata()).unwrap();
    assert_eq!(
        engineered,
        json!({
            "id": "engineered_features",
            "name": "Posture Features (1D)",
            "description": "Body proportion ratios with 1D soft binning (54 dims)",
            "dimensions": 54,
            "storageCost": 0,
            "computed": true,
            "modelType": "posture",
            "userSelectable": true,
            "requiresFitting": false
        })
    );
}

#[test]
fn feature_ids_round_trip_through_serde() {
    for id in FeatureId::ALL {
        let encoded = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<FeatureId>(&encoded).unwrap(), id);
        assert_eq!(encoded.trim_matches('"'), id.as_str());
    }
}
