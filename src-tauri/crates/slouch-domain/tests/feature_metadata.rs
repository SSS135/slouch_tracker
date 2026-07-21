use std::collections::BTreeSet;

use serde_json::json;
use slouch_domain::{feature_registry, FeatureId, ModelCategory};

#[test]
fn registry_contains_all_eighteen_unique_selectable_features() {
    let registry = feature_registry();
    assert_eq!(registry.len(), 18);
    assert_eq!(
        registry
            .iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>()
            .len(),
        18
    );
    assert!(registry.iter().all(|item| item.user_selectable));
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
