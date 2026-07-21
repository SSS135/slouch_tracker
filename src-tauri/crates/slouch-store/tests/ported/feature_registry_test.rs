use std::collections::{BTreeMap, BTreeSet};

use slouch_domain::{
    BoundingBox, FeatureId, FrameLabel, Keypoint, ModelCategory, PostureFrame, Thumbnail,
};
use slouch_ml::ported::constants::{
    RTMDET_EXTRACTED_DIMS, RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_GAU_POOLED_DIMS,
};
use slouch_store::ported::feature_registry::{
    extract_feature, get_all_feature_types, get_feature_dimensions,
    get_user_selectable_feature_types, is_computed_feature, is_feature_type,
    require_feature_definition, FeatureRegistryError, FEATURE_REGISTRY, FEATURE_TYPES,
};

fn frame(keypoints: Vec<Keypoint>, bbox: BoundingBox) -> PostureFrame {
    PostureFrame {
        id: "registry-frame".into(),
        timestamp: 1_700_000_000_000.0,
        features: BTreeMap::from([
            (
                FeatureId::BackboneFeatures,
                vec![0.25; FeatureId::BackboneFeatures.metadata().dimensions],
            ),
            (
                FeatureId::GauFeatures,
                vec![0.5; FeatureId::GauFeatures.metadata().dimensions],
            ),
            (
                FeatureId::RtmDetExtracted,
                vec![0.75; FeatureId::RtmDetExtracted.metadata().dimensions],
            ),
        ]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![1],
        },
        keypoints,
        bbox,
        label: FrameLabel::Good,
    }
}

fn valid_frame() -> PostureFrame {
    frame(
        (0..17)
            .map(|index| Keypoint::new(0.1 + index as f64 * 0.02, 0.2 + index as f64 * 0.015, 0.9))
            .collect(),
        BoundingBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.9,
            y2: 0.95,
            score: 0.95,
            width: 0.8,
            height: 0.85,
        },
    )
}

#[test]
fn registry_contains_every_feature_once_in_source_order() {
    assert_eq!(FEATURE_REGISTRY.len(), FeatureId::ALL.len());
    assert_eq!(FEATURE_TYPES, FeatureId::ALL);
    assert_eq!(get_all_feature_types(), FeatureId::ALL);
    let ids = FEATURE_REGISTRY
        .iter()
        .map(|definition| definition.id)
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), FeatureId::ALL.len());
    let names = FEATURE_REGISTRY
        .iter()
        .map(|definition| definition.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), FeatureId::ALL.len());
    for (index, id) in FeatureId::ALL.into_iter().enumerate() {
        assert_eq!(FEATURE_REGISTRY[index].id, id);
        assert!(is_feature_type(id.as_str()));
    }
    assert!(!is_feature_type("invalid_type"));
}

#[test]
fn helper_lookups_resolve_the_assigned_registry() {
    for definition in FEATURE_REGISTRY {
        assert_eq!(
            require_feature_definition(definition.id.as_str()).unwrap(),
            &definition,
        );
        assert_eq!(
            get_feature_dimensions(definition.id.as_str()).unwrap(),
            definition.dimensions,
        );
        assert_eq!(
            is_computed_feature(definition.id.as_str()).unwrap(),
            definition.computed,
        );
        assert!(definition.dimensions > 0);
        assert_eq!(definition.storage_cost == 0, definition.computed);
        // Oracle: stored features carry storageCost == dimensions * FLOAT32_BYTES (4).
        if !definition.computed {
            assert_eq!(
                definition.storage_cost,
                definition.dimensions * 4,
                "{}",
                definition.id.as_str()
            );
        }
    }

    let expected = format!(
        "Feature type \"missing\" not found in registry. Available types: {}",
        FeatureId::ALL
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );
    assert_eq!(
        require_feature_definition("missing")
            .unwrap_err()
            .to_string(),
        expected
    );
    assert_eq!(
        get_feature_dimensions("missing").unwrap_err().to_string(),
        expected
    );
    assert_eq!(
        is_computed_feature("missing").unwrap_err().to_string(),
        expected
    );
}

#[test]
fn stored_features_are_pinned_as_not_computed() {
    // Oracle: pooled features and rtmdet_extracted are stored, not computed
    // (featureRegistry.test.ts:65-73,144-155). Concrete `false` anchors guard against a
    // regression that flips a stored feature to computed (which self-referential asserts miss).
    for id in [
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
        FeatureId::RtmDetExtracted,
    ] {
        assert!(
            !is_computed_feature(id.as_str()).unwrap(),
            "{}",
            id.as_str()
        );
        let definition = require_feature_definition(id.as_str()).unwrap();
        assert!(!definition.computed, "{}", id.as_str());
        assert!(definition.storage_cost > 0, "{}", id.as_str());
    }
}

#[test]
fn selectable_features_are_derived_from_the_assigned_registry() {
    // Oracle: every registry entry has userSelectable: true, so all 16 ids surface.
    assert_eq!(get_user_selectable_feature_types(), FeatureId::ALL.to_vec());
    assert!(get_user_selectable_feature_types().contains(&FeatureId::RtmDetExtracted));

    // Oracle concrete metadata anchors for rtmdet_extracted (featureRegistry.test.ts:205-218).
    let rtmdet = require_feature_definition("rtmdet_extracted").unwrap();
    assert!(rtmdet.user_selectable);
    assert_eq!(rtmdet.model_type, Some(ModelCategory::Presence));
}

#[test]
fn stored_extractors_return_exact_vectors_and_preserve_missing_values() {
    // Oracle: each stored feature is anchored to its named dimension constant
    // (featureRegistry.test.ts:47-61,121-123).
    for id in [
        FeatureId::BackboneFeatures,
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
    ] {
        assert_eq!(
            require_feature_definition(id.as_str()).unwrap().dimensions,
            RTMPOSE_BACKBONE_POOLED_DIMS,
            "{}",
            id.as_str()
        );
    }
    for id in [
        FeatureId::GauFeatures,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
    ] {
        assert_eq!(
            require_feature_definition(id.as_str()).unwrap().dimensions,
            RTMPOSE_GAU_POOLED_DIMS,
            "{}",
            id.as_str()
        );
    }
    assert_eq!(
        require_feature_definition("rtmdet_extracted")
            .unwrap()
            .dimensions,
        RTMDET_EXTRACTED_DIMS,
    );

    let value = valid_frame();
    for (id, expected) in [
        (FeatureId::BackboneFeatures, 0.25_f32),
        (FeatureId::GauFeatures, 0.5_f32),
        (FeatureId::RtmDetExtracted, 0.75_f32),
    ] {
        let extracted = extract_feature(id.as_str(), &value).unwrap().unwrap();
        assert_eq!(extracted.len(), id.metadata().dimensions);
        assert!(extracted.iter().all(|lane| *lane == expected));
    }
    for id in [
        FeatureId::BackboneFeaturesMax,
        FeatureId::BackboneFeaturesStd,
        FeatureId::GauFeaturesMax,
        FeatureId::GauFeaturesStd,
    ] {
        assert_eq!(extract_feature(id.as_str(), &value).unwrap(), None);
    }
}

#[test]
fn every_computed_extractor_dispatches_with_its_registry_dimension() {
    let value = valid_frame();
    for definition in FEATURE_REGISTRY
        .iter()
        .filter(|definition| definition.computed)
    {
        let extracted = definition.extract(&value).unwrap_or_else(|error| {
            panic!("{} extraction failed: {error}", definition.id.as_str())
        });
        let extracted = extracted.unwrap_or_else(|| {
            panic!(
                "{} unexpectedly returned no feature",
                definition.id.as_str()
            )
        });
        assert_eq!(
            extracted.len(),
            definition.dimensions,
            "{}",
            definition.id.as_str()
        );
        assert!(extracted.iter().all(|lane| lane.is_finite()));
        assert_eq!(
            extract_feature(definition.id.as_str(), &value).unwrap(),
            Some(extracted)
        );
    }
}

#[test]
fn malformed_sources_return_errors_or_missing_values_without_panicking() {
    let short = frame(
        vec![Keypoint::new(0.1, 0.2, 0.9); 3],
        BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: 0.0,
            score: 0.0,
            width: 0.0,
            height: 0.0,
        },
    );
    for id in [
        FeatureId::EngineeredFeatures,
        FeatureId::Joint2d,
        FeatureId::Joint3d,
        FeatureId::Joint4d,
        FeatureId::PostureRaw,
        FeatureId::RawKeypoints,
        FeatureId::PostureGeometry,
        FeatureId::TorsoInvariant,
    ] {
        // Oracle contract: these keypoint-driven extractors return null (never throw)
        // on missing/short keypoints (featureRegistry.test.ts:250-264).
        assert_eq!(
            extract_feature(id.as_str(), &short).unwrap(),
            None,
            "{}",
            id.as_str()
        );
    }

    // Oracle: rtmdet_engineered yields a full-length finite default vector even from
    // degenerate input, while keypoint_scores follows the missing-value path (None).
    let rtmdet_dims = require_feature_definition("rtmdet_engineered")
        .unwrap()
        .dimensions;
    let rtmdet = extract_feature("rtmdet_engineered", &short)
        .unwrap()
        .expect("rtmdet_engineered returns a default vector for degenerate input");
    assert_eq!(rtmdet.len(), rtmdet_dims);
    assert!(rtmdet.iter().all(|lane| lane.is_finite()));

    assert_eq!(extract_feature("keypoint_scores", &short).unwrap(), None);

    let unknown = extract_feature("not_registered", &short).unwrap_err();
    assert!(matches!(unknown, FeatureRegistryError::Unknown(_)));
    assert!(unknown.to_string().contains("not_registered"));
}
