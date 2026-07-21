use std::collections::BTreeMap;

use serde_json::json;
use slouch_domain::{
    BoundingBox, CaptureAction, ClassifierConfig, ClassifierId, DatasetManifest,
    DimensionalityReductionConfig, DimensionalityReductionMethod, FeatureId, FrameLabel,
    FrameMetadata, PostureDataset, PostureFrame, PostureFrameMetadata, Thumbnail, TrainingSettings,
};

#[test]
fn plain_typescript_time_fields_preserve_fractional_numbers() {
    let archive = FrameMetadata {
        id: "archive-frame".into(),
        label: FrameLabel::Good,
        timestamp: 1_700_000_000_000.25,
        features: vec![FeatureId::GauFeatures],
    };
    assert_eq!(
        serde_json::to_value(archive).unwrap()["timestamp"],
        1_700_000_000_000.25
    );

    let capture = CaptureAction {
        frame_id: "capture-frame".into(),
        timestamp: 10.5,
        label: FrameLabel::Bad,
        thumbnail_url: "blob:thumbnail".into(),
    };
    assert_eq!(serde_json::to_value(capture).unwrap()["timestamp"], 10.5);

    let metadata = PostureFrameMetadata {
        id: "metadata-frame".into(),
        timestamp: 11.75,
        label: FrameLabel::Unused,
    };
    assert_eq!(serde_json::to_value(metadata).unwrap()["timestamp"], 11.75);

    let frame = PostureFrame {
        id: "frame".into(),
        timestamp: 12.125,
        features: BTreeMap::new(),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![1],
        },
        keypoints: Vec::new(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 1.0,
            width: 1.0,
            height: 1.0,
        },
        label: FrameLabel::Away,
    };
    assert_eq!(serde_json::to_value(frame).unwrap()["timestamp"], 12.125);

    let dataset = PostureDataset {
        frames: Vec::new(),
        version: 7,
        last_modified: 13.5,
    };
    let dataset_json = serde_json::to_value(dataset).unwrap();
    assert_eq!(dataset_json["version"], 7);
    assert_eq!(dataset_json["lastModified"], 13.5);

    let settings = TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id: ClassifierId::GaussianNb,
            params: BTreeMap::new(),
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: 16,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::RtmDetExtracted],
        feature_types: None,
        normalization_mode: None,
        cv_folds: 5,
        last_updated: 14.25,
    };
    let settings_json = serde_json::to_value(settings).unwrap();
    assert_eq!(settings_json["cvFolds"], 5);
    assert_eq!(settings_json["lastUpdated"], 14.25);
}

#[test]
fn integer_counters_still_reject_fractional_json_numbers() {
    assert!(serde_json::from_value::<PostureDataset>(json!({
        "frames": [],
        "version": 1.5,
        "lastModified": 10.25
    }))
    .is_err());

    assert!(serde_json::from_value::<DatasetManifest>(json!({
        "version": 1,
        "exportedAt": "2026-07-14T00:00:00.000Z",
        "frameCount": 1.5,
        "frames": []
    }))
    .is_err());

    assert!(serde_json::from_value::<TrainingSettings>(json!({
        "classifierConfig": { "classifierId": "gaussian_nb", "params": {} },
        "dimReductionConfig": { "method": "none", "components": 16 },
        "postureFeatureTypes": ["gau_features"],
        "presenceFeatureTypes": ["rtmdet_extracted"],
        "cvFolds": 2.5,
        "lastUpdated": 14.25
    }))
    .is_err());
}
