use std::collections::BTreeMap;

use slouch_domain::ported::guards::{is_feature_type, is_posture_frame};
use slouch_domain::{BoundingBox, FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail};

fn feature(feature_id: FeatureId, value: f32) -> (FeatureId, Vec<f32>) {
    (feature_id, vec![value; feature_id.metadata().dimensions])
}

fn valid_frame() -> PostureFrame {
    PostureFrame {
        id: "frame-123".to_owned(),
        timestamp: 1_700_000_000_000.0,
        keypoints: (0..17)
            .map(|index| Keypoint::new(10.5 + index as f64, 20.3 + index as f64, 0.95))
            .collect(),
        features: BTreeMap::from([feature(FeatureId::RtmDetExtracted, 0.1)]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: b"test".to_vec(),
        },
        label: FrameLabel::Good,
        bbox: BoundingBox {
            x1: 10.0,
            y1: 20.0,
            x2: 100.0,
            y2: 200.0,
            score: 0.95,
            width: 90.0,
            height: 180.0,
        },
    }
}

#[test]
fn accepts_valid_posture_frame() {
    assert!(is_posture_frame(&valid_frame()));
}

#[test]
fn accepts_frame_with_multiple_features() {
    let mut frame = valid_frame();
    frame.features.insert(
        FeatureId::GauFeatures,
        vec![0.2; FeatureId::GauFeatures.metadata().dimensions],
    );

    assert!(is_posture_frame(&frame));
}

#[test]
fn rejects_empty_thumbnail_bytes() {
    let mut frame = valid_frame();
    frame.thumbnail.bytes.clear();

    assert!(!is_posture_frame(&frame));
}

#[test]
fn rejects_non_image_thumbnail_mime_type() {
    let mut frame = valid_frame();
    frame.thumbnail.mime_type = "text/plain".to_owned();

    assert!(!is_posture_frame(&frame));
}

#[test]
fn accepts_image_mime_types_outside_common_whitelist() {
    for mime in ["image/gif", "image/avif", "image/bmp"] {
        let mut frame = valid_frame();
        frame.thumbnail.mime_type = mime.to_owned();
        assert!(is_posture_frame(&frame), "expected {mime} to be accepted");
    }
}

#[test]
fn accepts_different_labels() {
    for label in [
        FrameLabel::Good,
        FrameLabel::Bad,
        FrameLabel::Away,
        FrameLabel::Unused,
    ] {
        let mut frame = valid_frame();
        frame.label = label;
        assert!(is_posture_frame(&frame));
    }
}

#[test]
fn rejects_missing_required_id_shape() {
    let mut frame = valid_frame();
    frame.id.clear();

    assert!(!is_posture_frame(&frame));
}

#[test]
fn rejects_frames_without_keypoints() {
    let mut frame = valid_frame();
    frame.keypoints.clear();

    assert!(!is_posture_frame(&frame));
}

#[test]
fn accepts_frames_with_unclamped_bbox_extent() {
    // Inference clamps x1..y2 to the frame while width/height keep the UNCLAMPED
    // detector extent, so width != x2 - x1 is legitimate near frame edges and must
    // not be rejected. Requiring that span identity was the recurring false invariant.
    let mut frame = valid_frame();
    frame.bbox.width = 1.0;

    assert!(is_posture_frame(&frame));
}

#[test]
fn rejects_frames_with_invalid_bbox_geometry() {
    let mut reversed = valid_frame();
    reversed.bbox.x2 = reversed.bbox.x1 - 1.0;
    assert!(!is_posture_frame(&reversed));

    let mut negative_extent = valid_frame();
    negative_extent.bbox.width = -1.0;
    assert!(!is_posture_frame(&negative_extent));
}

#[test]
fn rejects_feature_vectors_with_invalid_dimensions() {
    let mut frame = valid_frame();
    frame
        .features
        .insert(FeatureId::GauFeatures, vec![0.0; 1_000]);

    assert!(!is_posture_frame(&frame));
}

#[test]
fn accepts_registry_sized_feature_vectors() {
    let mut frame = valid_frame();
    frame.features = BTreeMap::from([
        feature(FeatureId::GauFeatures, 0.0),
        feature(FeatureId::BackboneFeatures, 0.0),
    ]);

    assert!(is_posture_frame(&frame));
}

#[test]
fn rejects_non_finite_frame_values() {
    let mut frame = valid_frame();
    frame.timestamp = f64::NAN;

    assert!(!is_posture_frame(&frame));
}

#[test]
fn accepts_all_valid_feature_types() {
    for feature_type in FeatureId::ALL {
        assert!(is_feature_type(feature_type.as_str()));
    }
}

#[test]
fn rejects_empty_feature_type_strings() {
    assert!(!is_feature_type(""));
}

#[test]
fn rejects_unknown_feature_type_strings() {
    assert!(!is_feature_type("invalid"));
    assert!(!is_feature_type("backbone_features_unknown"));
}
