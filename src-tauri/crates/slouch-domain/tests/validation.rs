use slouch_domain::{
    has_required_frame_shape, is_feature_id, validate_bbox, validate_posture_frame, BoundingBox,
    FeatureId, FrameLabel, Keypoint, PostureFrame, Thumbnail, ValidationCode,
};

fn valid_frame() -> PostureFrame {
    let mut features = std::collections::BTreeMap::new();
    features.insert(FeatureId::GauFeatures, vec![0.25; 256]);
    PostureFrame {
        id: "frame-001".into(),
        timestamp: 1_700_000_000_123.5,
        features,
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![0, 1, 2, 255],
        },
        keypoints: (0..17)
            .map(|index| {
                Keypoint::new(
                    10.25 + 3.0 * index as f64,
                    200.5 - 2.0 * index as f64,
                    index as f64 / 16.0,
                )
            })
            .collect(),
        bbox: BoundingBox {
            x1: 10.25,
            y1: 20.5,
            x2: 110.75,
            y2: 220.5,
            score: 0.875,
            width: 100.5,
            height: 200.0,
        },
        label: FrameLabel::Good,
    }
}

fn has_code(frame: &PostureFrame, code: ValidationCode) -> bool {
    validate_posture_frame(frame)
        .unwrap_err()
        .issues
        .iter()
        .any(|issue| issue.code == code)
}

#[test]
fn accepts_valid_native_frame_and_all_labels() {
    for label in [
        FrameLabel::Good,
        FrameLabel::Bad,
        FrameLabel::Away,
        FrameLabel::Unused,
    ] {
        let mut frame = valid_frame();
        frame.label = label;
        assert!(validate_posture_frame(&frame).is_ok());
        assert!(has_required_frame_shape(&frame));
    }
}

#[test]
fn feature_id_guard_accepts_only_registry_ids() {
    assert!(FeatureId::ALL.iter().all(|id| is_feature_id(id.as_str())));
    assert!(!is_feature_id(""));
    assert!(!is_feature_id("gau_feature"));
}

#[test]
fn rejects_keypoint_count_and_nonfinite_score_but_accepts_activation_scores() {
    for count in [16, 18] {
        let mut frame = valid_frame();
        frame.keypoints.resize(count, Keypoint::new(0.0, 0.0, 1.0));
        assert!(has_code(&frame, ValidationCode::InvalidLength));
    }
    // Keypoint scores are NLF-calibrated confidences (normally in [0, 1]); validation
    // deliberately never range-checks them, so any finite value is accepted verbatim.
    for score in [3.2, 1.0001, -0.0001, 42.0] {
        let mut frame = valid_frame();
        frame.keypoints[0].score = score;
        assert!(validate_posture_frame(&frame).is_ok());
    }
    // Non-finite scores remain invalid.
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut frame = valid_frame();
        frame.keypoints[0].score = score;
        assert!(has_code(&frame, ValidationCode::NonFinite));
    }
}

#[test]
fn rejects_empty_id_nonpositive_timestamp_and_bad_thumbnail_metadata() {
    let mut frame = valid_frame();
    frame.id.clear();
    frame.timestamp = 0.0;
    frame.thumbnail.mime_type = "text/plain".into();
    frame.thumbnail.bytes.clear();
    let error = validate_posture_frame(&frame).unwrap_err();
    assert!(error.issues.iter().any(|issue| issue.path == "id"));
    assert!(error.issues.iter().any(|issue| issue.path == "timestamp"));
    assert!(error
        .issues
        .iter()
        .any(|issue| issue.path == "thumbnail.bytes"));
    assert!(error
        .issues
        .iter()
        .any(|issue| issue.path == "thumbnail.mimeType"));
}

#[test]
fn rejects_nonfinite_timestamp() {
    for timestamp in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut frame = valid_frame();
        frame.timestamp = timestamp;
        assert!(has_code(&frame, ValidationCode::NonFinite));
    }
}

#[test]
fn rejects_bbox_reversed_coordinates_and_out_of_range_values() {
    let mut frame = valid_frame();
    frame.bbox.x2 = 5.0; // x2 < x1 -> reversed coordinates
    frame.bbox.score = 1.1; // score outside [0, 1]
    frame.bbox.width = -1.0; // negative extent
    assert!(has_code(&frame, ValidationCode::OutOfRange));
}

#[test]
fn accepts_bbox_with_unclamped_detector_extent() {
    // Inference clamps x1..y2 to the frame while width/height keep the UNCLAMPED
    // detector extent (matching the frozen TS oracle), so width != x2 - x1 is
    // legitimate whenever the subject clips a frame edge. Requiring that identity
    // is the recurring false invariant and must never be validated.
    let mut frame = valid_frame();
    frame.bbox.width += 40.0;
    frame.bbox.height += 30.0;
    assert!(validate_posture_frame(&frame).is_ok());
    assert!(validate_bbox(&frame.bbox).is_ok());
}

#[test]
fn validate_bbox_rejects_only_genuine_defects() {
    let base = valid_frame().bbox;
    assert!(validate_bbox(&base).is_ok());

    let mut non_finite = base;
    non_finite.x1 = f64::NAN;
    assert!(validate_bbox(&non_finite).is_err());

    let mut reversed = base;
    reversed.x2 = base.x1 - 1.0;
    assert!(validate_bbox(&reversed).is_err());

    let mut negative_extent = base;
    negative_extent.width = -1.0;
    assert!(validate_bbox(&negative_extent).is_err());

    let mut bad_score = base;
    bad_score.score = 1.5;
    assert!(validate_bbox(&bad_score).is_err());

    // Unclamped extent (width != x2 - x1) is NOT a defect.
    let mut extent_mismatch = base;
    extent_mismatch.width += 25.0;
    extent_mismatch.height += 25.0;
    assert!(validate_bbox(&extent_mismatch).is_ok());
}

#[test]
fn rejects_unknown_dimensions_and_nonfinite_feature_values() {
    let mut frame = valid_frame();
    frame
        .features
        .insert(FeatureId::BackboneFeatures, vec![0.0; 1000]);
    frame.features.get_mut(&FeatureId::GauFeatures).unwrap()[0] = f32::NAN;
    let error = validate_posture_frame(&frame).unwrap_err();
    assert!(error
        .issues
        .iter()
        .any(|issue| issue.code == ValidationCode::InvalidDimensions));
    assert!(error
        .issues
        .iter()
        .any(|issue| issue.code == ValidationCode::NonFinite));
}
