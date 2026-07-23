use std::collections::BTreeMap;

use slouch_domain::{
    BoundingBox, FeatureId, FeatureMap, FeatureSource, FrameLabel, Keypoint, PostureFrame,
    Thumbnail, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_EYE, RIGHT_HIP,
    RIGHT_SHOULDER,
};
use slouch_ml::ported::{
    constants::{RTMPOSE_BACKBONE_POOLED_DIMS, RTMPOSE_GAU_POOLED_DIMS},
    feature_extraction::{
        build_feature_matrix, concatenate_features, extract_features,
        get_expected_concatenated_dimensions, get_expected_dimensions, is_feature_available,
        validate_frames_have_feature, validate_frames_have_features,
    },
};

struct MockFeatures {
    features: FeatureMap,
    keypoints: Vec<Keypoint>,
    bbox: BoundingBox,
}

impl FeatureSource for MockFeatures {
    type Bbox = BoundingBox;

    fn features(&self) -> &FeatureMap {
        &self.features
    }

    fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    fn bbox(&self) -> &Self::Bbox {
        &self.bbox
    }
}

fn create_mock_features(include_all: bool) -> MockFeatures {
    let mut features = BTreeMap::new();

    if include_all {
        features.insert(FeatureId::GauFeatures, vec![1.0; RTMPOSE_GAU_POOLED_DIMS]);
        features.insert(
            FeatureId::BackboneFeatures,
            vec![2.0; RTMPOSE_BACKBONE_POOLED_DIMS],
        );
    }

    MockFeatures {
        features,
        keypoints: Vec::new(),
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

fn create_mock_frame(id: &str, include_all: bool) -> PostureFrame {
    let container = create_mock_features(include_all);
    PostureFrame {
        id: id.to_owned(),
        label: FrameLabel::Good,
        timestamp: 1.0,
        features: container.features,
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: Vec::new(),
        },
        keypoints: Vec::new(),
        bbox: container.bbox,
    }
}

/// A container carrying the hidden 3D substrate plus 17 confident keypoints, so the computed
/// 3D posture features resolve through the public extraction dispatch. The substrate encodes a
/// minimal non-degenerate torso (hip-centered, `|SC - HC| = 1`) so the frame is real.
fn create_substrate_container(with_substrate: bool) -> MockFeatures {
    let mut features = BTreeMap::new();
    if with_substrate {
        let mut substrate = vec![0.0_f32; FeatureId::RawKeypoints3d.metadata().dimensions];
        let points = [
            (NOSE, [0.0_f32, -1.20, 0.10]),
            (LEFT_EYE, [-0.04, -1.25, 0.05]),
            (RIGHT_EYE, [0.04, -1.25, 0.05]),
            (LEFT_EAR, [-0.07, -1.25, 0.0]),
            (RIGHT_EAR, [0.07, -1.25, 0.0]),
            (LEFT_SHOULDER, [-0.20, -1.0, 0.0]),
            (RIGHT_SHOULDER, [0.20, -1.0, 0.0]),
            (LEFT_HIP, [-0.15, 0.0, 0.0]),
            (RIGHT_HIP, [0.15, 0.0, 0.0]),
        ];
        for (coco, xyz) in points {
            substrate[coco * 3] = xyz[0];
            substrate[coco * 3 + 1] = xyz[1];
            substrate[coco * 3 + 2] = xyz[2];
        }
        features.insert(FeatureId::RawKeypoints3d, substrate);
    }
    MockFeatures {
        features,
        keypoints: vec![Keypoint::new(0.0, 0.0, 0.9); 17],
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.9,
            width: 1.0,
            height: 1.0,
        },
    }
}

#[test]
fn extracts_computed_3d_features_from_the_hidden_substrate() {
    let container = create_substrate_container(true);

    // The substrate itself is a stored feature read straight from the container.
    let substrate = extract_features(&container, FeatureId::RawKeypoints3d).unwrap();
    assert_eq!(
        substrate.len(),
        FeatureId::RawKeypoints3d.metadata().dimensions
    );

    // The three computed 3D features resolve from the substrate + keypoints through dispatch.
    for feature in [
        FeatureId::PostureRaw3d,
        FeatureId::PostureGeometry3d,
        FeatureId::TorsoInvariant3d,
    ] {
        assert!(is_feature_available(&container, feature).unwrap());
        let values = extract_features(&container, feature).unwrap();
        assert_eq!(values.len(), feature.metadata().dimensions);
        assert!(values.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn computed_3d_feature_without_substrate_names_required_dependency() {
    let container = create_substrate_container(false);

    assert!(!is_feature_available(&container, FeatureId::PostureGeometry3d).unwrap());

    let error = extract_features(&container, FeatureId::PostureGeometry3d).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("not available in this container"));
    // required_dependencies is populated for the computed 3D features, so the substrate id is
    // named in the error instead of the "none" fallback.
    assert!(
        message.contains("raw_keypoints_3d"),
        "expected the required dependency to be named: {message}"
    );
}

#[test]
fn extracts_gau_features_stored_directly() {
    let container = create_mock_features(true);
    let features = extract_features(&container, "gau_features").unwrap();

    assert_eq!(features.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn extracts_backbone_features_stored_directly() {
    let container = create_mock_features(true);
    let features = extract_features(&container, "backbone_features").unwrap();

    assert_eq!(features.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn throws_on_unknown_feature_type() {
    let container = create_mock_features(true);

    let error = extract_features(&container, "unknown_feature").unwrap_err();
    assert!(error.to_string().contains("Unknown feature type"));
}

#[test]
fn throws_when_feature_dependency_is_missing() {
    let container = create_mock_features(false);

    let error = extract_features(&container, FeatureId::GauFeatures).unwrap_err();
    assert!(error
        .to_string()
        .contains("not available in this container"));
}

#[test]
fn validates_extracted_dimensions_match_registry() {
    let container = create_mock_features(true);
    let features = extract_features(&container, FeatureId::BackboneFeatures).unwrap();

    assert_eq!(features.len(), RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn concatenates_two_feature_types() {
    let container = create_mock_features(true);
    let concatenated = concatenate_features(
        &container,
        &[FeatureId::GauFeatures, FeatureId::BackboneFeatures],
    )
    .unwrap();

    let expected_length = RTMPOSE_GAU_POOLED_DIMS + RTMPOSE_BACKBONE_POOLED_DIMS;
    assert_eq!(concatenated.len(), expected_length);
}

#[test]
fn returns_single_feature_type_directly() {
    let container = create_mock_features(true);
    let concatenated = concatenate_features(&container, &[FeatureId::GauFeatures]).unwrap();

    assert_eq!(concatenated.len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn preserves_concatenation_order() {
    let container = create_mock_features(true);
    let features1 = extract_features(&container, FeatureId::BackboneFeatures).unwrap();
    let features2 = extract_features(&container, FeatureId::GauFeatures).unwrap();

    let concatenated = concatenate_features(
        &container,
        &[FeatureId::BackboneFeatures, FeatureId::GauFeatures],
    )
    .unwrap();

    assert_eq!(&concatenated[..features1.len()], features1.as_slice());
    assert_eq!(&concatenated[features1.len()..], features2.as_slice());
}

#[test]
fn throws_when_no_feature_types_are_specified() {
    let container = create_mock_features(true);

    let error = concatenate_features::<_, FeatureId>(&container, &[]).unwrap_err();
    assert!(error
        .to_string()
        .contains("At least one feature type must be specified"));
}

#[test]
fn throws_when_any_feature_type_is_missing() {
    let container = create_mock_features(false);

    let error = concatenate_features(
        &container,
        &[FeatureId::GauFeatures, FeatureId::BackboneFeatures],
    )
    .unwrap_err();
    assert!(error.to_string().contains("not available"));
}

#[test]
fn reports_feature_available_when_present() {
    let container = create_mock_features(true);

    assert!(is_feature_available(&container, "gau_features").unwrap());
    assert!(is_feature_available(&container, "backbone_features").unwrap());
}

#[test]
fn reports_feature_unavailable_when_dependency_is_missing() {
    let container = create_mock_features(false);

    assert!(!is_feature_available(&container, "gau_features").unwrap());
    assert!(!is_feature_available(&container, "backbone_features").unwrap());
}

#[test]
fn reports_unknown_feature_type_as_unavailable() {
    let container = create_mock_features(true);

    assert!(!is_feature_available(&container, "unknown_feature").unwrap());
}

#[test]
fn propagates_malformed_computed_feature_inputs() {
    let mut frame = create_mock_frame("malformed", false);
    frame.bbox.width = f64::NAN;
    let error = is_feature_available(&frame, "rtmdet_engineered").unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn validates_frames_with_feature_when_all_frames_have_it() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", true),
        create_mock_frame("frame3", true),
    ];

    let missing_ids = validate_frames_have_feature(&frames, "gau_features").unwrap();

    assert!(missing_ids.is_empty());
}

#[test]
fn returns_ids_of_frames_missing_a_feature() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", false),
        create_mock_frame("frame3", true),
        create_mock_frame("frame4", false),
    ];

    let missing_ids = validate_frames_have_feature(&frames, "gau_features").unwrap();

    assert_eq!(missing_ids, vec!["frame2".to_owned(), "frame4".to_owned()]);
}

#[test]
fn validates_empty_frame_array() {
    let missing_ids = validate_frames_have_feature(&[], "gau_features").unwrap();

    assert!(missing_ids.is_empty());
}

#[test]
fn reports_all_frames_when_all_are_missing_a_feature() {
    let frames = [
        create_mock_frame("frame1", false),
        create_mock_frame("frame2", false),
    ];

    let missing_ids = validate_frames_have_feature(&frames, "gau_features").unwrap();

    assert_eq!(missing_ids, vec!["frame1".to_owned(), "frame2".to_owned()]);
}

#[test]
fn validates_frames_with_all_features_when_complete() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", true),
    ];

    let missing_map =
        validate_frames_have_features(&frames, &["gau_features", "backbone_features"]).unwrap();

    assert!(missing_map.is_empty());
}

#[test]
fn maps_missing_features_to_frame_ids() {
    let frame1 = PostureFrame {
        id: "frame1".to_owned(),
        label: FrameLabel::Good,
        timestamp: 1.0,
        features: BTreeMap::from([(FeatureId::GauFeatures, vec![0.0; RTMPOSE_GAU_POOLED_DIMS])]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: Vec::new(),
        },
        keypoints: Vec::new(),
        bbox: create_mock_features(false).bbox,
    };
    let frame2 = PostureFrame {
        id: "frame2".to_owned(),
        label: FrameLabel::Good,
        timestamp: 1.0,
        features: BTreeMap::from([(
            FeatureId::BackboneFeatures,
            vec![0.0; RTMPOSE_BACKBONE_POOLED_DIMS],
        )]),
        thumbnail: Thumbnail {
            mime_type: "image/webp".to_owned(),
            bytes: Vec::new(),
        },
        keypoints: Vec::new(),
        bbox: create_mock_features(false).bbox,
    };

    let missing_map =
        validate_frames_have_features(&[frame1, frame2], &["gau_features", "backbone_features"])
            .unwrap();

    assert_eq!(missing_map.len(), 2);
    assert_eq!(
        missing_map.get("gau_features"),
        Some(&vec!["frame2".to_owned()])
    );
    assert_eq!(
        missing_map.get("backbone_features"),
        Some(&vec!["frame1".to_owned()])
    );
}

#[test]
fn validates_empty_frame_array_for_multiple_features() {
    let missing_map = validate_frames_have_features(&[], &["gau_features"]).unwrap();

    assert!(missing_map.is_empty());
}

#[test]
fn validates_empty_feature_types_array() {
    let frames = [create_mock_frame("frame1", true)];
    let feature_types: [FeatureId; 0] = [];

    let missing_map = validate_frames_have_features(&frames, &feature_types).unwrap();

    assert!(missing_map.is_empty());
}

#[test]
fn builds_matrix_with_single_feature_type() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", true),
        create_mock_frame("frame3", true),
    ];

    let matrix = build_feature_matrix(&frames, &[FeatureId::GauFeatures]).unwrap();

    assert_eq!(matrix.len(), 3);
    assert_eq!(matrix[0].len(), RTMPOSE_GAU_POOLED_DIMS);
    assert_eq!(matrix[1].len(), RTMPOSE_GAU_POOLED_DIMS);
    assert_eq!(matrix[2].len(), RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn builds_matrix_with_multiple_feature_types() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", true),
    ];

    let matrix = build_feature_matrix(
        &frames,
        &[FeatureId::GauFeatures, FeatureId::BackboneFeatures],
    )
    .unwrap();

    let expected_dimensions = RTMPOSE_GAU_POOLED_DIMS + RTMPOSE_BACKBONE_POOLED_DIMS;
    assert_eq!(matrix.len(), 2);
    assert_eq!(matrix[0].len(), expected_dimensions);
    assert_eq!(matrix[1].len(), expected_dimensions);
}

#[test]
fn builds_empty_matrix_for_empty_frame_array() {
    let matrix = build_feature_matrix(&[], &[FeatureId::GauFeatures]).unwrap();

    assert!(matrix.is_empty());
}

#[test]
fn throws_when_frames_are_missing_features() {
    let frames = [
        create_mock_frame("frame1", true),
        create_mock_frame("frame2", false),
    ];

    let error = build_feature_matrix(&frames, &["gau_features"]).unwrap_err();
    assert!(error.to_string().contains("Missing features in dataset"));
}

#[test]
fn preserves_requested_order_in_missing_feature_errors() {
    let frames = [create_mock_frame("frame1", false)];
    let first = build_feature_matrix(&frames, &["gau_features", "backbone_features"])
        .unwrap_err()
        .to_string();
    let second = build_feature_matrix(&frames, &["backbone_features", "gau_features"])
        .unwrap_err()
        .to_string();

    assert!(first.find("gau_features").unwrap() < first.find("backbone_features").unwrap());
    assert!(second.find("backbone_features").unwrap() < second.find("gau_features").unwrap());
}

#[test]
fn builds_matrix_with_all_feature_values() {
    let frames = [create_mock_frame("frame1", true)];

    let matrix = build_feature_matrix(&frames, &[FeatureId::GauFeatures]).unwrap();

    assert!(matrix[0].iter().all(|value| value.is_finite()));
}

#[test]
fn returns_correct_dimensions_for_gau() {
    let dimensions = get_expected_dimensions("gau_features").unwrap();

    assert_eq!(dimensions, RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn returns_correct_dimensions_for_backbone() {
    let dimensions = get_expected_dimensions("backbone_features").unwrap();

    assert_eq!(dimensions, RTMPOSE_BACKBONE_POOLED_DIMS);
}

#[test]
fn returns_correct_dimensions_for_a_single_feature() {
    let dimensions = get_expected_concatenated_dimensions(&[FeatureId::GauFeatures]).unwrap();

    assert_eq!(dimensions, RTMPOSE_GAU_POOLED_DIMS);
}

#[test]
fn returns_correct_dimensions_for_two_features() {
    let dimensions = get_expected_concatenated_dimensions(&[
        FeatureId::GauFeatures,
        FeatureId::BackboneFeatures,
    ])
    .unwrap();

    let expected = RTMPOSE_GAU_POOLED_DIMS + RTMPOSE_BACKBONE_POOLED_DIMS;
    assert_eq!(dimensions, expected);
}

#[test]
fn returns_zero_for_empty_feature_types() {
    let dimensions = get_expected_concatenated_dimensions::<FeatureId>(&[]).unwrap();

    assert_eq!(dimensions, 0);
}
