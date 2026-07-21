use serde::Deserialize;
use serde_json::json;
use slouch_domain::{
    validate_posture_frame, BoundingBox, FrameLabel, Keypoint, PostureFrame, Thumbnail,
    COCO_KEYPOINT_COUNT, LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE,
    LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP,
    RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    source_sha256: String,
    values: Vec<Keypoint>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../../fixtures/domain/keypoint-v1.json")).unwrap()
}

#[test]
fn constructor_preserves_signed_coordinates_and_unconstrained_scores() {
    let value = Keypoint::new(-12.5, -0.125, 1.5);
    assert_eq!(value.x, -12.5);
    assert_eq!(value.y, -0.125);
    assert_eq!(value.score, 1.5);
}

#[test]
fn plain_object_fixture_preserves_order_and_exact_values() {
    let fixture = fixture();
    assert_eq!(
        fixture.source_sha256,
        "3f411b325babc002f4d1ed34bca89e46a8335c8a71dc1131e2b1a141cc85e967"
    );
    let expected = [
        Keypoint::new(-12.5, 0.0, -0.25),
        Keypoint::new(9_007_199_254_740_991.0, -0.125, 1.5),
        Keypoint::new(10.25, 200.5, 0.875),
    ];
    assert_eq!(fixture.values, expected);
    assert_eq!(
        serde_json::to_value(&fixture.values).unwrap(),
        json!([
            { "x": -12.5, "y": 0.0, "score": -0.25 },
            { "x": 9007199254740991.0, "y": -0.125, "score": 1.5 },
            { "x": 10.25, "y": 200.5, "score": 0.875 }
        ])
    );
}

#[test]
fn coco_keypoint_indices_are_exact() {
    assert_eq!(
        [
            NOSE,
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_EAR,
            RIGHT_EAR,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_ELBOW,
            RIGHT_ELBOW,
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_HIP,
            RIGHT_HIP,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
    assert_eq!(COCO_KEYPOINT_COUNT, 17);
}

#[test]
fn serde_shape_is_exactly_x_y_score() {
    let point = Keypoint::new(10.25, 200.5, 0.875);
    assert_eq!(
        serde_json::to_value(point).unwrap(),
        json!({ "x": 10.25, "y": 200.5, "score": 0.875 })
    );
    assert_eq!(
        serde_json::from_value::<Keypoint>(json!({
          "x": 10.25,
          "y": 200.5,
          "score": 0.875
        }))
        .unwrap(),
        point
    );
}

#[test]
fn validation_is_separate_from_keypoint_construction() {
    // Construction stores any score verbatim; a SimCC activation mean can exceed 1.
    let point = Keypoint::new(1.0, 2.0, 1.5);
    assert_eq!(point.score, 1.5);
    let mut frame = PostureFrame {
        id: "frame-1".into(),
        timestamp: 1.0,
        features: Default::default(),
        thumbnail: Thumbnail {
            mime_type: "image/webp".into(),
            bytes: vec![1],
        },
        keypoints: vec![point; 17],
        bbox: BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            score: 0.5,
            width: 1.0,
            height: 1.0,
        },
        label: FrameLabel::Good,
    };
    // Out-of-unit-range keypoint scores are activations, not probabilities, so they pass.
    assert!(validate_posture_frame(&frame).is_ok());

    // Non-finite keypoint scores remain the only invalid case validation flags.
    frame.keypoints[0] = Keypoint::new(1.0, 2.0, f64::NAN);
    let error = validate_posture_frame(&frame).unwrap_err();
    assert!(error
        .issues
        .iter()
        .any(|issue| issue.path == "keypoints.0.score"));
}
