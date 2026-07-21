use slouch_domain::ported::messages::schemas::ImageData;
use slouch_domain::{BoundingBox, Keypoint};
use slouch_vision::ported::inference_worker::{
    compatibility_crop_pipeline, should_run_posture_for_presence,
};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual {actual}, expected {expected}"
    );
}

fn assert_f64_bits(actual: f64, expected: u64) {
    assert_eq!(
        actual.to_bits(),
        expected,
        "actual {actual} has unexpected IEEE-754 bits"
    );
}

fn source_image() -> ImageData {
    let mut data = Vec::new();
    for y in 0_u8..3 {
        for x in 0_u8..4 {
            let value = y * 10 + x;
            data.extend_from_slice(&[value, value + 40, value + 80, 255]);
        }
    }
    ImageData {
        data,
        width: 4,
        height: 3,
    }
}

#[test]
fn production_crop_pipeline_matches_browser_oracle_rounding_and_coordinates() {
    let image = source_image();
    let bbox = BoundingBox {
        x1: 0.5,
        y1: 0.5,
        x2: 2.5,
        y2: 2.0,
        score: 0.95,
        width: 2.0,
        height: 1.5,
    };
    let pose_keypoints = [
        Keypoint::new(96.0, 128.0, 0.9),
        Keypoint::new(0.0, 0.0, 0.8),
    ];

    let output = compatibility_crop_pipeline(&image, &bbox, &pose_keypoints).unwrap();
    let expanded = output.expanded_pixels.expanded;
    for (actual, expected) in [
        (expanded.x1, 0.1),
        (expanded.y1, 0.2),
        (expanded.x2, 2.9),
        (expanded.y2, 2.3),
        (expanded.width, 2.8),
        (expanded.height, 2.1),
    ] {
        close(actual, expected);
    }

    assert_eq!(output.expanded_pixels.original, bbox);
    assert_eq!(
        output.expanded_pixels.expanded.score.to_bits(),
        bbox.score.to_bits()
    );

    assert_eq!((output.cropped.width, output.cropped.height), (3, 3));
    let expected_pixels = image
        .data
        .chunks_exact(4)
        .enumerate()
        .filter(|(index, _)| index % 4 < 3)
        .flat_map(|(_, pixel)| pixel.iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(output.cropped.data, expected_pixels);

    let normalized_original = output.normalized_bbox.original;
    assert_f64_bits(normalized_original.x1, 0x3fc0000000000000);
    assert_f64_bits(normalized_original.y1, 0x3fc5555555555555);
    assert_f64_bits(normalized_original.x2, 0x3fe4000000000000);
    assert_f64_bits(normalized_original.y2, 0x3fe5555555555555);
    assert_f64_bits(normalized_original.score, 0x3fee666666666666);
    assert_f64_bits(normalized_original.width, 0x3fe0000000000000);
    assert_f64_bits(normalized_original.height, 0x3fe0000000000000);

    close(output.normalized_bbox.expanded.x1, 0.025);
    close(output.normalized_bbox.expanded.y1, 0.2 / 3.0);
    close(output.normalized_bbox.expanded.x2, 0.725);
    close(output.normalized_bbox.expanded.y2, 2.3 / 3.0);
    assert_eq!(
        output.normalized_bbox.expanded.score.to_bits(),
        bbox.score.to_bits()
    );
    assert_eq!(output.normalized_keypoints.len(), 2);
    assert_f64_bits(output.normalized_keypoints[0].x, 0x3fd999999999999a);
    assert_f64_bits(output.normalized_keypoints[0].y, 0x3fe2222222222222);
    assert_f64_bits(output.normalized_keypoints[0].score, 0x3feccccccccccccd);
    assert_f64_bits(output.normalized_keypoints[1].x, 0x3f99999999999998);
    assert_f64_bits(output.normalized_keypoints[1].y, 0x3fb1111111111110);
    assert_f64_bits(output.normalized_keypoints[1].score, 0x3fe999999999999a);
}

#[test]
fn production_crop_pipeline_clamps_expansion_to_image_bounds() {
    let image = source_image();
    // Padding (0.2) pushes both edges past the image on every side, so the
    // oracle's lower Math.max(0, ...) and upper Math.min(imageW/H, ...) clamps
    // in expandBbox must both fire (padX = 3.6*0.2 = 0.72, padY = 2.8*0.2 = 0.56).
    let bbox = BoundingBox {
        x1: 0.2,
        y1: 0.1,
        x2: 3.8,
        y2: 2.9,
        score: 0.75,
        width: 3.6,
        height: 2.8,
    };

    let output = compatibility_crop_pipeline(&image, &bbox, &[]).unwrap();
    let expanded = output.expanded_pixels.expanded;
    for (actual, expected) in [
        (expanded.x1, 0.0),
        (expanded.y1, 0.0),
        (expanded.x2, 4.0),
        (expanded.y2, 3.0),
        (expanded.width, 4.0),
        (expanded.height, 3.0),
    ] {
        close(actual, expected);
    }

    // Clamped expansion covers the whole image, so the cropImageData clamps
    // (Math.min(width/height, ceil(...))) resolve to the full 4x3 frame.
    assert_eq!((output.cropped.width, output.cropped.height), (4, 3));
    assert_eq!(output.cropped.data, image.data);

    let normalized = output.normalized_bbox.expanded;
    for (actual, expected) in [
        (normalized.x1, 0.0),
        (normalized.y1, 0.0),
        (normalized.x2, 1.0),
        (normalized.y2, 1.0),
        (normalized.width, 1.0),
        (normalized.height, 1.0),
    ] {
        close(actual, expected);
    }
    assert_eq!(
        output.normalized_bbox.expanded.score.to_bits(),
        bbox.score.to_bits()
    );
    assert!(output.normalized_keypoints.is_empty());
}

fn source_image_8x8() -> ImageData {
    let mut data = Vec::new();
    for y in 0_u8..8 {
        for x in 0_u8..8 {
            let value = y.wrapping_mul(8).wrapping_add(x);
            data.extend_from_slice(&[value, value.wrapping_add(40), value.wrapping_add(80), 255]);
        }
    }
    ImageData {
        data,
        width: 8,
        height: 8,
    }
}

#[test]
fn production_crop_pipeline_uses_correct_axis_for_padding_and_keypoint_scale() {
    // Non-square crop (5 wide, 4 tall) so a keypoint scale that swaps
    // crop_width/crop_height (scale_x = cropW/192, scale_y = cropH/256) diverges.
    let image = source_image_8x8();
    // width/height struct fields are deliberately bogus (not x2-x1 / y2-y1) so any
    // padding derived from the struct fields instead of the coordinate delta breaks
    // the expanded-box assertions below.
    let bbox = BoundingBox {
        x1: 1.0,
        y1: 1.0,
        x2: 4.0,
        y2: 3.0,
        score: 0.9,
        width: 99.0,
        height: 42.0,
    };
    // Keypoint at model-space (192, 256); both coords non-zero so each scale factor
    // contributes and an axis swap changes the transformed result.
    let pose_keypoints = [Keypoint::new(192.0, 256.0, 0.5)];

    let output = compatibility_crop_pipeline(&image, &bbox, &pose_keypoints).unwrap();

    // padding derives from coordinate delta (x2-x1 = 3 -> padX 0.6, y2-y1 = 2 -> padY 0.4),
    // NOT the bogus struct width/height fields.
    let expanded = output.expanded_pixels.expanded;
    for (actual, expected) in [
        (expanded.x1, 0.4),
        (expanded.y1, 0.6),
        (expanded.x2, 4.6),
        (expanded.y2, 3.4),
        (expanded.width, 4.2),
        (expanded.height, 2.8),
    ] {
        close(actual, expected);
    }

    assert_eq!((output.cropped.width, output.cropped.height), (5, 4));

    // scale_x = 5/192 (crop_width), scale_y = 4/256 (crop_height); swapping the crop
    // axis for either factor moves x off 0.675 and/or y off 0.575.
    assert_eq!(output.normalized_keypoints.len(), 1);
    close(output.normalized_keypoints[0].x, 5.4 / 8.0);
    close(output.normalized_keypoints[0].y, 4.6 / 8.0);
    close(output.normalized_keypoints[0].score, 0.5);
}

#[test]
fn production_presence_cascade_keeps_the_frozen_threshold_boundary() {
    assert!(!should_run_posture_for_presence(f64::from(f32::from_bits(
        0.5_f32.to_bits() - 1,
    ))));
    assert!(should_run_posture_for_presence(0.5));
    assert!(should_run_posture_for_presence(0.9));
}
