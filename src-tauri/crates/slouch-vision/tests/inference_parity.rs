use std::collections::BTreeSet;

use serde_json::Value;
use sha2::{Digest, Sha256};
use slouch_domain::ported::messages::schemas::ImageData;
use slouch_vision::ported::inference_worker::{
    compatibility_preprocess_rtmdet, compatibility_run_detector, compatibility_select_person_bbox,
    should_run_posture_for_presence,
};

const VISION_ABSOLUTE_TOLERANCE: f64 = 2e-4;
const VISION_RELATIVE_TOLERANCE: f64 = 2e-4;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/vision/vision-inference-v1.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn frame(case: &Value) -> ImageData {
    let relative = case["frame"]["path"]
        .as_str()
        .unwrap()
        .strip_prefix("src-tauri/")
        .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    ImageData {
        data: std::fs::read(path).unwrap(),
        width: case["frame"]["width"].as_u64().unwrap() as u32,
        height: case["frame"]["height"].as_u64().unwrap() as u32,
    }
}

fn f32_sha(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    format!("{:x}", Sha256::digest(bytes))
}

fn close(actual: f64, expected: f64) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= VISION_ABSOLUTE_TOLERANCE
            || difference <= VISION_RELATIVE_TOLERANCE * expected.abs(),
        "actual {actual} expected {expected}"
    );
}

fn compare_all(values: &[f32], summary: &Value) {
    let expected = summary["values"].as_array().unwrap();
    assert_eq!(values.len(), summary["length"].as_u64().unwrap() as usize);
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        close(f64::from(*actual), expected.as_f64().unwrap());
    }
}

#[test]
fn all_synthetic_frames_match_typescript_preprocessing_bit_exactly() {
    let oracle = fixture();
    assert_eq!(
        oracle["exactBehavior"]["detectionThreshold"]["operator"],
        ">"
    );
    assert_eq!(oracle["exactBehavior"]["detectionThreshold"]["value"], 0.3);
    assert_eq!(
        oracle["exactBehavior"]["presenceThreshold"]["operator"],
        ">="
    );
    assert_eq!(oracle["exactBehavior"]["presenceThreshold"]["value"], 0.5);
    assert_eq!(oracle["exactBehavior"]["bboxExpansion"], 0.2);
    assert_eq!(
        oracle["corpus"]["privacy"],
        "synthetic pixels only; no people, biometrics, camera captures, or personal data"
    );
    assert_eq!(
        oracle["corpus"]["license"],
        "MIT; generated entirely by repository code"
    );
    assert_eq!(oracle["corpus"]["licenseFile"], "LICENSE");
    assert_eq!(
        oracle["tolerances"]["vision"]["absolute"],
        VISION_ABSOLUTE_TOLERANCE
    );
    assert_eq!(
        oracle["tolerances"]["vision"]["relative"],
        VISION_RELATIVE_TOLERANCE
    );
    assert_eq!(
        oracle["ortWeb"]["artifact"]["sha256"],
        "3260fcdb33b4fc4ec33e89caf392e13625823e01049d3bf32c38464f9dbfe14c"
    );
    for case in oracle["cases"].as_array().unwrap() {
        let image = frame(case);
        let det = compatibility_preprocess_rtmdet(&image).unwrap();
        let expected = &case["preprocessing"]["rtmdet"];
        close(det.scale, expected["scale"].as_f64().unwrap());
        assert_eq!(
            det.scaled_width,
            expected["scaledW"].as_u64().unwrap() as usize
        );
        assert_eq!(
            det.scaled_height,
            expected["scaledH"].as_u64().unwrap() as usize
        );
        assert_eq!(det.pad_width as u64, expected["padW"].as_u64().unwrap());
        assert_eq!(det.pad_height as u64, expected["padH"].as_u64().unwrap());
        assert_eq!(f32_sha(&det.tensor), expected["tensor"]["sha256"]);
    }
}

#[test]
fn synthetic_postprocessing_and_presence_boundaries_are_executable() {
    let oracle = fixture();
    let expected_ids = oracle["postprocessingCases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut consumed_ids = BTreeSet::new();
    for case in oracle["postprocessingCases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        match id {
            "detector-score-exact-threshold" | "bbox-largest-area-tie" => {
                let dets = case["dets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                let labels = case["labels"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_i64().unwrap())
                    .collect::<Vec<_>>();
                let actual = compatibility_select_person_bbox(&dets, &labels, 100, 100).unwrap();
                assert_eq!(
                    actual.is_some(),
                    case["selection"]["personFound"].as_bool().unwrap(),
                    "{id}"
                );
                if let Some(actual) = actual {
                    let expected = &case["selection"]["selected"];
                    for (actual, expected) in [
                        (actual.x1, &expected["x1"]),
                        (actual.y1, &expected["y1"]),
                        (actual.x2, &expected["x2"]),
                        (actual.y2, &expected["y2"]),
                        (actual.score, &expected["score"]),
                        (actual.width, &expected["width"]),
                        (actual.height, &expected["height"]),
                    ] {
                        close(actual, expected.as_f64().unwrap());
                    }
                }
            }
            "simcc-first-index-tie" => {
                // RTMPose SimCC decoding is retired; keypoints now derive from NLF
                // coords2d (covered by the worker's `assemble_nlf_keypoints` unit
                // test). The fixture row is retained but no longer exercised here.
            }
            _ => panic!("unconsumed synthetic postprocessing case {id}"),
        }
        consumed_ids.insert(id.to_owned());
    }
    assert_eq!(consumed_ids, expected_ids);

    let expected_ids = oracle["cascadeCases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut consumed_ids = BTreeSet::new();
    for case in oracle["cascadeCases"].as_array().unwrap() {
        assert_eq!(
            should_run_posture_for_presence(case["presentProbability"].as_f64().unwrap()),
            case["postureRuns"].as_bool().unwrap(),
            "{}",
            case["id"]
        );
        consumed_ids.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(consumed_ids, expected_ids);
}

#[test]
fn every_synthetic_frame_runs_the_native_detector_cascade() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dll = root.join("resources/onnxruntime/windows-x86_64/onnxruntime.dll");
    assert!(ort::init_from(&dll).unwrap().commit());
    let detector_path = root.join("resources/models/rtmdet-nano.onnx");
    let oracle = fixture();
    let expected_ids = oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut consumed_ids = BTreeSet::new();
    for case in oracle["cases"].as_array().unwrap() {
        // RTMDet runs on the CPU kernels, so this parity path needs no GPU/DirectML;
        // the NLF-L pose model (keypoints/depth) is validated by the worker unit and
        // integration tests instead.
        let output =
            compatibility_run_detector(detector_path.to_str().unwrap(), &frame(case)).unwrap();
        consumed_ids.insert(case["id"].as_str().unwrap().to_owned());
        assert_eq!(
            output.bbox.is_some(),
            case["rtmdet"]["selection"]["personFound"]
                .as_bool()
                .unwrap(),
            "{}",
            case["id"],
        );
        compare_all(&output.rtmdet_pooled, &case["rtmdet"]["pooled"]);

        let Some(actual_bbox) = output.bbox.as_ref() else {
            continue;
        };
        let expected_bbox = &case["pipeline"]["bbox"];
        if matches!(
            case["id"].as_str().unwrap(),
            "edge-clipped-silhouette" | "boundary-crop-silhouette"
        ) {
            assert!(
                actual_bbox.expanded.x1 == 0.0 || actual_bbox.expanded.y1 == 0.0,
                "adversarial crop must clip an expanded boundary"
            );
            assert!(case["pipeline"]["crop"]["width"].as_u64().unwrap() > 0);
            assert!(case["pipeline"]["crop"]["height"].as_u64().unwrap() > 0);
        }
        for (actual, expected) in [
            (actual_bbox.original.x1, &expected_bbox["original"]["x1"]),
            (actual_bbox.original.y1, &expected_bbox["original"]["y1"]),
            (actual_bbox.original.x2, &expected_bbox["original"]["x2"]),
            (actual_bbox.original.y2, &expected_bbox["original"]["y2"]),
            (
                actual_bbox.original.score,
                &expected_bbox["original"]["score"],
            ),
            (
                actual_bbox.original.width,
                &expected_bbox["original"]["width"],
            ),
            (
                actual_bbox.original.height,
                &expected_bbox["original"]["height"],
            ),
            (actual_bbox.expanded.x1, &expected_bbox["expanded"]["x1"]),
            (actual_bbox.expanded.y1, &expected_bbox["expanded"]["y1"]),
            (actual_bbox.expanded.x2, &expected_bbox["expanded"]["x2"]),
            (actual_bbox.expanded.y2, &expected_bbox["expanded"]["y2"]),
            (
                actual_bbox.expanded.score,
                &expected_bbox["expanded"]["score"],
            ),
            (
                actual_bbox.expanded.width,
                &expected_bbox["expanded"]["width"],
            ),
            (
                actual_bbox.expanded.height,
                &expected_bbox["expanded"]["height"],
            ),
        ] {
            close(actual, expected.as_f64().unwrap());
        }
    }
    assert_eq!(
        consumed_ids, expected_ids,
        "every vision case must be consumed"
    );
}
