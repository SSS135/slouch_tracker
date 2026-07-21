//! Native ownership replacement for the browser-only transfer-list oracle
//! `src/workers/__tests__/inference-worker-buffer-transfer.test.ts`
//! (frozen SHA-256 `1f6252540bf6bd162bea758a02134bff12d1270113b7bc4faf52329fa8160429`).
//!
//! JavaScript `ArrayBuffer` detachment is intentionally unsupported by PORTING.md;
//! this drives the production worker and proves the typed native response owns
//! every resulting feature vector without a JSON/structured-clone replica.

use std::collections::VecDeque;

use slouch_domain::ported::messages::schemas::{
    InferenceWorkerMessage, InitializePayload, ProcessPayload,
};
use slouch_domain::FeatureId;
use slouch_vision::ported::inference_worker::{InferenceWorker, WorkerResponse};

use super::support::{
    detector_outputs, image, pose_outputs, CreateOutcome, TestFactory, TestLogger, TestRuntime,
};

#[test]
fn production_result_owns_all_native_feature_buffers_after_worker_drop() {
    let (runtime, _) = TestRuntime::new([
        CreateOutcome::Session(VecDeque::from([Ok(detector_outputs(
            vec![40.0, 40.0, 280.0, 280.0, 0.9],
            vec![0],
            0.25,
        ))])),
        CreateOutcome::Session(VecDeque::from([Ok(pose_outputs())])),
    ]);
    let mut worker =
        InferenceWorker::with_runtime(TestFactory::default(), TestLogger::default(), runtime);
    worker.handle_message(InferenceWorkerMessage::Initialize {
        payload: InitializePayload {
            rtmdet_path: "det".into(),
            rtmw3d_path: "pose".into(),
        },
    });
    let response = worker.handle_message(InferenceWorkerMessage::Process {
        payload: ProcessPayload {
            image_data: image(4, 4),
            request_id: 1,
        },
    });
    drop(worker);

    let [WorkerResponse::Result { result, .. }] = &response[..] else {
        panic!("expected result")
    };
    assert_eq!(result.features.len(), 7);
    let dimensions = result
        .features
        .iter()
        .map(|(kind, values)| (*kind, values.len()))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(dimensions[&FeatureId::BackboneFeatures], 768);
    assert_eq!(dimensions[&FeatureId::BackboneFeaturesMax], 768);
    assert_eq!(dimensions[&FeatureId::BackboneFeaturesStd], 768);
    assert_eq!(dimensions[&FeatureId::GauFeatures], 256);
    assert_eq!(dimensions[&FeatureId::GauFeaturesMax], 256);
    assert_eq!(dimensions[&FeatureId::GauFeaturesStd], 256);
    assert_eq!(dimensions[&FeatureId::RtmDetExtracted], 384);
}
