use std::fmt::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};
use slouch_vision::ported::inference_worker::WorkerResponse;

fn sha256(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[test]
fn package_shaped_startup_resolves_locked_runtime_and_loads_active_models() {
    let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock: Value = serde_json::from_slice(
        &std::fs::read(source_root.join("resource-lock.json")).expect("resource lock"),
    )
    .expect("valid resource lock");

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let fixture_root = std::env::temp_dir().join(format!("slouch-package-resources-{nonce}"));
    let data_dir = std::env::temp_dir().join(format!("slouch-native-runtime-{nonce}"));

    let mut checked = 0;
    for resource in lock["resources"].as_array().expect("resource inventory") {
        let Some(relative) = resource["packagedPath"].as_str() else {
            continue;
        };
        let source = source_root.join(relative);
        let bytes = std::fs::read(&source).unwrap_or_else(|error| {
            panic!("locked native resource {relative} is unavailable: {error}")
        });
        assert_eq!(
            bytes.len() as u64,
            resource["bytes"].as_u64().expect("locked byte length"),
            "{relative}"
        );
        assert_eq!(
            sha256(&bytes),
            resource["sha256"].as_str().expect("locked digest"),
            "{relative}"
        );

        let packaged = fixture_root.join(relative);
        std::fs::create_dir_all(packaged.parent().expect("packaged resource parent"))
            .expect("create package-shaped resource directory");
        std::fs::write(&packaged, bytes).expect("copy locked package resource");
        checked += 1;
    }
    assert_eq!(
        checked, 7,
        "runtime DLLs, notices, and the packaged detector model must be locked"
    );

    let packaged_runtime =
        fixture_root.join("resources/onnxruntime/windows-x86_64/onnxruntime.dll");
    let packaged_detector = fixture_root.join("resources/models/rtmdet-nano.onnx");
    assert!(packaged_runtime.is_file());
    assert!(packaged_detector.is_file());

    // NLF-L is no longer bundled into the installer: it is downloaded at first run
    // into the app data directory, where find_resource's data-dir search locates it.
    // Assert it is absent from the package, stage the model at the data-dir download
    // path from the on-disk (git LFS) copy, and prove a package-shaped startup loads
    // it for native inference from that runtime location rather than the bundle.
    assert!(
        !fixture_root
            .join("resources/models/nlf_l_crop_fp16.onnx")
            .is_file(),
        "the NLF pose model must not be packaged into the installer bundle",
    );
    let nlf_bytes = std::fs::read(source_root.join("resources/models/nlf_l_crop_fp16.onnx"))
        .expect("on-disk NLF model (git LFS) backing the data-dir download fixture");
    let downloaded_nlf = data_dir.join("models").join("nlf_l_crop_fp16.onnx");
    std::fs::create_dir_all(downloaded_nlf.parent().expect("data-dir model parent"))
        .expect("create the runtime download directory");
    std::fs::write(&downloaded_nlf, &nlf_bytes).expect("stage the runtime-downloaded NLF model");

    let state = crate::api::initialize_state(data_dir.clone(), fixture_root.clone())
        .expect("startup must resolve the package-shaped runtime and SQLite state");
    let responses = state
        .inference
        .send(crate::actors::initialize_message(
            packaged_detector,
            downloaded_nlf.clone(),
        ))
        .expect("active model initialization from the packaged detector and downloaded pose model");
    assert!(responses.iter().any(|response| matches!(
        response,
        WorkerResponse::Initialized { provider } if provider == "native"
    )));

    state.shutdown();
    drop(state);
    std::fs::remove_dir_all(data_dir).expect("remove temporary native data");
    if let Err(error) = std::fs::remove_dir_all(&fixture_root) {
        #[cfg(not(windows))]
        panic!("package cleanup failed off Windows: {error}");
        #[cfg(windows)]
        {
            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(
                packaged_runtime.is_file(),
                "only the process-loaded ONNX Runtime DLL may prevent Windows cleanup",
            );
        }
    }
}
