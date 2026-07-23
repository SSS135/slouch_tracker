const PUBLIC_BRIDGE: &str = include_str!("../../src/generated/bindings.ts");
const GENERATED: &str = include_str!("../../src/generated/bindings.generated.ts");
const REGISTRATION: &str = include_str!("../src/lib.rs");
const GENERATOR: &str = include_str!("../src/bindings.rs");

const COMMANDS: [(&str, &str); 39] = [
    ("app_status", "appStatus"),
    ("initialize_inference", "initializeInference"),
    ("infer_frame", "inferFrame"),
    ("train_models", "trainModels"),
    ("get_training_status", "getTrainingStatus"),
    ("cancel_training", "cancelTraining"),
    ("get_dataset_page", "getDatasetPage"),
    ("get_thumbnail", "getThumbnail"),
    ("get_dataset_stats", "getDatasetStats"),
    ("get_needs_retraining", "getNeedsRetraining"),
    ("get_reservoir_metadata", "getReservoirMetadata"),
    ("get_camera_settings", "getCameraSettings"),
    ("save_camera_settings", "saveCameraSettings"),
    ("reset_camera_settings", "resetCameraSettings"),
    ("get_ui_settings", "getUiSettings"),
    ("save_ui_settings", "saveUiSettings"),
    ("reset_ui_settings", "resetUiSettings"),
    ("get_training_settings", "getTrainingSettings"),
    ("reset_training_settings", "resetTrainingSettings"),
    ("save_training_settings", "saveTrainingSettings"),
    ("update_frame_label", "updateFrameLabel"),
    ("save_capture", "saveCapture"),
    ("cleanup_unused_frames", "cleanupUnusedFrames"),
    ("delete_frame", "deleteFrame"),
    ("undo_last_dataset_change", "undoLastDatasetChange"),
    ("get_undo_status", "getUndoStatus"),
    ("reset_dataset", "resetDataset"),
    ("reset_all_data", "resetAllData"),
    ("get_classifier_registry", "getClassifierRegistry"),
    ("get_feature_registry", "getFeatureRegistry"),
    ("get_active_model_metadata", "getActiveModelMetadata"),
    ("export_dataset", "exportDataset"),
    ("import_dataset", "importDataset"),
    ("get_shortcut_status", "getShortcutStatus"),
    ("get_autostart_enabled", "getAutostartEnabled"),
    ("set_autostart_enabled", "setAutostartEnabled"),
    ("start_camera", "startCamera"),
    ("stop_camera", "stopCamera"),
    ("list_cameras", "listCameras"),
];

#[test]
fn generated_bindings_are_fresh() {
    for (command, wrapper) in COMMANDS {
        assert!(
            REGISTRATION.contains(&format!("api::{command},")),
            "missing registration for {command}"
        );
        if command == "infer_frame" {
            // infer_frame is a raw-byte command retained ONLY as a native e2e
            // test harness: the e2e/native specs drive raw inference through it.
            // The frontend no longer calls it, so it is registered in lib.rs but
            // retired from the public bridge (bindings.ts) and never
            // specta-generated. Assert both absences to keep it off the public
            // surface.
            assert!(
                !GENERATOR.contains(&format!("crate::api::{command},")),
                "raw byte command {command} must not be generated as JSON"
            );
            assert!(
                !PUBLIC_BRIDGE.contains(wrapper),
                "retired test-harness command {command} must not appear in the public bridge"
            );
        } else if matches!(command, "get_thumbnail" | "save_capture") {
            assert!(
                PUBLIC_BRIDGE.contains(&format!("function {wrapper}(")),
                "raw byte command {command} is missing from the public bridge"
            );
            assert!(
                !GENERATOR.contains(&format!("crate::api::{command},")),
                "raw byte command {command} must not be generated as JSON"
            );
        } else {
            assert!(
                GENERATOR.contains(&format!("crate::api::{command},")),
                "JSON command {command} is missing from tauri-specta generation"
            );
            assert!(
                GENERATED.contains(&format!("\t{wrapper}:")),
                "generated JSON wrapper {wrapper} is missing"
            );
        }
    }

    assert!(PUBLIC_BRIDGE.contains("export { commands } from './bindings.generated';"));
    assert!(PUBLIC_BRIDGE.contains("export type * from './bindings.generated';"));
    assert!(!PUBLIC_BRIDGE.contains("NativeCommandMap"));
    assert!(!PUBLIC_BRIDGE.contains("callNative"));
    assert_eq!(
        PUBLIC_BRIDGE.matches("invoke<").count(),
        1,
        "only get_thumbnail has a typed handwritten invoke"
    );
    assert_eq!(
        PUBLIC_BRIDGE.matches("await invoke(").count(),
        1,
        "only save_capture has an untyped handwritten invoke"
    );
    assert!(PUBLIC_BRIDGE.contains("dataset-changed"));
    assert!(PUBLIC_BRIDGE.contains("shortcut-capture"));
    assert!(GENERATED.contains("getUndoStatus: () => typedError<UndoStatus"));
    assert!(GENERATED
        .contains("undoStatusChanged: makeEvent<UndoStatusChangedEvent>(\"undo-status-changed\")"));
    assert!(GENERATED.contains(
        "nativeStateChanged: makeEvent<NativeStateChangedEvent_Deserialize>(\"native-state-changed\")"
    ));
    assert!(GENERATED.contains("resetDataset: () => typedError<NativeStateSnapshot_Serialize"));
    assert!(GENERATED.contains("state: NativeStateSnapshot_Serialize"));
    assert!(GENERATED.contains("sequence: number"));
    assert!(GENERATED
        .contains("export type TrainingStage = \"processing\" | \"evaluating\" | \"deploying\""));

    for retired in [
        "load_posture_model",
        "load_presence_model",
        "get_posture_model",
        "get_presence_model",
        "save_frame",
        "export_dataset_raw",
        "import_dataset_raw",
    ] {
        assert!(!REGISTRATION.contains(&format!("api::{retired},")));
        assert!(!PUBLIC_BRIDGE.contains(retired));
    }

    let generated_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../src/generated/bindings.generated.ts");
    let temporary = std::env::temp_dir().join(format!(
        "slouch-bindings-{}-{}.ts",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    app_lib::export_bindings(&temporary).expect("export bindings with tauri-specta");
    let expected = std::fs::read_to_string(&generated_path)
        .expect("checked-in tauri-specta bindings are missing; run the export_bindings binary");
    let actual = std::fs::read_to_string(&temporary).expect("read generated bindings");
    let _ = std::fs::remove_file(&temporary);
    assert!(actual.contains("doCv"));
    assert!(actual.contains("onEvent"));
    assert_eq!(actual, expected, "tauri-specta bindings are stale");
}
