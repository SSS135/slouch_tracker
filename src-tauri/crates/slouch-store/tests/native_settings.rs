use std::time::{SystemTime, UNIX_EPOCH};

use slouch_domain::{CameraSettings, UiSettings};
use slouch_store::ported::storage::{DatasetStorage, StorageError};

#[test]
fn camera_and_ui_settings_use_rust_defaults_and_persist_independently() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    assert_eq!(
        storage.get_camera_settings().expect("camera defaults"),
        CameraSettings::default()
    );
    assert_eq!(
        storage.get_ui_settings().expect("UI defaults"),
        UiSettings::default()
    );

    let camera = CameraSettings {
        camera_width: 1280,
        privacy_mode: false,
        show_detection_overlay: true,
        ..CameraSettings::default()
    };
    storage.save_camera_settings(&camera).expect("save camera");

    let ui = UiSettings {
        alert_volume: 0.75,
        minimize_to_tray_on_close: false,
        ..UiSettings::default()
    };
    storage.save_ui_settings(&ui).expect("save UI");

    assert_eq!(storage.get_camera_settings().expect("load camera"), camera);
    assert_eq!(storage.get_ui_settings().expect("load UI"), ui);

    assert_eq!(
        storage.reset_camera_settings().expect("reset camera"),
        CameraSettings::default()
    );
    assert_eq!(storage.get_ui_settings().expect("UI remains"), ui);
    assert_eq!(
        storage.reset_ui_settings().expect("reset UI"),
        UiSettings::default()
    );
}

#[test]
fn camera_and_ui_settings_round_trip_through_the_native_archive() {
    let source = DatasetStorage::open_in_memory().expect("source");
    let camera = CameraSettings {
        camera_width: 1280,
        ..CameraSettings::default()
    };
    let ui = UiSettings {
        alert_delay_seconds: 12.0,
        start_hidden_on_login: false,
        ..UiSettings::default()
    };
    source.save_camera_settings(&camera).expect("camera");
    source.save_ui_settings(&ui).expect("UI");

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("slouch-settings-{nonce}.slouchpack"));
    source.export_archive(&archive, "test").expect("export");

    let target = DatasetStorage::open_in_memory().expect("target");
    target.import_archive(&archive).expect("import");
    assert_eq!(target.get_camera_settings().expect("camera"), camera);
    assert_eq!(target.get_ui_settings().expect("UI"), ui);
    std::fs::remove_file(archive).expect("remove archive");
}

#[test]
fn settings_reject_invalid_values_without_overwriting_saved_state() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    let camera = CameraSettings::default();
    storage
        .save_camera_settings(&camera)
        .expect("save valid camera");

    let mut invalid_camera = camera.clone();
    invalid_camera.gaussian_blur_kernel = 4;
    assert!(matches!(
        storage.save_camera_settings(&invalid_camera),
        Err(StorageError::Validation(_))
    ));
    assert_eq!(
        storage.get_camera_settings().expect("unchanged camera"),
        camera
    );

    let invalid_ui = UiSettings {
        alert_delay_seconds: f64::INFINITY,
        ..UiSettings::default()
    };
    assert!(matches!(
        storage.save_ui_settings(&invalid_ui),
        Err(StorageError::Validation(_))
    ));
}

#[test]
fn tray_notice_flag_defaults_false_and_round_trips_independently() {
    let storage = DatasetStorage::open_in_memory().expect("storage");
    assert!(!storage.get_tray_notice_shown().expect("default flag"));

    storage.set_tray_notice_shown(true).expect("set flag");
    assert!(storage.get_tray_notice_shown().expect("flag persists"));

    // A UI-settings reset must not re-arm the notice.
    storage.reset_ui_settings().expect("reset UI");
    assert!(storage
        .get_tray_notice_shown()
        .expect("flag survives UI reset"));

    // Reset All Data clears every settings key, including this flag.
    storage.reset_all().expect("reset all");
    assert!(!storage
        .get_tray_notice_shown()
        .expect("flag cleared by reset all"));
}
