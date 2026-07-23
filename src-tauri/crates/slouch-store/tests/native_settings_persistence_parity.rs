use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use slouch_domain::{CameraSettings, UiSettings};
use slouch_store::ported::storage::{DatasetStorage, StorageError};

fn database_path(test: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("slouch-settings-{test}-{nonce}.sqlite3"))
}

fn remove_database(path: &std::path::Path) {
    for candidate in [
        path.to_path_buf(),
        std::path::PathBuf::from(format!("{}-wal", path.display())),
        std::path::PathBuf::from(format!("{}-shm", path.display())),
    ] {
        if candidate.exists() {
            std::fs::remove_file(candidate).expect("remove SQLite test file");
        }
    }
}

#[test]
fn file_backed_sqlite_settings_survive_restart_and_reset_independently() {
    let path = database_path("restart");
    let camera = CameraSettings {
        camera_width: 1280,
        camera_height: 720,
        privacy_mode: false,
        smoothing_frames: 4,
        ..CameraSettings::default()
    };
    let ui = UiSettings {
        alert_volume: 0.75,
        alert_delay_seconds: 12.0,
        minimize_to_tray_on_close: false,
        start_hidden_on_login: false,
    };

    {
        let storage = DatasetStorage::open(&path).expect("create native SQLite database");
        assert_eq!(
            storage.get_camera_settings().unwrap(),
            CameraSettings::default()
        );
        assert_eq!(storage.get_ui_settings().unwrap(), UiSettings::default());
        storage
            .save_camera_settings(&camera)
            .expect("save camera settings");
        storage.save_ui_settings(&ui).expect("save UI settings");
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen native SQLite database");
        assert_eq!(storage.get_camera_settings().unwrap(), camera);
        assert_eq!(storage.get_ui_settings().unwrap(), ui);
        assert_eq!(
            storage.reset_camera_settings().unwrap(),
            CameraSettings::default()
        );
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen after reset");
        assert_eq!(
            storage.get_camera_settings().unwrap(),
            CameraSettings::default()
        );
        assert_eq!(storage.get_ui_settings().unwrap(), ui);
        assert_eq!(storage.reset_ui_settings().unwrap(), UiSettings::default());
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen after UI reset");
        assert_eq!(storage.get_ui_settings().unwrap(), UiSettings::default());
    }
    remove_database(&path);
}

#[test]
fn rejected_native_settings_never_replace_the_last_sqlite_value() {
    let path = database_path("validation");
    let saved = CameraSettings {
        camera_width: 640,
        ..CameraSettings::default()
    };
    {
        let storage = DatasetStorage::open(&path).expect("native SQLite database");
        storage
            .save_camera_settings(&saved)
            .expect("save valid settings");
        let invalid = CameraSettings {
            gaussian_blur_kernel: 4,
            ..saved.clone()
        };
        assert!(matches!(
            storage.save_camera_settings(&invalid),
            Err(StorageError::Validation(_))
        ));
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen after rejected write");
        assert_eq!(storage.get_camera_settings().unwrap(), saved);
    }
    remove_database(&path);
}

#[test]
fn future_setting_schema_versions_are_rejected_without_changing_payloads() {
    let path = database_path("forward-schema");
    let camera = CameraSettings {
        camera_width: 1280,
        ..CameraSettings::default()
    };
    let ui = UiSettings {
        alert_volume: 0.8,
        alert_delay_seconds: 9.0,
        ..UiSettings::default()
    };
    {
        let storage = DatasetStorage::open(&path).expect("native SQLite database");
        storage.save_camera_settings(&camera).unwrap();
        storage.save_ui_settings(&ui).unwrap();
    }
    {
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE settings SET schema_version = 2 WHERE key IN (?, ?)",
                params!["camera:settings", "ui:settings"],
            )
            .unwrap();
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen future-version database");
        assert!(matches!(
            storage.get_camera_settings(),
            Err(StorageError::InvalidData(_))
        ));
        assert!(matches!(
            storage.get_ui_settings(),
            Err(StorageError::InvalidData(_))
        ));
    }
    {
        let connection = Connection::open(&path).unwrap();
        connection
            .execute("UPDATE settings SET schema_version = 1", [])
            .unwrap();
    }
    {
        let storage = DatasetStorage::open(&path).expect("reopen restored-version database");
        assert_eq!(storage.get_camera_settings().unwrap(), camera);
        assert_eq!(storage.get_ui_settings().unwrap(), ui);
    }
    remove_database(&path);
}
