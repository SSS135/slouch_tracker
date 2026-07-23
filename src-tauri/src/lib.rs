mod actors;
mod api;
mod bindings;
mod errors;
mod model_download;
#[cfg(test)]
mod native_runtime_resources_parity;
mod power;
mod tray;

pub fn export_bindings(path: impl AsRef<std::path::Path>) -> Result<(), String> {
    bindings::export(path)
}

use std::path::PathBuf;

use tauri::{Emitter, Manager, RunEvent, WindowEvent};
use tauri_plugin_autostart::ManagerExt;
use tauri_plugin_global_shortcut::{Builder as GlobalShortcutBuilder, ShortcutState};
use tauri_plugin_log::{Target, TargetKind};
use tauri_plugin_notification::NotificationExt;

use crate::actors::CameraMode;

#[derive(Debug, Clone, serde::Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
struct ShortcutCaptureEvent {
    label: String,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shortcut_builder = match GlobalShortcutBuilder::new().with_shortcuts([
        "CTRL+SUPER+G",
        "CTRL+SUPER+B",
        "CTRL+SUPER+A",
    ]) {
        Ok(builder) => builder,
        Err(error) => {
            eprintln!("failed to configure global capture shortcuts: {error}");
            return;
        }
    };
    let shortcut_plugin = shortcut_builder
        .with_handler(|app, shortcut, event| {
            if event.state != ShortcutState::Pressed {
                return;
            }
            let value = shortcut.to_string().to_ascii_lowercase();
            let label = if value.ends_with("+g") {
                "good"
            } else if value.ends_with("+b") {
                "bad"
            } else if value.ends_with("+a") {
                "away"
            } else {
                return;
            };
            let _ = app.emit(
                "shortcut-capture",
                ShortcutCaptureEvent {
                    label: label.to_owned(),
                },
            );
        })
        .build();

    let builder = tauri::Builder::default();

    // Single-instance MUST be the first registered plugin (its hard requirement).
    // With a tray-resident app, a second launch focuses/shows the existing window
    // via the shared restore helper instead of spawning a duplicate.
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
        tray::restore_window(app);
    }));

    let builder = builder
        .plugin(shortcut_plugin)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        // Per-user Run-key autostart. The commands drive it via the plugin's Rust
        // ManagerExt (no guest JS plugin / capabilities). LaunchAgent is the macOS
        // launcher, irrelevant on Windows. Launch arg `--autostart` marks a login
        // launch so setup can start the window hidden in the tray.
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--autostart"]),
        ));

    // Embedded WebDriver server for packaged-native E2E (devbuild only). Must be
    // registered on the builder (not in setup) so its on_webview_ready hook runs
    // for the main window created from the config.
    #[cfg(feature = "devbuild")]
    let builder = builder.plugin(tauri_plugin_wdio_webdriver::init());

    let builder = builder
        // Per-frame PULL preview: the webview GETs `slouchcam://…/frame` each rAF
        // and gets the freshest MJPEG frame (foreground) or 204 (no frame /
        // background). A path containing "processed" serves the detector-input
        // frame instead (post-preprocessing JPEG). Each /processed pull also stamps
        // live demand so the capture loop refreshes that frame at capture rate while
        // it is watched, falling back to the ~1fps detection refresh when idle. A
        // path containing "inferred" serves the same detector-input JPEG WITHOUT
        // stamping demand, so it stays the dispatcher-written inferred frame at
        // detection cadence (used by the diagnostic detection overlay so the shown
        // frame matches the result's keypoints/bbox); any other path serves the raw
        // preview frame.
        // The webview origin (dev: http://127.0.0.1:5174; prod: tauri://localhost)
        // differs from this custom-scheme host, so the frame must be CORS-allowed;
        // without it privacy-mode grid sampling and thumbnail reads fail as opaque
        // cross-origin fetches. The stream is a local, in-process preview only.
        .register_asynchronous_uri_scheme_protocol("slouchcam", |ctx, request, responder| {
            let state = ctx.app_handle().state::<api::AppState>();
            let path = request.uri().path();
            // Processed/inferred frames carry a monotonic capture sequence; the raw
            // feed has none. `frame_seq` is echoed as `x-slouch-frame-seq` so the
            // webview can order commits against the Rust-authoritative ordering.
            let (bytes, frame_seq) = if path.contains("debug-tiles") {
                // Preprocessing debug heatmap. Stamps demand exactly like /processed
                // because the debug frame is produced inside `run_preview_processing`,
                // which only runs while processed-view demand is active — pulling only
                // /debug-tiles must keep that foreground pipeline alive.
                state.camera.note_processed_request();
                match state.camera.debug_frame_snapshot() {
                    Some((seq, bytes)) => (Some(bytes), Some(seq)),
                    None => (None, None),
                }
            } else if path.contains("inferred") {
                // Detection-overlay diagnostic: the exact preprocessed frame the
                // detector last ran on. Deliberately does NOT stamp demand, so
                // `processed_frame` stays the dispatcher-written inferred frame
                // (detection cadence) matching the keypoints/bbox in the same
                // InferenceUiResult — no 30fps capture-rate override.
                match state.camera.processed_frame_snapshot() {
                    Some((seq, bytes)) => (Some(bytes), Some(seq)),
                    None => (None, None),
                }
            } else if path.contains("processed") {
                // Stamp live demand so the capture loop keeps the processed frame
                // fresh at capture rate while the view is being pulled.
                state.camera.note_processed_request();
                match state.camera.processed_frame_snapshot() {
                    Some((seq, bytes)) => (Some(bytes), Some(seq)),
                    None => (None, None),
                }
            } else {
                (state.camera.latest_frame_bytes(), None)
            };
            let response = match bytes {
                Some(bytes) => {
                    let mut builder = tauri::http::Response::builder()
                        .status(200)
                        .header(tauri::http::header::CONTENT_TYPE, "image/jpeg")
                        .header(tauri::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");
                    if let Some(seq) = frame_seq {
                        // Expose the sequence to cross-origin `fetch()` readers too, so
                        // any frontend commit guard can read it (the <img> fast path
                        // stays header-free and monotonic-by-construction).
                        builder = builder
                            .header("x-slouch-frame-seq", seq.to_string())
                            .header(
                                tauri::http::header::ACCESS_CONTROL_EXPOSE_HEADERS,
                                "x-slouch-frame-seq",
                            );
                    }
                    builder.body(bytes)
                }
                None => tauri::http::Response::builder()
                    .status(204)
                    .header(tauri::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                    .body(Vec::new()),
            };
            match response {
                Ok(response) => responder.respond(response),
                Err(error) => {
                    log::error!(target: "camera", "camera preview response build failed: {error}")
                }
            }
        })
        // Focused → foreground (smooth ~30fps preview); unfocused → background (camera
        // throttled, but the preview cell still updates at the ~1fps detection rate, so
        // a visible-but-unfocused window keeps a live feed). Detection runs either way;
        // the frontend stops fetching only when the window is minimized/hidden.
        .on_window_event(|window, event| match event {
            WindowEvent::Focused(focused) => {
                let mode = if *focused {
                    CameraMode::Foreground
                } else {
                    CameraMode::Background
                };
                let _ = window.state::<api::AppState>().camera.set_mode(mode);
                // EcoQoS + idle priority when backgrounded, cleared when focused.
                // No startup call: the process defaults to normal priority /
                // no throttling, and the window's first Focused event applies
                // the correct state, so relying on it is both simplest and
                // correct whether the app launches focused or unfocused.
                power::set_efficiency_mode(!*focused);
            }
            // Closing hides the window to the tray (detection keeps running) when
            // the setting is on, showing a one-time toast the first time. With the
            // setting off the close proceeds to a full exit, exactly as before.
            WindowEvent::CloseRequested { api, .. } => {
                let state = window.state::<api::AppState>();
                let minimize = state
                    .storage
                    .get_ui_settings()
                    .map(|settings| settings.minimize_to_tray_on_close)
                    .unwrap_or(true);
                if minimize {
                    let _ = window.hide();
                    api.prevent_close();
                    if !state.storage.get_tray_notice_shown().unwrap_or(false) {
                        let _ = window
                            .app_handle()
                            .notification()
                            .builder()
                            .title("Slouch Tracker is still running")
                            .body("It minimized to the tray and keeps tracking your posture. Double-click the tray icon to open it, or right-click to pause or exit.")
                            .show();
                        let _ = state.storage.set_tray_notice_shown(true);
                    }
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            api::app_status,
            api::initialize_inference,
            api::infer_frame,
            api::train_models,
            api::get_training_status,
            api::cancel_training,
            api::get_dataset_page,
            api::get_thumbnail,
            api::get_dataset_stats,
            api::get_needs_retraining,
            api::get_reservoir_metadata,
            api::get_camera_settings,
            api::save_camera_settings,
            api::reset_camera_settings,
            api::get_ui_settings,
            api::save_ui_settings,
            api::reset_ui_settings,
            api::get_training_settings,
            api::reset_training_settings,
            api::save_training_settings,
            api::update_frame_label,
            api::save_capture,
            api::cleanup_unused_frames,
            api::delete_frame,
            api::undo_last_dataset_change,
            api::get_undo_status,
            api::reset_dataset,
            api::reset_all_data,
            api::get_classifier_registry,
            api::get_feature_registry,
            api::get_active_model_metadata,
            api::export_dataset,
            api::import_dataset,
            api::get_shortcut_status,
            api::get_autostart_enabled,
            api::set_autostart_enabled,
            api::start_camera,
            api::stop_camera,
            api::list_cameras,
            api::get_pose_model_status,
            api::ensure_pose_model,
        ])
        .setup(|app| {
            #[cfg(any(debug_assertions, feature = "devbuild"))]
            let log_level = log::LevelFilter::Debug;
            #[cfg(not(any(debug_assertions, feature = "devbuild")))]
            let log_level = log::LevelFilter::Info;

            // SLOUCH_APP_DATA_DIR redirects all persisted state (SQLite, models)
            // into an isolated directory; used by packaged inspection and E2E.
            let data_dir = match std::env::var_os("SLOUCH_APP_DATA_DIR") {
                Some(dir) if !dir.is_empty() => PathBuf::from(dir),
                _ => app.path().app_data_dir()?,
            };
            let resource_dir = app
                .path()
                .resource_dir()
                .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
            let state =
                api::initialize_state(data_dir, resource_dir).map_err(std::io::Error::other)?;
            app.manage(state);

            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log_level)
                    .targets([
                        Target::new(TargetKind::Stdout),
                        Target::new(TargetKind::Webview),
                        Target::new(TargetKind::LogDir { file_name: None }),
                    ])
                    .build(),
            )?;

            // WDIO execute/log plugin (devbuild only). Registered after
            // tauri-plugin-log so it never wins the global-logger race; its own
            // fallback logger install fails gracefully in that case.
            #[cfg(feature = "devbuild")]
            app.handle().plugin(tauri_plugin_wdio::init())?;

            // Defer inference initialization on a first run where the NLF pose model
            // has not been downloaded yet: a missing model must not crash startup
            // (the frontend downloads it via ensure_pose_model, then re-calls
            // initialize_inference). When the model IS present a genuine init failure
            // (for example no DirectX 12 GPU) still aborts startup as before.
            {
                let app_state = app.state::<api::AppState>();
                if app_state.pose_model_present() {
                    api::initialize_inference(app_state).map_err(std::io::Error::other)?;
                } else {
                    log::info!(
                        target: "inference",
                        "NLF pose model not present; deferring inference initialization until first-run download completes"
                    );
                }
            }

            tray::build_tray(app.handle())?;

            // Refresh the per-user Run-key entry so it carries the current exe path
            // plus the `--autostart` arg (older entries predate the arg). Idempotent.
            // Windows StartupApproved\Run (the Task Manager toggle) stays
            // authoritative: is_enabled() already folds it in, so a user who
            // disabled startup there reports false here and we skip — enable() never
            // overrides that kill switch.
            let autolaunch = app.autolaunch();
            if autolaunch.is_enabled().unwrap_or(false) {
                let _ = autolaunch.enable();
            }

            // The window is configured `visible: false` (this also removes the white
            // startup flash). Show it now UNLESS this is an autostart login launch
            // the user chose to keep hidden. A hidden launch still fully initializes:
            // the webview loads and the frontend starts the camera on mount
            // regardless of visibility, and detection runs off the window state.
            let start_hidden = std::env::args().any(|arg| arg == "--autostart")
                && app
                    .state::<api::AppState>()
                    .storage
                    .get_ui_settings()
                    .map(|settings| settings.start_hidden_on_login)
                    .unwrap_or(true);
            if !start_hidden {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                }
            }

            #[cfg(feature = "devbuild")]
            if let Some(window) = app.get_webview_window("main") {
                window.open_devtools();
            }

            Ok(())
        });

    let application = match builder.build(tauri::generate_context!()) {
        Ok(application) => application,
        Err(error) => {
            eprintln!("failed to build Tauri application: {error}");
            return;
        }
    };
    application.run(|app, event| {
        if matches!(event, RunEvent::Exit) {
            app.state::<api::AppState>().shutdown();
        }
    });
}
