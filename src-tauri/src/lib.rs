mod actors;
mod api;
mod bindings;
mod errors;
mod power;
#[cfg(test)]
mod native_runtime_resources_parity;

pub fn export_bindings(path: impl AsRef<std::path::Path>) -> Result<(), String> {
    bindings::export(path)
}

use std::path::PathBuf;

use tauri::{Emitter, Manager, RunEvent, WindowEvent};
use tauri_plugin_global_shortcut::{Builder as GlobalShortcutBuilder, ShortcutState};
use tauri_plugin_log::{Target, TargetKind};

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

    let builder = tauri::Builder::default()
        .plugin(shortcut_plugin)
        .plugin(tauri_plugin_dialog::init());

    // Embedded WebDriver server for packaged-native E2E (devbuild only). Must be
    // registered on the builder (not in setup) so its on_webview_ready hook runs
    // for the main window created from the config.
    #[cfg(feature = "devbuild")]
    let builder = builder.plugin(tauri_plugin_wdio_webdriver::init());

    let builder = builder
        // Per-frame PULL preview: the webview GETs `slouchcam://…/frame` each rAF
        // and gets the freshest MJPEG frame (foreground) or 204 (no frame /
        // background). A path containing "processed" serves the detector-input
        // frame instead (post-preprocessing JPEG, refreshed at detection rate);
        // any other path serves the raw preview frame.
        // The webview origin (dev: http://127.0.0.1:5174; prod: tauri://localhost)
        // differs from this custom-scheme host, so the frame must be CORS-allowed;
        // without it privacy-mode grid sampling and thumbnail reads fail as opaque
        // cross-origin fetches. The stream is a local, in-process preview only.
        .register_asynchronous_uri_scheme_protocol("slouchcam", |ctx, request, responder| {
            let state = ctx.app_handle().state::<api::AppState>();
            let bytes = if request.uri().path().contains("processed") {
                state.camera.processed_frame_bytes()
            } else {
                state.camera.latest_frame_bytes()
            };
            let response = match bytes {
                Some(bytes) => tauri::http::Response::builder()
                    .status(200)
                    .header(tauri::http::header::CONTENT_TYPE, "image/jpeg")
                    .header(tauri::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                    .body(bytes),
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
        .on_window_event(|window, event| {
            if let WindowEvent::Focused(focused) = event {
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
            api::start_camera,
            api::stop_camera,
            api::list_cameras,
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

            api::initialize_inference(app.state::<api::AppState>())
                .map_err(std::io::Error::other)?;

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
