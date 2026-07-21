use std::path::Path;

pub fn builder() -> tauri_specta::Builder<tauri::Wry> {
    tauri_specta::Builder::<tauri::Wry>::new()
        .commands(tauri_specta::collect_commands![
            crate::api::app_status,
            crate::api::initialize_inference,
            crate::api::train_models,
            crate::api::get_training_status,
            crate::api::cancel_training,
            crate::api::get_dataset_page,
            crate::api::get_dataset_stats,
            crate::api::get_needs_retraining,
            crate::api::get_reservoir_metadata,
            crate::api::get_camera_settings,
            crate::api::save_camera_settings,
            crate::api::reset_camera_settings,
            crate::api::get_ui_settings,
            crate::api::save_ui_settings,
            crate::api::reset_ui_settings,
            crate::api::get_training_settings,
            crate::api::reset_training_settings,
            crate::api::save_training_settings,
            crate::api::update_frame_label,
            crate::api::cleanup_unused_frames,
            crate::api::delete_frame,
            crate::api::undo_last_dataset_change,
            crate::api::get_undo_status,
            crate::api::reset_dataset,
            crate::api::reset_all_data,
            crate::api::get_classifier_registry,
            crate::api::get_feature_registry,
            crate::api::get_active_model_metadata,
            crate::api::export_dataset,
            crate::api::import_dataset,
            crate::api::get_shortcut_status,
            crate::api::start_camera,
            crate::api::stop_camera,
            crate::api::list_cameras,
        ])
        .events(tauri_specta::collect_events![
            crate::api::UndoStatusChangedEvent,
            crate::api::NativeStateChangedEvent,
        ])
}

pub fn export(path: impl AsRef<Path>) -> Result<(), String> {
    builder()
        .export(specta_typescript::Typescript::default(), path)
        .map_err(|error| error.to_string())
}
