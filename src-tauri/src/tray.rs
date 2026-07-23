//! System-tray integration: the tray icon, its menu (Open / Pause / Exit), window
//! restore, and the single source of truth for pausing/resuming tracking.
//!
//! Pause/resume flows through [`set_tracking_paused`] no matter who initiates it
//! (tray menu, or the `start_camera`/`stop_camera` commands), so the camera actor,
//! the tray menu text/icon, and the webview (`tracking-state-changed`) never
//! disagree.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, Wry,
};

use crate::{api::AppState, errors::ApiError};

/// Managed state that keeps the tray alive for the process lifetime. Dropping the
/// `TrayIcon` would remove the icon, so both it and the pause menu item are held
/// here in `app.manage(...)`.
pub struct TrayState {
    icon: TrayIcon<Wry>,
    pause_item: MenuItem<Wry>,
    normal_icon: Image<'static>,
    paused_icon: Image<'static>,
    // Session-only paused flag (never persisted).
    paused: AtomicBool,
    // Serializes a full pause/resume transition so the camera toggle, menu text,
    // icon swap, and event emission stay a single atomic unit.
    toggle_lock: Mutex<()>,
}

/// Builds the tray icon and its menu, then stores the handles in managed state.
pub fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open", true, None::<&str>)?;
    let pause = MenuItem::with_id(app, "pause", "Pause tracking", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Exit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &pause, &quit])?;

    let (normal_icon, paused_icon) = tray_icons(app)?;

    let tray = TrayIconBuilder::<Wry>::new()
        .icon(normal_icon.clone())
        .tooltip("Slouch Tracker")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "open" => restore_window(app),
            "pause" => {
                let paused = app.state::<TrayState>().paused.load(Ordering::Acquire);
                let _ = set_tracking_paused(app, !paused);
            }
            "quit" => app.exit(0),
            _ => {}
        })
        // Left double-click and left click-release both reopen the window. Keep the
        // catch-all arm: TrayIconEvent is non_exhaustive.
        .on_tray_icon_event(|tray, event| {
            let app = tray.app_handle();
            match event {
                TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                } => restore_window(app),
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } => restore_window(app),
                _ => {}
            }
        })
        .build(app)?;

    app.manage(TrayState {
        icon: tray,
        pause_item: pause,
        normal_icon,
        paused_icon,
        paused: AtomicBool::new(false),
        toggle_lock: Mutex::new(()),
    });
    Ok(())
}

/// Brings the main window back from the tray. All three calls are required to work
/// around a Windows restore bug (tauri#12392) where `show` alone leaves the window
/// minimized/unfocused.
pub fn restore_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Single source of truth for pause/resume. Toggles the camera actor, updates the
/// tray menu text + icon, records the session paused flag, and notifies the webview
/// via `tracking-state-changed`.
pub fn set_tracking_paused(app: &AppHandle, paused: bool) -> Result<(), ApiError> {
    let tray = app.state::<TrayState>();
    let _guard = tray
        .toggle_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let camera = &app.state::<AppState>().camera;
    if paused {
        camera.stop_capture()?;
    } else {
        camera.start_capture()?;
    }
    tray.paused.store(paused, Ordering::Release);

    let label = if paused {
        "Resume tracking"
    } else {
        "Pause tracking"
    };
    let _ = tray.pause_item.set_text(label);

    let icon = if paused {
        &tray.paused_icon
    } else {
        &tray.normal_icon
    };
    let _ = tray.icon.set_icon(Some(icon.clone()));

    let _ = app.emit(
        "tracking-state-changed",
        crate::api::TrackingStateChangedEvent { paused },
    );
    Ok(())
}

/// The normal (default window icon) and a runtime-desaturated grayscale copy used
/// while tracking is paused. No new asset: the paused icon is computed per pixel
/// from the same RGBA.
fn tray_icons(app: &AppHandle) -> tauri::Result<(Image<'static>, Image<'static>)> {
    let source = app
        .default_window_icon()
        .ok_or_else(|| tauri::Error::AssetNotFound("default window icon".into()))?;
    let (width, height) = (source.width(), source.height());
    let rgba = source.rgba().to_vec();
    let grayscale = desaturate_rgba(&rgba);
    Ok((
        Image::new_owned(rgba, width, height),
        Image::new_owned(grayscale, width, height),
    ))
}

/// Replaces each pixel's RGB with its Rec. 601 luma, preserving alpha.
fn desaturate_rgba(rgba: &[u8]) -> Vec<u8> {
    let mut out = rgba.to_vec();
    for pixel in out.chunks_exact_mut(4) {
        let luma = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        pixel[0] = luma;
        pixel[1] = luma;
        pixel[2] = luma;
    }
    out
}
