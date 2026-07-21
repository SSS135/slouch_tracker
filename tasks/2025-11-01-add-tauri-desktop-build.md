# Task 2025-11-01: Add Tauri Desktop Build Support
**STATUS:** COMPLETED

## User Request
"create finished task for adding tauri build"

## Critical Discoveries (Non-Obvious)

**1. Rust toolchain version jump required:**
Initial 1.62.0 incompatible → 1.91.0 needed. RLS component deprecated, switched to minimal profile + stable-x86_64-pc-windows-msvc.

**2. Build size confusion:**
`target/` directory shows 2+ GB but actual executable is 43 MB. MSI (38 MB) and NSIS (37 MB) installers even smaller due to compression.

**3. DevTools auto-open needs Manager trait:**
Auto-opening DevTools requires explicit `use tauri::Manager;` import. Without it, `window.open_devtools()` method unavailable.

**4. Web-dist vs dist directory:**
Expo web builds to `web-dist/`, not `dist/`. Tauri config must point to correct output directory or build fails silently.

## Solution

Added Tauri desktop application wrapper to existing Expo React Native web app:

**Setup:**
- Created `src-tauri/` Rust project with standard Tauri structure
- Configured `tauri.conf.json` for 1400x900 window (min 800x600)
- Set frontend dist path to `../web-dist/` (Expo's web output)
- Build command: `npm run build:web`
- Bundle targets: MSI + NSIS installers for Windows

**DevTools Integration:**
```rust
// src-tauri/src/lib.rs
use tauri::Manager;

#[cfg(debug_assertions)]
window.open_devtools();
```
Auto-opens DevTools in debug builds, F12 and right-click inspect work.

**Build Configuration:**
- Upgraded Rust toolchain 1.62.0 → 1.91.0
- Installed stable-x86_64-pc-windows-msvc
- MSVC toolchain (Windows-recommended)
- Embedded web assets in executable (standalone distribution)

## Lessons

- Tauri requires recent Rust (1.80+), check toolchain before initializing
- Build artifacts large but final executable compact (37-43 MB)
- Manager trait import required for window manipulation
- DevTools auto-open should be debug-only (`#[cfg(debug_assertions)]`)
- Dual installer formats (MSI + NSIS) provide flexibility for users

## Files Modified

- `src-tauri/tauri.conf.json` - App configuration, window settings, build targets
- `src-tauri/src/lib.rs` - Added Manager import, DevTools auto-open logic
- `src-tauri/src/main.rs` - Entry point (calls run() from lib.rs)
- `src-tauri/Cargo.toml` - Rust dependencies (Tauri, logging plugin)
- `.gitignore` - Added `src-tauri/target/`, `src-tauri/WixTools/`

## Impact

**Capabilities Added:**
- Desktop application distribution (Windows MSI/NSIS installers)
- Standalone executable (43 MB, no browser required)
- Native window chrome, system integration
- DevTools available in development builds

**Distribution:**
- MSI installer: 38 MB (`Slouch Tracker_1.0.0_x64_en-US.msi`)
- NSIS installer: 37 MB (`Slouch Tracker_1.0.0_x64-setup.exe`)
- Portable exe: 43 MB (no installation needed)

**Developer Experience:**
- `npm run tauri dev` - Development with auto-reload
- `npm run tauri build` - Production builds
- Auto-open DevTools in debug mode
