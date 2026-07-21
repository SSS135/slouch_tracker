# Slouch Tracker

**Privacy-first webcam posture tracker for Windows. All detection and model training run on-device — nothing ever leaves your machine.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D6)
![Tauri](https://img.shields.io/badge/Tauri-2-24C8DB)
![Rust](https://img.shields.io/badge/Rust-1.88%2B-DEA584)
![Svelte](https://img.shields.io/badge/Svelte-5-FF3E00)

Slouch Tracker watches your webcam, estimates your body pose in real time, and warns you when you start slouching. It runs entirely as a native Windows desktop app: the camera, ML inference, feature extraction, model training, and storage are all handled locally in Rust. There are no cloud services, no accounts, and no network calls.

## Features

- **Real-time posture detection** — RTMDet-nano (person detection) + NLF-L (17-keypoint + 3D depth) pose estimation on DirectML, running through native ONNX Runtime via the Rust `ort` crate.
- **Train your own models** — Collect and label your own frames, then train personalized classifiers in-app. Six registry-driven classifier types (`mlp`, `knn`, `svm`, `kmeans_prototype`, `gaussian_nb`, `kmeans_logistic`) with a UI that auto-generates parameter controls.
- **Flexible features** — 12 user-selectable feature types (RTMDet features, NLF-L 3D-depth features, and geometric/keypoint features) with normalization (`z_score`/`layer`/`none`) and dimensionality reduction (`pca`/`random_projection`/`none`).
- **In-app data collection** — Capture frames with `G` (good), `B` (bad), and `A` (away) keys, plus global hotkeys `Ctrl+Win+G` / `Ctrl+Win+B` / `Ctrl+Win+A` that work even when the app is not focused.
- **Cross-validated training** — One-click training with optional k-fold cross-validation and reported metrics; progress streams live over a Tauri channel.
- **Local SQLite storage** — Frames, keypoints, feature vectors, thumbnails, settings, and trained models are stored in a local SQLite database.
- **Dataset export/import** — Portable `.slouchpack` archives via native file dialogs.
- **Privacy mode** — Obscures the live preview while detection keeps running.
- **Focus-aware power behavior** — Smooth ~30 fps preview when focused, detection at ~1 fps in every mode, and EcoQoS / efficiency mode when the window is backgrounded. Typical CPU usage is ~1–2%.

## Screenshots

<!-- TODO: add screenshot/GIF -->

## Install

1. Download the latest NSIS installer (`.exe`) from the [GitHub Releases](https://github.com/SSS135/slouch_tracker/releases) page.
2. Run the installer and launch **Slouch Tracker**.

> **Note:** Release builds are **unsigned**. Windows SmartScreen will show an "unknown publisher" warning on first run — this is expected. Choose *More info → Run anyway* to proceed.

## Build from Source

### Prerequisites

- **Windows 10/11 (x64)**
- **Git LFS — mandatory.** The ONNX models (`rtmdet-nano.onnx` ~4 MB, `nlf_l_crop_fp16.onnx` ~233 MB) and the app icons are stored via Git LFS. Cloning **without** Git LFS installed produces pointer files instead of the real assets and the build will be broken.
- **Rust 1.88+** (with the `x86_64-pc-windows-msvc` toolchain)
- **Node.js 20+**
- **Visual Studio 2022** with the *Desktop development with C++* workload (any edition — Community works)
- **WebView2 runtime** (preinstalled on Windows 11; on Windows 10 install the Evergreen runtime from Microsoft)

### Steps

```bash
# 1. Install Git LFS once per machine
git lfs install

# 2. Clone (LFS assets are fetched automatically)
git clone https://github.com/SSS135/slouch_tracker.git
cd slouch_tracker

# 3. Install frontend dependencies
npm install

# 4a. Run in development
npm run tauri:dev

# 4b. Or build the Windows NSIS installer
npm run tauri:build:win
```

> Cargo must run inside a Visual Studio 2022 x64 developer environment (vcvars64). `npm run tauri:*` handles this when launched from a developer shell; for raw `cargo` commands, first run:
> `call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Usage

The app is a single window: a live camera viewport with overlay controls, plus a slide-in panel with **Settings**, **Collect**, and **Training** tabs.

**Capture keys** (while the app is focused):

| Key | Action |
|-----|--------|
| `G` | Capture a *good posture* frame |
| `B` | Capture a *bad posture* frame |
| `A` | Capture an *away* frame |
| `C` | Clear sampled frames |
| `U` | Undo last dataset change |

**Global hotkeys** (work while the app is unfocused): `Ctrl+Win+G` / `Ctrl+Win+B` / `Ctrl+Win+A` capture good / bad / away with an audio confirmation beep.

**Training workflow:**

1. Collect labeled frames for *good*, *bad*, and *away* postures.
2. Open the **Training** tab, pick features / classifier / normalization / reduction, and press **Train**.
3. Review the cross-validation metrics. The trained model is deployed automatically for live detection.

The app tracks whether the model needs retraining as you add data, so you can keep refining it as your dataset grows.

## Privacy

- **Everything is local.** Detection, feature extraction, and training all run on your own machine in native Rust. There are no network calls and no telemetry.
- **Your data stays on disk.** Frames, keypoints, feature vectors, thumbnails, settings, and trained models live in a local SQLite database under your user app-data directory.
- **Privacy mode** obscures the live preview (blurred) while detection continues, so you can keep tracking without a visible camera feed on screen.
- **Portable, not cloud.** Datasets move only when *you* export a `.slouchpack` file through a native save dialog.

## Architecture

Slouch Tracker is a Tauri 2 app with a Rust backend workspace and a deliberately thin Svelte 5 UI. The `src-tauri/` workspace splits into the `app` crate plus `slouch-domain` (DTOs, validation, registries), `slouch-ml` (classifiers, feature math, cross-validation, training), `slouch-vision` (ONNX sessions, preprocessing, the RTMDet + NLF-L inference worker), and `slouch-store` (SQLite storage, model container format, dataset archives). The camera is captured natively (nokhwa, MJPEG) and previewed in the webview through a custom `slouchcam://` URI scheme; detection runs on a dedicated Rust dispatcher thread. The frontend talks to Rust through generated [Specta](https://github.com/specta-rs/specta) bindings, with three raw-byte MessagePack commands reserved for moving bulk image data. See [specs.md](specs.md) for the full architecture.

## Development

```bash
# Frontend tests (Vitest)
npm run test:svelte
npm run test:svelte -- <pattern>      # single file / pattern

# Type checking and linting
npm run check:svelte
npm run check:svelte:plumbing
npm run lint:svelte

# Rust tests (must run inside a VS 2022 x64 dev environment / vcvars64)
node scripts/run-gate.mjs             # wraps fmt / clippy / test with vcvars64
# or directly:
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cargo test --manifest-path src-tauri/Cargo.toml --workspace

# End-to-end
npm run test:e2e:web                  # Playwright against the mock-Tauri browser harness
npm run tauri:build:dev:win && npm run test:e2e:native   # WebdriverIO against the devbuild binary
```

After changing any Rust command signature, DTO, or event, regenerate and verify the TypeScript bindings:

```bash
npm run bindings:generate
npm run bindings:check
```

## License

Slouch Tracker is released under the [MIT License](LICENSE).

Third-party components and their licenses are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

### Acknowledgments

- [OpenMMLab](https://github.com/open-mmlab) — RTMDet person-detection model.
- [NLF (Neural Localizer Fields)](https://github.com/isarandi/nlf) — NLF-L 3D human-pose model (weights are for non-commercial research use only).
- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) — native inference runtime.
- [Tauri](https://tauri.app) — the desktop application framework.
