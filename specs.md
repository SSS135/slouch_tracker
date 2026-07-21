## Architecture

### Tech Stack
- **Runtime**: Tauri 2 desktop application (Windows-first, MSVC target)
- **Backend**: Rust workspace in `src-tauri/` — the `app` crate plus `slouch-domain`, `slouch-ml`, `slouch-vision`, `slouch-store`
- **Frontend**: Svelte 5 (runes) + TypeScript in `src-svelte/` — a thin UI layer; all ML, storage, and vision logic lives in Rust
- **Inference**: native ONNX Runtime via `ort` (bundled `onnxruntime.dll` in resources)
- **ML**: pure-Rust classifiers, feature math, and cross-validation in `slouch-ml` (no TensorFlow.js)
- **Storage**: SQLite (STRICT schema) in the app data directory via `slouch-store`
- **IPC**: generated Specta TypeScript bindings (`src/generated/`) + 3 raw-byte commands with MessagePack responses
- **Frontend state**: TanStack `svelte-query` + Svelte runes stores
- **Tests**: Vitest (frontend), `cargo test` (backend), Playwright (browser-harness e2e), WebdriverIO (native e2e)

The frontend is a thin UI layer: it renders the native preview `<img>`, draws keypoint overlays, and issues commands. Camera capture, detection, feature extraction, training, persistence, and settings are all native Rust — behind Tauri commands plus a custom `slouchcam://` URI-scheme preview stream.

### Detection Pipeline

```
Webcam (nokhwa, MJPEG) → CameraActor (dedicated Rust thread)
→ DetectionDispatcher thread (~1 fps) → InferenceActor → RTMDet-nano (person detection) → crop
→ RTMPose-M (pose estimation) → keypoints + feature extraction → presence + posture classifiers
→ MessagePack InferenceUiResult → Svelte UI (overlays, status, smoothing)

Preview: CameraActor's freshest MJPEG frame is served over the custom `slouchcam://` URI scheme;
the webview pulls it into a native `<img>` each rAF — raw ~30 fps when focused, ~1 fps when
unfocused (the detection rate), off when the window is hidden/minimized.
```

**Detection runs at ~1 fps in every window mode** (the `DetectionDispatcher` rate). The ~30 fps is only the focused-window preview refresh — it is NOT the detection rate. The old frontend `getUserMedia` capture path is gone; the camera is owned natively by the `CameraActor`, and `infer_frame` remains registered only as a raw command for the test harness.

**Components:**
- `src-tauri/src/actors.rs` — `CameraActor`: owns the webcam on a dedicated "slouch-camera" thread (nokhwa, MJPEG capture); exposes the freshest raw/processed frames to the `slouchcam://` preview scheme; switches between `Foreground`/`Background` modes on window focus; runs a `DetectionDispatcher` thread that feeds frames to the `InferenceActor` at ~1 fps in all modes
- `src-tauri/src/actors.rs` — `InferenceActor`: owns the ONNX sessions on a dedicated thread behind a command channel; caches recent inference results keyed by one-use tokens so `save_capture` persists exactly the frame that was inferred (checkout/restore/commit lifecycle); feeds reservoir samples to the feature reservoir
- `src-tauri/src/power.rs` — Windows EcoQoS / idle-priority (efficiency mode) applied when the window is backgrounded, cleared when focused
- `src-tauri/crates/slouch-vision` — ONNX session management, preprocessing, the ported inference worker (RTMDet + RTMPose-M)
- `src-svelte/hooks/`, `src-svelte/components/` — pull the `slouchcam://` preview into a native `<img>`, draw keypoint overlays, and issue capture/training commands
- `src-svelte/lib/native/client.ts` — the single frontend gateway to all native commands (argument validation + typed error unwrapping)

**Model resources** (`src-tauri/resources/models/`): `rtmdet-nano.onnx`, `rtmpose-m.onnx`. RTMPose-M backbone output is `[1,768,8,6]` (pooled over spatial axes → 768 dims); GAU output is `[1,17,256]` (pooled over keypoints → 256 dims); std pooling uses `sqrt(population_variance + 1e-6)`.

### ML Training Pipeline

```
Collect (capture + label frames via save_capture) → SQLite dataset
→ Training tab (save_training_settings: features, classifier, normalization, reduction, CV)
→ train_models(doCv, Channel) → TrainingActor (dedicated Rust thread)
→ load frames/features from SQLite → normalization → dimensionality reduction
→ train presence + posture model pair → optional k-fold CV
→ serialize to model container v1 → persist generation to SQLite → publish to InferenceActor
```

- Training progress streams over a Tauri `Channel<TrainingEvent>`: `started` → `progress { stage: processing | evaluating | deploying, progress: 0–100 }` → `completed | failed | cancelled`. Events carry a job id and a monotonic sequence number.
- One training job at a time; `cancel_training` requests cooperative cancellation; `get_training_status` reports whether a job is running.
- Deployment is persist-then-publish: the model pair is written to SQLite first, then swapped into the `InferenceActor`; a failed publish never leaves storage and runtime disagreeing silently.

### IPC Contract

37 Tauri commands total, registered in `src-tauri/src/lib.rs`.

**Typed commands (34)** are annotated `#[specta::specta]` and exported through `tauri-specta` to `src/generated/bindings.generated.ts`. The frontend calls them only through `src-svelte/lib/native/client.ts`, which unwraps the `Result<T, ApiError>` envelope into `NativeCommandError`. Beyond the settings and dataset commands, these include the camera lifecycle commands `start_camera` / `stop_camera` / `list_cameras`, `get_needs_retraining` (auto-retrain hint), `get_reservoir_metadata` (feature reservoir state), and `cleanup_unused_frames`.

**Raw-byte commands (3)** — `infer_frame`, `get_thumbnail`, `save_capture` — move bulk binary data. `infer_frame` is retained only as a test-harness entry point now that capture is native; `get_thumbnail` and `save_capture` remain on the live path:
- Requests use raw bodies plus `x-slouch-*` headers: `x-slouch-ipc-version`, `x-slouch-pixel-format: rgba8`, `x-slouch-width/height/stride`, and for captures request id, one-use token, frame id, timestamp, label, MIME type.
- Frames are tightly packed RGBA (`stride = width × 4`), max 1920×1080, max 8 MiB. Thumbnails are 1 byte–2 MiB JPEG/PNG/WebP.
- Responses are MessagePack (`rmp_serde`, named fields).
- **Hard rule**: camera pixels, thumbnails, feature vectors, and model weights must never cross IPC as JSON arrays.

**Events**: `shortcut-capture` (global hotkeys), `dataset-changed`, `undoStatusChanged`, `nativeStateChanged`. Training progress uses a per-call `Channel`, not a global event.

### Persistence (SQLite + slouch-store)

**Live database** (`src-tauri/schema/live-v1.sql`, `application_id` = `SLCH`, STRICT tables):
- `app_meta` — dataset version + last-modified timestamp
- `frames` — id, capture time, label (`good`/`bad`/`away`/`unused`), bbox
- `frame_keypoints` — 17 keypoints per frame with scores
- `frame_features` — per-frame stored feature vectors as little-endian f32 blobs, dimension-checked
- `thumbnails` — JPEG/PNG/WebP bytes (≤ 2 MiB)
- `settings` — key + schema version + validated JSON (camera, UI, training settings)
- `model_generations` / model payload tables — versioned model pairs with dataset identity and training-config SHA-256 fingerprints; one active generation

**Dataset export/import**: `.slouchpack` files — a standalone SQLite archive (`src-tauri/schema/archive-v1.sql`, `application_id` = `SLPK`) selected through native file dialogs (`export_dataset` / `import_dataset`).

**Undo**: single authoritative native undo for dataset mutations (`undo_last_dataset_change`, `get_undo_status`, `undoStatusChanged` event).

### Model Format

Trained models are serialized as **model container v1** (`src-tauri/model-format-v1.md`): a canonical little-endian record container (magic `SLMD`), strictly ordered named records, exact per-classifier record allowlists, SHA-256 payload hashes, and a pair-level training-config fingerprint container (magic `SLCF`). Both role envelopes (presence + posture) in one generation share the same dataset version and config fingerprint. Validation rejects unknown, duplicate, missing, or extra records — the format is closed by design.

### Classifiers (6, registry-driven)

`mlp`, `knn` (cosine/RBF kernels), `svm`, `kmeans_prototype`, `gaussian_nb`, `kmeans_logistic`.

- Implementations: `src-tauri/crates/slouch-ml/src/ported/*_classifier.rs` + `classifier_factory.rs` + `classifier_registry.rs`
- Metadata and parameter schemas: `src-tauri/crates/slouch-domain/src/classifier.rs` (`ClassifierId`, `ClassifierMetadata`)
- The Svelte UI auto-generates parameter controls from `get_classifier_registry` — no frontend changes needed for new parameters.

### Feature Types (16, registry-driven)

Defined in `src-tauri/crates/slouch-domain/src/feature.rs`; served via `get_feature_registry`.

**Stored** (extracted at capture time, persisted in `frame_features`):
- `backbone_features` / `_max` / `_std` — RTMPose-M backbone avg/max/std pooling (768 dims each)
- `gau_features` / `_max` / `_std` — RTMPose-M GAU avg/max/std pooling (256 dims each)
- `rtmdet_extracted` — pooled (avg/std/max) RTMDet cls_p5 + reg_p5 (384 dims, presence)

**Computed** (derived from keypoints/bbox on demand):
- `rtmdet_engineered` — detection geometry (135 dims, presence)
- `engineered_features` — body-proportion ratios, 1D soft binning (54 dims)
- `joint_2d` / `joint_3d` / `joint_4d` — joint histograms (81 / 125 / 625 dims)
- `posture_raw` — 5 raw geometric features
- `posture_geometry` — 10 scale/translation-invariant geometric posture features from head and shoulder keypoints
- `keypoint_scores` (17 dims), `raw_keypoints` (34 dims)

### Normalization & Dimensionality Reduction

- **Normalization** (`normalization.mode`): `none`, `layer` (per-sample standardization), `z_score` (per-dimension standardization, recommended; mean/std tensors saved in the model container)
- **Reduction** (`reduction.method`): `none`, `random_projection` (seeded, matrix saved), `pca` (mean/components/explained-variance saved). PCA is available again — the Rust backend has SVD; the old "PCA removed, TensorFlow.js has no SVD" limitation no longer applies.

## Directory Structure

```
src-tauri/
  src/               app crate: lib.rs (setup, plugins, global shortcuts),
                     api.rs (AppState + 37 commands), actors.rs (Camera/Inference/TrainingActor),
                     bindings.rs (Specta builder), errors.rs, bin/export_bindings
  crates/
    slouch-domain/   DTOs, validation, labels, keypoints/bboxes, settings types,
                     feature + classifier registry metadata
    slouch-ml/       feature math, classifiers, k-means, cross-validation,
                     training worker, model-state serialization
    slouch-vision/   ort ONNX sessions, preprocessing, inference worker
    slouch-store/    SQLite storage, archive export/import, model container format,
                     feature reservoir
  schema/            live-v1.sql, archive-v1.sql
  resources/         models/ (rtmdet-nano.onnx, rtmpose-m.onnx), onnxruntime/ (dll + notices)
  model-format-v1.md model container specification

src/generated/       bindings.generated.ts (machine-written — never hand-edit),
                     bindings.ts (hand-written wrapper), bindings.contract.test.ts

src-svelte/
  pages/             PostureTrackerApp.svelte (single-page app, keyboard shortcuts)
  components/
    unified/         ControlPanel, SettingsTab, TrainingTab, CameraViewport,
                     CaptureButtonsOverlay, FrameListOverlay, LoggerSettings, UndoButton
    dataset/         frame grid, thumbnails, classifier/feature selectors, stats
    ui/              primitives (Slider, RadioGroup, Section, HelpText)
  hooks/             runes hooks (.svelte.ts implementation + .ts interface file per hook):
                     camera stream/settings, frame sampler/processor, canvas renderer,
                     dataset operations, model training, global shortcuts, notifications
  contexts/          Camera, TrainingConfig, Training (runes contexts)
  lib/
    native/          client.ts — the only IPC gateway (validation + error unwrapping)
    query/           svelte-query client + dataset queries
    state/           runes stores (inference, nativeApp, training)
  services/          logging/, dataset/ (thumbnail generation), ml/, posture/ constants
  harness/           mock-Tauri harness for browser e2e
  providers/         AppProviders.svelte

e2e/browser/         Playwright specs (svelte-plumbing, svelte-real-app)
e2e/native/          WebdriverIO specs (launch readiness, raw IPC inference,
                     persistence restart pair, dialog/lifecycle errors);
                     wdio.conf.ts drives the devbuild binary
scripts/             bindings check, run-gate.mjs (vcvars64 cargo wrapper),
                     verify-migration, security gates
```

## Key Workflows

### Adding a New Classifier

Spans the Rust workspace plus generated bindings:

1. Implement the classifier in `src-tauri/crates/slouch-ml/src/ported/<name>_classifier.rs`; wire it into `classifier_factory.rs` and `classifier_registry.rs`.
2. Add the `ClassifierId` variant plus metadata/parameter schema in `src-tauri/crates/slouch-domain/src/classifier.rs`.
3. Extend the model container: add the classifier's exact record allowlist (with a state version) in `slouch-store`'s `model_format.rs` and document it in `src-tauri/model-format-v1.md`.
4. Regenerate bindings: `npm run bindings:generate`, then `npm run bindings:check`.
5. The Svelte UI auto-generates parameter controls from `get_classifier_registry` — no UI changes needed.
6. Add Rust unit tests plus serialization round-trip coverage.

### Bindings Generation

- `npm run bindings:generate` — runs `cargo run --bin export_bindings` and rewrites `src/generated/bindings.generated.ts`
- `npm run bindings:check` — verifies the generated file matches the Rust API
- CI freshness gate: `cargo test -p app --test bindings_freshness -- --exact generated_bindings_are_fresh`
- Never hand-edit `bindings.generated.ts`. The hand-written surface is `src/generated/bindings.ts` and `src-svelte/lib/native/client.ts`.
- Any change to command signatures, DTOs, or events in Rust requires regenerating bindings before the frontend compiles.

## Navigation & Shortcuts

Single-window app: camera viewport with overlay controls plus a slide-in control panel (Settings / Collect / Training tabs).

- **In-app keys**: `G` (capture good), `B` (capture bad), `A` (capture away), `C` (clear sampled frames), `U` (undo)
- **Global hotkeys** (work while the app is unfocused): `Ctrl+Win+G` / `Ctrl+Win+B` / `Ctrl+Win+A` — registered natively via `tauri-plugin-global-shortcut`, emitted to the frontend as `shortcut-capture` events with an audio confirmation beep. `get_shortcut_status` reports registration state.
- Capture uses one-use tokens tied to the cached inference result, so a capture always stores the exact inferred frame.

## Settings

All settings are persisted **natively** through commands into the SQLite `settings` table — there is no `localStorage` persistence:

- `get_camera_settings` / `save_camera_settings` / `reset_camera_settings`
- `get_ui_settings` / `save_ui_settings` / `reset_ui_settings` (alert volume, alert delay)
- `get_training_settings` / `save_training_settings` / `reset_training_settings`

Settings are validated in `slouch-domain` before persisting; training-settings mutations are rejected while a training job is running.

## Logging

**Frontend** (`src-svelte/services/logging/`): category-based logger with URL-parameter control.
- Categories: `detection`, `training`, `worker`, `storage`, `debug`, `preprocessing`; levels `debug`/`info`/`warn`/`error`
- `?log=debug`, `?log=all`, `?log=detection:debug,training:info`, `?log=none`
- Runtime-adjustable without reload from the Settings tab (`LoggerSettings.svelte` updates the URL and reconfigures the logger)
- Default: warnings and errors only

**Backend**: `tauri-plugin-log` writes to stdout, the webview console, and a rotating file in the app log directory. Level is `Debug` in dev/devbuild builds and `Info` in release.

## Testing

```bash
npm run test:svelte          # Vitest frontend suite (vitest.svelte.config.ts)
npm run test:svelte -- <pattern>   # single file/pattern
npm run check:svelte         # svelte-check type checking
npm run check:svelte:plumbing
npm run lint:svelte
```

**Rust** — on Windows, cargo must run inside a VS 2022 x64 developer environment (vcvars64):

```
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cargo test --manifest-path src-tauri/Cargo.toml --workspace
```

(`scripts/run-gate.mjs` wraps the gate commands this way; use it for the standard fmt/clippy/test gates.)

**End-to-end:**
- `npm run test:e2e:web` — Playwright against the browser harness: builds the Svelte app with a mock Tauri backend and serves it at `127.0.0.1:4174` (`e2e/browser/`, `playwright.config.ts`). Exercises the real UI plumbing (capture tokens, paging, undo) without a native build.
- `npm run test:e2e:native` — WebdriverIO (`wdio.conf.ts`, `e2e/native/`) against the devbuild desktop binary; run `npm run tauri:build:dev:win` first to produce it.
- `npm run verify:migration` — migration acceptance verification.

## Build & Development Commands

**Development:**
- `npm run dev:svelte` — Vite dev server for the Svelte frontend
- `npm run tauri:dev` — full Tauri development build (frontend + Rust backend)

**Production:**
- `npm run build:svelte` — production frontend bundle
- `npm run tauri:build` / `npm run tauri:build:win` — desktop app (Windows MSVC target)
- `npm run tauri:build:dev` / `:dev:win` — dev-configured build with the `devbuild` feature (debug logging + devtools open on launch)

**Quality gates:**
- `npm run lint:svelte`, `npm run check:svelte`, `npm run test:svelte`
- `cargo fmt` / `cargo clippy -D warnings` / `cargo test` across the workspace (via vcvars64)
- `npm run bindings:check` — generated bindings freshness
- `npm run test:security`, `npm run package:inspect`
