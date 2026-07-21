# Native Rust GUI migration — research & decision record

Date: 2026-07-20. Status: **DEFERRED — keep Svelte/Tauri for now (hybrid: native camera + web UI).**
This document preserves the research so the "drop the webview, go pure-Rust GUI" decision can be
revisited later without re-doing the work.

## Why this was considered

Nearly all of the app's recent pain has been **webview-era pain**, not UI pain: the WebView2 camera
permission prompt (per-origin, re-prompts on origin change), the IPv6 dev-server load stall, JS-framework
context bugs, and the whole machinery needed to stream camera pixels Rust→webview over IPC. Since all
ML/vision/storage/camera is already native Rust, the question was whether a pure-Rust GUI (egui/iced/…)
would delete the webview and its pain.

## Decision (2026-07-20)

**Keep Svelte/Tauri for now.** Rationale: the app has a **complex, working UI** (dataset tab with
draggable thumbnail grids, buttons-in-cells, hover actions, etc.) that HTML/CSS is genuinely well-suited
to, and the *actual* pain point — camera — is being solved natively anyway (see "native camera" below).
A full ~13k-LOC frontend rewrite isn't justified right now. Research retained for a future revisit.

**What the webview pain reduces to once native camera lands:** just RAM/startup overhead + two languages
+ frame-streaming IPC. The camera permission prompt and getUserMedia are already gone (native capture).

## Framework comparison (if/when we go pure-native)

**Recommended if we migrate: egui/eframe.** Runner-up: iced 0.14. Everything else ruled out for this app.

| Framework | Verdict | Why |
|---|---|---|
| **egui/eframe** | **TOP PICK** | Immediate mode = natural fit for per-frame camera texture + overlays; **best interactive-grid ecosystem** (`egui_dnd` + `egui_virtual_list` + `egui_extras::Table`); zero-copy wgpu texture path (`register_native_texture`); Rerun is direct production proof; MIT/Apache; frictionless winit integration with our OS crates; AccessKit accessibility. Weakness: "tool-ish" aesthetic ceiling. |
| **iced 0.14** | Runner-up | Cleaner/more consumer aesthetics, best idle-power (reactive-by-default in 0.14), automatic tiny-skia software GPU fallback, MIT, proven (System76 COSMIC via libcosmic fork, Sniffnet). Costs: per-frame `image` widget is a **documented trap** — camera needs a custom `shader`/`Primitive` (hundreds of lines of wgpu, copyable from `iced_video_player`); drag-reorder grid is hand-rolled; `iced_aw` (tabs) immaturity. |
| **Slint** | 3rd | Best-looking (Fluent), virtualized ListView, simplest declarative camera path. Costs: **event-loop friction with `global-hotkey`/`tray-icon`** (one dev split into two processes); **license** is tri-licensed (GPLv3 / royalty-free-with-attribution / commercial) — closed desktop is free but requires an attribution badge, unusual for Rust; no built-in radio/tooltip/drag-reorder; desktop still hardening (late-2025). |
| **gpui** (Zed's) | **NO** | Windows only stabilized Oct 2025 (via a 3rd HLSL/DX backend, GPU-driver crashes, "behaves like a game"); pre-1.0 API churn; sparse docs; **no documented live-video-texture path**; effectively Zed-coupled. `gpui-component` (Longbridge) is nice but inherits the risk. Too much Windows GPU-driver surface for a background utility. |
| **Dioxus / Blitz** | Watch, don't adopt | Blitz (native React-in-Rust, no webview) is **alpha** (beta end-2025, prod "sometime 2026"); live-video not proven. **Dioxus *desktop* today is still `wry`/WebView2** — choosing it now keeps the webview. |
| **Xilem/Masonry, Floem, Makepad** | Too early / niche | Xilem/Masonry alpha with ongoing breaking changes; Floem pre-1.0 smaller ecosystem; Makepad = bespoke DSL + steep curve. |
| **Keep Tauri** | Baseline | Best for arbitrary complex web UI + polish; smallest to keep since UI exists. But it's the source of the webview/camera pain and keeps two languages + IPC. |
| **Leptos** | N/A | Web/WASM — on desktop it still needs a webview/Tauri. Does NOT remove the webview problem. |

### The interactive dataset grid (was the key worry) — egui is actually STRONGEST here
- **`egui_dnd`** (0.16): drag-to-reorder that mutates the Vec in place, animates the dragged tile, and
  **works in wrapped grid layouts** (its example is a reorderable thumbnail grid) — not just vertical lists.
- **Grid layout:** **datasets are usually <100 samples, so virtualization is NOT needed** — a plain
  `egui::Grid` or `egui_dnd` wrapped layout is fine (cells hold image + buttons + labels). Only reach for
  `egui_virtual_list`/`egui_extras::TableBuilder` (virtualized) if datasets ever grow to thousands.
- **Hover actions:** paint per-item buttons in the tile's fixed rect on `response.hovered()` (avoid
  clickable buttons in tooltips/`on_hover_ui` — issue #5519; reserve a fixed rect to avoid flicker #5363).
- iced and Slint both make you **hand-roll** drag-reorder for an arbitrary wrapped grid. Conclusion from
  research: *"the interactive-grid requirement does not justify switching away from egui — if anything it
  argues for egui."* Remaining manual burden in egui: thumbnail texture eviction (`forget_image`, images
  are cached indefinitely by default — issue #3291) + flicker-free hover overlays; "neither is hard."

### Camera path (native, no IPC) — same for egui/iced
`nokhwa` (MSMF/Win, AVFoundation/mac; feature `output-wgpu` copies a frame straight into a `wgpu::Texture`)
→ egui `register_native_texture`→`TextureId` (or iced `shader` widget) → composite with skeleton/bbox
overlay drawn via `Painter`/`Canvas`. Upload frames **off the UI thread** (`queue.write_texture`); drive
repaints via `ctx.request_repaint()` from the camera-actor thread on frame arrival. No getUserMedia, no
permission prompt, no pixels-over-IPC, one language.

### Background/idle power (this app runs for hours)
- iced 0.14 = best idle (reactive-by-default; go idle on Unfocused/minimize).
- egui = fine (reactive; repaint only on frame arrival while focused). **RED FLAG to validate:** egui
  issue **#7059** — on **macOS**, when occluded/backgrounded, CPU jumps 50%+ (glow AND wgpu, no fix).
  Windows-first + channel-driven 1-2 fps repaint should avoid it, but **must test a multi-hour
  backgrounded run on macOS** before committing to Mac.
- All wgpu-based: measure idle/minimized/occluded CPU+GPU on both OSes over hours (wgpu #3788 closed
  "not planned").

### GPU robustness
- egui/eframe defaults to wgpu with **no automatic software fallback** → blank-window/crash risk on odd
  drivers/RDP/VMs; mitigate by shipping a `glow` fallback build or validating the target GPU spread.
- iced has an automatic `tiny-skia` CPU fallback (robustness advantage).

## Reuse / rewrite / discard scope (if we migrate)

**KEEP unchanged (~30k LOC Rust, GUI-agnostic — confirmed no Tauri deps):**
- `src-tauri/crates/slouch-domain` (~3.0k), `slouch-ml` (~14.8k), `slouch-vision` (~2.8k, ort/ONNX),
  `slouch-store` (~7.1k, SQLite/archive/model-format). Settings + model-format v1 live here.
- The actors (`src-tauri/src/actors.rs`, ~2.2k LOC): Inference/Training/**CameraActor**. **Only 1 line of
  Tauri coupling in the whole file** — `use tauri::ipc::Channel` for `TrainingActor` progress; swap for a
  generic `Fn(TrainingEvent)` callback. `CameraActor` already uses a generic sink → no change.
  Extract these (+ `errors.rs`, `inference_cache.rs`, `ipc_validation.rs`) from the `app` bin into a lib.

**REWRITE — the ~13k-LOC Svelte frontend** (`src-svelte/`, 26 `.svelte` + 74 `.ts`; +~6.8k test LOC
discarded). Much is form plumbing (sliders/tabs/settings/modals) that shrinks a lot in immediate mode.
Real effort concentrates on: the **dataset thumbnail grid** (virtualized + texture caching + drag/hover),
the **live camera viewport + overlay**, and **notifications/sound**.

**DISCARD — Tauri/webview glue:** `api.rs` command wrappers (keep the bodies as direct calls), `lib.rs`
Tauri Builder + the `slouchcam` URI protocol + focus `on_window_event` + Wave-2 `start_camera`/`stop_camera`/
`list_cameras` command wrappers, `bindings.rs`/Specta + `src/generated/*`, the 3 raw-byte commands + the
MessagePack IPC, `tauri.conf*`, capabilities, wdio plugins.

**REPLACE — Tauri plugins with native crates:** global-shortcut → `global-hotkey` 0.8 (Carbon on macOS =
no accessibility prompt); dialog → `rfd` 0.17 (Tauri's dialog plugin wraps rfd); log → `env_logger`/`fern`;
bundling → **`cargo-packager`** (supports `infoPlistPath` for macOS `NSCameraUsageDescription` + bundle id
+ signing/notarization; emits NSIS/MSI + .app/.dmg); notifications → `notify-rust` (needs a Windows
AppUserModelID + a macOS `.app` bundle id — identity Tauri provided for free).

## Cross-platform note
`nokhwa input-native` covers mac + win (+ Linux). macOS needs `NSCameraUsageDescription` + a bundled
`.app` (bare CLI can't get a stable TCC identity) + `nokhwa_initialize` to trigger the permission flow.

## Risks to validate BEFORE committing to a native migration
1. egui macOS background-CPU (#7059) over a multi-hour minimized/occluded run.
2. wgpu blank-window/crash on the target GPU/driver spread + RDP (ship a `glow` fallback if it bites).
3. Idle/minimized/occluded CPU+GPU on both OSes (wgpu-layer reality, not framework-specific).
4. No published 30fps-720p benchmark exists — smoothness is strong inference (720p RGBA ≈105 MB/s upload,
   trivial) + video-player precedents (Rerun, iced_video_player), not a measured citation.

## Bottom line
If/when eliminating the webview itself becomes a hard goal worth a ~13k-LOC UI rewrite: **go egui**, and it
*can* handle the complex UI (the dataset grid is its strength via `egui_dnd`/`egui_virtual_list`). Until
then, the hybrid (native camera + Svelte web UI) captures the real win (kills the camera permission prompt
and getUserMedia) at a fraction of the cost.

---

## Appendix: Svelte 5 vs Leptos (keep the webview, Rust frontend) — researched 2026-07-20

Separate axis from egui: **keep Tauri/webview but replace Svelte/TS with Leptos (Rust→WASM)** so the
frontend is also Rust. **Verdict: keep Svelte 5.** Leptos is the weakest of the three options for this app.

- **It's a half-measure.** It keeps every webview cost (RAM, WebView2, startup, the IPC boundary, all
  HTML/CSS/DOM work) and only removes the *language split*. If the goal is "all Rust + kill the webview/IPC
  pain," **egui removes all three (webview + IPC + two languages); Leptos removes only the third.**
- **The one real win (shared DTOs) is thin and ~70% true.** Frontend can `use slouch_domain::…` directly
  (struct duplication + Specta/TS codegen + the freshness gate disappear; the 3 MessagePack commands
  deserialize into the same Rust structs via `rmp_serde`). BUT `invoke("cmd_name", …)` stays **stringly-typed
  on the command name** across the WASM/backend compilation boundary — args/return get checked against shared
  DTOs (via `tauri-sys`/`tauri-wasm`), but you hand-write a thin wrapper per command and keep names in sync
  manually. You're trading an *already-automated one-command* binding step for hand-written invoke wrappers.
- **Costs vs Svelte:** Leptos is **pre-1.0 (0.8.x, Jun 2026)** with real churn each minor; **UI ecosystem is
  thin** (Thaw ~60 Fluent components is the only live lib; Leptonic stale) → build most controls yourself;
  **DOM-heavy interactive UI (drag/hover/buttons-in-cells) is ~1:1 with JS but more verbose** (web-sys casts,
  cloned-signal closures, manual listener cleanup) — the highest-friction area; **HMR regresses** (WASM
  recompiles ~5-10s vs Vite instant); **WASM debugging is worse**. WASM bundle (~1-2 MB) is a non-issue for a
  bundled desktop app that runs for hours. CSS ports over largely as-is. The `slouchcam://` stream is
  webview-level → identical, no advantage either way.
- **Migration:** full ~13k-LOC component rewrite in Rust, hard cutover (no incremental path), **~4-8 weeks
  solo**, front-loaded by the learning curve + hand-rolling missing components.
- **Recommendation:** keep Svelte (best velocity + mature ecosystem; the binding "cost" is already automated).
  Consider Leptos only if single-language Rust is valued for its own sake over iteration speed + UI ecosystem.
  For a true "all-Rust" endgame, egui is the more coherent target than Leptos.
- Grounding: Leptos ~375k downloads/mo, ~482 dependent crates, 96 contributors (real, but ~10x smaller than
  the JS frameworks). Sources: leptos-rs/leptos releases, book.leptos.dev, v2.tauri.app/start/frontend/leptos,
  thaw-ui/thaw, lpotthast/leptonic (stale), tauri-sys docs.
