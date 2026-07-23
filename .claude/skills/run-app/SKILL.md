---
name: run-app
description: Launch and drive the Slouch Tracker Tauri 2 desktop app (webcam posture detector) on Windows. Use when asked to run/start/launch the app, screenshot it, or verify a change in the real app (not just tests).
---

# Running Slouch Tracker

Tauri 2 desktop app (Windows-first, MSVC): Svelte 5 frontend + Rust backend
(`ort`/ONNX Runtime, webcam). "Running" = launching the real window on the
physical display; it opens a WebView2 window and requests the webcam.

## Primary: full app (`tauri dev`)

Compiles the Rust backend and opens the app window with live detection.

**Two hard requirements / gotchas (both hit in practice):**
1. **Needs the MSVC x64 env.** `tauri dev` invokes `cargo`, which needs
   `vcvars64.bat` on PATH (for `link.exe`). Without it the Rust build fails.
2. **git-bash mangles Windows paths/`cmd` args.** Launching via
   `cmd //c 'cd /d <repo root> && call vcvars && npm run tauri:dev'` from git-bash
   fails (`/d` and paths get MSYS-translated → "path not found"). Run it from a
   native `cmd`/terminal, or via a `.bat` wrapper invoked by ABSOLUTE path.

**Human recipe** — in a normal `cmd` window (not git-bash):
```
cd /d <repo root>
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
npm run tauri:dev
```

**Programmatic launch (from the Bash tool / git-bash)** — write a `.bat` and run
it by absolute path in the background (it's a long-lived GUI process):
```
# launch_app.bat:
#   @echo off
#   cd /d <repo root>
#   call "C:\Program Files\...\VC\Auxiliary\Build\vcvars64.bat"
#   call npm run tauri:dev
cmd //c "C:\full\path\to\launch_app.bat"   # run_in_background: true
```
`tauri dev` auto-starts the Vite frontend (`beforeDevCommand`) — do NOT run
`dev:svelte` separately.

**Confirm launch from the log** (you can't screenshot the native window). Look for:
- `VITE v6.4.1 ready` + `Local: http://localhost:5174/`
- `Finished \`dev\`` … then `Running \`target\debug\app.exe\``
- `[inference][INFO] ... Both models loaded successfully` (RTMDet + RTMPose-M)

First build is slow; incremental is ~40s. **Benign warnings to ignore:** a
garbled `vswhere.exe` cp866 line during vcvars init, and
`Found version mismatched Tauri packages` (tauri crate vs `@tauri-apps/api`) —
both non-fatal.

## Alternate modes

- **UI only, no native build:** `npm run dev:svelte` → browser at
  `127.0.0.1:5174` with a MOCK Tauri backend (no real camera/inference). Good
  for frontend work; no cargo, no MSVC needed.
- **Devbuild binary + native e2e:** `npm run tauri:build:dev:win` (needs
  vcvars64) produces the devbuild exe, then `npm run test:e2e:native`
  (WebdriverIO) drives it headlessly.
- **Browser e2e (mock backend):** `npm run test:e2e:web` (Playwright).

## Driving it / what to check

Single window: camera viewport + slide-in control panel (Settings / Collect /
Training tabs). In-app keys: `G`/`B`/`A` capture good/bad/away, `C` clear, `U`
undo. To exercise the posture-quality work: **Training tab** → feature selector
has **"Posture Geometry (invariant)"**, normalization has **"Calibrated
(relative to good)"**; after `train_models` the CV card shows accuracy with a
**confidence interval**, balanced accuracy, and worst-fold.

## Stopping

Close the app window, or `Ctrl+C` in its terminal. If launched via the Bash
tool in the background, stop that background task (the window closes with the
process).

## Troubleshooting

- **Rust build link errors / `link.exe` not found** → vcvars64 wasn't loaded.
- **"path not found" launching from git-bash** → MSYS mangled the command; use
  the `.bat` wrapper or a native `cmd` window.
- **A stale absolute path from a previous checkout location in a build error** → leftover
  Tauri build-script cache from a project move. Fix with a targeted clean:
  `cargo clean -p tauri -p tauri-plugin-dialog -p tauri-plugin-fs -p tauri-plugin-log -p tauri-plugin-global-shortcut -p app` (avoids nuking the ~54 GB `target`).
