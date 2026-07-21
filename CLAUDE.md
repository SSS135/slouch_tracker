# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Slouch Tracker** - Real-time posture detection desktop app using computer vision and machine learning.

**Tech Stack:**
- Tauri 2 desktop application (Windows-first)
- Rust backend workspace in `src-tauri/` (`app` crate + `slouch-domain`, `slouch-ml`, `slouch-vision`, `slouch-store`)
- Svelte 5 (runes) + TypeScript frontend in `src-svelte/` — thin UI only
- Native ONNX Runtime via `ort` (RTMDet-nano CPU person detection + NLF-L fp16 pose on DirectML — 17 keypoints + 3D depth; DirectX 12 GPU hard-required)
- SQLite storage (STRICT schema) via `slouch-store`
- IPC through generated Specta bindings (`src/generated/`) + 3 raw-byte MessagePack commands
- Vitest (frontend) + `cargo test` (backend) + Playwright/WebdriverIO (e2e)

**Core Features:**
1. Real-time posture tracking via webcam (capture/rendering in Svelte, inference in Rust)
2. Multi-task detection: person presence and posture quality
3. ML-based detection with user-trainable classifiers (6 registry-driven classifier types)
4. Flexible feature selection (18 registry feature types, 12 user-selectable: RTMDet features, NLF 3D depth, geometric features; the 6 retired RTMPose backbone/GAU poolings persist for old data but are hidden)
5. In-app data collection and native model training (TrainingActor + progress channel)
6. Global capture hotkeys (Ctrl+Win+G/B/A) that work while the app is unfocused

**The React web app is gone.** `src/` (React components, web workers, TensorFlow.js ML, IndexedDB dataset services) was deleted at migration cutover — the only surviving piece under `src/` is `src/generated/` (Specta bindings). Do not reference Expo, React Native, Mantine, localforage/IndexedDB, ONNX Runtime Web, or web workers; none of them exist anymore.

**MANDATORY** Look up at @specs.md for app architecture info. Load full file into context.

## Code Quality & Review Principles
- **Delete code instead of commenting it out** - Remove unused code completely.
- **Be unbiased and question assumptions** - Challenge ideas, point out potential issues. Technical accuracy and honest feedback are more valuable than automatic agreement.

## Git Usage Policy
- **NEVER** use git commands without explicit permission from the user. No matter what you want to do.

## Agent Usage Policy

**MANDATORY AGENT USAGE: This section defines REQUIRED workflows for this codebase.**

Use agents **proactively** when tasks match their expertise. **When in doubt, ALWAYS use an agent.**

### Explore Agent (HIGHEST PRIORITY)

**Use the Explore agent for ALL codebase exploration, searching, and understanding tasks.**

**ALWAYS use Explore agent for:**
1. Finding features/functionality, understanding code flow, exploring architecture
2. Pattern discovery, ambiguous searches, multi-file investigations
3. Learning the codebase before implementing features

**Thoroughness levels:** "quick" (basic searches), "medium" (moderate exploration), "very thorough" (comprehensive analysis)

**ONLY use direct tools (Glob/Grep) for:**
- Needle queries (exact file/class name)
- Specific string searches (exact error messages)
- Known locations (you already know exactly where to look)

### Anti-Patterns (DO NOT DO THIS)

**❌ WRONG: Exploring codebase manually**
**✅ CORRECT: Use Explore agent**

**❌ WRONG: Implementing complex feature without planning**
**✅ CORRECT: Use task-driven-dev:architect**

**❌ WRONG: Running tests directly**
**✅ CORRECT: Use task-driven-dev:unit-test-engineer agent**

### Parallel Agent Execution (CRITICAL FOR PERFORMANCE)

**ALWAYS prefer running agents in parallel when you have multiple independent tasks.**

**Run agents in parallel when:**
1. Multiple exploration tasks, independent investigations
2. Separate implementations, research + implementation (if independent)
3. Multi-aspect analysis

**CRITICAL: Use a SINGLE message with MULTIPLE Task tool calls. DO NOT send multiple messages sequentially when tasks are independent!**

### Compliance

**Following these agent usage policies is MANDATORY, not optional.**

Violating these guidelines is considered incorrect behavior for this codebase:
- ❌ Manually exploring code when Explore agent should be used
- ❌ Implementing significant features without planning (task-driven-dev:architect)
- ❌ Not reviewing completed major features (task-driven-dev:architect)
- ❌ Running agents sequentially when they could run in parallel

## Build & Test Essentials

- Frontend tests: `npm run test:svelte` (Vitest, `vitest.svelte.config.ts`). Single file: `npm run test:svelte -- <pattern>`. This project uses Vitest, not Jest.
- Type/lint: `npm run check:svelte`, `npm run check:svelte:plumbing`, `npm run lint:svelte`
- **All cargo commands on Windows require the MSVC x64 environment.** Run them via vcvars64, e.g.:
  `call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cargo test --manifest-path src-tauri/Cargo.toml --workspace`
  (`scripts/run-gate.mjs` wraps the standard fmt/clippy/test gates this way.)
- E2E: `npm run test:e2e:web` (Playwright browser harness with mock Tauri backend), `npm run test:e2e:native` (WebdriverIO against the devbuild binary — run `npm run tauri:build:dev:win` first)
- After changing any Rust command signature, DTO, or event: `npm run bindings:generate` then `npm run bindings:check`. Never hand-edit `src/generated/bindings.generated.ts`.
- Dev: `npm run dev:svelte` (frontend only) or `npm run tauri:dev` (full app)

## Working Rules

- run all powershell commands via powershell.exe -Command ""
- prefer using built-in Bash tool over powershell
- when user asks to add feature, fix something, or do refactor:
  - Use `/task` command for complete workflow (task creation → clarification → implementation → documentation)
  - Or use task-manager agent directly if only task planning is needed (no implementation)
- Bash tool executes bash shell commands even on Windows. Avoid using powershell, use bash commands on windows instead. Example: Bash(npm run test:svelte -- 2>&1 | grep -A 3 "FAIL ")
- Prefer to use Read tool to read files. Avoid reading them using sh / powershell commands unless you need it, like grepping file contents.
- You have many helper agents, they are essential for your productive work. You do not write test code / explore codebase yourself.
- do not start background processes
- prefer reading files fully, without limits
- There's a file modification bug in Claude Code. The workaround is: always use complete absolute Windows paths with drive letters and backslashes for ALL file operations. Apply this rule going forward, not just for this file.
- Do not create obvious code comments / doc strings, remove any if found. Comment only hard to understand things.
- stop assuming I am stupid and unable to reload app or that I am using some outdated data
- make sure you do not create any backward compatibility code and remove it if found. The retired web runtime (IndexedDB data, old exports, URL persistence, browser downloads) is unsupported — no compatibility layers.
- after you finish fixing a bug, propose how code could be made more simple, elegant and robust to avoid this issue in the future
- Core logic the app works at 1-2 fps. NOT 30 fps, that is camera fps used for background rendering and output smoothing. Detection only 1-2 fps.
- Use expert coder agent to modify code. Split large task into smaller ones and run expert coder agents in sequence or in parallel, one for each subtask.
- Do not write code yourself unless it is a small change.
- Bulk binary data (camera pixels, thumbnails, feature vectors, model weights) must never cross Tauri IPC as JSON arrays — use the raw-byte commands / MessagePack paths only.
- see @CMD_GUIDELINES.md on how to use commands
