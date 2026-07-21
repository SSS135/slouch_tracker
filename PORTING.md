# TypeScript to Rust Porting Guide

> **Status: port executed.** The clean cut is complete — the React/Expo web runtime and its TypeScript ML/dataset services were deleted and the Svelte + Rust implementation is primary. This document is retained for its conventions and numerical tolerances; the scope sections below describe the historical trial phase.

## Scope and rule

Mechanically preserve current working-tree behavior before Rust cleanup. The preparation/trial covers domain contracts and metadata plus `Keypoint`, RTMPose pooling, and Gaussian Naive Bayes. React, workers, IndexedDB, model loading, K-Means, Tauri commands, and UI callers remain unchanged.

Every port records its TypeScript source SHA-256 in `src-tauri/port-manifest.json`; golden fixtures repeat the hashes for their direct sources. Never regenerate a fixture after a source change without reviewing the behavioral delta and updating both locations.

## Ownership

| Rust crate | Owns |
|---|---|
| `slouch-domain` | DTOs, validation, labels, keypoints/bboxes, feature and classifier UI metadata |
| `slouch-ml` | Feature math, classifier algorithms, serialized model state |
| `slouch-vision` | Native ONNX sessions and image processing (post-trial) |
| `slouch-store` | SQLite and archive persistence (post-trial) |
| `app` | Tauri state, commands, events, actors, resource resolution (post-trial) |

Browser-only canvas/display and camera capture stay in TypeScript. The retired standalone web runtime, IndexedDB data, old exports, URL persistence, and browser downloads are unsupported in the final native application; no compatibility layer is planned.

## Mechanical conventions

- Keep TypeScript names/formulas and source iteration order until parity passes.
- Plain TS `number` domain values use `f64`; tensor/model arrays use `f32`; priors, logarithms, exponentials, and probabilities use `f64`. RTMPose mean and variance reductions accumulate in `f64` over `f32` inputs/intermediates, then cast outputs to `f32`, matching the executable TensorFlow.js CPU oracle on cancellation-sensitive lanes.
- Serde JSON field names match existing camelCase/snake_case wire shapes. Camera pixels, thumbnails, feature vectors, and weights must not cross IPC as JSON arrays in integration phases.
- Errors are typed `Result` values. Malformed tensor lengths and native boundary validation must never panic.
- Project code may not use `unsafe`, `todo!`, `unimplemented!`, broad `allow` attributes, placeholder tests, or skipped legacy behavior.
- Stored native feature vectors must match registry dimensions. This intentionally tightens the TypeScript guards/Zod schemas, which accept arbitrary lengths.
- Native thumbnails are bytes plus MIME metadata, not browser `Blob` objects. The TypeScript fast guard's duck-typed Blob exception is not a native contract.
- Boundary validation is stricter than the legacy fast guard: finite numbers, 17 keypoints, score/probability ranges, ordered bbox coordinates, consistent derived dimensions, positive timestamps, nonempty IDs, image MIME metadata, and registry-sized finite features.

## Numerical oracle

Use `abs(actual - expected) <= abs_tol || abs(actual - expected) <= rel_tol * abs(expected)`.

| Output | Acceptance |
|---|---|
| IDs, labels, shapes, error kinds | exact |
| Keypoint `f64`, raw `f32`, max pooling, serialized `f32` | IEEE bit exact for fixture values |
| RTMPose mean/std | absolute `1e-6` or relative `1e-6` |
| Gaussian-NB probability | absolute `1e-6` |
| Later normalization/centroids/projection | absolute `2e-6` or relative `2e-6` |
| Later trained iterative models | absolute `2e-4`; exact decisions away from threshold |
| ORT-Web WASM vs native CPU vision outputs | absolute `2e-4` or relative `2e-4`; exact branches away from thresholds |

The compatibility baseline pins ORT-Web `1.23.0` to `ort-wasm-simd-threaded.wasm` (11,815,498 bytes, SHA-256 `3260fcdb33b4fc4ec33e89caf392e13625823e01049d3bf32c38464f9dbfe14c`), the lock-resolved npm integrity `sha512-w0bvC2RwDxphOUFF8jFGZ/dYw+duaX20jM6V4BIZJPCfK4QuCpB/pVREV+hjYbT3x4hyfa2ZbTaWx4e1Vot0fQ==`, WASM EP only, one thread, and graph optimization `all`.

RTMPose backbone is `[1,768,8,6]` (NCHW), pooled over axes `[2,3]`. GAU is `[1,17,256]` (keypoint-major), pooled over axis `[1]`. Standard deviation is `sqrt(population_variance + 1e-6)`. GAU width is **256**; comments saying 384 are stale.

Gaussian-NB uses `0 = good`; every nonzero label enters class 1. Means and population variances accumulate through `f32` in source order, epsilon is added after division, one-sample variance is epsilon, log probabilities are divided by `sqrt(feature_count)`, and the result is `P(class 0)`.

## Known source inconsistencies (do not silently redesign)

- `ml/types.ts` says KNN uses euclidean/manhattan; runtime metadata uses cosine/RBF plus gamma.
- K-Means parameter interfaces omit `nClusters`, while implementations and metadata support it.
- K-Means Prototype lacks a dedicated behavior test.
- Factory tests omit Gaussian-NB and K-Means-Logistic legacy/load coverage.
- K-Means-Logistic loading restores `weightDecay` as `1.0` regardless of state.
- `specs.md` and comments contain RTMPose-S/512/384 and PCA-removal drift; executable code selects RTMPose-M/768/256 and still imports PCA.
- The actual test runner is Vitest, despite stale Jest wording.

## Workflow gates

Each shard follows implement → two adversarial reviews → fixer pass → verification → manifest update. Review one attacks numeric/source parity; review two attacks contracts, validation, panic paths, and dependency boundaries. Findings remain recorded under `src-tauri/reviews/` and must be closed before status becomes `verified`.

Preparation baseline: the focused Vitest suite below passed at exactly `8 files / 157 tests`; `tsc --noEmit` reported the existing 13 diagnostics captured in `.plans/tsc-baseline-current.txt`. Trial work must not change either frontend result.

Run:

```bash
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --workspace --all-targets --all-features -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml --workspace
npx tsx scripts/check-wave1-oracles.ts
npm run test:fast -- src/services/ml/__tests__/rtmposeFeatures.test.ts src/services/ml/__tests__/naiveBayesClassifier.test.ts src/services/ml/__tests__/kmeansLogisticClassifier.test.ts src/services/ml/__tests__/classifierFactory.test.ts src/services/validation/__tests__/schemas.test.ts src/services/dataset/__tests__/featureRegistry.test.ts src/services/validation/__tests__/guards.test.ts src/services/posture/__tests__/detection.test.ts
```
