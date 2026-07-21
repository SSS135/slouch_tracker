# Repository Guidelines

## Project Overview
- **Slouch Tracker** is a real-time posture tracking web app that trains user-specific classifiers to assess posture quality, hand-to-face proximity, and mouth-open cues.
- Tech stack: Expo React Native (web) with TypeScript and React 19, ONNX Runtime Web for inference, web workers for off-main-thread processing, and IndexedDB via `localforage` for storage.
- Core features include webcam capture, multi-task pose analysis, in-browser dataset labeling, and cross-platform targets (web + Electron). Always load `specs.md` for architecture, ML flows, and data contracts before making changes.

## Project Structure & Module Organization
- Expo Router entry lives in `app/`. Core logic is organized under `src/` (`components/`, `contexts/`, `hooks/`, `services/`, `utils/`, `workers/`) with mocks in `src/__tests__/`.
- Static assets (`assets/`, `public/`) and Electron scaffolding (`electron/`, `electron-builder.yml`) sit at the repo root. Worker builds output to `public/inference-worker.js`.

## Build, Test, and Development Commands
- `npm run start` boots Expo after bundling workers. Use `npm run web`, `npm run android`, or `npm run ios` for platform-specific targets (each rebuilds workers).
- Desktop flows: `npm run electron` for dev shell, `npm run electron:build` or `npm run electron:build:win` for packaged builds.
- Quality gates: `npm run lint` for ESLint (Expo config) and `npm test`, `npm run test:watch`, `npm run test:coverage` for Jest suites. Maintain meaningful coverage on inference and data flows.

## Coding Style & Code Quality
- TypeScript-first with functional components and hook-based state. Use two-space indentation, trailing commas, and explicit return types on exports.
- Delete dead code instead of commenting it out, and challenge assumptions during review—technical accuracy outweighs agreement.
- Run `npm run lint` before submitting and keep naming consistent: `PascalCase` components/hooks, `camelCase` utilities, tests mirroring source with `.test.ts(x)`.

## Testing Guidelines
- Jest with Testing Library (`@testing-library/react`, `@testing-library/user-event`) powers UI tests; mocks and helpers live in `src/__tests__/utils/`.
- Co-locate new tests under `src/__tests__`, prefer deterministic async handling (fake timers when applicable), and prioritize coverage around posture inference pipelines and dataset tooling.

## Commit & Pull Request Guidelines
- Follow the concise, imperative commit style seen in history (`support electron`, `fix memory leaks`). Never run git commands from this environment—coordinate with the repository owner.
- Pull requests must describe behavior, reference issues/spec sections, document platform impact, and attach UI or inference evidence (screenshots, logs). Confirm lint, tests, and worker builds locally; note any skipped step with rationale.

## Agent Workflow & Operational Constraints
- Use the Explore agent for all code exploration, architecture tracing, or ambiguous searches; avoiding it is non-compliant. Direct glob/grep is for exact matches only.
- For substantial changes, invoke `task-driven-dev:architect`; use unit-test agents to run test suites; schedule independent agent tasks in parallel and record findings.
- Prefer the built-in Bash shell (`bash -lc "..."`), avoid PowerShell, and never spawn background processes. When in doubt, escalate through agents rather than manual intervention.
