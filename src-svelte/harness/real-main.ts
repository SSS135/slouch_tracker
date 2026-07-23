import { mount } from 'svelte';
import '../app.css';
import App from '../App.svelte';
import {
  emitPoseModelEvent,
  emitTrackingState,
  getHarnessMetrics,
  installMockTauri,
  setPoseModelDownloadRequired,
} from './mockTauri';

installMockTauri();

Object.defineProperty(window, '__SLOUCH_HARNESS_METRICS__', {
  configurable: true,
  value: getHarnessMetrics(),
});

// Lets Playwright simulate a native/tray-initiated pause/resume by firing the
// typed `tracking-state-changed` event through the mock bus, just as Rust would.
Object.defineProperty(window, '__SLOUCH_EMIT_TRACKING_STATE__', {
  configurable: true,
  value: (paused: boolean) => emitTrackingState(paused),
});

// Lets Playwright drive the first-run pose-model download: boot with
// `?poseModel=downloadRequired`, then step the scripted event sequence.
Object.defineProperty(window, '__SLOUCH_POSE_MODEL__', {
  configurable: true,
  value: {
    setDownloadRequired: () => setPoseModelDownloadRequired(),
    emit: (event: unknown) => emitPoseModelEvent(event as Parameters<typeof emitPoseModelEvent>[0]),
  },
});

const target = document.getElementById('root');
if (!target) throw new Error('Real application harness root element was not found.');
mount(App, { target });
