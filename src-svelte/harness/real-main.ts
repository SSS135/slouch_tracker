import { mount } from 'svelte';
import '../app.css';
import App from '../App.svelte';
import { emitTrackingState, getHarnessMetrics, installMockTauri } from './mockTauri';

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

const target = document.getElementById('root');
if (!target) throw new Error('Real application harness root element was not found.');
mount(App, { target });
