import { mount } from 'svelte';
import '../app.css';
import App from '../App.svelte';
import { getHarnessMetrics, installMockTauri } from './mockTauri';

installMockTauri();

Object.defineProperty(window, '__SLOUCH_HARNESS_METRICS__', {
  configurable: true,
  value: getHarnessMetrics(),
});

const target = document.getElementById('root');
if (!target) throw new Error('Real application harness root element was not found.');
mount(App, { target });
