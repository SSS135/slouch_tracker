import { mount } from 'svelte';
import '../app.css';
import MockTauriHarness from './MockTauriHarness.svelte';
import { installMockTauri } from './mockTauri';

installMockTauri();

const target = document.getElementById('root');
if (!target) {
  throw new Error('Mocked Tauri harness root element was not found.');
}

mount(MockTauriHarness, { target });
