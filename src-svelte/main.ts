import { mount } from 'svelte';
import App from './App.svelte';
import { applyThemeTokens } from './theme';
import './app.css';

const container = document.getElementById('root');

if (!container) {
  throw new Error('Failed to find root element for DOM bootstrap.');
}

applyThemeTokens();
mount(App, { target: container });
