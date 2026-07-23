import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const tauriConfigPath = path.resolve(
  testDirectory,
  '../../src-tauri/tauri.conf.json',
);

const mainWindow = JSON.parse(readFileSync(tauriConfigPath, 'utf8')).app
  .windows[0];

describe('tauri window config invariants', () => {
  it('keeps dragDropEnabled disabled so in-page HTML5 drag-and-drop works on Windows', () => {
    expect(mainWindow.dragDropEnabled).toBe(false);
  });

  it('starts the window hidden so the tray-driven startup can show it without a white flash', () => {
    expect(mainWindow.visible).toBe(false);
  });
});
