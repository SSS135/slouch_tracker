/**
 * App-wide color constants and theme
 */

export const Colors = {
  light: {
    background: '#ffffff',
    text: '#000000',
    tint: '#3498db',
    tabIconDefault: '#9E9E9E',
    tabIconSelected: '#3498db',
  },
  dark: {
    background: '#000000',
    text: '#ffffff',
    tint: '#3498db',
    tabIconDefault: '#9E9E9E',
    tabIconSelected: '#3498db',
  },
  posture: {
    good: '#4CAF50',
    bad: '#FF5722',
    noPerson: '#9E9E9E',
    webviewBg: '#596e73',
  },
  // Posture badge colors (muted, less saturated)
  postureBadge: {
    good: 'green.7',
    bad: 'red.7',
    noModel: 'gray.7',
    personAway: 'blue.7',
  } as const,
  // UI component colors (dark theme)
  ui: {
    // Primary colors
    primary: '#007bff',
    secondary: '#6c757d',

    // Backgrounds (dark theme)
    background: '#0a0a0a',
    backgroundElevated: '#1a1a1a',
    backgroundInput: '#2a2a2a',

    // Borders
    border: '#333',
    borderHover: '#4A90E2',

    // Text
    text: '#ffffff',
    textMuted: '#999',
    textSubtle: '#666',

    // Slider-specific
    slider: {
      activeTrack: '#007bff',
      inactiveTrack: '#333',
      thumb: '#007bff',
    },

    // Status colors
    success: '#28a745',
    warning: '#ffc107',
    danger: '#dc3545',
    info: '#17a2b8',
  },
} as const;
