export const themeTokens = {
  color: {
    background: '#0a0a0a',
    surface: '#171717',
    surfaceRaised: '#1f1f1f',
    border: '#334155',
    text: '#f8fafc',
    textMuted: '#cbd5e1',
    primary: '#148fff',
    primaryHover: '#3da8ff',
    success: '#12b886',
    warning: '#fab005',
    danger: '#fa5252',
  },
  radius: {
    small: '0.375rem',
    medium: '0.5rem',
    large: '0.75rem',
  },
  fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
} as const;

export type ThemeTokens = typeof themeTokens;

/** Applies the canonical theme contract before the Svelte tree mounts. */
export function applyThemeTokens(root: HTMLElement = document.documentElement): void {
  const { color, radius, fontFamily } = themeTokens;
  const values: Record<string, string> = {
    '--color-background': color.background,
    '--color-surface': color.surface,
    '--color-surface-raised': color.surfaceRaised,
    '--color-border': color.border,
    '--color-text': color.text,
    '--color-text-muted': color.textMuted,
    '--color-primary': color.primary,
    '--color-primary-hover': color.primaryHover,
    '--color-success': color.success,
    '--color-warning': color.warning,
    '--color-danger': color.danger,
    '--radius-sm': radius.small,
    '--radius-md': radius.medium,
    '--radius-lg': radius.large,
    '--font-family': fontFamily,
  };
  for (const [name, value] of Object.entries(values)) root.style.setProperty(name, value);
}
