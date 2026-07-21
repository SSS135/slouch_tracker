import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, vi } from 'vitest';
import Slider from '../Slider.svelte';

afterEach(cleanup);

function valueText(container: HTMLElement): string {
  return container.querySelector('.slider-value')?.textContent?.trim() ?? '';
}

// Regression: the slider used to capture `value` once at mount and never
// re-sync. Because SettingsTab mounts while the native settings are still an
// all-zero placeholder, every slider froze on that placeholder (0% volume, 1s
// delay, 0s auto-capture) even after the real values loaded.
describe('Slider re-syncs when its bound value changes after mount', () => {
  it('reflects a continuous value that arrives after mount', async () => {
    const props = {
      label: 'Audio Volume',
      value: 0,
      minimumValue: 0,
      maximumValue: 100,
      step: 1,
      formatValue: (v: number) => `${Math.round(v)}%`,
      onValueChange: vi.fn(),
    };
    const { container, rerender } = render(Slider, { props });
    expect(valueText(container)).toBe('0%');

    await rerender({ ...props, value: 30 });

    expect(valueText(container)).toBe('30%');
    expect((container.querySelector('input[type="range"]') as HTMLInputElement).value).toBe('30');
  });

  it('reflects a fixed-values selection that arrives after mount', async () => {
    const props = {
      label: 'Alert Delay',
      value: 0,
      minimumValue: 1,
      maximumValue: 15,
      step: 1,
      fixedValues: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
      formatValue: (v: number) => `${v}s`,
      onValueChange: vi.fn(),
      showMinMax: true,
    };
    const { container, rerender } = render(Slider, { props });
    expect(valueText(container)).toBe('1s');

    await rerender({ ...props, value: 5 });

    expect(valueText(container)).toBe('5s');
  });

  it('does not clobber an in-progress interaction, then re-syncs after blur', async () => {
    const props = {
      label: 'Audio Volume',
      value: 0,
      minimumValue: 0,
      maximumValue: 100,
      step: 1,
      formatValue: (v: number) => `${Math.round(v)}%`,
      onValueChange: vi.fn(),
    };
    const { container, rerender } = render(Slider, { props });
    const range = container.querySelector('input[type="range"]') as HTMLInputElement;

    await fireEvent.focus(range);
    range.value = '70';
    await fireEvent.input(range);
    expect(valueText(container)).toBe('70%');

    // A prop echo arriving mid-interaction must not yank the thumb back.
    await rerender({ ...props, value: 10 });
    expect(valueText(container)).toBe('70%');

    // Once the interaction ends, the authoritative prop wins again.
    await fireEvent.blur(range);
    expect(valueText(container)).toBe('10%');
  });
});

describe('Slider Component', () => {
  describe('Basic Rendering', () => {
    it('should display help text when provided', () => {
      render(Slider, {
        props: {
          label: 'Test Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          onValueChange: () => {},
          helpText: 'This is helpful text',
        },
      });

      expect(screen.getByText('This is helpful text')).toBeInTheDocument();
    });
  });

  describe('Continuous Slider (no fixedValues)', () => {
    it('should work without fixedValues', () => {
      render(Slider, {
        props: {
          label: 'Continuous Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Continuous Slider')).toBeInTheDocument();
      expect(screen.getByDisplayValue('50')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty fixedValues array', () => {
      render(Slider, {
        props: {
          label: 'Test Slider',
          value: 50,
          minimumValue: 0,
          maximumValue: 100,
          step: 1,
          fixedValues: [],
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Test Slider')).toBeInTheDocument();
    });

    it('should handle single fixed value', () => {
      render(Slider, {
        props: {
          label: 'Single Value',
          value: 5,
          minimumValue: 1,
          maximumValue: 10,
          step: 1,
          fixedValues: [5],
          onValueChange: () => {},
        },
      });

      expect(screen.getByText('Single Value')).toBeInTheDocument();
    });
  });
});
