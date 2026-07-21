import { fireEvent, render, screen } from '@testing-library/svelte';
import { vi } from 'vitest';
import { logger } from '../../../services/logging/logger';
import Slider from '../../ui/Slider.svelte';

describe('Slider scale transformations', () => {
  it('maps exponential values to logarithmic range positions and back through production wiring', async () => {
    const onValueChange = vi.fn();
    const { container } = render(Slider, {
      props: {
        label: 'Regularization',
        value: 1,
        minimumValue: 0.001,
        maximumValue: 1000,
        step: 0.001,
        scale: 'exponential',
        editable: true,
        onValueChange,
      },
    });
    const input = container.querySelector('input[type="range"]') as HTMLInputElement;

    expect(Number(input.value)).toBeCloseTo(0.5, 5);
    expect(input.min).toBe('0');
    expect(input.max).toBe('1');
    expect(input.step).toBe('0.001');
    expect(screen.getByRole('button', { name: 'Regularization: 1.00' })).toBeInTheDocument();

    await fireEvent.input(input, { target: { value: String(4 / 6) } });
    expect(onValueChange).toHaveBeenLastCalledWith(expect.closeTo(10, 8));
  });

  it('clamps mount values outside the exponential bounds in both directions', () => {
    // Assert on the rendered formatted value (derived from the internal localValue
    // float via positionToValue), not just the range input.value which jsdom
    // sanitizes to [min,max]. This is what genuinely exercises the
    // Math.max(min, Math.min(max, value)) clamp in valueToPosition.
    const below = render(Slider, {
      props: {
        label: 'Below',
        value: 0.0001,
        minimumValue: 0.001,
        maximumValue: 1000,
        scale: 'exponential',
        editable: true,
        onValueChange: vi.fn(),
      },
    });
    const belowInput = below.container.querySelector('input[type="range"]') as HTMLInputElement;
    expect(Number(belowInput.value)).toBeCloseTo(0, 5);
    // Clamped to the lower bound 0.001, not the raw 0.0001.
    expect(screen.getByRole('button', { name: 'Below: 0.0010' })).toBeInTheDocument();
    below.unmount();

    const above = render(Slider, {
      props: {
        label: 'Above',
        value: 10_000,
        minimumValue: 0.001,
        maximumValue: 1000,
        scale: 'exponential',
        editable: true,
        onValueChange: vi.fn(),
      },
    });
    const aboveInput = above.container.querySelector('input[type="range"]') as HTMLInputElement;
    expect(Number(aboveInput.value)).toBeCloseTo(1, 5);
    // Clamped to the upper bound 1000, not the raw 10000.
    expect(screen.getByRole('button', { name: 'Above: 1000' })).toBeInTheDocument();
    above.unmount();
  });

  it('falls back to linear production mapping when exponential minimum is nonpositive', async () => {
    const warning = vi.spyOn(logger, 'warn');
    const onValueChange = vi.fn();
    const { container } = render(Slider, {
      props: {
        label: 'Fallback',
        value: 5,
        minimumValue: 0,
        maximumValue: 10,
        scale: 'exponential',
        onValueChange,
      },
    });
    const input = container.querySelector('input[type="range"]') as HTMLInputElement;
    expect(Number(input.value)).toBe(0.5);
    expect(warning).toHaveBeenCalledWith(
      'debug',
      'Exponential scale requires min > 0, falling back to linear',
    );

    await fireEvent.input(input, { target: { value: '0.75' } });
    expect(onValueChange).toHaveBeenLastCalledWith(7.5);
  });

  it('uses the production adaptive formatting at exponential magnitude boundaries', () => {
    const cases: Array<[number, string]> = [
      [0.001, '0.0010'],
      [0.005, '0.0050'],
      [0.01, '0.010'],
      [0.1, '0.100'],
      [0.999, '0.999'],
      [1, '1.00'],
      [9.99, '9.99'],
      [10, '10.0'],
      [50, '50.0'],
      [100, '100'],
      [1000, '1000'],
    ];
    for (const [value, formatted] of cases) {
      const { unmount } = render(Slider, {
        props: {
          label: `Value ${value}`,
          value,
          minimumValue: 0.001,
          maximumValue: 1000,
          scale: 'exponential',
          editable: true,
          onValueChange: vi.fn(),
        },
      });
      expect(screen.getByRole('button', { name: `Value ${value}: ${formatted}` })).toBeInTheDocument();
      unmount();
    }
  });

  it('preserves linear step formatting through production rendering', () => {
    render(Slider, {
      props: {
        label: 'Linear',
        value: 10.5,
        minimumValue: 0,
        maximumValue: 20,
        step: 0.1,
        editable: true,
        onValueChange: vi.fn(),
      },
    });
    expect(screen.getByRole('button', { name: 'Linear: 10.5000' })).toBeInTheDocument();
  });

  it('applies linear integer formatting through the toFixed(0) branch', () => {
    // step >= 1 (or the default step of 1) takes the toFixed(0) path, including
    // rounding (5.5 -> '6'). The step<1 case above covers only the toFixed(4) path.
    const cases: Array<[number, number | undefined, string]> = [
      [5.5, 1, '6'],
      [10, 1, '10'],
      [100, undefined, '100'],
    ];
    for (const [value, step, formatted] of cases) {
      const { unmount } = render(Slider, {
        props: {
          label: `Lin ${value}`,
          value,
          minimumValue: 0,
          maximumValue: 200,
          ...(step === undefined ? {} : { step }),
          editable: true,
          onValueChange: vi.fn(),
        },
      });
      expect(screen.getByRole('button', { name: `Lin ${value}: ${formatted}` })).toBeInTheDocument();
      unmount();
    }
  });

  it('formats linear zero and negative mount values without altering sign', () => {
    // Oracle 'edge cases': formatParamValue(0,'linear')='0', (-1,'linear')='-1'.
    // In linear mode displayValue = localValue = the untracked mount `value`, so a
    // zero/negative mount value flows straight to the toFixed(0) branch. (The
    // exponential zero/negative sign-preservation cases from the oracle are
    // structurally unreachable through the component: displayValue in exponential
    // mode is positionToValue -> Math.exp(...), which is always > 0, so no mount or
    // interaction can drive a nonpositive value into formatParamValue's exponential
    // branch. Those branches are already exercised by the positive cases above.)
    const cases: Array<[number, string]> = [
      [0, '0'],
      [-1, '-1'],
    ];
    for (const [value, formatted] of cases) {
      const { unmount } = render(Slider, {
        props: {
          label: `Edge ${value}`,
          value,
          minimumValue: -10,
          maximumValue: 10,
          editable: true,
          onValueChange: vi.fn(),
        },
      });
      expect(screen.getByRole('button', { name: `Edge ${value}: ${formatted}` })).toBeInTheDocument();
      unmount();
    }
  });
});
