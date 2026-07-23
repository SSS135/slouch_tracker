import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import TrackingToggleButton from '../TrackingToggleButton.svelte';

afterEach(() => {
  cleanup();
});

describe('TrackingToggleButton', () => {
  const onToggle = vi.fn();

  beforeEach(() => {
    onToggle.mockClear();
  });

  const renderButton = (props: { paused: boolean; disabled?: boolean }) =>
    render(TrackingToggleButton, { props: { ...props, onToggle } });

  it('reads as "Pause" while active', () => {
    renderButton({ paused: false });
    const button = screen.getByRole('button');
    expect(button).toHaveAccessibleName('Pause tracking');
    expect(button).toHaveAttribute('aria-pressed', 'false');
    expect(button).toHaveTextContent('Pause');
  });

  it('reads as "Resume" while paused', () => {
    renderButton({ paused: true });
    const button = screen.getByRole('button');
    expect(button).toHaveAccessibleName('Resume tracking');
    expect(button).toHaveAttribute('aria-pressed', 'true');
    expect(button).toHaveTextContent('Resume');
  });

  it('is keyboard focusable (native button)', () => {
    renderButton({ paused: false });
    const button = screen.getByRole('button');
    button.focus();
    expect(button).toHaveFocus();
  });

  it('calls onToggle when clicked while enabled', () => {
    renderButton({ paused: false });
    fireEvent.click(screen.getByRole('button'));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it('does not call onToggle while disabled', () => {
    renderButton({ paused: false, disabled: true });
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onToggle).not.toHaveBeenCalled();
  });
});
