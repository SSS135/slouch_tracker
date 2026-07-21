import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import UndoButton from '../UndoButton.svelte';
import type { CaptureAction } from '@/services/dataset/types';
import { FrameLabel } from '@/services/dataset/types';

afterEach(() => {
  cleanup();
});

describe('UndoButton Component', () => {
  const createMockAction = (
    overrides: Partial<CaptureAction> = {},
  ): CaptureAction => ({
    frameId: 'frame-123',
    timestamp: Date.now(),
    label: FrameLabel.GOOD,
    thumbnailUrl: 'data:image/webp;base64,mockdata',
    ...overrides,
  });

  const mockOnUndo = vi.fn();

  beforeEach(() => {
    mockOnUndo.mockClear();
  });

  const renderButton = (props: {
    onUndo: () => void;
    canUndo: boolean;
    lastAction: CaptureAction | null;
  }) => render(UndoButton, { props });

  describe('rendering', () => {
    it('should be disabled when canUndo is false', () => {
      renderButton({ onUndo: mockOnUndo, canUndo: false, lastAction: null });

      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('should be enabled when canUndo is true', () => {
      const action = createMockAction();
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      expect(button).not.toBeDisabled();
    });
  });

  describe('click behavior', () => {
    it('should call onUndo when clicked while enabled', () => {
      const action = createMockAction();
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(mockOnUndo).toHaveBeenCalledTimes(1);
    });

    it('should not call onUndo when disabled', () => {
      renderButton({ onUndo: mockOnUndo, canUndo: false, lastAction: null });

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(mockOnUndo).not.toHaveBeenCalled();
    });
  });

  describe('popover content', () => {
    it('should show label in popover', async () => {
      const action = createMockAction({ label: FrameLabel.GOOD });
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      fireEvent.mouseEnter(button);

      await waitFor(() => {
        expect(screen.getByText(/Undo: Remove/i)).toBeInTheDocument();
        expect(screen.getByText('good')).toBeInTheDocument();
      });
    });

    it('should show thumbnail image in popover', async () => {
      const action = createMockAction({
        thumbnailUrl: 'data:image/webp;base64,test123',
      });
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      fireEvent.mouseEnter(button);

      await waitFor(() => {
        const image = screen.getByAltText('Undo preview');
        expect(image).toBeInTheDocument();
        expect(image).toHaveAttribute(
          'src',
          'data:image/webp;base64,test123',
        );
      });
    });
  });

  describe('edge cases', () => {
    it('opens an empty popover shell when canUndo is true with null lastAction', async () => {
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: null });

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'false');
      expect(button).not.toHaveAttribute('aria-describedby');
      await fireEvent.focus(button);
      const tooltip = screen.getByRole('tooltip');
      expect(button).toHaveAttribute('aria-expanded', 'true');
      expect(button).toHaveAttribute('aria-describedby', tooltip.id);
    });

    it('shows populated disclosure on focus and closes it on blur', async () => {
      const action = createMockAction({ label: FrameLabel.GOOD });
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      await fireEvent.focus(button);
      const tooltip = screen.getByRole('tooltip');
      expect(button).toHaveAttribute('aria-expanded', 'true');
      expect(button).toHaveAttribute('aria-describedby', tooltip.id);
      expect(screen.getByText(/Undo: Remove/i)).toBeInTheDocument();
      expect(screen.getByAltText('Undo preview')).toBeInTheDocument();

      await fireEvent.blur(button);
      expect(button).toHaveAttribute('aria-expanded', 'false');
      expect(button).not.toHaveAttribute('aria-describedby');
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });

    it('dismisses an open popover with Escape and an outside pointer', async () => {
      const action = createMockAction();
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const button = screen.getByRole('button');
      await fireEvent.focus(button);
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
      await fireEvent.keyDown(window, { key: 'Escape' });
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();

      await fireEvent.mouseEnter(button);
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
      await fireEvent.pointerDown(document.body);
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });

    it('uses instance-scoped tooltip ids', async () => {
      const action = createMockAction();
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });
      renderButton({ onUndo: mockOnUndo, canUndo: true, lastAction: action });

      const buttons = screen.getAllByRole('button');
      await fireEvent.focus(buttons[0]);
      const firstId = buttons[0].getAttribute('aria-describedby');
      await fireEvent.blur(buttons[0]);
      await fireEvent.focus(buttons[1]);
      const secondId = buttons[1].getAttribute('aria-describedby');
      expect(firstId).toBeTruthy();
      expect(secondId).toBeTruthy();
      expect(secondId).not.toBe(firstId);
    });

    it('should update when lastAction changes', async () => {
      const action1 = createMockAction({ label: FrameLabel.GOOD });
      const view = renderButton({
        onUndo: mockOnUndo,
        canUndo: true,
        lastAction: action1,
      });

      let button = screen.getByRole('button');
      fireEvent.mouseEnter(button);
      await waitFor(() => {
        expect(screen.getByText('good')).toBeInTheDocument();
      });

      fireEvent.mouseLeave(button);

      const action2 = createMockAction({ label: FrameLabel.BAD });
      await view.rerender({
        onUndo: mockOnUndo,
        canUndo: true,
        lastAction: action2,
      });

      button = screen.getByRole('button');
      fireEvent.mouseEnter(button);
      await waitFor(() => {
        expect(screen.getByText('bad')).toBeInTheDocument();
      });
    });
  });
});
