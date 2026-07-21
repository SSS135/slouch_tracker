import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import PostureStatusBadge from '../PostureStatusBadge.svelte';
import type { ClassificationResult } from '@/hooks/usePostureClassifier';

afterEach(() => {
  cleanup();
});

const createMockClassificationResult = (
  presentProbability: number,
  goodProbability: number | null,
): ClassificationResult => ({
  presentProbability,
  goodProbability,
});

type PostureStatusBadgeProps = {
  data?: ClassificationResult;
  hasModel: boolean;
  presenceThreshold?: number;
};

const renderBadge = (props: PostureStatusBadgeProps) =>
  render(PostureStatusBadge, { props });

describe('PostureStatusBadge Component', () => {
  describe('No Model State', () => {
    it('should display "No Model Trained" message when hasModel is false', () => {
      renderBadge({ hasModel: false });

      expect(screen.getByText('No Model Trained')).toBeInTheDocument();
      expect(
        screen.getByText('Train a classifier to enable posture scoring.'),
      ).toBeInTheDocument();
    });
  });

  describe('Posture-only deployment', () => {
    it('never shows "No Model Trained" once a posture model is active, even with no presence model', () => {
      // A posture-only pair (0 away frames -> presence training skipped) still has
      // hasModel true; presence probability comes from the RTMDet fallback. The
      // badge must present scoring, not the no-model state.
      const data = createMockClassificationResult(0.9, 0.8);

      renderBadge({ hasModel: true, data, presenceThreshold: 0.5 });

      expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument();
      expect(screen.getByText('Good Posture')).toBeInTheDocument();
    });
  });

  describe('Person Away State', () => {
    it('should display "Person Away" when person is not detected', () => {
      renderBadge({ hasModel: true, data: undefined });

      expect(screen.getByText('Person Away')).toBeInTheDocument();
    });

    it('should display "Person Away" when presentProbability is below threshold', () => {
      const data = createMockClassificationResult(0.3, 0.8);

      renderBadge({ hasModel: true, data, presenceThreshold: 0.5 });

      expect(screen.getByText('Person Away')).toBeInTheDocument();
    });

    it('should show present probability bar when data is provided', () => {
      const data = createMockClassificationResult(0.3, 0.8);

      renderBadge({ hasModel: true, data, presenceThreshold: 0.5 });

      expect(screen.getByText('Present')).toBeInTheDocument();
      expect(screen.getByText('30%')).toBeInTheDocument();
    });
  });

  describe('Good Posture State', () => {
    it('should display "Good Posture" when goodProbability exceeds threshold', () => {
      const data = createMockClassificationResult(0.9, 0.85);

      renderBadge({ hasModel: true, data, presenceThreshold: 0.5 });

      expect(screen.getByText('Good Posture')).toBeInTheDocument();
    });

    it('should display both good and present probability bars', () => {
      const data = createMockClassificationResult(0.9, 0.85);

      renderBadge({ hasModel: true, data });

      expect(screen.getByText('Good')).toBeInTheDocument();
      expect(screen.getByText('Present')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(screen.getByText('90%')).toBeInTheDocument();
    });
  });

  describe('Bad Posture State', () => {
    it('should display "Bad Posture" when goodProbability is below threshold', () => {
      const data = createMockClassificationResult(0.9, 0.3);

      renderBadge({ hasModel: true, data });

      expect(screen.getByText('Bad Posture')).toBeInTheDocument();
    });

    it('should display good probability percentage for bad posture', () => {
      const data = createMockClassificationResult(0.9, 0.3);

      renderBadge({ hasModel: true, data });

      expect(screen.getByText('Good')).toBeInTheDocument();
      expect(screen.getByText('30%')).toBeInTheDocument();
    });
  });

  describe('Threshold Behavior', () => {
    it('should use presenceThreshold to determine person away state', async () => {
      const data = createMockClassificationResult(0.6, 0.8);

      const view = renderBadge({
        hasModel: true,
        data,
        presenceThreshold: 0.7,
      });

      expect(screen.getByText('Person Away')).toBeInTheDocument();

      await view.rerender({
        hasModel: true,
        data,
        presenceThreshold: 0.5,
      });

      expect(screen.getByText('Good Posture')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle null goodProbability gracefully', () => {
      const data = createMockClassificationResult(0.9, null);

      renderBadge({ hasModel: true, data });

      expect(screen.getByText('0%')).toBeInTheDocument();
    });

    it('should handle very low presentProbability gracefully', () => {
      const data = createMockClassificationResult(0.0, 0.8);

      renderBadge({ hasModel: true, data, presenceThreshold: 0.5 });

      expect(screen.getByText('Person Away')).toBeInTheDocument();
    });

    it('should treat the exact posture threshold as good', () => {
      const data = createMockClassificationResult(0.5, 0.5);

      renderBadge({
        hasModel: true,
        data,
        presenceThreshold: 0.5,
      });

      expect(screen.getByText('Good Posture')).toBeInTheDocument();
    });

    it('should treat a value below the posture threshold as bad', () => {
      const data = createMockClassificationResult(0.5, 0.499);

      renderBadge({
        hasModel: true,
        data,
        presenceThreshold: 0.5,
      });

      expect(screen.getByText('Bad Posture')).toBeInTheDocument();
    });

    it('should clamp probability percentages to 0-100 range', () => {
      const data = createMockClassificationResult(1.2, -0.1);

      renderBadge({ hasModel: true, data });

      expect(screen.getByText('100%')).toBeInTheDocument();
      expect(screen.getByText('0%')).toBeInTheDocument();
    });
  });
});
