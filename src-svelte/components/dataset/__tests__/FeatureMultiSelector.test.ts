import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../contexts/TrainingConfigContext', () => ({
  useTrainingConfig: () => ({
    features: [{
      id: 'engineered_features',
      name: 'Engineered features',
      description: 'Geometry features',
      dimensions: 20,
      storageCost: 80,
      computed: true,
      modelType: null,
      userSelectable: true,
      requiresFitting: false,
    }],
  }),
}));

import FeatureMultiSelector from '../FeatureMultiSelector.svelte';

afterEach(cleanup);

describe('FeatureMultiSelector accessibility', () => {
  it('gives every model checkbox a feature-specific accessible name', () => {
    render(FeatureMultiSelector, {
      props: {
        postureSelected: ['engineered_features'],
        presenceSelected: ['engineered_features'],
        onPostureChange: vi.fn(),
        onPresenceChange: vi.fn(),
      },
    });
    expect(screen.getByRole('checkbox', { name: 'Engineered features for posture model' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Engineered features for presence model' })).toBeChecked();
    expect(screen.queryByRole('checkbox', { name: 'Posture' })).not.toBeInTheDocument();
  });
});
