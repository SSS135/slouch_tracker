import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

// Feature list in registry order (get_feature_registry order), so the selector's canonicalization
// has a meaningful ranking to sort against.
const makeFeature = (id: string, name: string) => ({
  id,
  name,
  description: `${name} description`,
  dimensions: 20,
  storageCost: 80,
  computed: true,
  modelType: null,
  userSelectable: true,
  requiresFitting: false,
});

vi.mock('../../../contexts/TrainingConfigContext', () => ({
  useTrainingConfig: () => ({
    features: [
      makeFeature('backbone_features_max', 'Backbone Max'),
      makeFeature('gau_features_max', 'GAU Max'),
      makeFeature('engineered_features', 'Engineered features'),
      makeFeature('torso_invariant', 'Torso Invariant'),
    ],
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

describe('FeatureMultiSelector canonical emission', () => {
  it('emits an ascending, unique posture list when a lower-index feature is added after a higher one', async () => {
    const onPostureChange = vi.fn();
    render(FeatureMultiSelector, {
      props: {
        // torso_invariant is the highest registry index; adding a lower-index feature next used to
        // append in click order and break the backend's ascending-order contract.
        postureSelected: ['torso_invariant'],
        presenceSelected: ['torso_invariant'],
        onPostureChange,
        onPresenceChange: vi.fn(),
      },
    });

    await fireEvent.click(screen.getByRole('checkbox', { name: 'Backbone Max for posture model' }));

    expect(onPostureChange).toHaveBeenCalledWith(['backbone_features_max', 'torso_invariant']);
  });

  it('keeps the remaining posture list canonical when a feature is removed', async () => {
    const onPostureChange = vi.fn();
    render(FeatureMultiSelector, {
      props: {
        postureSelected: ['backbone_features_max', 'gau_features_max', 'torso_invariant'],
        presenceSelected: ['torso_invariant'],
        onPostureChange,
        onPresenceChange: vi.fn(),
      },
    });

    await fireEvent.click(screen.getByRole('checkbox', { name: 'GAU Max for posture model' }));

    expect(onPostureChange).toHaveBeenCalledWith(['backbone_features_max', 'torso_invariant']);
  });
});
