import { describe, expect, it } from 'vitest';
import type { FeatureId } from '@generated/bindings';
import { canonicalizeFeatureIds } from '../featureOrder';

// A representative slice of the real registry order (feature.rs / get_feature_registry). The helper
// must never hardcode this list; the tests supply it exactly as the live registry would.
const REGISTRY_ORDER: FeatureId[] = [
  'backbone_features',
  'backbone_features_max',
  'gau_features',
  'gau_features_max',
  'engineered_features',
  'keypoint_scores',
  'raw_keypoints',
  'posture_geometry',
  'torso_invariant',
];

describe('canonicalizeFeatureIds', () => {
  it('sorts click-order selections into ascending registry order', () => {
    expect(canonicalizeFeatureIds(['gau_features_max', 'backbone_features_max'], REGISTRY_ORDER)).toEqual([
      'backbone_features_max',
      'gau_features_max',
    ]);
  });

  it('produces an identical result for every click-order permutation of the same set', () => {
    const set: FeatureId[] = ['torso_invariant', 'backbone_features_max', 'gau_features_max', 'posture_geometry'];
    const canonical = ['backbone_features_max', 'gau_features_max', 'posture_geometry', 'torso_invariant'];
    const permutations: FeatureId[][] = [
      ['backbone_features_max', 'gau_features_max', 'posture_geometry', 'torso_invariant'],
      ['torso_invariant', 'posture_geometry', 'gau_features_max', 'backbone_features_max'],
      ['gau_features_max', 'torso_invariant', 'backbone_features_max', 'posture_geometry'],
      ['posture_geometry', 'backbone_features_max', 'torso_invariant', 'gau_features_max'],
    ];
    for (const permutation of permutations) {
      expect([...permutation].sort()).toEqual([...set].sort()); // same underlying set
      expect(canonicalizeFeatureIds(permutation, REGISTRY_ORDER)).toEqual(canonical);
    }
  });

  it('reproduces the reported break: appending torso_invariant then a lower-index feature stays canonical', () => {
    // The selector used to append in click order: [...selected, featureType]. Adding a feature with a
    // lower registry index than the current tail produced a non-ascending list the backend rejected.
    const clickOrder: FeatureId[] = ['torso_invariant', 'backbone_features_max'];
    expect(canonicalizeFeatureIds(clickOrder, REGISTRY_ORDER)).toEqual([
      'backbone_features_max',
      'torso_invariant',
    ]);
  });

  it('collapses duplicates to a single occurrence', () => {
    expect(
      canonicalizeFeatureIds(
        ['gau_features_max', 'backbone_features_max', 'gau_features_max', 'backbone_features_max'],
        REGISTRY_ORDER,
      ),
    ).toEqual(['backbone_features_max', 'gau_features_max']);
  });

  it('drops ids that are not present in the registry order', () => {
    expect(
      canonicalizeFeatureIds(
        ['gau_features_max', 'not_a_real_feature' as FeatureId, 'backbone_features_max'],
        REGISTRY_ORDER,
      ),
    ).toEqual(['backbone_features_max', 'gau_features_max']);
  });

  it('passes the input through unchanged when the registry order is unavailable', () => {
    // An empty registry means "cannot rank" — emptying the list would violate the non-empty contract
    // and discard the user's selection, so the input is returned verbatim (a fresh array).
    const input: FeatureId[] = ['gau_features_max', 'backbone_features_max'];
    const result = canonicalizeFeatureIds(input, []);
    expect(result).toEqual(input);
    expect(result).not.toBe(input);
  });

  it('returns a canonical list unchanged (idempotent)', () => {
    const canonical: FeatureId[] = ['backbone_features_max', 'gau_features_max', 'torso_invariant'];
    expect(canonicalizeFeatureIds(canonical, REGISTRY_ORDER)).toEqual(canonical);
    expect(canonicalizeFeatureIds(canonicalizeFeatureIds(canonical, REGISTRY_ORDER), REGISTRY_ORDER)).toEqual(canonical);
  });
});
