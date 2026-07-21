import type { FeatureId } from '@generated/bindings';

// The native backend enforces a closed contract on training feature selections: each list must be
// unique and ascending in FeatureId registry order (feature.rs / get_feature_registry). That order
// feeds the training-config fingerprint, so the contract is intentionally strict and is rejected in
// three native places (save_training_settings validation, storage persist, model fingerprint). The
// UI, however, builds selections in click order, which can produce out-of-order or duplicate lists.
// This helper canonicalizes any selection into the exact form the backend accepts.
//
// `registryOrder` is the live id order from get_feature_registry (never hardcoded), so new features
// are ordered correctly the moment the registry ships them. Ids absent from the registry cannot be
// ranked or trained and are dropped. When the registry is unavailable (empty), the input is passed
// through unchanged rather than emptied — dropping every id would violate the non-empty contract and
// discard the user's selection; a later canonicalize once the registry loads will normalize it.
export function canonicalizeFeatureIds(
  ids: readonly FeatureId[],
  registryOrder: readonly FeatureId[],
): FeatureId[] {
  if (registryOrder.length === 0) return [...ids];

  const rank = new Map<FeatureId, number>();
  registryOrder.forEach((id, index) => {
    if (!rank.has(id)) rank.set(id, index);
  });

  const seen = new Set<FeatureId>();
  const kept: FeatureId[] = [];
  for (const id of ids) {
    if (rank.has(id) && !seen.has(id)) {
      seen.add(id);
      kept.push(id);
    }
  }
  kept.sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0));
  return kept;
}
