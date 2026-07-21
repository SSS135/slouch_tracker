use crate::FeatureMap;

/// Clones every canonical, registry-keyed feature vector without sharing its
/// backing allocation.
///
/// `Vec<f32>::clone` performs a deep copy, matching the TypeScript utility's
/// use of `new Float32Array(array)` before buffers are transferred to a worker.
/// The returned `BTreeMap<FeatureId, _>` intentionally uses the domain
/// registry's `FeatureId` order; native callers never observe an accidental
/// lexicographic ordering of string aliases.
pub fn clone_inference_features(features: &FeatureMap) -> FeatureMap {
    features.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeatureId;

    #[test]
    fn clones_canonical_feature_ids_and_owns_independent_vectors() {
        let mut original = FeatureMap::new();
        original.insert(FeatureId::RtmDetExtracted, vec![3.0; 384]);
        original.insert(FeatureId::BackboneFeatures, vec![1.0; 768]);

        let cloned = clone_inference_features(&original);
        assert_eq!(cloned, original);
        assert_eq!(
            cloned.keys().copied().collect::<Vec<_>>(),
            vec![FeatureId::BackboneFeatures, FeatureId::RtmDetExtracted]
        );
        assert_ne!(
            cloned[&FeatureId::BackboneFeatures].as_ptr(),
            original[&FeatureId::BackboneFeatures].as_ptr()
        );
    }
}
