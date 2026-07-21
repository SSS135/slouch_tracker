//! Classifier factory and legacy deserialization compatibility.
//!
//! Mechanical Rust port of `src/services/ml/classifierFactory.ts`. Creation
//! and loading deliberately delegate to the canonical registry so defaults,
//! parameter validation, and compatibility behavior cannot drift by entrypoint.

use std::collections::BTreeMap;

use slouch_domain::{ClassifierConfig, ClassifierLookupError, ParameterValue};

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::classifier_registry::{
    create_classifier as registry_create_classifier, load_classifier_state,
    normalize_registry_classifier_id,
};
use super::types::SerializedClassifierState;

/// Creates a classifier from the schema-driven classifier configuration.
pub fn create_classifier(
    config: ClassifierConfig,
) -> Result<Box<dyn BaseClassifier>, ClassifierError> {
    registry_create_classifier(&config)
}

/// Deserializes a classifier state from a current ID or legacy class name.
///
/// `_params` remains intentionally ignored, matching the source factory: the
/// registry loader reconstructs its compatibility parameters from state.
pub fn deserialize_classifier(
    classifier_id: &str,
    state: SerializedClassifierState,
    _params: &BTreeMap<String, ParameterValue>,
) -> Result<Box<dyn BaseClassifier>, ClassifierError> {
    let normalized = normalize_registry_classifier_id(classifier_id).map_err(|error| {
        ClassifierError::InvalidState(match error {
            ClassifierLookupError::DeprecatedLogisticRegression => error.to_string(),
            ClassifierLookupError::Unknown(_) => {
                format!("Unknown classifier ID: {classifier_id}")
            }
        })
    })?;

    load_classifier_state(normalized, state)
}
