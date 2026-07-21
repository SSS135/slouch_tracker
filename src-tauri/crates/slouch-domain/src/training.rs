use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{ClassifierId, FeatureId, ParameterValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum DimensionalityReductionMethod {
    RandomProjection,
    Pca,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    None,
    Layer,
    ZScore,
    Calibrated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
pub struct DimensionalityReductionConfig {
    pub method: DimensionalityReductionMethod,
    #[specta(type = specta_typescript::Number)]
    pub components: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierConfig {
    pub classifier_id: ClassifierId,
    pub params: BTreeMap<String, ParameterValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainingSettings {
    pub classifier_config: ClassifierConfig,
    pub dim_reduction_config: DimensionalityReductionConfig,
    pub posture_feature_types: Vec<FeatureId>,
    pub presence_feature_types: Vec<FeatureId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_types: Option<Vec<FeatureId>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalization_mode: Option<NormalizationMode>,
    #[specta(type = specta_typescript::Number)]
    pub cv_folds: usize,
    pub last_updated: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainingMetrics {
    pub cv_accuracy: f64,
    pub cv_std: f64,
    pub mcc: f64,
    pub f1_score: f64,
    #[specta(type = Vec<Vec<specta_typescript::Number>>)]
    pub confusion_matrix: Vec<Vec<u64>>,
    pub fold_accuracies: Vec<f64>,
    pub balanced_accuracy: f64,
    pub accuracy_ci_low: f64,
    pub accuracy_ci_high: f64,
    pub worst_fold_accuracy: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cv_type: Option<CrossValidationType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
pub enum CrossValidationType {
    #[serde(rename = "temporal block")]
    TemporalBlock,
    #[serde(rename = "shuffled stratified")]
    ShuffledStratified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct TrainingResult {
    pub success: bool,
    pub metrics: TrainingMetrics,
    pub dim_reduction_method: DimensionalityReductionMethod,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}
