use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationResult {
    pub present_probability: f64,
    pub good_probability: Option<f64>,
}
