//! Bounded debug logging for classifier predictions.
//!
//! The native logging policy intentionally exposes only diagnostic metadata and
//! classifier results. Feature samples, model weights, and per-feature
//! contributions never cross this logging boundary.

const DETECTION_CATEGORY: &str = "detection";

/// Bounded values emitted by the prediction diagnostics logger.
#[derive(Debug, Clone, PartialEq)]
pub enum DebugValue {
    Number(f64),
    Boolean(bool),
}

/// Logging boundary used by the native ML code.
pub trait PredictionDebugLogger {
    fn is_debug_enabled(&self, category: &str) -> bool;

    fn debug(&self, category: &str, message: &str, value: Option<DebugValue>);
}

/// Inputs to [`log_prediction_debug`].
#[derive(Debug, Clone, Copy)]
pub struct PredictionDebugParams<'a> {
    pub classifier_name: &'a str,
    pub features: &'a [f32],
    pub transformed_features: &'a [f32],
    pub dim_reduction_applied: bool,
    pub weights_data: Option<&'a [f32]>,
    pub bias_value: Option<f64>,
    pub prob_good: f64,
}

/// Logs bounded classifier prediction diagnostics when detection debug logging
/// is enabled.
///
/// The caller must provide the configured application logger. There is no
/// default or no-op logger here, so an enabled diagnostic path cannot silently
/// discard its output.
pub fn log_prediction_debug<L>(params: PredictionDebugParams<'_>, logger: &L)
where
    L: PredictionDebugLogger + ?Sized,
{
    if !logger.is_debug_enabled(DETECTION_CATEGORY) {
        return;
    }

    let prefix = format!("[{}_PREDICT_DEBUG] ", params.classifier_name);
    debug(
        logger,
        &format!("{prefix}========== PREDICT PROBA START =========="),
        None,
    );

    debug(
        logger,
        &format!("{prefix}Input features length:"),
        Some(DebugValue::Number(params.features.len() as f64)),
    );
    debug(
        logger,
        &format!("{prefix}Input has NaN:"),
        Some(DebugValue::Boolean(
            params.features.iter().any(|value| value.is_nan()),
        )),
    );
    debug(
        logger,
        &format!("{prefix}Input has Inf:"),
        Some(DebugValue::Boolean(
            params.features.iter().any(|value| !value.is_finite()),
        )),
    );

    if params.dim_reduction_applied {
        debug(
            logger,
            &format!("{prefix}Dimensionality reduction was applied"),
            None,
        );
        debug(
            logger,
            &format!("{prefix}After reduction length:"),
            Some(DebugValue::Number(params.transformed_features.len() as f64)),
        );
        debug(
            logger,
            &format!("{prefix}After reduction has NaN:"),
            Some(DebugValue::Boolean(
                params
                    .transformed_features
                    .iter()
                    .any(|value| value.is_nan()),
            )),
        );
        debug(
            logger,
            &format!("{prefix}After reduction has Inf:"),
            Some(DebugValue::Boolean(
                params
                    .transformed_features
                    .iter()
                    .any(|value| !value.is_finite()),
            )),
        );
    }

    if let (Some(weights), Some(bias)) = (params.weights_data, params.bias_value) {
        debug(
            logger,
            &format!("{prefix}Weights length:"),
            Some(DebugValue::Number(weights.len() as f64)),
        );
        debug(
            logger,
            &format!("{prefix}Weights has NaN:"),
            Some(DebugValue::Boolean(
                weights.iter().any(|value| value.is_nan()),
            )),
        );
        debug(
            logger,
            &format!("{prefix}Weights has Inf:"),
            Some(DebugValue::Boolean(
                weights.iter().any(|value| !value.is_finite()),
            )),
        );
        debug(
            logger,
            &format!("{prefix}Bias is NaN:"),
            Some(DebugValue::Boolean(bias.is_nan())),
        );
        debug(
            logger,
            &format!("{prefix}Bias is Inf:"),
            Some(DebugValue::Boolean(!bias.is_finite())),
        );

        // Preserve the source decision arithmetic while keeping all raw model
        // values and per-feature contributions out of the log.
        let mut decision = bias;
        for (index, &value) in params.transformed_features.iter().enumerate() {
            let Some(&raw_weight) = weights.get(index) else {
                break;
            };
            let weight = if raw_weight == 0.0 || raw_weight.is_nan() {
                0.0
            } else {
                f64::from(raw_weight)
            };
            decision += f64::from(value) * weight;
        }

        debug(
            logger,
            &format!("{prefix}Decision is NaN:"),
            Some(DebugValue::Boolean(decision.is_nan())),
        );
        debug(
            logger,
            &format!("{prefix}Decision is Inf:"),
            Some(DebugValue::Boolean(!decision.is_finite())),
        );
    }

    debug(
        logger,
        &format!("{prefix}Final probability: good ="),
        Some(DebugValue::Number(params.prob_good)),
    );
    debug(
        logger,
        &format!("{prefix}probGood is NaN:"),
        Some(DebugValue::Boolean(params.prob_good.is_nan())),
    );
    debug(
        logger,
        &format!("{prefix}========== PREDICT PROBA END =========="),
        None,
    );
}

fn debug<L>(logger: &L, message: &str, value: Option<DebugValue>)
where
    L: PredictionDebugLogger + ?Sized,
{
    logger.debug(DETECTION_CATEGORY, message, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct TestLogger {
        enabled: bool,
        entries: RefCell<Vec<(String, String, Option<DebugValue>)>>,
    }

    impl PredictionDebugLogger for TestLogger {
        fn is_debug_enabled(&self, category: &str) -> bool {
            self.enabled && category == DETECTION_CATEGORY
        }

        fn debug(&self, category: &str, message: &str, value: Option<DebugValue>) {
            self.entries
                .borrow_mut()
                .push((category.to_owned(), message.to_owned(), value));
        }
    }

    #[test]
    fn skips_all_work_when_detection_debug_is_disabled() {
        let logger = TestLogger::default();
        log_prediction_debug(
            PredictionDebugParams {
                classifier_name: "LogisticRegression",
                features: &[1.0],
                transformed_features: &[2.0],
                dim_reduction_applied: false,
                weights_data: Some(&[3.0_f32]),
                bias_value: Some(0.5),
                prob_good: 0.9,
            },
            &logger,
        );

        assert!(logger.entries.borrow().is_empty());
    }

    #[test]
    fn logs_bounded_metadata_without_raw_model_or_feature_values() {
        let logger = TestLogger {
            enabled: true,
            ..TestLogger::default()
        };
        log_prediction_debug(
            PredictionDebugParams {
                classifier_name: "LogisticRegression",
                features: &[1.2345679_f32, f32::NAN, f32::INFINITY],
                transformed_features: &[2.0, -1.0],
                dim_reduction_applied: true,
                weights_data: Some(&[3.0_f32, f32::NEG_INFINITY]),
                bias_value: Some(f64::NAN),
                prob_good: 0.9,
            },
            &logger,
        );

        let entries = logger.entries.borrow();
        assert!(entries.iter().any(|(_, message, value)| {
            message.ends_with("Input has NaN:") && *value == Some(DebugValue::Boolean(true))
        }));
        assert!(entries.iter().any(|(_, message, value)| {
            message.ends_with("Weights has Inf:") && *value == Some(DebugValue::Boolean(true))
        }));
        assert!(entries.iter().any(|(_, message, value)| {
            message.ends_with("Decision is NaN:") && *value == Some(DebugValue::Boolean(true))
        }));
        assert!(entries
            .iter()
            .all(|(_, message, _)| !message.contains("first 5")
                && !message.ends_with("Input min:")
                && !message.ends_with("Input max:")
                && !message.ends_with("Weights min:")
                && !message.ends_with("Weights max:")
                && !message.ends_with("Bias:")));
        assert!(entries.iter().all(|(_, _, value)| {
            matches!(
                value,
                None | Some(DebugValue::Number(_)) | Some(DebugValue::Boolean(_))
            )
        }));
    }
}
