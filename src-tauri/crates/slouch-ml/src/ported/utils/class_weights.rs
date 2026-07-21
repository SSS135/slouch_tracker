//! Balanced class-weight computation.
//!
//! This is the Rust equivalent of
//! `src/services/ml/utils/classWeights.ts`. Class labels remain signed
//! integers at the boundary, while the returned weights use `f64` to match
//! TypeScript's `number` arithmetic.

use std::fmt;

const TRAINING_CATEGORY: &str = "training";
/// Prevents a sparse, attacker-controlled class index from driving an
/// unbounded dense allocation while retaining the general class-index
/// behavior used by the TypeScript helper.
pub const MAX_CLASS_COUNT: usize = 1_000_000;

/// Errors raised for labels that cannot represent a class distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassWeightsError {
    EmptyLabels,
    NegativeLabel {
        index: usize,
        label: i32,
    },
    ClassCountTooLarge {
        maximum_label: i32,
        class_count: usize,
        max_class_count: usize,
    },
    AllocationFailed {
        class_count: usize,
    },
}

impl fmt::Display for ClassWeightsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLabels => formatter.write_str("labels must contain at least one value"),
            Self::NegativeLabel { index, label } => write!(
                formatter,
                "label at index {index} is negative ({label}); labels must be nonnegative"
            ),
            Self::ClassCountTooLarge {
                maximum_label,
                class_count,
                max_class_count,
            } => write!(
                formatter,
                "label {maximum_label} requires {class_count} classes; the maximum is {max_class_count}"
            ),
            Self::AllocationFailed { class_count } => write!(
                formatter,
                "could not allocate class counts for {class_count} classes"
            ),
        }
    }
}

impl std::error::Error for ClassWeightsError {}

/// Logging boundary for native training diagnostics.
///
/// The application can adapt its training logger to this trait without
/// making the ML crate depend on Tauri or a particular logging backend.
pub trait ClassWeightsLogger {
    fn is_info_enabled(&self, category: &str) -> bool;

    fn info(&self, category: &str, message: &str);
}

/// Default logger for callers that do not need training diagnostics.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopClassWeightsLogger;

impl ClassWeightsLogger for NoopClassWeightsLogger {
    fn is_info_enabled(&self, _category: &str) -> bool {
        false
    }

    fn info(&self, _category: &str, _message: &str) {}
}

/// Computes balanced class weights using the scikit-learn formula:
/// `n_samples / (n_classes * class_count)`.
///
/// Class indices are zero-based. As in the source implementation, classes
/// between zero and the maximum observed label are included; an unobserved
/// class consequently receives an infinite weight. This convenience path
/// uses [`NoopClassWeightsLogger`].
pub fn compute_balanced_class_weights(labels: &[i32]) -> Result<Vec<f64>, ClassWeightsError> {
    compute_balanced_class_weights_with_logger(labels, &NoopClassWeightsLogger)
}

/// Computes balanced class weights and emits the source-compatible training
/// diagnostics through `logger` when training info logging is enabled.
pub fn compute_balanced_class_weights_with_logger<L>(
    labels: &[i32],
    logger: &L,
) -> Result<Vec<f64>, ClassWeightsError>
where
    L: ClassWeightsLogger,
{
    let n_samples = labels.len();
    let Some(&maximum_label) = labels.iter().max() else {
        return Err(ClassWeightsError::EmptyLabels);
    };

    for (index, &label) in labels.iter().enumerate() {
        if label < 0 {
            return Err(ClassWeightsError::NegativeLabel { index, label });
        }
    }

    let class_count_i64 = i64::from(maximum_label) + 1;
    if class_count_i64 > MAX_CLASS_COUNT as i64 {
        return Err(ClassWeightsError::ClassCountTooLarge {
            maximum_label,
            class_count: usize::try_from(class_count_i64).unwrap_or(usize::MAX),
            max_class_count: MAX_CLASS_COUNT,
        });
    }
    let n_classes =
        usize::try_from(class_count_i64).map_err(|_| ClassWeightsError::ClassCountTooLarge {
            maximum_label,
            class_count: usize::MAX,
            max_class_count: MAX_CLASS_COUNT,
        })?;

    let mut class_counts = Vec::new();
    class_counts
        .try_reserve_exact(n_classes)
        .map_err(|_| ClassWeightsError::AllocationFailed {
            class_count: n_classes,
        })?;
    class_counts.resize(n_classes, 0_usize);
    for &label in labels {
        class_counts[label as usize] += 1;
    }

    let mut weights = Vec::new();
    weights
        .try_reserve_exact(n_classes)
        .map_err(|_| ClassWeightsError::AllocationFailed {
            class_count: n_classes,
        })?;
    for &count in &class_counts {
        weights.push(n_samples as f64 / (n_classes as f64 * count as f64));
    }

    if logger.is_info_enabled(TRAINING_CATEGORY) {
        let weights_message = weights
            .iter()
            .enumerate()
            .map(|(index, weight)| format!("class_{index}={}", format_weight(*weight)))
            .collect::<Vec<_>>()
            .join(", ");
        logger.info(
            TRAINING_CATEGORY,
            &format!("[CLASS_WEIGHTS] Computed class weights: {weights_message}"),
        );

        let distribution_message = class_counts
            .iter()
            .enumerate()
            .map(|(index, count)| format!("class_{index}={count}"))
            .collect::<Vec<_>>()
            .join(", ");
        logger.info(
            TRAINING_CATEGORY,
            &format!("[CLASS_WEIGHTS] Class distribution: {distribution_message}"),
        );
    }

    Ok(weights)
}

fn format_weight(weight: f64) -> String {
    if weight.is_nan() {
        "NaN".to_owned()
    } else if weight.is_infinite() {
        if weight.is_sign_negative() {
            "-Infinity".to_owned()
        } else {
            "Infinity".to_owned()
        }
    } else {
        // Match JS Number.prototype.toFixed(3): decimal ties resolve to the larger
        // candidate (round half away from zero for non-negative values), whereas Rust's
        // `{:.3}` rounds half to even. Weights here are always non-negative.
        let scaled = (weight * 1000.0).round();
        let magnitude = scaled.abs() as u64;
        let int_part = magnitude / 1000;
        let frac_part = magnitude % 1000;
        let sign = if scaled.is_sign_negative() && scaled != 0.0 {
            "-"
        } else {
            ""
        };
        format!("{sign}{int_part}.{frac_part:03}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct RecordingLogger {
        enabled: bool,
        messages: RefCell<Vec<(String, String)>>,
    }

    impl ClassWeightsLogger for RecordingLogger {
        fn is_info_enabled(&self, category: &str) -> bool {
            self.enabled && category == TRAINING_CATEGORY
        }

        fn info(&self, category: &str, message: &str) {
            self.messages
                .borrow_mut()
                .push((category.to_owned(), message.to_owned()));
        }
    }

    #[test]
    fn preserves_balanced_weight_formula_for_binary_and_sparse_classes() {
        assert_eq!(
            compute_balanced_class_weights(&[0, 1]).expect("valid labels"),
            vec![1.0, 1.0]
        );
        assert_eq!(
            compute_balanced_class_weights(&[0, 0, 1]).expect("valid labels"),
            vec![0.75, 1.5]
        );

        let weights = compute_balanced_class_weights(&[0, 2]).expect("valid labels");
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0], 2.0 / 3.0);
        assert!(weights[1].is_infinite());
        assert_eq!(weights[2], 2.0 / 3.0);
    }

    #[test]
    fn rejects_overlarge_class_count_before_allocation() {
        assert_eq!(
            compute_balanced_class_weights(&[i32::MAX]),
            Err(ClassWeightsError::ClassCountTooLarge {
                maximum_label: i32::MAX,
                class_count: (i32::MAX as usize) + 1,
                max_class_count: MAX_CLASS_COUNT,
            })
        );
        assert!(matches!(
            compute_balanced_class_weights(&[0, MAX_CLASS_COUNT as i32]),
            Err(ClassWeightsError::ClassCountTooLarge { .. })
        ));
    }

    #[test]
    fn emits_source_compatible_training_messages_when_enabled() {
        let logger = RecordingLogger {
            enabled: true,
            ..RecordingLogger::default()
        };

        let weights = compute_balanced_class_weights_with_logger(&[0, 0, 0, 1], &logger)
            .expect("valid labels");

        assert_eq!(weights, vec![2.0 / 3.0, 2.0]);
        assert_eq!(
            logger.messages.borrow().as_slice(),
            [
                (
                    "training".to_owned(),
                    "[CLASS_WEIGHTS] Computed class weights: class_0=0.667, class_1=2.000"
                        .to_owned(),
                ),
                (
                    "training".to_owned(),
                    "[CLASS_WEIGHTS] Class distribution: class_0=3, class_1=1".to_owned(),
                ),
            ]
        );
    }

    #[test]
    fn matches_js_tofixed_round_half_up_for_exact_ties() {
        let logger = RecordingLogger {
            enabled: true,
            ..RecordingLogger::default()
        };

        let mut labels = vec![0i32; 8];
        labels.extend(std::iter::repeat_n(31i32, 8));

        compute_balanced_class_weights_with_logger(&labels, &logger).expect("valid labels");

        let messages = logger.messages.borrow();
        let computed = &messages[0].1;
        // weight[0] = 16 / (32 * 8) = 0.0625; JS toFixed(3) rounds this half-up to "0.063",
        // whereas Rust's `{:.3}` half-to-even would emit "0.062".
        assert!(
            computed.contains("class_0=0.063"),
            "expected class_0=0.063, got: {computed}"
        );
        assert!(computed.contains("class_31=0.063"));
    }

    #[test]
    fn disabled_logger_emits_nothing() {
        let logger = RecordingLogger::default();

        compute_balanced_class_weights_with_logger(&[0, 0, 0, 1], &logger).expect("valid labels");

        assert!(logger.messages.borrow().is_empty());
    }
}
