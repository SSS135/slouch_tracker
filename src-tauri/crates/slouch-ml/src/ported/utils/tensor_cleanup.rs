//! Tensor cleanup utilities for native model resources.

/// A tensor-like resource that can release its backing allocation explicitly.
pub trait DisposableTensor {
    fn dispose(&mut self);
}

impl DisposableTensor for Vec<f32> {
    fn dispose(&mut self) {
        drop(std::mem::take(self));
    }
}

/// Native equivalent of the TensorFlow.js model shape used by the source utility.
#[derive(Debug, PartialEq)]
pub struct ModelWeights<T = Vec<f32>> {
    pub weights: T,
    pub bias: T,
}

/// Logging boundary for tensor cleanup diagnostics.
pub trait TensorCleanupLogger {
    fn debug(&self, category: &str, message: &str);
}

/// Logger used by the source-compatible convenience function when no native
/// logger is supplied.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopTensorCleanupLogger;

impl TensorCleanupLogger for NoopTensorCleanupLogger {
    fn debug(&self, _category: &str, _message: &str) {}
}

/// Dispose both model variables when a model is present.
pub fn dispose_model_weights<T>(model: Option<&mut ModelWeights<T>>)
where
    T: DisposableTensor,
{
    dispose_model_weights_with_logger(model, &NoopTensorCleanupLogger);
}

/// Dispose both model variables and report the same diagnostic as the source
/// logger call.
pub fn dispose_model_weights_with_logger<T, L>(model: Option<&mut ModelWeights<T>>, logger: &L)
where
    T: DisposableTensor,
    L: TensorCleanupLogger,
{
    if let Some(model) = model {
        model.weights.dispose();
        model.bias.dispose();
        logger.debug(
            "training",
            "[TENSOR_CLEANUP] Disposed model weights and bias",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestLogger {
        entries: std::cell::RefCell<Vec<(String, String)>>,
    }

    impl TensorCleanupLogger for TestLogger {
        fn debug(&self, category: &str, message: &str) {
            self.entries
                .borrow_mut()
                .push((category.to_owned(), message.to_owned()));
        }
    }

    #[test]
    fn disposes_weights_and_bias_and_logs_once() {
        let mut model = ModelWeights {
            weights: vec![1.0, 2.0],
            bias: vec![3.0],
        };
        let logger = TestLogger::default();

        dispose_model_weights_with_logger(Some(&mut model), &logger);

        assert!(model.weights.is_empty());
        assert!(model.bias.is_empty());
        assert_eq!(
            logger.entries.into_inner(),
            vec![(
                "training".to_owned(),
                "[TENSOR_CLEANUP] Disposed model weights and bias".to_owned(),
            )]
        );
    }

    #[test]
    fn ignores_missing_model() {
        let logger = TestLogger::default();

        dispose_model_weights_with_logger::<Vec<f32>, _>(None, &logger);

        assert!(logger.entries.into_inner().is_empty());
    }
}
