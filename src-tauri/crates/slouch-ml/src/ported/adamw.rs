//! AdamW optimizer with decoupled weight decay.
//!
//! This is the Rust equivalent of the TensorFlow.js AdamW optimizer. Model
//! values and optimizer moments remain `f32`, matching TensorFlow.js tensor
//! arithmetic, while configuration values use `f64` as ordinary TypeScript
//! numbers do elsewhere in the native core.

use std::fmt;

use serde::{Deserialize, Serialize};

const DEFAULT_LEARNING_RATE: f64 = 0.001;
const DEFAULT_BETA1: f64 = 0.9;
const DEFAULT_BETA2: f64 = 0.999;
const DEFAULT_EPSILON: f64 = 1e-7;
const DEFAULT_WEIGHT_DECAY: f64 = 0.0;

/// A named parameter or gradient tensor.
///
/// The native optimizer operates on flat tensors because shape handling is
/// owned by the model layer. The values are kept as `f32`, just like the
/// TensorFlow.js tensors used by the source implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedTensor {
    pub name: String,
    pub values: Vec<f32>,
}

impl NamedTensor {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

/// Serializable AdamW configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdamWConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: DEFAULT_BETA1,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_EPSILON,
            weight_decay: DEFAULT_WEIGHT_DECAY,
        }
    }
}

/// Errors raised by malformed optimizer inputs or state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdamWError {
    InvalidParameter(&'static str),
    EmptyName {
        index: usize,
    },
    GradientCountMismatch {
        variables: usize,
        gradients: usize,
    },
    NonFiniteValue {
        name: String,
        index: usize,
    },
    ShapeMismatch {
        name: String,
        parameter: usize,
        gradient: usize,
    },
    InvalidWeights(&'static str),
    InvalidIteration,
    Disposed,
}

impl fmt::Display for AdamWError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(formatter, "{name} must be finite and valid"),
            Self::EmptyName { index } => {
                write!(formatter, "variable at index {index} has an empty name")
            }
            Self::GradientCountMismatch {
                variables,
                gradients,
            } => write!(
                formatter,
                "gradient count {gradients} does not match variable count {variables}"
            ),
            Self::NonFiniteValue { name, index } => {
                write!(
                    formatter,
                    "tensor {name} value at index {index} is not finite"
                )
            }
            Self::ShapeMismatch {
                name,
                parameter,
                gradient,
            } => write!(
                formatter,
                "gradient for {name} has {gradient} values, expected {parameter}"
            ),
            Self::InvalidWeights(message) => {
                write!(formatter, "invalid optimizer weights: {message}")
            }
            Self::InvalidIteration => {
                formatter.write_str("optimizer iteration is not a finite nonnegative integer")
            }
            Self::Disposed => formatter.write_str("optimizer has been disposed"),
        }
    }
}

impl std::error::Error for AdamWError {}

#[derive(Debug, Clone, PartialEq)]
struct OptimizerVariable {
    original_name: String,
    values: Vec<f32>,
}

/// AdamW (Adam with decoupled weight decay).
#[derive(Debug, Clone, PartialEq)]
pub struct AdamWOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    acc_beta1: f32,
    acc_beta2: f32,
    iterations: u64,
    accumulated_first_moment: Vec<OptimizerVariable>,
    accumulated_second_moment: Vec<OptimizerVariable>,
    disposed: bool,
}

impl Default for AdamWOptimizer {
    fn default() -> Self {
        let config = AdamWConfig::default();
        Self {
            learning_rate: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            weight_decay: config.weight_decay,
            acc_beta1: config.beta1 as f32,
            acc_beta2: config.beta2 as f32,
            iterations: 0,
            accumulated_first_moment: Vec::new(),
            accumulated_second_moment: Vec::new(),
            disposed: false,
        }
    }
}

impl AdamWOptimizer {
    /// Creates an optimizer with explicit parameters.
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Result<Self, AdamWError> {
        Self::from_config(AdamWConfig {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        })
    }

    /// Creates an optimizer from the serialized TensorFlow.js-style config.
    pub fn from_config(config: AdamWConfig) -> Result<Self, AdamWError> {
        validate_config(&config)?;
        Ok(Self {
            learning_rate: config.learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            weight_decay: config.weight_decay,
            acc_beta1: config.beta1 as f32,
            acc_beta2: config.beta2 as f32,
            iterations: 0,
            accumulated_first_moment: Vec::new(),
            accumulated_second_moment: Vec::new(),
            disposed: false,
        })
    }

    pub fn class_name() -> &'static str {
        "AdamW"
    }

    pub fn get_config(&self) -> AdamWConfig {
        AdamWConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        }
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Applies one gradient step to the supplied variables.
    ///
    /// `None` gradients are intentionally accepted. TensorFlow.js initializes
    /// that variable's moment buffers before skipping its update, and still
    /// advances the shared bias-correction factors; this method preserves that
    /// behavior. Moment buffers are indexed in the same order as the supplied
    /// variables, matching the source optimizer's order-based state handling.
    pub fn apply_gradients(
        &mut self,
        variables: &mut [NamedTensor],
        gradients: &[Option<NamedTensor>],
    ) -> Result<(), AdamWError> {
        if self.disposed {
            return Err(AdamWError::Disposed);
        }
        if variables.len() != gradients.len() {
            return Err(AdamWError::GradientCountMismatch {
                variables: variables.len(),
                gradients: gradients.len(),
            });
        }

        for (index, variable) in variables.iter().enumerate() {
            validate_tensor(variable, index)?;
        }
        for (index, gradient) in gradients.iter().enumerate() {
            if let Some(gradient) = gradient {
                validate_tensor(gradient, index)?;
                if gradient.values.len() != variables[index].values.len() {
                    return Err(AdamWError::ShapeMismatch {
                        name: variables[index].name.clone(),
                        parameter: variables[index].values.len(),
                        gradient: gradient.values.len(),
                    });
                }
            }
        }

        for (index, variable) in variables.iter().enumerate() {
            if let Some(moment) = self.accumulated_first_moment.get(index) {
                if moment.values.len() != variable.values.len() {
                    return Err(AdamWError::ShapeMismatch {
                        name: variable.name.clone(),
                        parameter: variable.values.len(),
                        gradient: moment.values.len(),
                    });
                }
            }
            if let Some(moment) = self.accumulated_second_moment.get(index) {
                if moment.values.len() != variable.values.len() {
                    return Err(AdamWError::ShapeMismatch {
                        name: variable.name.clone(),
                        parameter: variable.values.len(),
                        gradient: moment.values.len(),
                    });
                }
            }
        }

        let one_minus_acc_beta1 = 1.0_f32 - self.acc_beta1;
        let one_minus_acc_beta2 = 1.0_f32 - self.acc_beta2;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let one_minus_beta1 = (1.0 - self.beta1) as f32;
        let one_minus_beta2 = (1.0 - self.beta2) as f32;
        let epsilon = self.epsilon as f32;
        let learning_rate = self.learning_rate as f32;
        let decay_factor = (1.0 - self.learning_rate * self.weight_decay) as f32;

        for (index, (variable, gradient)) in variables.iter_mut().zip(gradients).enumerate() {
            if self.accumulated_first_moment.len() <= index {
                self.accumulated_first_moment.push(OptimizerVariable {
                    original_name: format!("{}/m", variable.name),
                    values: vec![0.0; variable.values.len()],
                });
            }
            if self.accumulated_second_moment.len() <= index {
                self.accumulated_second_moment.push(OptimizerVariable {
                    original_name: format!("{}/v", variable.name),
                    values: vec![0.0; variable.values.len()],
                });
            }

            let Some(gradient) = gradient else {
                continue;
            };
            let first_moment = &mut self.accumulated_first_moment[index].values;
            let second_moment = &mut self.accumulated_second_moment[index].values;

            for ((value, first), (second, gradient)) in variable
                .values
                .iter_mut()
                .zip(first_moment.iter_mut())
                .zip(second_moment.iter_mut().zip(&gradient.values))
            {
                // m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
                let new_first = beta1 * *first + one_minus_beta1 * *gradient;
                // v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
                let new_second = beta2 * *second + one_minus_beta2 * (*gradient * *gradient);
                *first = new_first;
                *second = new_second;

                // Bias correction uses the factors before this step's update,
                // exactly as accBeta1/accBeta2 do in the TensorFlow.js source.
                let bias_corrected_first = new_first / one_minus_acc_beta1;
                let bias_corrected_second = new_second / one_minus_acc_beta2;
                let normalized_update =
                    bias_corrected_first / (bias_corrected_second.sqrt() + epsilon);
                let update = (-learning_rate) * normalized_update;

                if self.weight_decay != 0.0 && !variable.name.contains("bias") {
                    *value *= decay_factor;
                }
                *value += update;
            }
        }

        self.acc_beta1 *= beta1;
        self.acc_beta2 *= beta2;
        self.iterations = self.iterations.saturating_add(1);
        Ok(())
    }

    /// Returns TensorFlow.js-compatible optimizer state ordering:
    /// iterations, all first moments, then all second moments.
    pub fn get_weights(&self) -> Vec<NamedTensor> {
        let mut weights = Vec::with_capacity(
            1 + self.accumulated_first_moment.len() + self.accumulated_second_moment.len(),
        );
        weights.push(NamedTensor::new("iter", vec![self.iterations as f32]));
        weights.extend(
            self.accumulated_first_moment
                .iter()
                .map(|value| NamedTensor::new(value.original_name.clone(), value.values.clone())),
        );
        weights.extend(
            self.accumulated_second_moment
                .iter()
                .map(|value| NamedTensor::new(value.original_name.clone(), value.values.clone())),
        );
        weights
    }

    /// Restores state produced by [`Self::get_weights`].
    pub fn set_weights(&mut self, weights: &[NamedTensor]) -> Result<(), AdamWError> {
        if self.disposed {
            return Err(AdamWError::Disposed);
        }
        let Some(iteration) = weights.first() else {
            return Err(AdamWError::InvalidWeights("missing iteration tensor"));
        };
        if iteration.values.len() != 1
            || !iteration.values[0].is_finite()
            || iteration.values[0] < 0.0
            || iteration.values[0].fract() != 0.0
            || iteration.values[0] > u64::MAX as f32
        {
            return Err(AdamWError::InvalidIteration);
        }
        let state = &weights[1..];
        if !state.len().is_multiple_of(2) {
            return Err(AdamWError::InvalidWeights(
                "first and second moment counts must match",
            ));
        }

        let variable_count = state.len() / 2;
        let first = &state[..variable_count];
        let second = &state[variable_count..];
        for (index, (first, second)) in first.iter().zip(second).enumerate() {
            if first.values.len() != second.values.len() {
                return Err(AdamWError::InvalidWeights("moment shapes must match"));
            }
            validate_tensor(first, index)?;
            validate_tensor(second, index)?;
        }

        self.iterations = iteration.values[0] as u64;
        self.acc_beta1 = (self.beta1 as f32).powf(self.iterations as f32 + 1.0);
        self.acc_beta2 = (self.beta2 as f32).powf(self.iterations as f32 + 1.0);
        self.accumulated_first_moment = first
            .iter()
            .map(|value| OptimizerVariable {
                original_name: value.name.clone(),
                values: value.values.clone(),
            })
            .collect();
        self.accumulated_second_moment = second
            .iter()
            .map(|value| OptimizerVariable {
                original_name: value.name.clone(),
                values: value.values.clone(),
            })
            .collect();
        Ok(())
    }

    /// Releases optimizer state. Disposal is idempotent; subsequent gradient
    /// application and state restoration return [`AdamWError::Disposed`].
    /// `get_weights` retains only the iteration marker for observability.
    pub fn dispose(&mut self) {
        if self.disposed {
            return;
        }
        self.accumulated_first_moment.clear();
        self.accumulated_second_moment.clear();
        self.disposed = true;
    }
}

/// Creates an AdamW optimizer with the source implementation's defaults.
pub fn adamw() -> AdamWOptimizer {
    AdamWOptimizer::default()
}

/// Creates an AdamW optimizer with explicit parameters.
pub fn adamw_with_parameters(
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
) -> Result<AdamWOptimizer, AdamWError> {
    AdamWOptimizer::new(learning_rate, beta1, beta2, epsilon, weight_decay)
}

fn validate_config(config: &AdamWConfig) -> Result<(), AdamWError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(AdamWError::InvalidParameter("learning rate"));
    }
    if !config.beta1.is_finite() || !(0.0..1.0).contains(&config.beta1) {
        return Err(AdamWError::InvalidParameter("beta1"));
    }
    if !config.beta2.is_finite() || !(0.0..1.0).contains(&config.beta2) {
        return Err(AdamWError::InvalidParameter("beta2"));
    }
    if !config.epsilon.is_finite() || config.epsilon <= 0.0 {
        return Err(AdamWError::InvalidParameter("epsilon"));
    }
    if !config.weight_decay.is_finite() {
        return Err(AdamWError::InvalidParameter("weight decay"));
    }
    Ok(())
}

fn validate_tensor(tensor: &NamedTensor, index: usize) -> Result<(), AdamWError> {
    if tensor.name.is_empty() {
        return Err(AdamWError::EmptyName { index });
    }
    if let Some(value_index) = tensor.values.iter().position(|value| !value.is_finite()) {
        return Err(AdamWError::NonFiniteValue {
            name: tensor.name.clone(),
            index: value_index,
        });
    }
    Ok(())
}
