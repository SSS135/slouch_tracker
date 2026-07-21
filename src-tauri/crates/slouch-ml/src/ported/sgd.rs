use serde::{Deserialize, Serialize};

pub const CLASS_NAME: &str = "SGD";
const ITERATIONS_NAME: &str = "iter";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedTensor {
    pub name: String,
    pub tensor: Vec<f32>,
}

impl NamedTensor {
    pub fn new(name: impl Into<String>, tensor: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            tensor,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub values: Vec<f32>,
}

impl Variable {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    pub name: String,
    pub values: Option<Vec<f32>>,
}

impl Gradient {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values: Some(values),
        }
    }

    pub fn missing(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: None,
        }
    }
}

/// Object-form variables in JavaScript property-enumeration order.
/// Replacing an existing property retains its original position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NamedVariableMap(Vec<(String, Variable)>);

impl NamedVariableMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: String, variable: Variable) -> Option<Variable> {
        if let Some((_, current)) = self.0.iter_mut().find(|(current, _)| current == &key) {
            return Some(std::mem::replace(current, variable));
        }
        self.0.push((key, variable));
        None
    }

    pub fn get(&self, key: &str) -> Option<&Variable> {
        self.0
            .iter()
            .find(|(current, _)| current == key)
            .map(|(_, variable)| variable)
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Variable> {
        self.0
            .iter_mut()
            .find(|(current, _)| current == key)
            .map(|(_, variable)| variable)
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &Variable)> {
        self.0
            .iter()
            .map(|(key, variable)| (key.as_str(), variable))
    }
}

impl<const N: usize> From<[(String, Variable); N]> for NamedVariableMap {
    fn from(entries: [(String, Variable); N]) -> Self {
        let mut map = Self::new();
        for (key, variable) in entries {
            map.insert(key, variable);
        }
        map
    }
}

/// Object-form gradients in JavaScript property-enumeration order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NamedGradientMap(Vec<(String, Option<Vec<f32>>)>);

impl NamedGradientMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: String, gradient: Option<Vec<f32>>) -> Option<Option<Vec<f32>>> {
        if let Some((_, current)) = self.0.iter_mut().find(|(current, _)| current == &key) {
            return Some(std::mem::replace(current, gradient));
        }
        self.0.push((key, gradient));
        None
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &Option<Vec<f32>>)> {
        self.0
            .iter()
            .map(|(key, gradient)| (key.as_str(), gradient))
    }
}

impl<const N: usize> From<[(String, Option<Vec<f32>>); N]> for NamedGradientMap {
    fn from(entries: [(String, Option<Vec<f32>>); N]) -> Self {
        let mut map = Self::new();
        for (key, gradient) in entries {
            map.insert(key, gradient);
        }
        map
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SgdError {
    MissingVariable(String),
    GradientLengthMismatch {
        name: String,
        variable: usize,
        gradient: usize,
    },
    VelocityLengthMismatch {
        name: String,
        variable: usize,
        velocity: usize,
    },
    VariableKeyNameMismatch {
        key: String,
        name: String,
    },
    InvalidIterations,
    InvalidWeightState(String),
}

impl std::fmt::Display for SgdError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVariable(name) => write!(formatter, "variable '{name}' is not registered"),
            Self::GradientLengthMismatch {
                name,
                variable,
                gradient,
            } => write!(
                formatter,
                "gradient for '{name}' has length {gradient}, expected {variable}"
            ),
            Self::VelocityLengthMismatch {
                name,
                variable,
                velocity,
            } => write!(
                formatter,
                "velocity for '{name}' has length {velocity}, expected {variable}"
            ),
            Self::VariableKeyNameMismatch { key, name } => write!(
                formatter,
                "registered variable key '{key}' does not match variable name '{name}'"
            ),
            Self::InvalidIterations => formatter.write_str("optimizer iteration value is invalid"),
            Self::InvalidWeightState(message) => {
                write!(formatter, "invalid optimizer state: {message}")
            }
        }
    }
}

impl std::error::Error for SgdError {}

#[derive(Debug, Clone)]
struct OptimizerVariable {
    original_name: String,
    variable: Vec<f32>,
}

/// Stochastic gradient descent with momentum, Nesterov momentum, and decoupled
/// weight decay.
#[derive(Debug, Clone)]
pub struct SgdOptimizer {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    iterations: u64,
    velocity_buffers: Vec<OptimizerVariable>,
}

impl Default for SgdOptimizer {
    fn default() -> Self {
        Self::new(0.01, 0.9, 0.0, false)
    }
}

impl SgdOptimizer {
    pub fn new(learning_rate: f64, momentum: f64, weight_decay: f64, nesterov: bool) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            nesterov,
            iterations: 0,
            velocity_buffers: Vec::new(),
        }
    }

    pub fn class_name() -> &'static str {
        CLASS_NAME
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn get_config(&self) -> OptimizerConfig {
        OptimizerConfig {
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            nesterov: self.nesterov,
        }
    }

    pub fn from_config(config: OptimizerConfig) -> Self {
        Self::new(
            config.learning_rate,
            config.momentum,
            config.weight_decay,
            config.nesterov,
        )
    }

    /// Apply gradients in the same order as the TensorFlow.js array form.
    /// Missing gradients leave the corresponding variable unchanged.
    pub fn apply_gradients(
        &mut self,
        variables: &mut [Variable],
        gradients: &[Gradient],
    ) -> Result<(), SgdError> {
        for (index, gradient) in gradients.iter().enumerate() {
            let variable = variables
                .iter()
                .find(|variable| variable.name == gradient.name)
                .ok_or_else(|| SgdError::MissingVariable(gradient.name.clone()))?;
            self.validate_update(index, variable, gradient.values.as_deref())?;
        }

        for (index, gradient) in gradients.iter().enumerate() {
            let variable = variables
                .iter_mut()
                .find(|variable| variable.name == gradient.name)
                .ok_or_else(|| SgdError::MissingVariable(gradient.name.clone()))?;
            self.ensure_velocity(index, &variable.name, variable.values.len());
            let Some(gradient_values) = gradient.values.as_deref() else {
                continue;
            };
            self.apply_to_variable(index, variable, gradient_values);
        }
        self.iterations = self.iterations.saturating_add(1);
        Ok(())
    }

    /// Apply gradients in the same order as the TensorFlow.js object form.
    pub fn apply_gradients_map(
        &mut self,
        variables: &mut NamedVariableMap,
        gradients: &NamedGradientMap,
    ) -> Result<(), SgdError> {
        for (key, variable) in variables.iter() {
            if key != variable.name.as_str() {
                return Err(SgdError::VariableKeyNameMismatch {
                    key: key.to_owned(),
                    name: variable.name.clone(),
                });
            }
        }
        for (index, (name, values)) in gradients.iter().enumerate() {
            let variable = variables
                .get(name)
                .ok_or_else(|| SgdError::MissingVariable(name.to_owned()))?;
            self.validate_update(index, variable, values.as_deref())?;
        }

        for (index, (name, values)) in gradients.iter().enumerate() {
            let variable = variables
                .get_mut(name)
                .ok_or_else(|| SgdError::MissingVariable(name.to_owned()))?;
            self.ensure_velocity(index, name, variable.values.len());
            let Some(gradient_values) = values.as_deref() else {
                continue;
            };
            self.apply_to_variable(index, variable, gradient_values);
        }
        self.iterations = self.iterations.saturating_add(1);
        Ok(())
    }

    fn validate_update(
        &self,
        index: usize,
        variable: &Variable,
        gradient: Option<&[f32]>,
    ) -> Result<(), SgdError> {
        if let Some(gradient) = gradient {
            if variable.values.len() != gradient.len() {
                return Err(SgdError::GradientLengthMismatch {
                    name: variable.name.clone(),
                    variable: variable.values.len(),
                    gradient: gradient.len(),
                });
            }
        }
        if let Some(velocity) = self.velocity_buffers.get(index) {
            if velocity.variable.len() != variable.values.len() {
                return Err(SgdError::VelocityLengthMismatch {
                    name: variable.name.clone(),
                    variable: variable.values.len(),
                    velocity: velocity.variable.len(),
                });
            }
        }
        Ok(())
    }

    fn ensure_velocity(&mut self, index: usize, name: &str, length: usize) {
        if self.velocity_buffers.len() <= index {
            self.velocity_buffers
                .resize_with(index + 1, || OptimizerVariable {
                    original_name: String::new(),
                    variable: Vec::new(),
                });
        }
        if self.velocity_buffers[index].variable.is_empty()
            && self.velocity_buffers[index].original_name.is_empty()
        {
            self.velocity_buffers[index] = OptimizerVariable {
                original_name: format!("{name}/velocity"),
                variable: vec![0.0; length],
            };
        }
    }

    fn apply_to_variable(&mut self, index: usize, variable: &mut Variable, gradient: &[f32]) {
        let learning_rate = self.learning_rate as f32;
        let momentum = self.momentum as f32;
        let decay_factor = (1.0 - self.learning_rate * self.weight_decay) as f32;
        let velocity = &mut self.velocity_buffers[index].variable;

        for ((value, velocity), gradient) in variable
            .values
            .iter_mut()
            .zip(velocity.iter_mut())
            .zip(gradient)
        {
            let mut new_value = if self.weight_decay != 0.0 {
                *value * decay_factor
            } else {
                *value
            };

            if self.momentum != 0.0 {
                *velocity = momentum * *velocity + *gradient;
                let update = if self.nesterov {
                    *gradient + momentum * *velocity
                } else {
                    *velocity
                };
                new_value -= learning_rate * update;
            } else {
                new_value -= learning_rate * *gradient;
            }
            *value = new_value;
        }
    }

    /// Return TensorFlow.js-compatible optimizer weights: iteration state first,
    /// followed by velocity buffers in their positional update order.
    pub fn get_weights(&self) -> Vec<NamedTensor> {
        let mut weights = Vec::with_capacity(self.velocity_buffers.len() + 1);
        weights.push(NamedTensor::new(
            ITERATIONS_NAME,
            vec![self.iterations as f32],
        ));
        weights.extend(
            self.velocity_buffers
                .iter()
                .map(|buffer| NamedTensor::new(&buffer.original_name, buffer.variable.clone())),
        );
        weights
    }

    pub fn set_weights(&mut self, weights: &[NamedTensor]) -> Result<(), SgdError> {
        let Some(iteration_weight) = weights.first() else {
            return Err(SgdError::InvalidWeightState(
                "iteration state is missing".into(),
            ));
        };
        if iteration_weight.name != ITERATIONS_NAME || iteration_weight.tensor.len() != 1 {
            return Err(SgdError::InvalidWeightState(
                "first weight must be the scalar iteration state".into(),
            ));
        }
        let iterations = iteration_weight.tensor[0];
        if !iterations.is_finite() || iterations < 0.0 || iterations.fract() != 0.0 {
            return Err(SgdError::InvalidIterations);
        }

        let velocity_buffers = weights[1..]
            .iter()
            .map(|weight| OptimizerVariable {
                original_name: weight.name.clone(),
                variable: weight.tensor.clone(),
            })
            .collect();

        self.iterations = iterations as u64;
        self.velocity_buffers = velocity_buffers;
        Ok(())
    }

    pub fn dispose(&mut self) {
        self.velocity_buffers.clear();
    }
}

pub fn sgd(learning_rate: f64, momentum: f64, weight_decay: f64, nesterov: bool) -> SgdOptimizer {
    SgdOptimizer::new(learning_rate, momentum, weight_decay, nesterov)
}

pub fn default_sgd() -> SgdOptimizer {
    SgdOptimizer::default()
}
