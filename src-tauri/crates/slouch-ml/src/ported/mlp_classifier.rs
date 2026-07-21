//! Multi-layer perceptron classifier.
//!
//! This is a mechanical Rust port of `src/services/ml/mlpClassifier.ts`.
//! Tensor/model values stay in `f32`, while configuration and serialized
//! JavaScript-number values use `f64`.

use super::adamw::{adamw_with_parameters, NamedTensor};
use super::base_classifier::{BaseClassifier, ClassifierError};
use super::config::TRAINING_CONFIG;
use super::types::{MlpParams, SerializedClassifierState, SerializedMlp};
use super::utils::class_weights::compute_balanced_class_weights;

const DEFAULT_WEIGHT_DECAY: f64 = 0.01;
const DEFAULT_MAX_ITERATIONS: usize = 1000;
const DEFAULT_LEARNING_RATE: f64 = 0.01;
const DEFAULT_LABEL_SMOOTHING: f64 = 0.1;
const DEFAULT_HIDDEN_LAYERS: usize = 0;
const DEFAULT_HIDDEN_SIZE: usize = 64;
const MAX_HIDDEN_SIZE: usize = 16_384;
const MAX_ITERATIONS: usize = 1_000_000;
const MAX_MODEL_PARAMETERS: usize = 16_777_216;

/// MLP training parameters, including the source implementation's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlpConfig {
    pub weight_decay: f64,
    pub max_iterations: usize,
    pub learning_rate: f64,
    pub use_class_weights: bool,
    pub label_smoothing: f64,
    pub hidden_layers: usize,
    pub hidden_size: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            weight_decay: DEFAULT_WEIGHT_DECAY,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            learning_rate: DEFAULT_LEARNING_RATE,
            use_class_weights: false,
            label_smoothing: DEFAULT_LABEL_SMOOTHING,
            hidden_layers: DEFAULT_HIDDEN_LAYERS,
            hidden_size: DEFAULT_HIDDEN_SIZE,
        }
    }
}

impl From<MlpParams> for MlpConfig {
    fn from(params: MlpParams) -> Self {
        Self {
            weight_decay: params.weight_decay,
            max_iterations: params.max_iterations,
            learning_rate: params.learning_rate,
            use_class_weights: params.use_class_weights.unwrap_or(false),
            label_smoothing: params.label_smoothing.unwrap_or(DEFAULT_LABEL_SMOOTHING),
            hidden_layers: params.hidden_layers.unwrap_or(DEFAULT_HIDDEN_LAYERS),
            hidden_size: params.hidden_size.unwrap_or(DEFAULT_HIDDEN_SIZE),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Layer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    fan_in: usize,
    fan_out: usize,
}

impl Layer {
    fn new(fan_in: usize, fan_out: usize, weights: Vec<f32>, bias: Vec<f32>) -> Self {
        Self {
            weights,
            bias,
            fan_in,
            fan_out,
        }
    }
}

/// MLP classifier with zero, one, or two ReLU hidden layers.
#[derive(Debug, Clone)]
pub struct MlpClassifier {
    layers: Vec<Layer>,
    params: MlpConfig,
    class_weights: [f64; 2],
    input_dimensions: Option<usize>,
}

impl MlpClassifier {
    /// Creates an untrained classifier. Accepts either [`MlpConfig`],
    /// [`MlpParams`], or any other configuration converted into `MlpConfig`.
    pub fn new<T: Into<MlpConfig>>(params: T) -> Result<Self, ClassifierError> {
        let params = params.into();
        validate_config(&params)?;
        Ok(Self {
            layers: Vec::new(),
            params,
            class_weights: [1.0, 1.0],
            input_dimensions: None,
        })
    }

    /// Creates an untrained classifier using source-style defaults.
    pub fn default_classifier() -> Result<Self, ClassifierError> {
        Self::new(MlpConfig::default())
    }

    /// Restores a serialized MLP with the supplied training configuration.
    pub fn from_json<T: Into<MlpConfig>>(
        data: SerializedMlp,
        params: T,
    ) -> Result<Self, ClassifierError> {
        let mut classifier = Self::new(params)?;
        classifier.load_state(data)?;
        Ok(classifier)
    }

    /// Restores a model using the architecture and class weights in its state.
    pub fn from_state(data: SerializedMlp) -> Result<Self, ClassifierError> {
        let config = MlpConfig {
            hidden_layers: data.hidden_layers,
            hidden_size: data.hidden_size,
            ..MlpConfig::default()
        };
        Self::from_json(data, config)
    }

    fn load_state(&mut self, data: SerializedMlp) -> Result<(), ClassifierError> {
        if data.layer_weights.is_empty()
            || data.layer_weights.len() != data.layer_biases.len()
            || data.layer_shapes.len() != data.layer_weights.len() + 1
        {
            return Err(ClassifierError::InvalidState(
                "MLP layers and shapes have inconsistent lengths".into(),
            ));
        }

        if data.hidden_layers > 2
            || data.layer_weights.len() != data.hidden_layers + 1
            || data.layer_shapes.last().copied() != Some(1)
            || (data.hidden_layers > 0 && data.hidden_size == 0)
            || data.layer_shapes[1..data.layer_shapes.len() - 1]
                .iter()
                .any(|width| *width != data.hidden_size)
        {
            return Err(ClassifierError::InvalidState(
                "MLP architecture metadata does not match the layer graph".into(),
            ));
        }

        let mut layers = Vec::with_capacity(data.layer_weights.len());
        let mut parameter_count = 0_usize;
        for index in 0..data.layer_weights.len() {
            let fan_in = data.layer_shapes[index];
            let fan_out = data.layer_shapes[index + 1];
            if fan_in == 0 || fan_out == 0 {
                return Err(ClassifierError::InvalidState(
                    "MLP layer dimensions must be positive".into(),
                ));
            }
            let expected_weights = fan_in.checked_mul(fan_out).ok_or_else(|| {
                ClassifierError::InvalidState("MLP layer dimensions overflow".into())
            })?;
            parameter_count = parameter_count
                .checked_add(expected_weights)
                .and_then(|count| count.checked_add(fan_out))
                .ok_or_else(|| {
                    ClassifierError::InvalidState("MLP parameter count overflow".into())
                })?;
            if parameter_count > MAX_MODEL_PARAMETERS {
                return Err(ClassifierError::InvalidState(
                    "MLP model exceeds the native parameter limit".into(),
                ));
            }
            if data.layer_weights[index].len() != expected_weights
                || data.layer_biases[index].len() != fan_out
            {
                return Err(ClassifierError::InvalidState(format!(
                    "MLP layer {index} has invalid tensor lengths"
                )));
            }
            let weights = data.layer_weights[index]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            let biases = data.layer_biases[index]
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            if weights
                .iter()
                .chain(biases.iter())
                .any(|value| !value.is_finite())
            {
                return Err(ClassifierError::InvalidState(
                    "MLP state contains values outside the finite f32 range".into(),
                ));
            }
            layers.push(Layer::new(fan_in, fan_out, weights, biases));
        }

        if data
            .class_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ClassifierError::InvalidState(
                "MLP class weights must be finite and positive".into(),
            ));
        }
        self.input_dimensions = data.layer_shapes.first().copied();
        self.layers = layers;
        self.params.hidden_layers = data.hidden_layers;
        self.params.hidden_size = data.hidden_size;
        self.class_weights = data.class_weights;
        Ok(())
    }

    fn dispose_model(&mut self) {
        self.layers.clear();
        self.input_dimensions = None;
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut activations = vec![input.to_vec()];
        let mut current = input.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut next = vec![0.0_f32; layer.fan_out];
            for (output, next_value) in next.iter_mut().enumerate() {
                let mut value = layer.bias[output];
                for (input_index, input_value) in current.iter().enumerate() {
                    value += input_value * layer.weights[input_index * layer.fan_out + output];
                }
                *next_value = value;
            }
            if index + 1 != self.layers.len() {
                for value in &mut next {
                    *value = value.max(0.0);
                }
                activations.push(next.clone());
            }
            current = next;
        }
        (activations, current)
    }

    fn initialize_layers(&mut self, input_dimensions: usize) -> Result<(), ClassifierError> {
        let mut layer_sizes = Vec::with_capacity(self.params.hidden_layers + 2);
        layer_sizes.push(input_dimensions);
        for _ in 0..self.params.hidden_layers {
            layer_sizes.push(self.params.hidden_size);
        }
        layer_sizes.push(1);

        let mut parameter_count = 0_usize;
        for window in layer_sizes.windows(2) {
            let weights = window[0].checked_mul(window[1]).ok_or_else(|| {
                ClassifierError::InvalidState("MLP layer dimensions overflow".into())
            })?;
            parameter_count = parameter_count
                .checked_add(weights)
                .and_then(|count| count.checked_add(window[1]))
                .ok_or_else(|| {
                    ClassifierError::InvalidState("MLP parameter count overflow".into())
                })?;
        }
        if parameter_count > MAX_MODEL_PARAMETERS {
            return Err(ClassifierError::InvalidState(
                "MLP model exceeds the native parameter limit".into(),
            ));
        }

        self.layers.clear();
        for index in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[index];
            let fan_out = layer_sizes[index + 1];
            let standard_deviation = if index + 1 == layer_sizes.len() - 1 {
                0.01
            } else {
                let fan_sum = fan_in.checked_add(fan_out).ok_or_else(|| {
                    ClassifierError::InvalidState("MLP layer dimensions overflow".into())
                })?;
                (2.0 / fan_sum as f64).sqrt()
            };
            // TensorFlow.js constructs a separately seeded MPRandGauss for
            // every randomNormal call, so each layer restarts the Alea stream.
            let mut rng = NormalRng::new(TRAINING_CONFIG.random_seed);
            let weight_count = fan_in.checked_mul(fan_out).ok_or_else(|| {
                ClassifierError::InvalidState("MLP layer dimensions overflow".into())
            })?;
            let weights = (0..weight_count)
                .map(|_| rng.normal(standard_deviation))
                .collect();
            self.layers
                .push(Layer::new(fan_in, fan_out, weights, vec![0.0; fan_out]));
        }
        self.input_dimensions = Some(input_dimensions);
        Ok(())
    }

    fn validate_prediction(&self, features: &[f32]) -> Result<(), ClassifierError> {
        let expected = self.input_dimensions.ok_or(ClassifierError::Untrained)?;
        if features.len() != expected {
            return Err(ClassifierError::PredictionDimension {
                expected,
                actual: features.len(),
            });
        }
        if let Some(column) = features.iter().position(|value| !value.is_finite()) {
            return Err(ClassifierError::NonFiniteFeature { row: 0, column });
        }
        Ok(())
    }

    /// Returns the serialized MLP payload without the classifier union wrapper.
    pub fn to_mlp_json(&self) -> Result<SerializedMlp, ClassifierError> {
        if self.layers.is_empty() {
            return Err(ClassifierError::Untrained);
        }
        let mut layer_weights = Vec::with_capacity(self.layers.len());
        let mut layer_biases = Vec::with_capacity(self.layers.len());
        let mut layer_shapes = Vec::with_capacity(self.layers.len() + 1);
        for (index, layer) in self.layers.iter().enumerate() {
            if index == 0 {
                layer_shapes.push(layer.fan_in);
            }
            layer_shapes.push(layer.fan_out);
            layer_weights.push(
                layer
                    .weights
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
            );
            layer_biases.push(layer.bias.iter().map(|value| f64::from(*value)).collect());
        }
        Ok(SerializedMlp {
            layer_weights,
            layer_biases,
            layer_shapes,
            hidden_layers: self.params.hidden_layers,
            hidden_size: self.params.hidden_size,
            class_weights: self.class_weights,
        })
    }
}

impl BaseClassifier for MlpClassifier {
    fn classifier_id(&self) -> &'static str {
        "mlp"
    }

    fn train(&mut self, features: &[Vec<f32>], labels: &[i32]) -> Result<(), ClassifierError> {
        if features.is_empty() {
            return Err(ClassifierError::EmptyDataset);
        }
        if features.len() != labels.len() {
            return Err(ClassifierError::LengthMismatch {
                features: features.len(),
                labels: labels.len(),
            });
        }

        // TensorFlow.js disposes the previous layers before shape and label
        // validation. A failed retrain therefore leaves the classifier
        // untrained rather than retaining a stale model.
        self.dispose_model();
        let dimensions = validate_features(features)?;
        for (index, label) in labels.iter().enumerate() {
            if *label != 0 && *label != 1 {
                return Err(ClassifierError::InvalidState(format!(
                    "label at index {index} must be 0 or 1"
                )));
            }
        }

        self.class_weights = if self.params.use_class_weights {
            let weights = compute_balanced_class_weights(labels)
                .map_err(|error| ClassifierError::InvalidState(error.to_string()))?;
            [
                weights.first().copied().unwrap_or(1.0),
                weights.get(1).copied().unwrap_or(1.0),
            ]
        } else {
            [1.0, 1.0]
        };
        self.initialize_layers(dimensions)?;

        let mut optimizer = adamw_with_parameters(
            self.params.learning_rate,
            0.9,
            0.999,
            1e-7,
            self.params.weight_decay,
        )
        .map_err(|error| ClassifierError::InvalidState(error.to_string()))?;
        let smoothing = self.params.label_smoothing as f32;
        // `sampleWeights` is rank 1 while the TFJS sigmoid-cross-entropy
        // losses are `[samples, 1]`. Broadcasting makes every loss see every
        // class weight, so the weights cancel in SUM_BY_NONZERO_WEIGHTS. Keep
        // the serialized class weights, but preserve that effective gradient.
        let denominator = labels.len() as f32;

        for _ in 0..self.params.max_iterations {
            // TFJS CPU reduction tensors stay f32. The source-order rounding
            // is observable when balanced-class bias gradients nearly cancel.
            let mut layer_weight_gradients = self
                .layers
                .iter()
                .map(|layer| vec![0.0_f32; layer.weights.len()])
                .collect::<Vec<_>>();
            let mut layer_bias_gradients = self
                .layers
                .iter()
                .map(|layer| vec![0.0_f32; layer.bias.len()])
                .collect::<Vec<_>>();

            for (sample, label) in features.iter().zip(labels) {
                let (activations, logits) = self.forward(sample);
                let target = *label as f32 * (1.0 - smoothing) + smoothing / 2.0;
                let sample_weight = 1.0_f32;
                // Autodiff follows the stable loss graph kernel by kernel.
                // Scale enters before Exp/Log1p backprop, and positive logits
                // accumulate the ReLU branch last. Both f32 orders are visible
                // in cancellation-sensitive bias gradients.
                let scale = sample_weight / denominator;
                let exponential = f64::from(-logits[0].abs()).exp() as f32;
                let log_gradient = (scale / (1.0_f32 + exponential)) * exponential;
                let target_gradient = -(target * scale);
                let scaled_error = if logits[0] >= 0.0 {
                    scale + (target_gradient - log_gradient)
                } else {
                    target_gradient + log_gradient
                };
                let mut delta = vec![scaled_error];

                for layer_index in (0..self.layers.len()).rev() {
                    let layer = &self.layers[layer_index];
                    let input = &activations[layer_index];
                    for (output, delta_value) in delta.iter().copied().enumerate() {
                        layer_bias_gradients[layer_index][output] += delta_value;
                        for (input_index, input_value) in input.iter().copied().enumerate() {
                            layer_weight_gradients[layer_index]
                                [input_index * layer.fan_out + output] += input_value * delta_value;
                        }
                    }

                    if layer_index > 0 {
                        let mut previous_delta = vec![0.0_f32; layer.fan_in];
                        let previous_activation = &activations[layer_index];
                        for (input_index, previous_delta_value) in
                            previous_delta.iter_mut().enumerate()
                        {
                            let mut value = 0.0_f32;
                            for (output, delta_value) in delta.iter().copied().enumerate() {
                                value += delta_value
                                    * layer.weights[input_index * layer.fan_out + output];
                            }
                            *previous_delta_value = if previous_activation[input_index] > 0.0 {
                                value
                            } else {
                                0.0
                            };
                        }
                        delta = previous_delta;
                    }
                }
            }

            let mut variables = Vec::with_capacity(self.layers.len() * 2);
            let mut gradients = Vec::with_capacity(self.layers.len() * 2);
            for (index, layer) in self.layers.iter().enumerate() {
                variables.push(NamedTensor::new(
                    format!("layers.{index}.weights"),
                    layer.weights.clone(),
                ));
                gradients.push(Some(NamedTensor::new(
                    format!("layers.{index}.weights"),
                    layer_weight_gradients[index].clone(),
                )));
                variables.push(NamedTensor::new(
                    format!("layers.{index}.bias"),
                    layer.bias.clone(),
                ));
                gradients.push(Some(NamedTensor::new(
                    format!("layers.{index}.bias"),
                    layer_bias_gradients[index].clone(),
                )));
            }
            optimizer
                .apply_gradients(&mut variables, &gradients)
                .map_err(|error| ClassifierError::InvalidState(error.to_string()))?;

            let mut variable_index = 0;
            for layer in &mut self.layers {
                layer.weights.clone_from(&variables[variable_index].values);
                layer.bias.clone_from(&variables[variable_index + 1].values);
                variable_index += 2;
            }
        }

        optimizer.dispose();
        if self
            .layers
            .iter()
            .flat_map(|layer| layer.weights.iter().chain(layer.bias.iter()))
            .any(|value| !value.is_finite())
        {
            return Err(ClassifierError::InvalidState(
                "training failed: model weights contain non-finite values".into(),
            ));
        }
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        self.validate_prediction(features)?;
        let (_, logits) = self.forward(features);
        Ok(1.0 - f64::from(sigmoid(logits[0])))
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        Ok(SerializedClassifierState::Mlp(self.to_mlp_json()?))
    }

    fn dispose(&mut self) {
        self.dispose_model();
    }
}

fn validate_config(config: &MlpConfig) -> Result<(), ClassifierError> {
    if config.hidden_layers > 2 {
        return Err(ClassifierError::InvalidState(
            "hiddenLayers must be 0, 1, or 2".into(),
        ));
    }
    if config.hidden_layers > 0 && config.hidden_size == 0 {
        return Err(ClassifierError::InvalidState(
            "hiddenSize must be positive when hidden layers are enabled".into(),
        ));
    }
    if config.hidden_size > MAX_HIDDEN_SIZE || config.max_iterations > MAX_ITERATIONS {
        return Err(ClassifierError::InvalidState(
            "MLP resource limits are exceeded".into(),
        ));
    }
    if !config.weight_decay.is_finite()
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
        || !config.label_smoothing.is_finite()
        || !(0.0..=1.0).contains(&config.label_smoothing)
    {
        return Err(ClassifierError::InvalidState(
            "MLP numeric parameters are invalid".into(),
        ));
    }
    Ok(())
}

fn validate_features(features: &[Vec<f32>]) -> Result<usize, ClassifierError> {
    let dimensions = features.first().ok_or(ClassifierError::EmptyDataset)?.len();
    if dimensions == 0 {
        return Err(ClassifierError::EmptyFeatureVector);
    }
    for (row, feature) in features.iter().enumerate() {
        if feature.len() != dimensions {
            return Err(ClassifierError::RaggedFeatures {
                row,
                expected: dimensions,
                actual: feature.len(),
            });
        }
        if let Some(column) = feature.iter().position(|value| !value.is_finite()) {
            return Err(ClassifierError::NonFiniteFeature { row, column });
        }
    }
    Ok(dimensions)
}

fn sigmoid(value: f32) -> f32 {
    // TensorFlow.js CPU uses this direct JavaScript-number expression for
    // both signs, then narrows the kernel output to f32.
    (1.0_f64 / (1.0 + (-f64::from(value)).exp())) as f32
}

/// TensorFlow.js 4.22's seeded `MPRandGauss`: Alea plus the Marsaglia polar
/// method. All generator math remains `f64`; tensor assignment narrows to f32.
#[derive(Debug, Clone)]
struct NormalRng {
    random: Alea,
    spare: Option<f64>,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self {
            random: Alea::new(&seed.to_string()),
            spare: None,
        }
    }

    fn normal(&mut self, standard_deviation: f64) -> f32 {
        if let Some(value) = self.spare.take() {
            return (value * standard_deviation) as f32;
        }
        loop {
            let first = 2.0 * self.random.next_f64() - 1.0;
            let second = 2.0 * self.random.next_f64() - 1.0;
            let squared_radius = first * first + second * second;
            if squared_radius >= 1.0 || squared_radius == 0.0 {
                continue;
            }
            let multiplier = (-2.0 * squared_radius.ln() / squared_radius).sqrt();
            self.spare = Some(second * multiplier);
            return (standard_deviation * first * multiplier) as f32;
        }
    }
}

#[derive(Debug, Clone)]
struct Alea {
    s0: f64,
    s1: f64,
    s2: f64,
    carry: i32,
}

impl Alea {
    fn new(seed: &str) -> Self {
        let mut mash = Mash::new();
        let blank = mash.apply(" ");
        let mut generator = Self {
            s0: blank,
            s1: mash.apply(" "),
            s2: mash.apply(" "),
            carry: 1,
        };
        let seed_value = mash.apply(seed);
        generator.s0 = (generator.s0 - seed_value).rem_euclid(1.0);
        let seed_value = mash.apply(seed);
        generator.s1 = (generator.s1 - seed_value).rem_euclid(1.0);
        let seed_value = mash.apply(seed);
        generator.s2 = (generator.s2 - seed_value).rem_euclid(1.0);
        generator
    }

    fn next_f64(&mut self) -> f64 {
        const TWO_POW_NEG_32: f64 = 2.328_306_436_538_696_3e-10;
        let value = 2_091_639.0 * self.s0 + f64::from(self.carry) * TWO_POW_NEG_32;
        self.s0 = self.s1;
        self.s1 = self.s2;
        self.carry = value.trunc() as i32;
        self.s2 = value - f64::from(self.carry);
        self.s2
    }
}

#[derive(Debug, Clone)]
struct Mash {
    state: f64,
}

impl Mash {
    fn new() -> Self {
        Self {
            state: 0xefc8_249d_u32 as f64,
        }
    }

    fn apply(&mut self, value: &str) -> f64 {
        const TWO_POW_32: f64 = 4_294_967_296.0;
        const TWO_POW_NEG_32: f64 = 2.328_306_436_538_696_3e-10;
        for code_unit in value.encode_utf16() {
            self.state += f64::from(code_unit);
            let mut intermediate = 0.025_196_032_824_169_38 * self.state;
            self.state = f64::from(js_to_u32(intermediate));
            intermediate -= self.state;
            intermediate *= self.state;
            self.state = f64::from(js_to_u32(intermediate));
            intermediate -= self.state;
            self.state += intermediate * TWO_POW_32;
        }
        f64::from(js_to_u32(self.state)) * TWO_POW_NEG_32
    }
}

fn js_to_u32(value: f64) -> u32 {
    value.trunc().rem_euclid(4_294_967_296.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ported::types::MlpClassifierId;

    #[test]
    fn tensor_flow_seeded_normal_sequence_matches_cpu_oracle() {
        let mut generator = NormalRng::new(42);
        let expected = [
            1.205_023_4_f32,
            0.301_963_72,
            0.281_725_05,
            -0.245_303_2,
            0.044_384_7,
            0.322_790_2,
        ];
        let actual = (0..expected.len())
            .map(|_| generator.normal((2.0_f64 / 5.0).sqrt()))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn mlp_params_preserve_architecture_and_label_smoothing() {
        let params = MlpParams {
            classifier_id: MlpClassifierId::Mlp,
            weight_decay: 0.01,
            max_iterations: 0,
            learning_rate: 0.01,
            use_class_weights: Some(false),
            label_smoothing: Some(0.25),
            hidden_layers: Some(1),
            hidden_size: Some(8),
        };
        let mut classifier = MlpClassifier::new(params).unwrap();
        classifier
            .train(&[vec![0.0, 1.0], vec![1.0, 0.0]], &[0, 1])
            .unwrap();
        let state = classifier.to_mlp_json().unwrap();
        assert_eq!(state.layer_shapes, vec![2, 8, 1]);
        assert_eq!(state.hidden_layers, 1);
        assert_eq!(state.hidden_size, 8);
        assert_eq!(classifier.params.label_smoothing, 0.25);
    }

    #[test]
    fn oversized_training_budgets_are_rejected_before_allocation() {
        assert!(MlpClassifier::new(MlpConfig {
            hidden_layers: 1,
            hidden_size: usize::MAX,
            ..MlpConfig::default()
        })
        .is_err());
        assert!(MlpClassifier::new(MlpConfig {
            max_iterations: usize::MAX,
            ..MlpConfig::default()
        })
        .is_err());
    }

    #[test]
    fn malformed_architecture_and_zero_class_weights_are_rejected() {
        let malformed = SerializedMlp {
            layer_weights: vec![vec![0.0, 0.0, 0.0, 0.0]],
            layer_biases: vec![vec![0.0, 0.0]],
            layer_shapes: vec![2, 2],
            hidden_layers: 0,
            hidden_size: 64,
            class_weights: [1.0, 1.0],
        };
        assert!(MlpClassifier::from_state(malformed).is_err());

        let zero_weight = SerializedMlp {
            layer_weights: vec![vec![0.0, 0.0]],
            layer_biases: vec![vec![0.0]],
            layer_shapes: vec![2, 1],
            hidden_layers: 0,
            hidden_size: 64,
            class_weights: [0.0, 1.0],
        };
        assert!(MlpClassifier::from_state(zero_weight).is_err());
    }
}
