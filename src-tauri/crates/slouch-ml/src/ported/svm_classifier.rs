//! Linear support-vector-machine classifier.
//!
//! Mechanical Rust port of `src/services/ml/svmClassifier.ts`. Feature
//! vectors and model tensors use `f32`, while serialized values and returned
//! probabilities use `f64` at the native boundary.

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::config::TRAINING_CONFIG;
use super::sgd::{sgd, Gradient, Variable};
use super::types::{SerializedClassifierState, SerializedSvm, SvmParams};
use super::utils::class_weights::compute_balanced_class_weights;

const DEFAULT_C: f64 = 1.0;
const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Training parameters for the linear SVM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvmConfig {
    pub c: f64,
    pub max_iterations: usize,
    pub use_class_weights: bool,
}

impl Default for SvmConfig {
    fn default() -> Self {
        Self {
            c: DEFAULT_C,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            use_class_weights: false,
        }
    }
}

impl From<SvmParams> for SvmConfig {
    fn from(params: SvmParams) -> Self {
        Self {
            c: params.c,
            max_iterations: params.max_iterations,
            use_class_weights: params.use_class_weights.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
struct TrainedModel {
    weights: Vec<f32>,
    bias: f32,
}

/// Binary linear SVM trained with weighted hinge loss and L2 regularization.
#[derive(Debug, Clone)]
pub struct SvmClassifier {
    model: Option<TrainedModel>,
    params: SvmConfig,
    class_weights: [f64; 2],
}

impl SvmClassifier {
    /// Creates an untrained classifier from explicit parameters or [`SvmParams`].
    pub fn new<T: Into<SvmConfig>>(params: T) -> Result<Self, ClassifierError> {
        let params = params.into();
        validate_config(&params)?;
        Ok(Self {
            model: None,
            params,
            class_weights: [1.0, 1.0],
        })
    }

    /// Creates an untrained classifier with the source implementation defaults.
    pub fn default_classifier() -> Result<Self, ClassifierError> {
        Self::new(SvmConfig::default())
    }

    /// Restores a serialized classifier with the supplied training parameters.
    pub fn from_json<T: Into<SvmConfig>>(
        data: SerializedSvm,
        params: T,
    ) -> Result<Self, ClassifierError> {
        let mut classifier = Self::new(params)?;
        if data.weights.is_empty() {
            return Err(ClassifierError::InvalidState(
                "SVM weights must not be empty".into(),
            ));
        }
        let weights = data
            .weights
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let bias = data.bias as f32;
        if weights.iter().any(|value| !value.is_finite()) || !bias.is_finite() {
            return Err(ClassifierError::InvalidState(
                "SVM state contains values outside the finite f32 range".into(),
            ));
        }

        if data
            .class_weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ClassifierError::InvalidState(
                "SVM class weights must be finite and nonnegative".into(),
            ));
        }

        classifier.model = Some(TrainedModel { weights, bias });
        classifier.class_weights = data.class_weights;
        Ok(classifier)
    }

    /// Restores a classifier with the source registry loader's retraining
    /// parameters rather than constructor defaults.
    pub fn from_state(data: SerializedSvm) -> Result<Self, ClassifierError> {
        Self::from_json(
            data,
            SvmConfig {
                c: 0.01,
                max_iterations: 100,
                use_class_weights: true,
            },
        )
    }

    fn dispose_model(&mut self) {
        self.model = None;
    }

    fn validate_prediction(&self, features: &[f32]) -> Result<(), ClassifierError> {
        let expected = self
            .model
            .as_ref()
            .ok_or(ClassifierError::Untrained)?
            .weights
            .len();
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
}

impl BaseClassifier for SvmClassifier {
    fn classifier_id(&self) -> &'static str {
        "svm"
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
        let dimensions = validate_features(features)?;
        for (index, label) in labels.iter().enumerate() {
            if *label != 0 && *label != 1 {
                return Err(ClassifierError::InvalidState(format!(
                    "label at index {index} must be 0 or 1"
                )));
            }
        }

        if self.params.use_class_weights && (!labels.contains(&0) || !labels.contains(&1)) {
            return Err(ClassifierError::MissingClass);
        }

        self.dispose_model();
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

        let mut rng = NormalRng::new(TRAINING_CONFIG.random_seed);
        let mut weights = (0..dimensions)
            .map(|_| rng.normal(0.01))
            .collect::<Vec<_>>();
        let mut bias = 0.0_f32;
        let sample_weights = labels
            .iter()
            .map(|label| self.class_weights[*label as usize] as f32)
            .collect::<Vec<_>>();
        let total_weight = sample_weights.iter().sum::<f32>();

        let mut optimizer = sgd(0.01, 0.9, 0.0, false);
        for _ in 0..self.params.max_iterations {
            let mut weight_gradients = vec![0.0_f32; dimensions];
            let mut bias_gradient = 0.0_f32;

            for ((sample, label), sample_weight) in features.iter().zip(labels).zip(&sample_weights)
            {
                let label_sign = if *label == 0 { -1.0_f32 } else { 1.0_f32 };
                let logit = sample
                    .iter()
                    .zip(&weights)
                    .fold(bias, |sum, (feature, weight)| sum + feature * weight);
                let margin = label_sign * logit;
                if margin < 1.0 {
                    let scale = -label_sign * self.params.c as f32 * *sample_weight / total_weight;
                    for (gradient, feature) in weight_gradients.iter_mut().zip(sample) {
                        *gradient += scale * *feature;
                    }
                    bias_gradient += scale;
                }
            }

            for (gradient, weight) in weight_gradients.iter_mut().zip(&weights) {
                *gradient += 2.0 * *weight;
            }

            let mut variables = vec![
                Variable::new("weights", weights.clone()),
                Variable::new("bias", vec![bias]),
            ];
            let gradients = vec![
                Gradient::new("weights", weight_gradients),
                Gradient::new("bias", vec![bias_gradient]),
            ];
            optimizer
                .apply_gradients(&mut variables, &gradients)
                .map_err(|error| ClassifierError::InvalidState(error.to_string()))?;
            weights.clone_from(&variables[0].values);
            bias = variables[1].values[0];
        }

        optimizer.dispose();
        if weights.iter().any(|value| !value.is_finite()) || !bias.is_finite() {
            return Err(ClassifierError::InvalidState(
                "training failed: model weights contain non-finite values. This indicates numerical instability. Try reducing learning rate or increasing regularization.".into(),
            ));
        }

        self.model = Some(TrainedModel { weights, bias });
        Ok(())
    }

    fn predict_proba(&self, features: &[f32]) -> Result<f64, ClassifierError> {
        self.validate_prediction(features)?;
        let model = self.model.as_ref().ok_or(ClassifierError::Untrained)?;
        let decision = model
            .weights
            .iter()
            .zip(features)
            .fold(model.bias, |sum, (weight, feature)| sum + weight * feature);
        Ok(1.0 - f64::from(sigmoid(decision)))
    }

    fn to_json(&self) -> Result<SerializedClassifierState, ClassifierError> {
        let model = self.model.as_ref().ok_or(ClassifierError::Untrained)?;
        Ok(SerializedClassifierState::Svm(SerializedSvm {
            weights: model
                .weights
                .iter()
                .map(|value| f64::from(*value))
                .collect(),
            bias: f64::from(model.bias),
            class_weights: self.class_weights,
        }))
    }

    fn dispose(&mut self) {
        self.dispose_model();
    }
}

fn validate_config(config: &SvmConfig) -> Result<(), ClassifierError> {
    if !config.c.is_finite() || config.c < 0.0 {
        return Err(ClassifierError::InvalidState(
            "SVM C must be finite and nonnegative".into(),
        ));
    }
    if config.max_iterations > 3_000 {
        return Err(ClassifierError::InvalidState(
            "SVM maxIterations must be at most 3000".into(),
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
    (1.0_f64 / (1.0 + (-f64::from(value)).exp())) as f32
}

/// TensorFlow.js 4.22's seeded `MPRandGauss`: Alea plus the Marsaglia polar
/// method. Each `tf.randomNormal` call starts a fresh generator.
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

    fn normal(&mut self, standard_deviation: f32) -> f32 {
        if let Some(value) = self.spare.take() {
            return (value * f64::from(standard_deviation)) as f32;
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
            return (f64::from(standard_deviation) * first * multiplier) as f32;
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
        generator.s0 = (generator.s0 - mash.apply(seed)).rem_euclid(1.0);
        generator.s1 = (generator.s1 - mash.apply(seed)).rem_euclid(1.0);
        generator.s2 = (generator.s2 - mash.apply(seed)).rem_euclid(1.0);
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
