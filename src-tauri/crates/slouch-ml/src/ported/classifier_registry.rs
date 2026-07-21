//! Registry and factory helpers for the posture classifiers.
//!
//! Mechanical Rust port of `src/services/ml/classifierRegistry.ts`. The
//! schema metadata remains owned by `slouch-domain`; this module adds the
//! classifier constructors and serialized-model loaders around the shared
//! `BaseClassifier` contract.

use slouch_domain::{
    classifier_registry as domain_classifier_registry, normalize_classifier_id, ClassifierConfig,
    ClassifierId, ClassifierLookupError, OrderedParameters, ParameterValue,
};

use super::base_classifier::{BaseClassifier, ClassifierError};
use super::kmeans_logistic_classifier::{KMeansLogisticClassifier, KMeansLogisticConfig};
use super::kmeans_prototype_classifier::KMeansPrototypeClassifier;
use super::knn_classifier::{KnnClassifier, KnnConfig};
use super::mlp_classifier::{MlpClassifier, MlpConfig};
use super::naive_bayes_classifier::GaussianNbClassifier;
use super::svm_classifier::{SvmClassifier, SvmConfig};
use super::types::{KnnKernel, SerializedClassifierState, SerializedModel};

pub use slouch_domain::{
    ParameterCondition, ParameterDefinition as ParamDefinition, ParameterOption, ParameterScale,
    ParameterType as ParamType,
};

const DEFAULT_MLP_WEIGHT_DECAY: f64 = 0.01;
const DEFAULT_MLP_MAX_ITERATIONS: usize = 1000;
const DEFAULT_MLP_LEARNING_RATE: f64 = 0.01;
const DEFAULT_MLP_LABEL_SMOOTHING: f64 = 0.1;
const LOAD_MLP_WEIGHT_DECAY: f64 = 1.0;
const LOAD_MLP_LABEL_SMOOTHING: f64 = 0.1;
const LOAD_MLP_MAX_ITERATIONS: usize = 1000;
const DEFAULT_MLP_HIDDEN_LAYERS: usize = 0;
const DEFAULT_MLP_HIDDEN_SIZE: usize = 64;
const DEFAULT_KNN_K: usize = 5;
const LOAD_KNN_K: usize = 3;
const DEFAULT_KNN_GAMMA: f64 = 1.0;
const DEFAULT_SVM_C: f64 = 1.0;
const DEFAULT_SVM_MAX_ITERATIONS: usize = 1000;
const LOAD_SVM_C: f64 = 0.01;
const LOAD_SVM_MAX_ITERATIONS: usize = 100;
const DEFAULT_PROTOTYPE_TEMPERATURE: f64 = 1.0;
const DEFAULT_GAUSSIAN_NB_VARIANCE_SMOOTHING: f64 = 1e-6;
const DEFAULT_KMEANS_LOGISTIC_WEIGHT_DECAY: f64 = 1.0;
const DEFAULT_KMEANS_LOGISTIC_MAX_ITERATIONS: usize = 100;
const DEFAULT_KMEANS_LOGISTIC_TEMPERATURE: f64 = 1.0;

pub type ClassifierInstance = Box<dyn BaseClassifier>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OrderedParameterValues(Vec<(String, ParameterValue)>);

impl OrderedParameterValues {
    pub fn get(&self, key: &str) -> Option<&ParameterValue> {
        self.0
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then_some(value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(key, _)| key.as_str())
    }
}

impl Extend<(String, ParameterValue)> for OrderedParameterValues {
    fn extend<T: IntoIterator<Item = (String, ParameterValue)>>(&mut self, iter: T) {
        for (key, value) in iter {
            if let Some(index) = self.0.iter().position(|(candidate, _)| candidate == &key) {
                self.0[index].1 = value;
            } else {
                self.0.push((key, value));
            }
        }
    }
}

impl<'a> IntoIterator for &'a OrderedParameterValues {
    type Item = &'a (String, ParameterValue);
    type IntoIter = std::slice::Iter<'a, (String, ParameterValue)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

type CreateFn = fn(&ClassifierConfig) -> Result<ClassifierInstance, ClassifierError>;
type LoadFn = fn(&SerializedModel) -> Result<ClassifierInstance, ClassifierError>;

/// Classifier metadata plus the constructor and model-loader for that ID.
pub struct ClassifierDefinition {
    pub id: ClassifierId,
    pub name: String,
    pub description: String,
    pub params: OrderedParameters,
    create_fn: CreateFn,
    load_fn: LoadFn,
}

impl ClassifierDefinition {
    pub fn create(&self, config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
        validate_classifier_config(self, config)?;
        (self.create_fn)(config)
    }

    pub fn load(&self, model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
        (self.load_fn)(model)
    }
}

/// Returns the registry in the same declaration order as the TypeScript
/// object. Each call returns owned metadata so callers can safely retain it.
pub fn classifier_registry() -> Vec<ClassifierDefinition> {
    domain_classifier_registry()
        .into_iter()
        .map(|metadata| {
            let (create_fn, load_fn) = functions_for(metadata.id);
            ClassifierDefinition {
                id: metadata.id,
                name: metadata.name,
                description: metadata.description,
                params: metadata.params,
                create_fn,
                load_fn,
            }
        })
        .collect()
}

/// Gets the default parameter values for a classifier.
pub fn get_default_params(classifier_id: &str) -> Result<OrderedParameterValues, ClassifierError> {
    let definition = get_classifier_definition(classifier_id)?;
    let params = definition
        .params
        .keys()
        .filter_map(|key| {
            definition
                .params
                .get(key)
                .map(|value| (key.to_owned(), value.default.clone()))
        })
        .collect::<Vec<_>>();
    Ok(OrderedParameterValues(params))
}

/// Creates a classifier from a schema-driven training configuration.
pub fn create_classifier(config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
    let definition = get_classifier_definition(config.classifier_id.as_str())?;
    definition.create(config)
}

/// Loads a classifier from a complete serialized model.
pub fn load_classifier(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    let definition = get_classifier_definition(&model.classifier.classifier_id)?;
    definition.load(model)
}

/// Loads one serialized classifier state through the registry's canonical
/// source-compatible loader defaults.
pub fn load_classifier_state(
    classifier_id: ClassifierId,
    state: SerializedClassifierState,
) -> Result<ClassifierInstance, ClassifierError> {
    match (classifier_id, state) {
        (ClassifierId::Mlp, SerializedClassifierState::Mlp(state)) => {
            let config = MlpConfig {
                weight_decay: LOAD_MLP_WEIGHT_DECAY,
                max_iterations: LOAD_MLP_MAX_ITERATIONS,
                learning_rate: DEFAULT_MLP_LEARNING_RATE,
                use_class_weights: true,
                label_smoothing: LOAD_MLP_LABEL_SMOOTHING,
                hidden_layers: state.hidden_layers,
                hidden_size: state.hidden_size,
            };
            Ok(Box::new(MlpClassifier::from_json(state, config)?))
        }
        (ClassifierId::Knn, SerializedClassifierState::Knn(state)) => {
            let config = KnnConfig {
                k: if state.k == 0 { LOAD_KNN_K } else { state.k },
                kernel: state.kernel.unwrap_or(KnnKernel::Cosine),
                gamma: state.gamma.unwrap_or(DEFAULT_KNN_GAMMA),
            };
            Ok(Box::new(KnnClassifier::from_json(state, config)?))
        }
        (ClassifierId::Svm, SerializedClassifierState::Svm(state)) => {
            let config = SvmConfig {
                c: LOAD_SVM_C,
                max_iterations: LOAD_SVM_MAX_ITERATIONS,
                use_class_weights: true,
            };
            Ok(Box::new(SvmClassifier::from_json(state, config)?))
        }
        (ClassifierId::KmeansPrototype, SerializedClassifierState::KMeansPrototype(state)) => {
            let temperature = state.temperature;
            Ok(Box::new(KMeansPrototypeClassifier::from_json(
                state,
                temperature,
            )?))
        }
        (ClassifierId::GaussianNb, SerializedClassifierState::GaussianNb(state)) => {
            let epsilon = state.epsilon;
            Ok(Box::new(GaussianNbClassifier::from_json(state, epsilon)?))
        }
        (ClassifierId::KmeansLogistic, SerializedClassifierState::KMeansLogistic(state)) => {
            let config = KMeansLogisticConfig {
                temperature: state.temperature,
                weight_decay: DEFAULT_KMEANS_LOGISTIC_WEIGHT_DECAY,
                max_iterations: DEFAULT_KMEANS_LOGISTIC_MAX_ITERATIONS,
                use_class_weights: false,
                n_clusters: 0,
            };
            Ok(Box::new(KMeansLogisticClassifier::from_json(
                state, config,
            )?))
        }
        (classifier_id, _) => Err(state_mismatch(classifier_id.as_str())),
    }
}

/// Gets one classifier definition by its current registry ID.
pub fn get_classifier_definition(
    classifier_id: &str,
) -> Result<ClassifierDefinition, ClassifierError> {
    classifier_registry()
        .into_iter()
        .find(|definition| definition.id.as_str() == classifier_id)
        .ok_or_else(|| unknown_classifier(classifier_id))
}

/// Gets all available classifier definitions.
pub fn get_all_classifiers() -> Vec<ClassifierDefinition> {
    classifier_registry()
}

/// Resolves current and legacy classifier names for the compatibility factory.
pub fn normalize_registry_classifier_id(
    value: &str,
) -> Result<ClassifierId, ClassifierLookupError> {
    normalize_classifier_id(value)
}

fn functions_for(id: ClassifierId) -> (CreateFn, LoadFn) {
    match id {
        ClassifierId::Mlp => (create_mlp, load_mlp),
        ClassifierId::Knn => (create_knn, load_knn),
        ClassifierId::Svm => (create_svm, load_svm),
        ClassifierId::KmeansPrototype => (create_kmeans_prototype, load_kmeans_prototype),
        ClassifierId::GaussianNb => (create_gaussian_nb, load_gaussian_nb),
        ClassifierId::KmeansLogistic => (create_kmeans_logistic, load_kmeans_logistic),
    }
}

fn create_mlp(config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(MlpClassifier::new(MlpConfig {
        weight_decay: number(config, "weightDecay", DEFAULT_MLP_WEIGHT_DECAY),
        max_iterations: usize_number(config, "maxIterations", DEFAULT_MLP_MAX_ITERATIONS),
        learning_rate: number(config, "learningRate", DEFAULT_MLP_LEARNING_RATE),
        use_class_weights: boolean(config, "useClassWeights", false),
        label_smoothing: number(config, "labelSmoothing", DEFAULT_MLP_LABEL_SMOOTHING),
        hidden_layers: usize_number(config, "hiddenLayers", DEFAULT_MLP_HIDDEN_LAYERS),
        hidden_size: usize_number(config, "hiddenSize", DEFAULT_MLP_HIDDEN_SIZE),
    })?))
}

fn create_knn(config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(KnnClassifier::with_config(KnnConfig {
        k: usize_number(config, "k", DEFAULT_KNN_K),
        kernel: kernel(config),
        gamma: number(config, "gamma", DEFAULT_KNN_GAMMA),
    })?))
}

fn create_svm(config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(SvmClassifier::new(SvmConfig {
        c: number(config, "C", DEFAULT_SVM_C),
        max_iterations: usize_number(config, "maxIterations", DEFAULT_SVM_MAX_ITERATIONS),
        use_class_weights: boolean(config, "useClassWeights", false),
    })?))
}

fn create_kmeans_prototype(
    config: &ClassifierConfig,
) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(KMeansPrototypeClassifier::new(
        number(config, "temperature", DEFAULT_PROTOTYPE_TEMPERATURE),
        usize_number(config, "nClusters", 0),
    )?))
}

fn create_gaussian_nb(config: &ClassifierConfig) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(GaussianNbClassifier::new(number(
        config,
        "varianceSmoothing",
        DEFAULT_GAUSSIAN_NB_VARIANCE_SMOOTHING,
    ))?))
}

fn create_kmeans_logistic(
    config: &ClassifierConfig,
) -> Result<ClassifierInstance, ClassifierError> {
    Ok(Box::new(KMeansLogisticClassifier::new(
        number(config, "temperature", DEFAULT_KMEANS_LOGISTIC_TEMPERATURE),
        number(config, "weightDecay", DEFAULT_KMEANS_LOGISTIC_WEIGHT_DECAY),
        usize_number(
            config,
            "maxIterations",
            DEFAULT_KMEANS_LOGISTIC_MAX_ITERATIONS,
        ),
        boolean(config, "useClassWeights", false),
        usize_number(config, "nClusters", 0),
    )?))
}

fn load_mlp(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(ClassifierId::Mlp, model.classifier.state.clone())
}

fn load_knn(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(ClassifierId::Knn, model.classifier.state.clone())
}

fn load_svm(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(ClassifierId::Svm, model.classifier.state.clone())
}

fn load_kmeans_prototype(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(
        ClassifierId::KmeansPrototype,
        model.classifier.state.clone(),
    )
}

fn load_gaussian_nb(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(ClassifierId::GaussianNb, model.classifier.state.clone())
}

fn load_kmeans_logistic(model: &SerializedModel) -> Result<ClassifierInstance, ClassifierError> {
    load_classifier_state(ClassifierId::KmeansLogistic, model.classifier.state.clone())
}

fn validate_classifier_config(
    definition: &ClassifierDefinition,
    config: &ClassifierConfig,
) -> Result<(), ClassifierError> {
    if config.classifier_id != definition.id {
        return Err(ClassifierError::InvalidState(
            "classifier configuration ID does not match the selected registry definition"
                .to_owned(),
        ));
    }

    for (key, value) in &config.params {
        let parameter = definition.params.get(key).ok_or_else(|| {
            ClassifierError::InvalidState(format!("unknown classifier parameter {key}"))
        })?;

        match (&parameter.parameter_type, value) {
            (ParamType::Boolean, ParameterValue::Boolean(_)) => {}
            (ParamType::Select, candidate) => {
                if !parameter
                    .options
                    .iter()
                    .any(|option| option.value == *candidate)
                {
                    return Err(ClassifierError::InvalidState(format!(
                        "classifier parameter {key} is not one of the declared options"
                    )));
                }
            }
            (ParamType::Range | ParamType::Number, ParameterValue::Number(number)) => {
                if !number.is_finite() {
                    return Err(ClassifierError::InvalidState(format!(
                        "classifier parameter {key} must be finite"
                    )));
                }
                if parameter.min.is_some_and(|minimum| *number < minimum)
                    || parameter.max.is_some_and(|maximum| *number > maximum)
                {
                    return Err(ClassifierError::InvalidState(format!(
                        "classifier parameter {key} is outside the declared registry bounds"
                    )));
                }
                if is_integer_parameter(key) && number.fract() != 0.0 {
                    return Err(ClassifierError::InvalidState(format!(
                        "classifier parameter {key} must be an integer"
                    )));
                }
            }
            _ => {
                return Err(ClassifierError::InvalidState(format!(
                    "classifier parameter {key} has the wrong type"
                )));
            }
        }
    }

    Ok(())
}

fn is_integer_parameter(key: &str) -> bool {
    matches!(
        key,
        "hiddenLayers" | "hiddenSize" | "maxIterations" | "k" | "nClusters"
    )
}

fn state_mismatch(classifier_id: &str) -> ClassifierError {
    ClassifierError::InvalidState(format!(
        "serialized state does not match classifier {classifier_id}"
    ))
}

fn unknown_classifier(classifier_id: &str) -> ClassifierError {
    ClassifierError::InvalidState(format!("Unknown classifier ID: {classifier_id}"))
}

fn number(config: &ClassifierConfig, key: &str, default: f64) -> f64 {
    match config.params.get(key) {
        Some(ParameterValue::Number(value)) => *value,
        _ => default,
    }
}

fn usize_number(config: &ClassifierConfig, key: &str, default: usize) -> usize {
    number(config, key, default as f64) as usize
}

fn boolean(config: &ClassifierConfig, key: &str, default: bool) -> bool {
    match config.params.get(key) {
        Some(ParameterValue::Boolean(value)) => *value,
        _ => default,
    }
}

fn kernel(config: &ClassifierConfig) -> KnnKernel {
    match config.params.get("kernel") {
        Some(ParameterValue::String(value)) if value == "rbf" => KnnKernel::Rbf,
        _ => KnnKernel::Cosine,
    }
}
