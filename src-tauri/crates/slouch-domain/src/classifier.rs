use std::{collections::BTreeMap, fmt, str::FromStr};

use serde::{ser::SerializeMap, Deserialize, Serialize, Serializer};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, specta::Type,
)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierId {
    Mlp,
    Knn,
    Svm,
    KmeansPrototype,
    GaussianNb,
    KmeansLogistic,
}

impl ClassifierId {
    pub const ALL: [Self; 6] = [
        Self::Mlp,
        Self::Knn,
        Self::Svm,
        Self::KmeansPrototype,
        Self::GaussianNb,
        Self::KmeansLogistic,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::Knn => "knn",
            Self::Svm => "svm",
            Self::KmeansPrototype => "kmeans_prototype",
            Self::GaussianNb => "gaussian_nb",
            Self::KmeansLogistic => "kmeans_logistic",
        }
    }
}

impl fmt::Display for ClassifierId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ClassifierId {
    type Err = ClassifierLookupError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        normalize_classifier_id(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierLookupError {
    DeprecatedLogisticRegression,
    Unknown(String),
}

impl fmt::Display for ClassifierLookupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
      Self::DeprecatedLogisticRegression => formatter.write_str(
        "LogisticRegression models are no longer supported. Please retrain your model using the MLP classifier.",
      ),
      Self::Unknown(id) => write!(formatter, "unknown classifier ID: {id}"),
    }
    }
}

impl std::error::Error for ClassifierLookupError {}

pub fn normalize_classifier_id(value: &str) -> Result<ClassifierId, ClassifierLookupError> {
    match value {
        "mlp" | "MLPClassifier" => Ok(ClassifierId::Mlp),
        "knn" | "KNNClassifier" => Ok(ClassifierId::Knn),
        "svm" | "SVMClassifier" => Ok(ClassifierId::Svm),
        "kmeans_prototype"
        | "PrototypicalClassifier"
        | "prototypical"
        | "KMeansPrototypeClassifier" => Ok(ClassifierId::KmeansPrototype),
        "gaussian_nb" | "GaussianNBClassifier" => Ok(ClassifierId::GaussianNb),
        "kmeans_logistic" | "KMeansLogisticClassifier" => Ok(ClassifierId::KmeansLogistic),
        "LogisticRegressionClassifier" | "logistic_regression" => {
            Err(ClassifierLookupError::DeprecatedLogisticRegression)
        }
        id => Err(ClassifierLookupError::Unknown(id.to_owned())),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(untagged)]
pub enum ParameterValue {
    Number(f64),
    String(String),
    Boolean(bool),
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnnDistance {
    Euclidean,
    Manhattan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "classifierId", rename_all = "snake_case")]
pub enum ClassifierParameters {
    Mlp {
        #[serde(rename = "weightDecay")]
        weight_decay: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "learningRate")]
        learning_rate: f64,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
        #[serde(rename = "hiddenLayers", skip_serializing_if = "Option::is_none")]
        hidden_layers: Option<usize>,
        #[serde(rename = "hiddenSize", skip_serializing_if = "Option::is_none")]
        hidden_size: Option<usize>,
    },
    Knn {
        k: usize,
        distance: KnnDistance,
    },
    Svm {
        #[serde(rename = "C")]
        c: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
    },
    KmeansPrototype {
        temperature: f64,
    },
    GaussianNb {
        #[serde(rename = "varianceSmoothing")]
        variance_smoothing: f64,
    },
    KmeansLogistic {
        temperature: f64,
        #[serde(rename = "weightDecay")]
        weight_decay: f64,
        #[serde(rename = "maxIterations")]
        max_iterations: usize,
        #[serde(rename = "useClassWeights", skip_serializing_if = "Option::is_none")]
        use_class_weights: Option<bool>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum ParameterType {
    Range,
    Select,
    Number,
    Boolean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "snake_case")]
pub enum ParameterScale {
    Linear,
    Exponential,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct ParameterOption {
    pub value: ParameterValue,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(untagged)]
pub enum ParameterCondition {
    One(ParameterValue),
    Any(Vec<ParameterValue>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct ParameterDefinition {
    #[serde(rename = "type")]
    pub parameter_type: ParameterType,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub default: ParameterValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<ParameterScale>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub options: Vec<ParameterOption>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub show_when: BTreeMap<String, ParameterCondition>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OrderedParameters(Vec<(String, ParameterDefinition)>);

impl OrderedParameters {
    pub fn insert(&mut self, key: String, value: ParameterDefinition) {
        self.0.push((key, value));
    }

    pub fn get(&self, key: &str) -> Option<&ParameterDefinition> {
        self.0
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then_some(value))
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(key, _)| key.as_str())
    }
}

impl Serialize for OrderedParameters {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, specta::Type)]
pub struct ClassifierMetadata {
    pub id: ClassifierId,
    pub name: String,
    pub description: String,
    #[specta(type = BTreeMap<String, ParameterDefinition>)]
    pub params: OrderedParameters,
}

fn parameter(
    parameter_type: ParameterType,
    label: &str,
    description: &str,
    default: impl Into<ParameterValue>,
) -> ParameterDefinition {
    ParameterDefinition {
        parameter_type,
        label: label.to_owned(),
        description: Some(description.to_owned()),
        default: default.into(),
        min: None,
        max: None,
        step: None,
        scale: None,
        options: Vec::new(),
        show_when: BTreeMap::new(),
    }
}

fn range(
    label: &str,
    description: &str,
    default: f64,
    min: f64,
    max: f64,
    step: Option<f64>,
    scale: Option<ParameterScale>,
) -> ParameterDefinition {
    ParameterDefinition {
        min: Some(min),
        max: Some(max),
        step,
        scale,
        ..parameter(ParameterType::Range, label, description, default)
    }
}

fn option(value: impl Into<ParameterValue>, label: &str) -> ParameterOption {
    ParameterOption {
        value: value.into(),
        label: label.to_owned(),
    }
}

pub fn classifier_registry() -> Vec<ClassifierMetadata> {
    vec![
        mlp(),
        knn(),
        svm(),
        kmeans_prototype(),
        gaussian_nb(),
        kmeans_logistic(),
    ]
}

fn mlp() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    let mut hidden_layers = parameter(
        ParameterType::Select,
        "Hidden Layers",
        "0 = logistic regression, 1-2 = MLP with hidden layers",
        0.0,
    );
    hidden_layers.options = vec![
        option(0.0, "None (Logistic)"),
        option(1.0, "1 Layer"),
        option(2.0, "2 Layers"),
    ];
    params.insert("hiddenLayers".into(), hidden_layers);
    let mut hidden_size = parameter(
        ParameterType::Select,
        "Hidden Size",
        "Neurons per hidden layer",
        64.0,
    );
    hidden_size.options = vec![
        option(32.0, "32"),
        option(64.0, "64"),
        option(128.0, "128"),
        option(256.0, "256"),
    ];
    hidden_size.show_when.insert(
        "hiddenLayers".into(),
        ParameterCondition::Any(vec![1.0.into(), 2.0.into()]),
    );
    params.insert("hiddenSize".into(), hidden_size);
    params.insert(
        "weightDecay".into(),
        range(
            "Weight Decay",
            "AdamW weight decay strength (0.01 = moderate, 0.0001 = light)",
            1.0,
            0.01,
            100.0,
            None,
            Some(ParameterScale::Exponential),
        ),
    );
    params.insert(
        "maxIterations".into(),
        range(
            "Max Iterations",
            "Training iterations for convergence",
            100.0,
            10.0,
            1000.0,
            Some(10.0),
            None,
        ),
    );
    params.insert(
        "learningRate".into(),
        range(
            "Learning Rate",
            "Step size for Adam optimizer",
            0.01,
            0.0001,
            0.1,
            None,
            Some(ParameterScale::Exponential),
        ),
    );
    params.insert(
        "useClassWeights".into(),
        parameter(
            ParameterType::Boolean,
            "Use Class Weight Balancing",
            "Automatically adjust training weights for imbalanced datasets (scikit-learn style)",
            false,
        ),
    );
    params.insert(
        "labelSmoothing".into(),
        range(
            "Label Smoothing",
            "Smooth labels to reduce outlier sensitivity (0=off)",
            0.05,
            0.0,
            0.5,
            Some(0.01),
            None,
        ),
    );
    ClassifierMetadata {
        id: ClassifierId::Mlp,
        name: "MLP".into(),
        description: "Multi-layer perceptron (0 hidden layers = logistic regression)".into(),
        params,
    }
}

fn knn() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    params.insert(
        "k".into(),
        range(
            "k (Number of Neighbors)",
            "How many nearest neighbors to consider",
            3.0,
            1.0,
            20.0,
            Some(1.0),
            None,
        ),
    );
    let mut kernel = parameter(
        ParameterType::Select,
        "Distance Kernel",
        "Kernel for computing neighbor similarities",
        "cosine",
    );
    kernel.options = vec![
        option("cosine", "Cosine Similarity"),
        option("rbf", "RBF (Gaussian)"),
    ];
    params.insert("kernel".into(), kernel);
    let mut gamma = range(
        "RBF Gamma",
        "Kernel width (higher = sharper, more local)",
        1.0,
        0.001,
        10.0,
        Some(0.001),
        Some(ParameterScale::Exponential),
    );
    gamma
        .show_when
        .insert("kernel".into(), ParameterCondition::One("rbf".into()));
    params.insert("gamma".into(), gamma);
    ClassifierMetadata {
        id: ClassifierId::Knn,
        name: "K-Nearest Neighbors".into(),
        description: "Instance-based learning with configurable distance kernel".into(),
        params,
    }
}

fn svm() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    params.insert(
        "C".into(),
        range(
            "C (Regularization)",
            "Lower C = wider margin, more regularization (0.1 = strong, 10 = weak)",
            0.01,
            0.0001,
            100.0,
            None,
            Some(ParameterScale::Exponential),
        ),
    );
    params.insert(
        "maxIterations".into(),
        range(
            "Max Iterations",
            "Training iterations for convergence",
            100.0,
            10.0,
            3000.0,
            Some(10.0),
            None,
        ),
    );
    params.insert(
        "useClassWeights".into(),
        parameter(
            ParameterType::Boolean,
            "Use Class Weight Balancing",
            "Automatically adjust training weights for imbalanced datasets (scikit-learn style)",
            true,
        ),
    );
    ClassifierMetadata {
        id: ClassifierId::Svm,
        name: "Support Vector Machine".into(),
        description: "Linear SVM with hinge loss and maximum-margin separation".into(),
        params,
    }
}

fn kmeans_prototype() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    params.insert(
        "nClusters".into(),
        range(
            "Number of Clusters",
            "0 = auto-select based on data, 1-16 = manual selection",
            0.0,
            0.0,
            16.0,
            Some(1.0),
            None,
        ),
    );
    params.insert(
        "temperature".into(),
        range(
            "Temperature",
            "Softmax temperature for routing (lower = sharper routing)",
            1.0,
            0.1,
            5.0,
            Some(0.1),
            None,
        ),
    );
    ClassifierMetadata { id: ClassifierId::KmeansPrototype, name: "K-Means Prototype".into(), description: "K-means clustering to discover modes, then prototype matching (class mean) per cluster. Soft routing via distance to centroids.".into(), params }
}

fn gaussian_nb() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    params.insert(
        "varianceSmoothing".into(),
        range(
            "Variance Smoothing",
            "Stability term added to variances to prevent division by zero",
            1e-6,
            1e-12,
            1e-2,
            None,
            Some(ParameterScale::Exponential),
        ),
    );
    ClassifierMetadata {
        id: ClassifierId::GaussianNb,
        name: "Gaussian Naive Bayes".into(),
        description:
            "Probabilistic classifier ideal for small datasets with low-dimensional features".into(),
        params,
    }
}

fn kmeans_logistic() -> ClassifierMetadata {
    let mut params = OrderedParameters::default();
    params.insert(
        "nClusters".into(),
        range(
            "Number of Clusters",
            "0 = auto-select based on data, 1-16 = manual selection",
            0.0,
            0.0,
            16.0,
            Some(1.0),
            None,
        ),
    );
    params.insert(
        "temperature".into(),
        range(
            "Temperature",
            "Soft routing sharpness (lower = harder routing to nearest cluster)",
            1.0,
            0.1,
            5.0,
            Some(0.1),
            None,
        ),
    );
    params.insert(
        "weightDecay".into(),
        range(
            "Weight Decay",
            "L2 regularization for logistic models",
            1.0,
            0.01,
            100.0,
            None,
            Some(ParameterScale::Exponential),
        ),
    );
    params.insert(
        "maxIterations".into(),
        range(
            "Max Iterations",
            "Training iterations per logistic model",
            100.0,
            10.0,
            1000.0,
            Some(10.0),
            None,
        ),
    );
    params.insert(
        "useClassWeights".into(),
        parameter(
            ParameterType::Boolean,
            "Use Class Weight Balancing",
            "Balance classes in logistic training",
            false,
        ),
    );
    ClassifierMetadata { id: ClassifierId::KmeansLogistic, name: "K-Means Logistic".into(), description: "Mode-aware classifier: K-means discovers clusters, logistic regression classifies within each. Best when data has distinct modes.".into(), params }
}
