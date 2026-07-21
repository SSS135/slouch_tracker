//! Canonical `SLMD` model records and `SLCF` training-configuration fingerprints.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};
use slouch_domain::{
    CrossValidationType, FeatureId, ParameterValue, TrainingMetrics, TrainingSettings,
};
use slouch_ml::ported::{
    classifier_registry::get_default_params,
    pca::SerializedPca,
    random_projection::RandomProjectionState,
    types::{
        DimReductionTransformer, DimensionalityReductionMethod, KnnKernel, NormalizationMode,
        SerializedClassifier, SerializedClassifierState, SerializedFeatureExtractor,
        SerializedGaussianNb, SerializedKMeansLogistic, SerializedKMeansPrototype, SerializedKnn,
        SerializedMlp, SerializedModel, SerializedSvm,
    },
};

const VERSION: u16 = 1;
const MAX_RECORDS: usize = 256;
const MAX_RECORD_NAME: usize = 64;
const MAX_CONTAINER_BYTES: usize = 256 * 1024 * 1024;
const MAX_VALUE_BYTES: usize = 1024 * 1024;
const MAX_TENSOR_BYTES: usize = 64 * 1024 * 1024;
const MAX_TENSOR_ELEMENTS: usize = 16_777_216;
const MAX_SAFE_JS_INTEGER: i64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Posture,
    Presence,
}

impl ModelRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Posture => "posture",
            Self::Presence => "presence",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEnvelopeMetadata {
    pub role: ModelRole,
    pub dataset_version: u64,
    pub training_config_sha256: [u8; 32],
    pub classifier_id: String,
}

#[derive(Debug, Clone, PartialEq)]
struct Record {
    name: String,
    kind: u8,
    dimensions: Vec<u32>,
    bytes: Vec<u8>,
}

impl Record {
    fn scalar(name: impl Into<String>, kind: u8, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            kind,
            dimensions: Vec::new(),
            bytes,
        }
    }
    fn utf8(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::scalar(name, 7, value.into().into_bytes())
    }
    fn bytes(name: impl Into<String>, value: Vec<u8>) -> Self {
        Self::scalar(name, 8, value)
    }
    fn u8(name: impl Into<String>, value: u8) -> Self {
        Self::scalar(name, 1, vec![value])
    }
    fn u32(name: impl Into<String>, value: u32) -> Self {
        Self::scalar(name, 2, value.to_le_bytes().to_vec())
    }
    fn u64(name: impl Into<String>, value: u64) -> Self {
        Self::scalar(name, 3, value.to_le_bytes().to_vec())
    }
    fn i64(name: impl Into<String>, value: i64) -> Self {
        Self::scalar(name, 4, value.to_le_bytes().to_vec())
    }
    fn f32(name: impl Into<String>, value: f32) -> Result<Self, String> {
        finite_f32(value)?;
        Ok(Self::scalar(name, 5, value.to_le_bytes().to_vec()))
    }
    fn f64(name: impl Into<String>, value: f64) -> Result<Self, String> {
        finite_f64(value)?;
        Ok(Self::scalar(name, 6, value.to_le_bytes().to_vec()))
    }
    fn tensor_f32(
        name: impl Into<String>,
        dimensions: Vec<u32>,
        values: &[f64],
    ) -> Result<Self, String> {
        check_tensor_shape(&dimensions, values.len())?;
        let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
        for value in values {
            let converted = *value as f32;
            if !value.is_finite() || !converted.is_finite() {
                return Err("model tensor contains a non-finite or out-of-range value".into());
            }
            bytes.extend_from_slice(&converted.to_le_bytes());
        }
        Ok(Self {
            name: name.into(),
            kind: 9,
            dimensions,
            bytes,
        })
    }
    fn tensor_u8(
        name: impl Into<String>,
        dimensions: Vec<u32>,
        values: Vec<u8>,
    ) -> Result<Self, String> {
        check_tensor_shape(&dimensions, values.len())?;
        Ok(Self {
            name: name.into(),
            kind: 10,
            dimensions,
            bytes: values,
        })
    }
    fn tensor_u32(
        name: impl Into<String>,
        dimensions: Vec<u32>,
        values: &[usize],
    ) -> Result<Self, String> {
        check_tensor_shape(&dimensions, values.len())?;
        let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
        for value in values {
            bytes.extend_from_slice(
                &u32::try_from(*value)
                    .map_err(|_| "u32 tensor value is too large")?
                    .to_le_bytes(),
            );
        }
        Ok(Self {
            name: name.into(),
            kind: 11,
            dimensions,
            bytes,
        })
    }
}

pub fn training_config_fingerprint(
    settings: &TrainingSettings,
    do_cv: bool,
) -> Result<[u8; 32], String> {
    let mut records = Vec::new();
    records.push(Record::utf8("pair.dataset_selection", "all_labeled"));
    records.push(Record::u8("pair.include_reservoir", 1));
    let mut effective = get_default_params(settings.classifier_config.classifier_id.as_str())
        .map_err(|error| error.to_string())?;
    effective.extend(settings.classifier_config.params.clone());
    for (prefix, features) in [
        ("posture", &settings.posture_feature_types),
        ("presence", &settings.presence_feature_types),
    ] {
        validate_training_feature_order(features).map_err(|error| format!("{prefix} {error}"))?;
        let ids = features
            .iter()
            .map(|feature| feature.as_str())
            .collect::<Vec<_>>();
        records.push(Record::utf8(format!("{prefix}.feature.ids"), ids.join(",")));
        records.push(Record::tensor_u32(
            format!("{prefix}.feature.dimensions"),
            vec![u32::try_from(features.len()).map_err(|_| "too many features")?],
            &features
                .iter()
                .map(|feature| feature.metadata().dimensions)
                .collect::<Vec<_>>(),
        )?);
        records.push(Record::utf8(
            format!("{prefix}.normalization.mode"),
            match settings
                .normalization_mode
                .unwrap_or(slouch_domain::NormalizationMode::None)
            {
                slouch_domain::NormalizationMode::None => "none",
                slouch_domain::NormalizationMode::Layer => "layer",
                slouch_domain::NormalizationMode::ZScore => "z_score",
                slouch_domain::NormalizationMode::Calibrated => "calibrated",
            },
        ));
        let reduction = match settings.dim_reduction_config.method {
            slouch_domain::DimensionalityReductionMethod::None => "none",
            slouch_domain::DimensionalityReductionMethod::RandomProjection => "random_projection",
            slouch_domain::DimensionalityReductionMethod::Pca => "pca",
        };
        records.push(Record::utf8(
            format!("{prefix}.reduction.method"),
            reduction,
        ));
        records.push(Record::u32(
            format!("{prefix}.reduction.components"),
            u32::try_from(settings.dim_reduction_config.components)
                .map_err(|_| "reduction components exceed u32")?,
        ));
        if reduction == "random_projection" {
            records.push(Record::utf8(
                format!("{prefix}.reduction.rng"),
                "seedrandom",
            ));
            records.push(Record::utf8(format!("{prefix}.reduction.seed"), "42"));
        }
        records.push(Record::utf8(
            format!("{prefix}.classifier.id"),
            settings.classifier_config.classifier_id.as_str(),
        ));
        for (name, value) in &effective {
            let name = canonical_parameter_name(name)?;
            let record_name = format!("{prefix}.classifier.{name}");
            records.push(match value {
                ParameterValue::Number(value) => Record::f64(record_name, *value)?,
                ParameterValue::String(value) => Record::utf8(record_name, value.to_lowercase()),
                ParameterValue::Boolean(value) => Record::u8(record_name, u8::from(*value)),
            });
        }
        records.push(Record::u8(format!("{prefix}.cv.enabled"), u8::from(do_cv)));
        records.push(Record::u32(
            format!("{prefix}.cv.folds"),
            u32::try_from(settings.cv_folds).map_err(|_| "CV folds exceed u32")?,
        ));
        records.push(Record::utf8(format!("{prefix}.cv.seed"), "42"));
        let class_weights = effective
            .get("useClassWeights")
            .and_then(|value| match value {
                ParameterValue::Boolean(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(false);
        records.push(Record::u8(
            format!("{prefix}.class_weights"),
            u8::from(class_weights),
        ));
        records.push(Record::f64(format!("{prefix}.probability_threshold"), 0.5)?);
    }
    let bytes = encode_config_container(&mut records)?;
    Ok(sha256(&bytes))
}

pub fn encode_model(
    model: &SerializedModel,
    role: ModelRole,
    dataset_version: u64,
    training_config_sha256: [u8; 32],
    metrics: Option<&TrainingMetrics>,
) -> Result<Vec<u8>, String> {
    let extractor = &model.feature_extractor;
    let input_dimension = validate_extractor(extractor)?;
    let classifier_id = canonical_classifier_id(&model.classifier)?;
    let trained_at = finite_f64(model.trained_at)?;
    if trained_at <= 0.0 || trained_at.fract() != 0.0 || trained_at > MAX_SAFE_JS_INTEGER as f64 {
        return Err(
            "trained_at must be a positive JavaScript-safe integer millisecond timestamp".into(),
        );
    }
    let mut records = vec![
        Record::utf8("classifier.id", classifier_id),
        Record::u32("classifier.state_version", 1),
        Record::u64("dataset.version", dataset_version),
        Record::u32(
            "feature.input_dimension",
            u32::try_from(input_dimension).map_err(|_| "input dimension exceeds u32")?,
        ),
        Record::utf8("feature.ids", extractor.feature_types.join(",")),
        Record::utf8("role", role.as_str()),
        Record::i64("trained_at_ms", trained_at as i64),
        Record::bytes("training_config.sha256", training_config_sha256.to_vec()),
    ];
    encode_extractor(extractor, input_dimension, &mut records)?;
    if let Some(metrics) = metrics {
        encode_metrics(metrics, &mut records)?;
    }
    encode_classifier(
        &model.classifier,
        extractor_output_dimension(extractor, input_dimension)?,
        &mut records,
    )?;
    encode_container(*b"SLMD", VERSION, &mut records)
}

pub fn decode_model(payload: &[u8]) -> Result<(SerializedModel, ModelEnvelopeMetadata), String> {
    let (state_version, records) = decode_container(payload, *b"SLMD")?;
    if state_version != 1 {
        return Err(format!(
            "unsupported classifier state version {state_version}"
        ));
    }
    let role = match string(&records, "role")? {
        "posture" => ModelRole::Posture,
        "presence" => ModelRole::Presence,
        _ => return Err("model role must be posture or presence".into()),
    };
    let dataset_version = scalar_u64(&records, "dataset.version")?;
    let config = raw(&records, "training_config.sha256", 8)?;
    if config.len() != 32 {
        return Err("training_config.sha256 must contain 32 bytes".into());
    }
    let mut training_config_sha256 = [0_u8; 32];
    training_config_sha256.copy_from_slice(config);
    let classifier_id = string(&records, "classifier.id")?.to_owned();
    if scalar_u32(&records, "classifier.state_version")? != 1 {
        return Err("classifier.state_version must be 1".into());
    }
    let feature_types = string(&records, "feature.ids")?
        .split(',')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if feature_types.is_empty() || feature_types.iter().any(String::is_empty) {
        return Err("feature.ids is invalid".into());
    }
    let input_dimension = usize::try_from(scalar_u32(&records, "feature.input_dimension")?)
        .map_err(|_| "input dimension is invalid")?;
    let extractor = decode_extractor(&records, feature_types, input_dimension)?;
    if validate_extractor(&extractor)? != input_dimension {
        return Err("feature.input_dimension does not match the feature registry".into());
    }
    let output_dimension = extractor_output_dimension(&extractor, input_dimension)?;
    let classifier = decode_classifier(&records, &classifier_id, output_dimension)?;
    validate_metrics(&records)?;
    let trained_at = scalar_i64(&records, "trained_at_ms")?;
    if !(1..=MAX_SAFE_JS_INTEGER).contains(&trained_at) {
        return Err("trained_at_ms must be a positive JavaScript-safe integer".into());
    }
    validate_exact_allowlist(&records, &extractor, &classifier)?;
    Ok((
        SerializedModel {
            feature_extractor: extractor,
            classifier,
            trained_at: trained_at as f64,
            version: 1.0,
        },
        ModelEnvelopeMetadata {
            role,
            dataset_version,
            training_config_sha256,
            classifier_id,
        },
    ))
}

pub fn sha256(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    hasher.finalize().into()
}

fn encode_metrics(metrics: &TrainingMetrics, records: &mut Vec<Record>) -> Result<(), String> {
    records.push(Record::f64("metric.cv_accuracy", metrics.cv_accuracy)?);
    records.push(Record::f64("metric.cv_std", metrics.cv_std)?);
    records.push(Record::f64("metric.mcc", metrics.mcc)?);
    records.push(Record::f64("metric.f1_score", metrics.f1_score)?);
    if metrics.confusion_matrix.len() != 2
        || metrics.confusion_matrix.iter().any(|row| row.len() != 2)
    {
        return Err("cross-validation confusion matrix must be 2x2".into());
    }
    let confusion = metrics
        .confusion_matrix
        .iter()
        .flatten()
        .map(|value| {
            usize::try_from(*value).map_err(|_| "confusion-matrix value exceeds platform limits")
        })
        .collect::<Result<Vec<_>, _>>()?;
    records.push(Record::tensor_u32(
        "metric.confusion_matrix",
        vec![2, 2],
        &confusion,
    )?);
    if metrics.fold_accuracies.is_empty() {
        return Err("cross-validation fold accuracies must not be empty".into());
    }
    records.push(Record::tensor_f32(
        "metric.fold_accuracies",
        vec![u32::try_from(metrics.fold_accuracies.len()).map_err(|_| "too many CV folds")?],
        &metrics.fold_accuracies,
    )?);
    let cv_type = match metrics.cv_type.ok_or("cross-validation type is missing")? {
        CrossValidationType::TemporalBlock => "temporal_block",
        CrossValidationType::ShuffledStratified => "shuffled_stratified",
    };
    records.push(Record::utf8("metric.cv_type", cv_type));
    Ok(())
}

fn validate_metrics(records: &BTreeMap<String, Record>) -> Result<(), String> {
    let names = [
        "metric.cv_accuracy",
        "metric.cv_std",
        "metric.mcc",
        "metric.f1_score",
        "metric.confusion_matrix",
        "metric.fold_accuracies",
        "metric.cv_type",
    ];
    let present = names
        .iter()
        .filter(|name| records.contains_key(**name))
        .count();
    if present == 0 {
        return Ok(());
    }
    if present != names.len() {
        return Err("cross-validation metric records are incomplete".into());
    }
    for name in &names[..4] {
        scalar_f64(records, name)?;
    }
    tensor_usize(records, "metric.confusion_matrix", &[2, 2])?;
    let folds = record(records, "metric.fold_accuracies", 9)?;
    if folds.dimensions.len() != 1 || folds.dimensions[0] == 0 {
        return Err("cross-validation fold accuracies shape is invalid".into());
    }
    tensor_f64(records, "metric.fold_accuracies", &folds.dimensions)?;
    if !matches!(
        string(records, "metric.cv_type")?,
        "temporal_block" | "shuffled_stratified"
    ) {
        return Err("cross-validation type is invalid".into());
    }
    Ok(())
}

fn encode_extractor(
    extractor: &SerializedFeatureExtractor,
    input: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    let normalization = match extractor.normalization_mode {
        NormalizationMode::None => "none",
        NormalizationMode::Layer => "layer",
        NormalizationMode::ZScore => "z_score",
        NormalizationMode::Calibrated => "calibrated",
    };
    records.push(Record::utf8("normalization.mode", normalization));
    match extractor.normalization_mode {
        NormalizationMode::ZScore | NormalizationMode::Calibrated => {
            records.push(Record::tensor_f32(
                "normalization.mean",
                vec![input as u32],
                extractor
                    .normalization_mean
                    .as_deref()
                    .ok_or("z-score normalization mean is missing")?,
            )?);
            let std = extractor
                .normalization_std
                .as_deref()
                .ok_or("z-score normalization std is missing")?;
            if std.iter().any(|value| !value.is_finite() || *value <= 0.0) {
                return Err("z-score normalization std values must be positive and finite".into());
            }
            records.push(Record::tensor_f32(
                "normalization.std",
                vec![input as u32],
                std,
            )?);
        }
        _ if extractor.normalization_mean.is_some() || extractor.normalization_std.is_some() => {
            return Err("normalization tensors are forbidden for this mode".into())
        }
        _ => {}
    }
    let output = extractor_output_dimension(extractor, input)?;
    let method = match extractor.dim_reduction_config.method {
        DimensionalityReductionMethod::None => "none",
        DimensionalityReductionMethod::RandomProjection => "random_projection",
        DimensionalityReductionMethod::Pca => "pca",
        _ => return Err("unsupported reduction method".into()),
    };
    records.push(Record::utf8("reduction.method", method));
    records.push(Record::u32(
        "reduction.output_dimension",
        u32::try_from(output).map_err(|_| "output dimension exceeds u32")?,
    ));
    match (
        &extractor.dim_reduction_config.method,
        &extractor.dim_reduction_transformer,
    ) {
        (DimensionalityReductionMethod::None, None) => {}
        (
            DimensionalityReductionMethod::RandomProjection,
            Some(DimReductionTransformer::RandomProjection(state)),
        ) => {
            validate_projection(state, input, output)?;
            records.push(Record::tensor_f32(
                "reduction.matrix",
                vec![output as u32, input as u32],
                &flatten(&state.projection_matrix),
            )?);
            records.push(Record::utf8("reduction.rng", "seedrandom"));
            records.push(Record::utf8("reduction.seed", canonical_f64(state.seed)?));
        }
        (DimensionalityReductionMethod::Pca, Some(DimReductionTransformer::Pca(state))) => {
            validate_pca(state, input, output)?;
            records.push(Record::tensor_f32(
                "reduction.mean",
                vec![input as u32],
                &state.mean,
            )?);
            records.push(Record::tensor_f32(
                "reduction.components",
                vec![output as u32, input as u32],
                &flatten(&state.components),
            )?);
            records.push(Record::tensor_f32(
                "reduction.explained_variance",
                vec![output as u32],
                state
                    .explained_variance
                    .as_deref()
                    .ok_or("PCA explained variance is missing")?,
            )?);
        }
        _ => return Err("reduction configuration and fitted transformer do not match".into()),
    }
    Ok(())
}

fn encode_classifier(
    classifier: &SerializedClassifier,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    match (&*classifier.classifier_id, &classifier.state) {
        ("mlp", SerializedClassifierState::Mlp(state)) => {
            encode_mlp("mlp", state, dimension, records)
        }
        ("knn", SerializedClassifierState::Knn(state)) => encode_knn(state, dimension, records),
        ("svm", SerializedClassifierState::Svm(state)) => encode_svm(state, dimension, records),
        ("kmeans_prototype", SerializedClassifierState::KMeansPrototype(state)) => {
            encode_kmp(state, dimension, records)
        }
        ("gaussian_nb", SerializedClassifierState::GaussianNb(state)) => {
            encode_gnb(state, dimension, records)
        }
        ("kmeans_logistic", SerializedClassifierState::KMeansLogistic(state)) => {
            encode_kml(state, dimension, records)
        }
        _ => Err("classifier ID and state variant do not match".into()),
    }
}

fn encode_mlp(
    prefix: &str,
    state: &SerializedMlp,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    if state.layer_shapes.len() < 2
        || state.layer_shapes[0] != dimension
        || state.layer_weights.len() + 1 != state.layer_shapes.len()
        || state.layer_biases.len() != state.layer_weights.len()
    {
        return Err("MLP layer shapes are inconsistent".into());
    }
    records.push(Record::tensor_u32(
        format!("{prefix}.layer_shapes"),
        vec![state.layer_shapes.len() as u32],
        &state.layer_shapes,
    )?);
    for index in 0..state.layer_weights.len() {
        let input = state.layer_shapes[index];
        let output = state.layer_shapes[index + 1];
        let canonical_weights =
            transpose_input_output_to_output_input(&state.layer_weights[index], input, output)?;
        records.push(Record::tensor_f32(
            format!("{prefix}.{index}.weights"),
            vec![output as u32, input as u32],
            &canonical_weights,
        )?);
        records.push(Record::tensor_f32(
            format!("{prefix}.{index}.biases"),
            vec![output as u32],
            &state.layer_biases[index],
        )?);
    }
    records.push(Record::u32(
        format!("{prefix}.hidden_layers"),
        state.hidden_layers as u32,
    ));
    records.push(Record::u32(
        format!("{prefix}.hidden_size"),
        state.hidden_size as u32,
    ));
    records.push(Record::tensor_f32(
        format!("{prefix}.class_weights"),
        vec![2],
        &state.class_weights,
    )?);
    Ok(())
}

fn encode_knn(
    state: &SerializedKnn,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    let samples = state.training_data.len();
    if samples == 0
        || state.training_labels.len() != samples
        || state.training_data.iter().any(|row| row.len() != dimension)
        || state.k == 0
        || state.k > samples
    {
        return Err("KNN state dimensions are invalid".into());
    }
    if state
        .training_labels
        .iter()
        .any(|value| *value != 0.0 && *value != 1.0)
    {
        return Err("KNN labels must be 0 or 1".into());
    }
    records.push(Record::tensor_f32(
        "knn.training_data",
        vec![samples as u32, dimension as u32],
        &flatten(&state.training_data),
    )?);
    records.push(Record::tensor_u8(
        "knn.training_labels",
        vec![samples as u32],
        state
            .training_labels
            .iter()
            .map(|value| *value as u8)
            .collect(),
    )?);
    records.push(Record::u32("knn.k", state.k as u32));
    let kernel = state.kernel.unwrap_or(KnnKernel::Cosine);
    records.push(Record::utf8(
        "knn.kernel",
        match kernel {
            KnnKernel::Cosine => "cosine",
            KnnKernel::Rbf => "rbf",
        },
    ));
    let gamma = state.gamma.unwrap_or(1.0);
    if kernel == KnnKernel::Rbf && gamma <= 0.0 {
        return Err("KNN RBF gamma must be positive".into());
    }
    records.push(Record::f64("knn.gamma", gamma)?);
    Ok(())
}

fn encode_svm(
    state: &SerializedSvm,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    records.push(Record::tensor_f32(
        "svm.weights",
        vec![dimension as u32],
        &state.weights,
    )?);
    records.push(Record::f32("svm.bias", state.bias as f32)?);
    records.push(Record::tensor_f32(
        "svm.class_weights",
        vec![2],
        &state.class_weights,
    )?);
    Ok(())
}

fn encode_kmp(
    state: &SerializedKMeansPrototype,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    let clusters = state.clusters.len();
    if clusters == 0 {
        return Err("K-means prototype requires clusters".into());
    }
    let mut centroids = Vec::new();
    let mut present = Vec::new();
    let mut prototypes = Vec::new();
    for cluster in &state.clusters {
        if cluster.centroid.len() != dimension {
            return Err("prototype centroid dimension mismatch".into());
        }
        centroids.extend_from_slice(&cluster.centroid);
        for value in [&cluster.prototype_good, &cluster.prototype_bad] {
            present.push(u8::from(value.is_some()));
            if let Some(value) = value {
                if value.len() != dimension {
                    return Err("prototype dimension mismatch".into());
                }
                prototypes.extend_from_slice(value);
            } else {
                prototypes.resize(prototypes.len() + dimension, 0.0);
            }
        }
    }
    records.push(Record::tensor_f32(
        "kmp.centroids",
        vec![clusters as u32, dimension as u32],
        &centroids,
    )?);
    records.push(Record::tensor_u8(
        "kmp.prototype_present",
        vec![clusters as u32, 2],
        present,
    )?);
    records.push(Record::tensor_f32(
        "kmp.prototypes",
        vec![clusters as u32, 2, dimension as u32],
        &prototypes,
    )?);
    records.push(Record::tensor_f32(
        "kmp.global_good",
        vec![dimension as u32],
        &state.global_prototype_good,
    )?);
    records.push(Record::tensor_f32(
        "kmp.global_bad",
        vec![dimension as u32],
        &state.global_prototype_bad,
    )?);
    records.push(Record::f64("kmp.temperature", state.temperature)?);
    Ok(())
}

fn encode_gnb(
    state: &SerializedGaussianNb,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    let means = [
        state.class_means[0].as_slice(),
        state.class_means[1].as_slice(),
    ]
    .concat();
    let variances = [
        state.class_variances[0].as_slice(),
        state.class_variances[1].as_slice(),
    ]
    .concat();
    if state
        .class_variances
        .iter()
        .flatten()
        .any(|value| *value <= 0.0)
        || state.class_priors.iter().any(|value| *value <= 0.0)
        || (state.class_priors.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err("Gaussian NB variances/priors are invalid".into());
    }
    records.push(Record::tensor_f32(
        "gnb.means",
        vec![2, dimension as u32],
        &means,
    )?);
    records.push(Record::tensor_f32(
        "gnb.variances",
        vec![2, dimension as u32],
        &variances,
    )?);
    records.push(Record::tensor_f32(
        "gnb.priors",
        vec![2],
        &state.class_priors,
    )?);
    records.push(Record::f32("gnb.epsilon", state.epsilon as f32)?);
    Ok(())
}

fn encode_kml(
    state: &SerializedKMeansLogistic,
    dimension: usize,
    records: &mut Vec<Record>,
) -> Result<(), String> {
    let clusters = state.centroids.len();
    if clusters == 0 || state.cluster_models.len() != clusters {
        return Err("K-means logistic cluster state is invalid".into());
    }
    records.push(Record::tensor_f32(
        "kml.centroids",
        vec![clusters as u32, dimension as u32],
        &flatten(&state.centroids),
    )?);
    records.push(Record::tensor_u8(
        "kml.model_present",
        vec![clusters as u32],
        state
            .cluster_models
            .iter()
            .map(|value| u8::from(value.is_some()))
            .collect(),
    )?);
    for (index, model) in state.cluster_models.iter().enumerate() {
        if let Some(model) = model {
            encode_mlp(&format!("kml.cluster.{index}"), model, dimension, records)?;
        }
    }
    encode_mlp("kml.global", &state.global_model, dimension, records)?;
    records.push(Record::f64("kml.temperature", state.temperature)?);
    Ok(())
}

fn decode_extractor(
    records: &BTreeMap<String, Record>,
    feature_types: Vec<String>,
    input: usize,
) -> Result<SerializedFeatureExtractor, String> {
    let normalization_mode = match string(records, "normalization.mode")? {
        "none" => NormalizationMode::None,
        "layer" => NormalizationMode::Layer,
        "z_score" => NormalizationMode::ZScore,
        "calibrated" => NormalizationMode::Calibrated,
        _ => return Err("unknown normalization mode".into()),
    };
    let has_normalization_tensors = matches!(
        normalization_mode,
        NormalizationMode::ZScore | NormalizationMode::Calibrated
    );
    let normalization_mean = has_normalization_tensors
        .then(|| tensor_f64(records, "normalization.mean", &[input as u32]))
        .transpose()?;
    let normalization_std = has_normalization_tensors
        .then(|| tensor_f64(records, "normalization.std", &[input as u32]))
        .transpose()?;
    if normalization_std
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| *value <= 0.0))
    {
        return Err("normalization std must be positive".into());
    }
    let output = scalar_u32(records, "reduction.output_dimension")? as usize;
    let (method, transformer) = match string(records, "reduction.method")? {
        "none" if output == input => (DimensionalityReductionMethod::None, None),
        "random_projection" => {
            let matrix = tensor_matrix(records, "reduction.matrix", output, input)?;
            if string(records, "reduction.rng")? != "seedrandom" {
                return Err("unsupported projection RNG".into());
            }
            let seed = string(records, "reduction.seed")?
                .parse::<f64>()
                .map_err(|_| "projection seed is invalid")?;
            (
                DimensionalityReductionMethod::RandomProjection,
                Some(DimReductionTransformer::RandomProjection(
                    RandomProjectionState {
                        projection_matrix: matrix,
                        n_components: output,
                        n_features: input,
                        seed,
                    },
                )),
            )
        }
        "pca" => (
            DimensionalityReductionMethod::Pca,
            Some(DimReductionTransformer::Pca(SerializedPca {
                components: tensor_matrix(records, "reduction.components", output, input)?,
                mean: tensor_f64(records, "reduction.mean", &[input as u32])?,
                n_components: output,
                n_features: input,
                explained_variance: Some(tensor_f64(
                    records,
                    "reduction.explained_variance",
                    &[output as u32],
                )?),
            })),
        ),
        _ => return Err("invalid reduction method or dimensions".into()),
    };
    Ok(SerializedFeatureExtractor {
        feature_types,
        normalization_mode,
        dim_reduction_config: slouch_ml::ported::types::DimensionalityReductionConfig {
            method,
            components: output,
        },
        concatenated_dimensions: input,
        normalization_mean,
        normalization_std,
        dim_reduction_transformer: transformer,
    })
}

fn decode_classifier(
    records: &BTreeMap<String, Record>,
    id: &str,
    dimension: usize,
) -> Result<SerializedClassifier, String> {
    let state = match id {
        "mlp" => SerializedClassifierState::Mlp(decode_mlp(records, "mlp", dimension)?),
        "knn" => {
            let data_record = record(records, "knn.training_data", 9)?;
            if data_record.dimensions.len() != 2 || data_record.dimensions[1] as usize != dimension
            {
                return Err("KNN data shape is invalid".into());
            }
            let samples = data_record.dimensions[0] as usize;
            let flat = tensor_f64(
                records,
                "knn.training_data",
                &[samples as u32, dimension as u32],
            )?;
            let labels = raw(records, "knn.training_labels", 10)?
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>();
            if labels.len() != samples || labels.iter().any(|value| *value != 0.0 && *value != 1.0)
            {
                return Err("KNN labels are invalid".into());
            }
            let k = scalar_u32(records, "knn.k")? as usize;
            if k == 0 || k > samples {
                return Err("KNN k is invalid".into());
            }
            let kernel = match string(records, "knn.kernel")? {
                "cosine" => KnnKernel::Cosine,
                "rbf" => KnnKernel::Rbf,
                _ => return Err("KNN kernel is invalid".into()),
            };
            let gamma = scalar_f64(records, "knn.gamma")?;
            if kernel == KnnKernel::Rbf && gamma <= 0.0 {
                return Err("KNN gamma is invalid".into());
            }
            SerializedClassifierState::Knn(SerializedKnn {
                training_data: flat.chunks(dimension).map(<[f64]>::to_vec).collect(),
                training_labels: labels,
                k,
                kernel: Some(kernel),
                gamma: Some(gamma),
            })
        }
        "svm" => SerializedClassifierState::Svm(SerializedSvm {
            weights: tensor_f64(records, "svm.weights", &[dimension as u32])?,
            bias: scalar_f32(records, "svm.bias")? as f64,
            class_weights: array2(tensor_f64(records, "svm.class_weights", &[2])?)?,
        }),
        "kmeans_prototype" => decode_kmp(records, dimension)?,
        "gaussian_nb" => decode_gnb(records, dimension)?,
        "kmeans_logistic" => decode_kml(records, dimension)?,
        _ => return Err("unknown classifier ID".into()),
    };
    Ok(SerializedClassifier {
        classifier_id: id.to_owned(),
        state,
    })
}

fn decode_mlp(
    records: &BTreeMap<String, Record>,
    prefix: &str,
    dimension: usize,
) -> Result<SerializedMlp, String> {
    let shape_record = record(records, &format!("{prefix}.layer_shapes"), 11)?;
    if shape_record.dimensions.len() != 1 {
        return Err("MLP layer shape rank is invalid".into());
    }
    let shapes = tensor_usize(
        records,
        &format!("{prefix}.layer_shapes"),
        &shape_record.dimensions,
    )?;
    if shapes.len() < 2 || shapes[0] != dimension {
        return Err("MLP input dimension is invalid".into());
    }
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for index in 0..shapes.len() - 1 {
        let input = shapes[index];
        let output = shapes[index + 1];
        let canonical_weights = tensor_f64(
            records,
            &format!("{prefix}.{index}.weights"),
            &[output as u32, input as u32],
        )?;
        weights.push(transpose_output_input_to_input_output(
            &canonical_weights,
            output,
            input,
        )?);
        biases.push(tensor_f64(
            records,
            &format!("{prefix}.{index}.biases"),
            &[shapes[index + 1] as u32],
        )?);
    }
    Ok(SerializedMlp {
        layer_weights: weights,
        layer_biases: biases,
        layer_shapes: shapes,
        hidden_layers: scalar_u32(records, &format!("{prefix}.hidden_layers"))? as usize,
        hidden_size: scalar_u32(records, &format!("{prefix}.hidden_size"))? as usize,
        class_weights: array2(tensor_f64(
            records,
            &format!("{prefix}.class_weights"),
            &[2],
        )?)?,
    })
}
fn decode_kmp(
    records: &BTreeMap<String, Record>,
    dimension: usize,
) -> Result<SerializedClassifierState, String> {
    let cent = record(records, "kmp.centroids", 9)?;
    if cent.dimensions.len() != 2 || cent.dimensions[1] as usize != dimension {
        return Err("prototype centroids shape is invalid".into());
    }
    let clusters = cent.dimensions[0] as usize;
    let c = tensor_f64(
        records,
        "kmp.centroids",
        &[clusters as u32, dimension as u32],
    )?;
    let present = raw(records, "kmp.prototype_present", 10)?;
    if present.len() != clusters * 2 || present.iter().any(|v| *v > 1) {
        return Err("prototype presence flags are invalid".into());
    }
    let prot = tensor_f64(
        records,
        "kmp.prototypes",
        &[clusters as u32, 2, dimension as u32],
    )?;
    let mut values = Vec::new();
    for i in 0..clusters {
        values.push(slouch_ml::ported::types::KMeansPrototypeCluster {
            centroid: c[i * dimension..(i + 1) * dimension].to_vec(),
            prototype_good: (present[i * 2] == 1)
                .then(|| prot[(i * 2) * dimension..(i * 2 + 1) * dimension].to_vec()),
            prototype_bad: (present[i * 2 + 1] == 1)
                .then(|| prot[(i * 2 + 1) * dimension..(i * 2 + 2) * dimension].to_vec()),
        });
    }
    Ok(SerializedClassifierState::KMeansPrototype(
        SerializedKMeansPrototype {
            clusters: values,
            global_prototype_good: tensor_f64(records, "kmp.global_good", &[dimension as u32])?,
            global_prototype_bad: tensor_f64(records, "kmp.global_bad", &[dimension as u32])?,
            temperature: scalar_f64(records, "kmp.temperature")?,
        },
    ))
}
fn decode_gnb(
    records: &BTreeMap<String, Record>,
    dimension: usize,
) -> Result<SerializedClassifierState, String> {
    let means = tensor_f64(records, "gnb.means", &[2, dimension as u32])?;
    let vars = tensor_f64(records, "gnb.variances", &[2, dimension as u32])?;
    let priors = array2(tensor_f64(records, "gnb.priors", &[2])?)?;
    if vars.iter().any(|v| *v <= 0.0)
        || priors.iter().any(|v| *v <= 0.0)
        || (priors.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err("Gaussian NB state is invalid".into());
    }
    Ok(SerializedClassifierState::GaussianNb(
        SerializedGaussianNb {
            class_means: [means[..dimension].to_vec(), means[dimension..].to_vec()],
            class_variances: [vars[..dimension].to_vec(), vars[dimension..].to_vec()],
            class_priors: priors,
            epsilon: scalar_f32(records, "gnb.epsilon")? as f64,
        },
    ))
}
fn decode_kml(
    records: &BTreeMap<String, Record>,
    dimension: usize,
) -> Result<SerializedClassifierState, String> {
    let cent = record(records, "kml.centroids", 9)?;
    if cent.dimensions.len() != 2 || cent.dimensions[1] as usize != dimension {
        return Err("KML centroids shape is invalid".into());
    }
    let clusters = cent.dimensions[0] as usize;
    let flat = tensor_f64(
        records,
        "kml.centroids",
        &[clusters as u32, dimension as u32],
    )?;
    let present = raw(records, "kml.model_present", 10)?;
    if present.len() != clusters || present.iter().any(|v| *v > 1) {
        return Err("KML model flags are invalid".into());
    }
    let mut models = Vec::new();
    for (i, value) in present.iter().enumerate() {
        models.push(if *value == 1 {
            Some(decode_mlp(records, &format!("kml.cluster.{i}"), dimension)?)
        } else {
            None
        });
    }
    Ok(SerializedClassifierState::KMeansLogistic(
        SerializedKMeansLogistic {
            centroids: flat.chunks(dimension).map(<[f64]>::to_vec).collect(),
            cluster_models: models,
            global_model: decode_mlp(records, "kml.global", dimension)?,
            temperature: scalar_f64(records, "kml.temperature")?,
        },
    ))
}

fn validate_exact_allowlist(
    records: &BTreeMap<String, Record>,
    extractor: &SerializedFeatureExtractor,
    classifier: &SerializedClassifier,
) -> Result<(), String> {
    let mut expected = BTreeSet::from([
        "classifier.id".into(),
        "classifier.state_version".into(),
        "dataset.version".into(),
        "feature.ids".into(),
        "feature.input_dimension".into(),
        "role".into(),
        "trained_at_ms".into(),
        "training_config.sha256".into(),
        "normalization.mode".into(),
        "reduction.method".into(),
        "reduction.output_dimension".into(),
    ]);
    if matches!(
        extractor.normalization_mode,
        NormalizationMode::ZScore | NormalizationMode::Calibrated
    ) {
        expected.extend(["normalization.mean".into(), "normalization.std".into()]);
    }
    match extractor.dim_reduction_config.method {
        DimensionalityReductionMethod::RandomProjection => expected.extend([
            "reduction.matrix".into(),
            "reduction.rng".into(),
            "reduction.seed".into(),
        ]),
        DimensionalityReductionMethod::Pca => expected.extend([
            "reduction.mean".into(),
            "reduction.components".into(),
            "reduction.explained_variance".into(),
        ]),
        _ => {}
    }
    if records.contains_key("metric.cv_accuracy") {
        expected.extend([
            "metric.cv_accuracy".into(),
            "metric.cv_std".into(),
            "metric.mcc".into(),
            "metric.f1_score".into(),
            "metric.confusion_matrix".into(),
            "metric.fold_accuracies".into(),
            "metric.cv_type".into(),
        ]);
    }
    let mut generated = Vec::new();
    encode_classifier(
        classifier,
        extractor_output_dimension(extractor, extractor.concatenated_dimensions)?,
        &mut generated,
    )?;
    expected.extend(generated.into_iter().map(|r| r.name));
    let actual = records.keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "model record allowlist mismatch: unexpected={:?}, missing={:?}",
            actual.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&actual).collect::<Vec<_>>()
        ));
    }
    Ok(())
}

fn encode_config_container(records: &mut [Record]) -> Result<Vec<u8>, String> {
    if records.is_empty() || records.len() > MAX_RECORDS {
        return Err("record count is outside 1..=256".into());
    }
    records.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    for pair in records.windows(2) {
        if pair[0].name == pair[1].name {
            return Err("duplicate configuration record".into());
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"SLCF");
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for record in records.iter() {
        validate_record(record)?;
        out.extend_from_slice(&(record.name.len() as u16).to_le_bytes());
        out.extend_from_slice(record.name.as_bytes());
        out.push(record.kind);
        out.push(record.dimensions.len() as u8);
        for dimension in &record.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&(record.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&record.bytes);
        if out.len() > MAX_CONTAINER_BYTES {
            return Err("configuration container exceeds 256 MiB".into());
        }
    }
    Ok(out)
}

fn encode_container(
    magic: [u8; 4],
    state_version: u16,
    records: &mut [Record],
) -> Result<Vec<u8>, String> {
    if records.is_empty() || records.len() > MAX_RECORDS {
        return Err("record count is outside 1..=256".into());
    }
    records.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    for pair in records.windows(2) {
        if pair[0].name == pair[1].name {
            return Err("duplicate model record".into());
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&magic);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&state_version.to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for record in records.iter() {
        validate_record(record)?;
        out.extend_from_slice(&(record.name.len() as u16).to_le_bytes());
        out.extend_from_slice(record.name.as_bytes());
        out.push(record.kind);
        out.push(record.dimensions.len() as u8);
        for dim in &record.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&(record.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&record.bytes);
        if out.len() > MAX_CONTAINER_BYTES {
            return Err("model container exceeds 256 MiB".into());
        }
    }
    Ok(out)
}
fn decode_container(
    payload: &[u8],
    magic: [u8; 4],
) -> Result<(u16, BTreeMap<String, Record>), String> {
    if payload.len() > MAX_CONTAINER_BYTES || payload.len() < 12 {
        return Err("model container size is invalid".into());
    }
    let mut cursor = 0;
    let actual = take(payload, &mut cursor, 4)?;
    if actual != magic {
        return Err("model container magic is invalid".into());
    }
    if read_u16(payload, &mut cursor)? != VERSION {
        return Err("model envelope version is unsupported".into());
    }
    let state = read_u16(payload, &mut cursor)?;
    let count = read_u32(payload, &mut cursor)? as usize;
    if count == 0 || count > MAX_RECORDS {
        return Err("model record count is invalid".into());
    }
    let mut map = BTreeMap::new();
    let mut previous = None::<String>;
    for _ in 0..count {
        let len = read_u16(payload, &mut cursor)? as usize;
        let name = std::str::from_utf8(take(payload, &mut cursor, len)?)
            .map_err(|_| "record name is not UTF-8")?
            .to_owned();
        let kind = *take(payload, &mut cursor, 1)?
            .first()
            .ok_or("missing record kind")?;
        let rank = *take(payload, &mut cursor, 1)?
            .first()
            .ok_or("missing record rank")? as usize;
        let mut dimensions = Vec::with_capacity(rank);
        for _ in 0..rank {
            dimensions.push(read_u32(payload, &mut cursor)?);
        }
        let byte_len = usize::try_from(read_u64(payload, &mut cursor)?)
            .map_err(|_| "record length is too large")?;
        let bytes = take(payload, &mut cursor, byte_len)?.to_vec();
        let record = Record {
            name: name.clone(),
            kind,
            dimensions,
            bytes,
        };
        validate_record(&record)?;
        if previous
            .as_ref()
            .is_some_and(|value| value.as_bytes() >= name.as_bytes())
        {
            return Err("records are not in canonical order".into());
        }
        previous = Some(name.clone());
        if map.insert(name, record).is_some() {
            return Err("duplicate model record".into());
        }
    }
    if cursor != payload.len() {
        return Err("model container has trailing bytes".into());
    }
    Ok((state, map))
}
fn validate_record(record: &Record) -> Result<(), String> {
    if record.name.is_empty()
        || record.name.len() > MAX_RECORD_NAME
        || !record.name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
    {
        return Err("record name is invalid".into());
    }
    if !(1..=11).contains(&record.kind) {
        return Err("record kind is invalid".into());
    }
    let tensor = record.kind >= 9;
    if tensor {
        if record.dimensions.is_empty() || record.dimensions.len() > 4 {
            return Err("tensor rank is invalid".into());
        }
        let elements = element_count(&record.dimensions)?;
        let width = if record.kind == 10 { 1 } else { 4 };
        if elements > MAX_TENSOR_ELEMENTS
            || record.bytes.len() != elements.saturating_mul(width)
            || record.bytes.len() > MAX_TENSOR_BYTES
        {
            return Err("tensor shape or size is invalid".into());
        }
    } else if !record.dimensions.is_empty() {
        return Err("scalar record rank must be zero".into());
    } else {
        let width = match record.kind {
            1 => Some(1),
            2 | 5 => Some(4),
            3 | 4 | 6 => Some(8),
            _ => None,
        };
        if width.is_some_and(|width| record.bytes.len() != width) {
            return Err("scalar byte length is invalid".into());
        }
        if width.is_none() && record.bytes.len() > MAX_VALUE_BYTES {
            return Err("string/byte record exceeds 1 MiB".into());
        }
    }
    match record.kind {
        1 if record.name.ends_with(".enabled")
            || record.name.ends_with(".present")
            || record.name.ends_with("class_weights") =>
        {
            if record.bytes[0] > 1 {
                return Err("boolean u8 record is invalid".into());
            }
        }
        5 => {
            finite_f32(f32::from_le_bytes(
                record.bytes.clone().try_into().map_err(|_| "invalid f32")?,
            ))?;
        }
        6 => {
            finite_f64(f64::from_le_bytes(
                record.bytes.clone().try_into().map_err(|_| "invalid f64")?,
            ))?;
        }
        7 => {
            std::str::from_utf8(&record.bytes).map_err(|_| "text record is not UTF-8")?;
        }
        9 => {
            for chunk in record.bytes.chunks_exact(4) {
                finite_f32(f32::from_le_bytes(
                    chunk.try_into().map_err(|_| "invalid f32 tensor")?,
                ))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_extractor(extractor: &SerializedFeatureExtractor) -> Result<usize, String> {
    if extractor.feature_types.is_empty() {
        return Err("model feature list is empty".into());
    }
    let mut total = 0usize;
    let mut previous_index = None;
    for id in &extractor.feature_types {
        let feature = id
            .parse::<FeatureId>()
            .map_err(|_| format!("unknown feature ID {id}"))?;
        let index = registry_index(feature);
        if previous_index.is_some_and(|previous| index <= previous) {
            return Err("model feature IDs must be unique and in registry order".into());
        }
        previous_index = Some(index);
        total = total
            .checked_add(feature.metadata().dimensions)
            .ok_or("feature dimension overflow")?;
    }
    if total != extractor.concatenated_dimensions {
        return Err("concatenated feature dimension does not match registry".into());
    }
    Ok(total)
}

fn validate_training_feature_order(features: &[FeatureId]) -> Result<(), String> {
    if features.is_empty() {
        return Err("feature list must not be empty".into());
    }
    if features
        .windows(2)
        .any(|pair| registry_index(pair[0]) >= registry_index(pair[1]))
    {
        return Err("feature IDs must be unique and in registry order".into());
    }
    Ok(())
}

fn registry_index(feature: FeatureId) -> usize {
    feature as usize
}
fn extractor_output_dimension(
    extractor: &SerializedFeatureExtractor,
    input: usize,
) -> Result<usize, String> {
    let output = match extractor.dim_reduction_config.method {
        DimensionalityReductionMethod::None => input,
        _ => extractor.dim_reduction_config.components,
    };
    if output == 0 {
        return Err("model output dimension must be positive".into());
    }
    Ok(output)
}
fn validate_projection(
    state: &RandomProjectionState,
    input: usize,
    output: usize,
) -> Result<(), String> {
    if state.n_features != input
        || state.n_components != output
        || state.projection_matrix.len() != output
        || state.projection_matrix.iter().any(|r| r.len() != input)
    {
        return Err("random projection dimensions are inconsistent".into());
    }
    finite_f64(state.seed)?;
    Ok(())
}
fn validate_pca(state: &SerializedPca, input: usize, output: usize) -> Result<(), String> {
    if state.n_features != input
        || state.n_components != output
        || state.components.len() != output
        || state.components.iter().any(|r| r.len() != input)
        || state.mean.len() != input
        || state
            .explained_variance
            .as_ref()
            .is_none_or(|v| v.len() != output)
    {
        return Err("PCA dimensions are inconsistent".into());
    }
    Ok(())
}
fn canonical_classifier_id(classifier: &SerializedClassifier) -> Result<&str, String> {
    match classifier.classifier_id.as_str() {
        "mlp" | "knn" | "svm" | "kmeans_prototype" | "gaussian_nb" | "kmeans_logistic" => {
            Ok(&classifier.classifier_id)
        }
        _ => Err("unknown classifier ID".into()),
    }
}
fn canonical_parameter_name(value: &str) -> Result<String, String> {
    let value = value
        .chars()
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'-'))
    {
        return Err("classifier parameter name cannot be canonicalized".into());
    }
    Ok(value)
}
fn canonical_f64(value: f64) -> Result<String, String> {
    finite_f64(value)?;
    Ok(format!("{value:.17}"))
}
fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn transpose_input_output_to_output_input(
    values: &[f64],
    input: usize,
    output: usize,
) -> Result<Vec<f64>, String> {
    if values.len() != input.checked_mul(output).ok_or("MLP dimensions overflow")? {
        return Err("MLP weight tensor length is invalid".into());
    }
    let mut transposed = vec![0.0; values.len()];
    for input_index in 0..input {
        for output_index in 0..output {
            transposed[output_index * input + input_index] =
                values[input_index * output + output_index];
        }
    }
    Ok(transposed)
}

fn transpose_output_input_to_input_output(
    values: &[f64],
    output: usize,
    input: usize,
) -> Result<Vec<f64>, String> {
    if values.len() != output.checked_mul(input).ok_or("MLP dimensions overflow")? {
        return Err("MLP weight tensor length is invalid".into());
    }
    let mut transposed = vec![0.0; values.len()];
    for output_index in 0..output {
        for input_index in 0..input {
            transposed[input_index * output + output_index] =
                values[output_index * input + input_index];
        }
    }
    Ok(transposed)
}
fn finite_f64(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("floating value must be finite".into())
    }
}
fn finite_f32(value: f32) -> Result<f32, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("floating value must be finite".into())
    }
}
fn check_tensor_shape(dimensions: &[u32], len: usize) -> Result<(), String> {
    if element_count(dimensions)? != len {
        return Err("tensor dimensions do not match values".into());
    }
    Ok(())
}
fn element_count(dimensions: &[u32]) -> Result<usize, String> {
    if dimensions.is_empty() || dimensions.len() > 4 || dimensions.contains(&0) {
        return Err("tensor dimensions are invalid".into());
    }
    dimensions.iter().try_fold(1usize, |a, d| {
        a.checked_mul(*d as usize)
            .ok_or_else(|| "tensor dimension overflow".into())
    })
}
fn record<'a>(
    records: &'a BTreeMap<String, Record>,
    name: &str,
    kind: u8,
) -> Result<&'a Record, String> {
    let value = records
        .get(name)
        .ok_or_else(|| format!("required record {name} is missing"))?;
    if value.kind != kind {
        return Err(format!("record {name} has the wrong kind"));
    }
    Ok(value)
}
fn raw<'a>(
    records: &'a BTreeMap<String, Record>,
    name: &str,
    kind: u8,
) -> Result<&'a [u8], String> {
    Ok(&record(records, name, kind)?.bytes)
}
fn string<'a>(records: &'a BTreeMap<String, Record>, name: &str) -> Result<&'a str, String> {
    std::str::from_utf8(raw(records, name, 7)?).map_err(|_| format!("record {name} is not UTF-8"))
}
fn scalar_u32(records: &BTreeMap<String, Record>, name: &str) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        raw(records, name, 2)?
            .try_into()
            .map_err(|_| "invalid u32")?,
    ))
}
fn scalar_u64(records: &BTreeMap<String, Record>, name: &str) -> Result<u64, String> {
    Ok(u64::from_le_bytes(
        raw(records, name, 3)?
            .try_into()
            .map_err(|_| "invalid u64")?,
    ))
}
fn scalar_i64(records: &BTreeMap<String, Record>, name: &str) -> Result<i64, String> {
    Ok(i64::from_le_bytes(
        raw(records, name, 4)?
            .try_into()
            .map_err(|_| "invalid i64")?,
    ))
}
fn scalar_f32(records: &BTreeMap<String, Record>, name: &str) -> Result<f32, String> {
    Ok(f32::from_le_bytes(
        raw(records, name, 5)?
            .try_into()
            .map_err(|_| "invalid f32")?,
    ))
}
fn scalar_f64(records: &BTreeMap<String, Record>, name: &str) -> Result<f64, String> {
    Ok(f64::from_le_bytes(
        raw(records, name, 6)?
            .try_into()
            .map_err(|_| "invalid f64")?,
    ))
}
fn tensor_f64(
    records: &BTreeMap<String, Record>,
    name: &str,
    dimensions: &[u32],
) -> Result<Vec<f64>, String> {
    let r = record(records, name, 9)?;
    if r.dimensions != dimensions {
        return Err(format!("record {name} has the wrong dimensions"));
    }
    Ok(r.bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks exact")) as f64)
        .collect())
}
fn tensor_usize(
    records: &BTreeMap<String, Record>,
    name: &str,
    dimensions: &[u32],
) -> Result<Vec<usize>, String> {
    let r = record(records, name, 11)?;
    if r.dimensions != dimensions {
        return Err(format!("record {name} has the wrong dimensions"));
    }
    Ok(r.bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks exact")) as usize)
        .collect())
}
fn tensor_matrix(
    records: &BTreeMap<String, Record>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let values = tensor_f64(records, name, &[rows as u32, cols as u32])?;
    Ok(values.chunks(cols).map(<[f64]>::to_vec).collect())
}
fn array2(values: Vec<f64>) -> Result<[f64; 2], String> {
    values.try_into().map_err(|_| "expected two values".into())
}
fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let end = cursor.checked_add(len).ok_or("container offset overflow")?;
    let value = bytes
        .get(*cursor..end)
        .ok_or("model container is truncated")?;
    *cursor = end;
    Ok(value)
}
fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, String> {
    Ok(u16::from_le_bytes(
        take(bytes, cursor, 2)?
            .try_into()
            .map_err(|_| "invalid u16")?,
    ))
}
fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        take(bytes, cursor, 4)?
            .try_into()
            .map_err(|_| "invalid u32")?,
    ))
}
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    Ok(u64::from_le_bytes(
        take(bytes, cursor, 8)?
            .try_into()
            .map_err(|_| "invalid u64")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        encode_config_container, encode_container, transpose_input_output_to_output_input,
        transpose_output_input_to_input_output, Record, MAX_TENSOR_ELEMENTS, VERSION,
    };

    #[test]
    fn mlp_weights_use_canonical_output_input_order() {
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let canonical = transpose_input_output_to_output_input(&source, 2, 3).expect("transpose");
        assert_eq!(canonical, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(
            transpose_output_input_to_input_output(&canonical, 3, 2).expect("transpose back"),
            source
        );
    }

    #[test]
    fn canonical_container_rejects_duplicate_records() {
        let mut records = vec![Record::u8("duplicate", 0), Record::u8("duplicate", 1)];
        assert!(encode_config_container(&mut records).is_err());
    }

    #[test]
    fn canonical_container_rejects_tensor_element_limit_and_shape_mismatch() {
        let too_many = u32::try_from(MAX_TENSOR_ELEMENTS + 1).expect("test limit fits u32");
        let mut records = vec![Record {
            name: "tensor".into(),
            kind: 10,
            dimensions: vec![too_many],
            bytes: Vec::new(),
        }];
        assert!(encode_container(*b"SLMD", VERSION, &mut records).is_err());

        let mut records = vec![Record {
            name: "tensor".into(),
            kind: 9,
            dimensions: vec![2],
            bytes: 0.0_f32.to_le_bytes().to_vec(),
        }];
        assert!(encode_container(*b"SLMD", VERSION, &mut records).is_err());
    }
}
