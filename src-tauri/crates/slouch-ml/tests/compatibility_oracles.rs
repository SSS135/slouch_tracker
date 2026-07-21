use std::collections::BTreeSet;

use serde_json::Value;
use slouch_ml::ported::{
    adamw::{AdamWOptimizer, NamedTensor},
    base_classifier::BaseClassifier,
    cross_validation::{
        create_stratified_k_fold, create_temporal_block_k_fold, detect_temporal_blocks,
        validate_folds,
    },
    kmeans,
    kmeans_logistic_classifier::{KMeansLogisticClassifier, KMeansLogisticConfig},
    kmeans_prototype_classifier::KMeansPrototypeClassifier,
    mlp_classifier::{MlpClassifier, MlpConfig},
    pca::{PcaTransformer, SerializedPca},
    random_projection::{
        compatibility_seedrandom_trace, RandomProjectionState, RandomProjectionTransformer,
    },
    svm_classifier::{SvmClassifier, SvmConfig},
    types::{SerializedKMeansLogistic, SerializedKMeansPrototype, SerializedMlp, SerializedSvm},
};

const NUMERIC_TOLERANCE: f64 = 2e-6;
const ITERATIVE_TOLERANCE: f64 = 2e-4;

fn fixture(relative: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(relative);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn within_numeric(actual: f64, expected: f64, tolerance: f64) -> bool {
    let difference = (actual - expected).abs();
    difference <= tolerance || difference <= tolerance * expected.abs()
}

fn within_iterative(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    let accepted = if tolerance == ITERATIVE_TOLERANCE {
        within_iterative(actual, expected, tolerance)
    } else {
        within_numeric(actual, expected, tolerance)
    };
    assert!(
        accepted,
        "actual {actual} differs from expected {expected} by more than {tolerance}"
    );
}

#[test]
fn iterative_tolerance_is_absolute_only() {
    let expected = 10_000.0;
    let actual = expected + 1.0;
    assert!(within_numeric(actual, expected, ITERATIVE_TOLERANCE));
    assert!(!within_iterative(actual, expected, ITERATIVE_TOLERANCE));
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_f64)
        .map(Option::unwrap)
        .collect()
}

fn encoded_number(value: &Value) -> f64 {
    if let Some(number) = value.as_f64() {
        return number;
    }
    match value.as_str().unwrap() {
        "NaN" => f64::NAN,
        "+Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        other => panic!("unknown encoded number {other}"),
    }
}

fn encoded_numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(encoded_number)
        .collect()
}

fn rows(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|value| encoded_number(value) as f32)
                .collect()
        })
        .collect()
}

fn expected_ids(value: &Value, section: &str) -> BTreeSet<String> {
    value[section]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["id"].as_str().unwrap().to_owned())
        .collect()
}

fn assert_consumed(value: &Value, section: &str, consumed: BTreeSet<String>) {
    assert_eq!(
        consumed,
        expected_ids(value, section),
        "unconsumed {section} rows"
    );
}

fn error_variant<E: std::fmt::Debug>(error: E) -> String {
    format!("{error:?}")
        .split([' ', '{', '('])
        .next()
        .unwrap()
        .to_owned()
}

fn assert_native_error<E: std::fmt::Debug>(result: Result<(), E>, case: &Value) {
    let actual = error_variant(result.expect_err(case["id"].as_str().unwrap()));
    assert_eq!(actual, case["nativeError"], "{}", case["id"]);
    assert!(
        case["typescript"]["ok"].is_boolean(),
        "missing observed TypeScript result"
    );
}

fn serialized_pca(value: &Value) -> SerializedPca {
    SerializedPca {
        components: value["components"]
            .as_array()
            .unwrap()
            .iter()
            .map(encoded_numbers)
            .collect(),
        mean: encoded_numbers(&value["mean"]),
        n_components: value["nComponents"].as_u64().unwrap() as usize,
        n_features: value["nFeatures"].as_u64().unwrap() as usize,
        explained_variance: value
            .get("explainedVariance")
            .filter(|item| !item.is_null())
            .map(encoded_numbers),
    }
}

fn random_projection_state(value: &Value) -> RandomProjectionState {
    RandomProjectionState {
        projection_matrix: value["projectionMatrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(encoded_numbers)
            .collect(),
        n_components: value["nComponents"].as_u64().unwrap() as usize,
        n_features: value["nFeatures"].as_u64().unwrap() as usize,
        seed: encoded_number(&value["seed"]),
    }
}

fn serialized_mlp(value: &Value) -> SerializedMlp {
    SerializedMlp {
        layer_weights: value["layerWeights"]
            .as_array()
            .unwrap()
            .iter()
            .map(encoded_numbers)
            .collect(),
        layer_biases: value["layerBiases"]
            .as_array()
            .unwrap()
            .iter()
            .map(encoded_numbers)
            .collect(),
        layer_shapes: value["layerShapes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_u64().unwrap() as usize)
            .collect(),
        hidden_layers: value["hiddenLayers"].as_u64().unwrap() as usize,
        hidden_size: value["hiddenSize"].as_u64().unwrap() as usize,
        class_weights: [
            encoded_number(&value["classWeights"][0]),
            encoded_number(&value["classWeights"][1]),
        ],
    }
}

fn serialized_kmeans_logistic(value: &Value) -> SerializedKMeansLogistic {
    SerializedKMeansLogistic {
        centroids: value["centroids"]
            .as_array()
            .unwrap()
            .iter()
            .map(encoded_numbers)
            .collect(),
        cluster_models: value["clusterModels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|state| {
                if state.is_null() {
                    None
                } else {
                    Some(serialized_mlp(state))
                }
            })
            .collect(),
        global_model: serialized_mlp(&value["globalModel"]),
        temperature: encoded_number(&value["temperature"]),
    }
}

fn serialized_kmeans_prototype(value: &Value) -> SerializedKMeansPrototype {
    SerializedKMeansPrototype {
        clusters: value["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|cluster| slouch_ml::ported::types::KMeansPrototypeCluster {
                centroid: encoded_numbers(&cluster["centroid"]),
                prototype_good: cluster
                    .get("prototypeGood")
                    .filter(|item| !item.is_null())
                    .map(encoded_numbers),
                prototype_bad: cluster
                    .get("prototypeBad")
                    .filter(|item| !item.is_null())
                    .map(encoded_numbers),
            })
            .collect(),
        global_prototype_good: encoded_numbers(&value["globalPrototypeGood"]),
        global_prototype_bad: encoded_numbers(&value["globalPrototypeBad"]),
        temperature: encoded_number(&value["temperature"]),
    }
}

fn serialized_svm(value: &Value) -> SerializedSvm {
    SerializedSvm {
        weights: encoded_numbers(&value["weights"]),
        bias: encoded_number(&value["bias"]),
        class_weights: [
            encoded_number(&value["classWeights"][0]),
            encoded_number(&value["classWeights"][1]),
        ],
    }
}

fn mlp_config(value: &Value) -> MlpConfig {
    MlpConfig {
        weight_decay: value["weightDecay"].as_f64().unwrap_or(0.01),
        max_iterations: value["maxIterations"].as_u64().unwrap_or(0) as usize,
        learning_rate: value["learningRate"].as_f64().unwrap_or(0.01),
        use_class_weights: value["useClassWeights"].as_bool().unwrap_or(false),
        label_smoothing: value["labelSmoothing"].as_f64().unwrap_or(0.1),
        hidden_layers: value["hiddenLayers"].as_u64().unwrap_or(0) as usize,
        hidden_size: value["hiddenSize"].as_u64().unwrap_or(64) as usize,
    }
}

fn svm_config(value: &Value) -> SvmConfig {
    SvmConfig {
        c: value["C"].as_f64().unwrap_or(1.0),
        max_iterations: value["maxIterations"].as_u64().unwrap_or(0) as usize,
        use_class_weights: value["useClassWeights"].as_bool().unwrap_or(false),
    }
}

fn compare_json(actual: &Value, expected: &Value, tolerance: f64) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => close(
            actual.as_f64().unwrap(),
            expected.as_f64().unwrap(),
            tolerance,
        ),
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                compare_json(actual, expected, tolerance);
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (key, expected) in expected {
                compare_json(actual.get(key).unwrap(), expected, tolerance);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

fn labels(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(Value::as_i64)
        .map(Option::unwrap)
        .map(|value| value as i32)
        .collect()
}

fn compare_f64(actual: &[f64], expected: &Value, tolerance: f64) {
    let expected = numbers(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        close(*actual, expected, tolerance);
    }
}

fn compare_f32(actual: &[f32], expected: &Value, tolerance: f64) {
    let actual = actual
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    compare_f64(&actual, expected, tolerance);
}

fn assert_rng_stream(actual: &[f64], expected: &Value, expected_bits: &Value, id: &str) {
    let expected = numbers(expected);
    let expected_bits = expected_bits.as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{id} RNG value count");
    assert_eq!(actual.len(), expected_bits.len(), "{id} RNG bit count");
    for ((actual, expected), bits) in actual.iter().zip(expected).zip(expected_bits) {
        assert!(
            (*actual - expected).abs() <= f64::EPSILON,
            "{id} RNG value {actual} != {expected}"
        );
        assert_eq!(
            format!("{:016x}", actual.to_bits()),
            bits.as_str().unwrap(),
            "{id} RNG bits"
        );
    }
}

fn routing_weights(centroids: &[Vec<f64>], probe: &[f32], temperature: f64) -> Vec<f64> {
    let logits = centroids
        .iter()
        .map(|centroid| {
            -centroid
                .iter()
                .zip(probe)
                .map(|(value, probe)| (f64::from(*probe) - value).powi(2))
                .sum::<f64>()
                .sqrt()
                / temperature
        })
        .collect::<Vec<_>>();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

#[test]
fn kmeans_matches_typescript_fixtures_and_iteration_checkpoints() {
    let value = fixture("math/kmeans-v1.json");
    for case in value["cases"].as_array().unwrap() {
        if case["id"] == "select-best-k" {
            let features = rows(&case["rows"]);
            let candidates = case["kValues"]
                .as_array()
                .unwrap()
                .iter()
                .map(Value::as_u64)
                .map(Option::unwrap)
                .map(|value| value as usize)
                .collect::<Vec<_>>();
            let result =
                kmeans::select_best_k(&features, &candidates, case["seed"].as_f64().unwrap())
                    .unwrap();
            compare_kmeans(&result, &case["result"]);
            continue;
        }
        let features = rows(&case["rows"]);
        let result = kmeans::kmeans(
            &features,
            case["k"].as_u64().unwrap() as usize,
            case["maxIter"].as_u64().unwrap() as usize,
            case["seed"].as_f64().unwrap(),
        );
        if case
            .get("typescriptError")
            .is_some_and(|value| value.is_string())
        {
            assert!(
                result.is_err(),
                "{} must reject like TypeScript",
                case["id"]
            );
        } else {
            compare_kmeans(&result.unwrap(), &case["result"]);
        }
    }
}

#[test]
fn kmeans_selection_traces_and_invalid_rows_are_consumed() {
    let value = fixture("math/kmeans-v1.json");
    let mut consumed_cases = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let features = rows(&case["rows"]);
        if let Some(expected_rng) = case.get("rngFirst32") {
            let (seed_string, actual_rng) = kmeans::compatibility_seedrandom_trace(
                case["seed"].as_f64().unwrap(),
                expected_rng.as_array().unwrap().len(),
            )
            .unwrap();
            assert_eq!(seed_string, case["seedString"], "{id} seed string");
            assert_rng_stream(&actual_rng, expected_rng, &case["rngFirst32Bits"], id);
        }
        if id == "select-best-k" {
            let candidates = case["kValues"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let (result, traces) = kmeans::select_best_k_with_trace(
                &features,
                &candidates,
                case["seed"].as_f64().unwrap(),
            )
            .unwrap();
            compare_kmeans(&result, &case["result"]);
            assert_eq!(traces.len(), case["runTrace"].as_array().unwrap().len());
            for (actual, expected) in traces.iter().zip(case["runTrace"].as_array().unwrap()) {
                assert_eq!(
                    actual.candidate_k,
                    expected["candidateK"].as_u64().unwrap() as usize
                );
                assert_eq!(actual.run, expected["run"].as_u64().unwrap() as usize);
                assert_eq!(actual.seed, expected["seed"].as_f64().unwrap());
                assert_eq!(
                    actual.selected_for_candidate,
                    expected["selectedForCandidate"].as_bool().unwrap()
                );
                assert_eq!(
                    actual.selected_overall,
                    expected["selectedOverall"].as_bool().unwrap()
                );
                assert_eq!(
                    actual.initial_centroid_indices,
                    expected["initialCentroidIndices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|item| item.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual.initialization_steps.len(),
                    expected["initializationSteps"].as_array().unwrap().len()
                );
                for (step, expected_step) in actual
                    .initialization_steps
                    .iter()
                    .zip(expected["initializationSteps"].as_array().unwrap())
                {
                    close(
                        step.total_distance_squared,
                        expected_step["totalDistanceSquared"].as_f64().unwrap(),
                        NUMERIC_TOLERANCE,
                    );
                    close(
                        step.threshold,
                        expected_step["threshold"].as_f64().unwrap(),
                        NUMERIC_TOLERANCE,
                    );
                    assert_eq!(
                        step.selected_index,
                        expected_step["selectedIndex"].as_u64().unwrap() as usize
                    );
                }
            }
        } else {
            let traced = kmeans::kmeans_with_trace(
                &features,
                case["k"].as_u64().unwrap() as usize,
                case["maxIter"].as_u64().unwrap() as usize,
                case["seed"].as_f64().unwrap(),
            );
            if case.get("typescriptError").is_some_and(Value::is_string) {
                assert!(traced.is_err(), "{} must fail like TypeScript", case["id"]);
                consumed_cases.insert(id.to_owned());
                continue;
            }
            let (_, trace) = traced.unwrap();
            let expected = &case["trace"];
            assert_eq!(
                trace.initial_centroid_indices,
                expected["initialCentroidIndices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                trace.initialization_steps.len(),
                expected["initializationSteps"].as_array().unwrap().len(),
                "{id} initialization step count"
            );
            for (step, expected_step) in trace
                .initialization_steps
                .iter()
                .zip(expected["initializationSteps"].as_array().unwrap())
            {
                close(
                    step.total_distance_squared,
                    expected_step["totalDistanceSquared"].as_f64().unwrap(),
                    NUMERIC_TOLERANCE,
                );
                close(
                    step.threshold,
                    expected_step["threshold"].as_f64().unwrap(),
                    NUMERIC_TOLERANCE,
                );
                assert_eq!(
                    step.selected_index,
                    expected_step["selectedIndex"].as_u64().unwrap() as usize
                );
            }
        }
        consumed_cases.insert(id.to_owned());
    }
    assert_consumed(&value, "cases", consumed_cases);

    let mut consumed_invalid = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let result = kmeans::kmeans(&rows(&case["input"]), 2, 2, 42.0).map(|_| ());
        assert_native_error(result, case);
        consumed_invalid.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed_invalid);
}

fn compare_kmeans(actual: &kmeans::KMeansResult, expected: &Value) {
    assert_eq!(actual.k, expected["k"].as_u64().unwrap() as usize);
    assert_eq!(
        actual.assignments,
        expected["assignments"]
            .as_array()
            .unwrap()
            .iter()
            .map(Value::as_u64)
            .map(Option::unwrap)
            .map(|value| value as usize)
            .collect::<Vec<_>>()
    );
    let expected_centroids = expected["centroids"].as_array().unwrap();
    assert_eq!(actual.centroids.len(), expected_centroids.len());
    for (centroid, expected) in actual.centroids.iter().zip(expected_centroids) {
        compare_f32(centroid, expected, NUMERIC_TOLERANCE);
    }
    close(
        actual.silhouette_score,
        expected["silhouetteScore"].as_f64().unwrap(),
        NUMERIC_TOLERANCE,
    );
}

#[test]
fn pca_order_sign_convergence_and_serialized_transform_match_ml_pca() {
    let value = fixture("math/pca-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let data = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(numbers)
            .collect::<Vec<_>>();
        let mut transformer = PcaTransformer::new();
        transformer
            .fit(
                &data,
                case["requestedComponents"].as_i64().unwrap() as isize,
            )
            .unwrap();
        let state = transformer.to_json().unwrap();
        let expected = &case["state"];
        assert_eq!(
            state.n_components,
            expected["nComponents"].as_u64().unwrap() as usize,
            "{} component count",
            case["id"]
        );
        compare_f64(&state.mean, &expected["mean"], NUMERIC_TOLERANCE);
        assert_eq!(
            state.components.len(),
            expected["components"].as_array().unwrap().len()
        );
        for (component, expected_component) in state
            .components
            .iter()
            .zip(expected["components"].as_array().unwrap())
        {
            compare_f64(component, expected_component, NUMERIC_TOLERANCE);
        }
        compare_f64(
            state.explained_variance.as_ref().unwrap(),
            &expected["explainedVariance"],
            NUMERIC_TOLERANCE,
        );
        let transformed = transformer.transform(&data).unwrap();
        assert_eq!(
            transformed.len(),
            case["transformed"].as_array().unwrap().len()
        );
        for (row, expected_row) in transformed
            .iter()
            .zip(case["transformed"].as_array().unwrap())
        {
            compare_f64(row, expected_row, NUMERIC_TOLERANCE);
        }
        let loaded = PcaTransformer::from_json(state).unwrap();
        let loaded_transform = loaded.transform(&data).unwrap();
        assert_eq!(
            loaded_transform.len(),
            case["loadedTransform"].as_array().unwrap().len()
        );
        for (row, expected_row) in loaded_transform
            .iter()
            .zip(case["loadedTransform"].as_array().unwrap())
        {
            compare_f64(row, expected_row, NUMERIC_TOLERANCE);
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}

#[test]
fn pca_multiple_fit_and_corruption_matrix_are_consumed() {
    let value = fixture("math/pca-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["adversarialCases"].as_array().unwrap() {
        let mut transformer = PcaTransformer::new();
        let first_rows = case["first"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(numbers)
            .collect::<Vec<_>>();
        transformer
            .fit(
                &first_rows,
                case["first"]["requestedComponents"].as_i64().unwrap() as isize,
            )
            .unwrap();
        compare_json(
            &serde_json::to_value(transformer.to_json().unwrap()).unwrap(),
            &case["first"]["state"],
            NUMERIC_TOLERANCE,
        );
        let second_rows = case["second"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(numbers)
            .collect::<Vec<_>>();
        transformer
            .fit(
                &second_rows,
                case["second"]["requestedComponents"].as_i64().unwrap() as isize,
            )
            .unwrap();
        compare_json(
            &serde_json::to_value(transformer.to_json().unwrap()).unwrap(),
            &case["second"]["state"],
            NUMERIC_TOLERANCE,
        );
        compare_json(
            &serde_json::to_value(transformer.transform(&second_rows).unwrap()).unwrap(),
            &case["second"]["transformed"],
            NUMERIC_TOLERANCE,
        );
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "adversarialCases", consumed);

    let mut consumed = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let result = if let Some(state) = case.get("state") {
            PcaTransformer::from_json(serialized_pca(state)).map(|_| ())
        } else {
            let data = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(encoded_numbers)
                .collect::<Vec<_>>();
            let mut transformer = PcaTransformer::new();
            transformer.fit(
                &data,
                case["requestedComponents"].as_i64().unwrap() as isize,
            )
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed);
}

#[test]
fn random_projection_matrix_bits_and_outputs_match_tensorflowjs() {
    let value = fixture("math/random-projection-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let samples = rows(&case["rows"]);
        let expected_rng = numbers(&case["rngFirst32"]);
        let (seed_string, actual_rng) =
            compatibility_seedrandom_trace(case["seed"].as_f64().unwrap(), expected_rng.len())
                .unwrap();
        assert_eq!(
            seed_string, case["seedString"],
            "{} seed string",
            case["id"]
        );
        assert_rng_stream(
            &actual_rng,
            &case["rngFirst32"],
            &case["rngFirst32Bits"],
            case["id"].as_str().unwrap(),
        );
        let mut transformer = RandomProjectionTransformer::new(
            case["nComponents"].as_u64().unwrap() as usize,
            case["seed"].as_f64().unwrap(),
        )
        .unwrap();
        let transformed = transformer.fit_transform(&samples).unwrap();
        let state = transformer.to_json().unwrap();
        compare_json(
            &serde_json::to_value(&state).unwrap(),
            &case["state"],
            NUMERIC_TOLERANCE,
        );
        assert_eq!(
            state.projection_matrix.len(),
            case["matrixF32Bits"].as_array().unwrap().len()
        );
        for (row, expected_bits) in state
            .projection_matrix
            .iter()
            .zip(case["matrixF32Bits"].as_array().unwrap())
        {
            let actual_bits = row
                .iter()
                .map(|value| (*value as f32).to_bits())
                .collect::<Vec<_>>();
            let expected_bits = expected_bits
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(actual_bits, expected_bits, "{} matrix f32 bits", case["id"]);
        }
        assert_eq!(
            transformed.len(),
            case["transformed"].as_array().unwrap().len()
        );
        for ((row, expected), expected_bits) in transformed
            .iter()
            .zip(case["transformed"].as_array().unwrap())
            .zip(case["transformedBits"].as_array().unwrap())
        {
            compare_f32(row, expected, NUMERIC_TOLERANCE);
            assert_eq!(
                row.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected_bits
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>(),
                "{} transformed f32 bits",
                case["id"]
            );
        }
        let loaded = RandomProjectionTransformer::from_json(state).unwrap();
        assert_eq!(
            samples.len(),
            case["loadedTransform"].as_array().unwrap().len()
        );
        for (sample, expected) in samples
            .iter()
            .zip(case["loadedTransform"].as_array().unwrap())
        {
            compare_f32(
                &loaded.transform(sample).unwrap(),
                expected,
                NUMERIC_TOLERANCE,
            );
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}

#[test]
fn random_projection_lifecycle_dimension_and_state_errors_are_consumed() {
    let value = fixture("math/random-projection-v1.json");
    let mut consumed_refits = BTreeSet::new();
    for case in value["adversarialCases"].as_array().unwrap() {
        let mut transformer = RandomProjectionTransformer::new(
            case["nComponents"].as_u64().unwrap() as usize,
            case["seed"].as_f64().unwrap(),
        )
        .unwrap();
        for stage in ["first", "second"] {
            let transformed = transformer
                .fit_transform(&rows(&case[stage]["rows"]))
                .unwrap();
            compare_json(
                &serde_json::to_value(transformer.to_json().unwrap()).unwrap(),
                &case[stage]["state"],
                NUMERIC_TOLERANCE,
            );
            assert_eq!(
                transformed.len(),
                case[stage]["transformed"].as_array().unwrap().len()
            );
            for (actual, expected) in transformed
                .iter()
                .zip(case[stage]["transformed"].as_array().unwrap())
            {
                compare_f32(actual, expected, NUMERIC_TOLERANCE);
            }
            let state = transformer.to_json().unwrap();
            for (actual, expected) in state
                .projection_matrix
                .iter()
                .zip(case[stage]["matrixF32Bits"].as_array().unwrap())
            {
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| (*value as f32).to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap() as u32)
                        .collect::<Vec<_>>(),
                    "{} {stage} matrix bits",
                    case["id"]
                );
            }
        }
        consumed_refits.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "adversarialCases", consumed_refits);

    let mut consumed = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let result = match case["operation"].as_str().unwrap() {
            "construct" => RandomProjectionTransformer::new(
                case["nComponents"].as_u64().unwrap() as usize,
                case["seed"].as_f64().unwrap(),
            )
            .map(|_| ()),
            "fit" | "fitTransform" => {
                let mut transformer = RandomProjectionTransformer::new(
                    case["nComponents"].as_u64().unwrap() as usize,
                    case["seed"].as_f64().unwrap(),
                )
                .unwrap();
                transformer.fit_transform(&rows(&case["rows"])).map(|_| ())
            }
            "transform" => {
                let transformer =
                    RandomProjectionTransformer::from_json(random_projection_state(&case["state"]))
                        .unwrap();
                transformer
                    .transform(
                        &encoded_numbers(&case["probe"])
                            .into_iter()
                            .map(|number| number as f32)
                            .collect::<Vec<_>>(),
                    )
                    .map(|_| ())
            }
            "transform-unfitted" => {
                let transformer = RandomProjectionTransformer::new(
                    case["nComponents"].as_u64().unwrap() as usize,
                    case["seed"].as_f64().unwrap(),
                )
                .unwrap();
                transformer.transform(&[1.0, 2.0]).map(|_| ())
            }
            "serialize-unfitted" => {
                let transformer = RandomProjectionTransformer::new(
                    case["nComponents"].as_u64().unwrap() as usize,
                    case["seed"].as_f64().unwrap(),
                )
                .unwrap();
                transformer.to_json().map(|_| ())
            }
            "disposed-transform" => {
                let mut transformer =
                    RandomProjectionTransformer::from_json(random_projection_state(&case["state"]))
                        .unwrap();
                transformer.dispose();
                assert!(transformer.is_fitted());
                assert!(transformer.to_json().is_ok());
                transformer.transform(&[1.0, 2.0, 3.0]).map(|_| ())
            }
            "load" => {
                RandomProjectionTransformer::from_json(random_projection_state(&case["state"]))
                    .map(|_| ())
            }
            operation => panic!("unconsumed random projection operation {operation}"),
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed);
}

#[test]
fn adamw_checkpoints_match_tensorflowjs() {
    let value = fixture("classifiers/adamw-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let mut optimizer =
            AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-7, case["decay"].as_f64().unwrap()).unwrap();
        let name = case["name"].as_str().unwrap();
        let mut variables = vec![NamedTensor::new(
            name,
            numbers(&case["initial"])
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        )];
        let gradient_values = numbers(&case["gradient"])
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        for step in 1..=10 {
            optimizer
                .apply_gradients(
                    &mut variables,
                    &[Some(NamedTensor::new(name, gradient_values.clone()))],
                )
                .unwrap();
            if [1, 2, 5, 10].contains(&step) {
                compare_f32(
                    &variables[0].values,
                    &case["checkpoints"][step.to_string()],
                    ITERATIVE_TOLERANCE,
                );
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}

fn compare_optimizer_weights(actual: &[NamedTensor], expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.name, expected["name"]);
        compare_f32(&actual.values, &expected["values"], ITERATIVE_TOLERANCE);
    }
}

#[test]
fn adamw_null_gradient_order_restore_and_invalid_state_matrix_are_consumed() {
    let value = fixture("classifiers/adamw-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["adversarialCases"].as_array().unwrap() {
        match case["id"].as_str().unwrap() {
            "null-gradient-source-order" => {
                let mut optimizer =
                    AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-7, case["decay"].as_f64().unwrap())
                        .unwrap();
                let names = case["names"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_str().unwrap())
                    .collect::<Vec<_>>();
                let mut variables = names
                    .iter()
                    .zip(case["initial"].as_array().unwrap())
                    .map(|(name, values)| {
                        NamedTensor::new(
                            *name,
                            encoded_numbers(values)
                                .into_iter()
                                .map(|value| value as f32)
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                for (step, expected) in case["gradientSteps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(case["checkpoints"].as_array().unwrap())
                {
                    let gradients = step
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(&names)
                        .map(|(values, name)| {
                            if values.is_null() {
                                None
                            } else {
                                Some(NamedTensor::new(
                                    *name,
                                    encoded_numbers(values)
                                        .into_iter()
                                        .map(|value| value as f32)
                                        .collect(),
                                ))
                            }
                        })
                        .collect::<Vec<_>>();
                    optimizer
                        .apply_gradients(&mut variables, &gradients)
                        .unwrap();
                    for (actual, expected_values) in variables
                        .iter()
                        .zip(expected["variables"].as_array().unwrap())
                    {
                        compare_f32(&actual.values, expected_values, ITERATIVE_TOLERANCE);
                    }
                    compare_optimizer_weights(
                        &optimizer.get_weights(),
                        &expected["optimizerWeights"],
                    );
                }
            }
            "save-restore-continuation" => {
                let name = case["name"].as_str().unwrap();
                let gradient = encoded_numbers(&case["gradient"])
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                let mut original = vec![NamedTensor::new(
                    name,
                    encoded_numbers(&case["initial"])
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                )];
                let mut optimizer =
                    AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-7, case["decay"].as_f64().unwrap())
                        .unwrap();
                for _ in 0..case["saveAfterStep"].as_u64().unwrap() {
                    optimizer
                        .apply_gradients(
                            &mut original,
                            &[Some(NamedTensor::new(name, gradient.clone()))],
                        )
                        .unwrap();
                }
                compare_f32(
                    &original[0].values,
                    &case["savedVariable"],
                    ITERATIVE_TOLERANCE,
                );
                compare_optimizer_weights(&optimizer.get_weights(), &case["savedWeights"]);

                let restored_name = format!("{name}_restored");
                let mut restored_variables =
                    vec![NamedTensor::new(&restored_name, original[0].values.clone())];
                let mut restored =
                    AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-7, case["decay"].as_f64().unwrap())
                        .unwrap();
                restored.set_weights(&optimizer.get_weights()).unwrap();
                for expected in case["checkpoints"].as_array().unwrap() {
                    optimizer
                        .apply_gradients(
                            &mut original,
                            &[Some(NamedTensor::new(name, gradient.clone()))],
                        )
                        .unwrap();
                    restored
                        .apply_gradients(
                            &mut restored_variables,
                            &[Some(NamedTensor::new(&restored_name, gradient.clone()))],
                        )
                        .unwrap();
                    compare_f32(
                        &original[0].values,
                        &expected["original"],
                        ITERATIVE_TOLERANCE,
                    );
                    compare_f32(
                        &restored_variables[0].values,
                        &expected["restored"],
                        ITERATIVE_TOLERANCE,
                    );
                    compare_optimizer_weights(
                        &restored.get_weights(),
                        &expected["restoredWeights"],
                    );
                }
            }
            id => panic!("unconsumed AdamW adversarial case {id}"),
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "adversarialCases", consumed);

    let mut consumed = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let weights = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|weight| {
                NamedTensor::new(
                    weight["name"].as_str().unwrap(),
                    encoded_numbers(&weight["values"])
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let mut optimizer = AdamWOptimizer::new(0.01, 0.9, 0.999, 1e-7, 0.01).unwrap();
        assert_native_error(optimizer.set_weights(&weights), case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed);
}

#[test]
fn mlp_iterative_checkpoints_and_round_trips_match_tensorflowjs() {
    let value = fixture("classifiers/mlp-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let params = &case["params"];
        let config = MlpConfig {
            weight_decay: params["weightDecay"].as_f64().unwrap_or(0.01),
            max_iterations: params["maxIterations"].as_u64().unwrap() as usize,
            learning_rate: params["learningRate"].as_f64().unwrap_or(0.01),
            use_class_weights: params["useClassWeights"].as_bool().unwrap_or(false),
            label_smoothing: params["labelSmoothing"].as_f64().unwrap_or(0.1),
            hidden_layers: params["hiddenLayers"].as_u64().unwrap() as usize,
            hidden_size: params["hiddenSize"].as_u64().unwrap() as usize,
        };
        let mut classifier = MlpClassifier::new(config).unwrap();
        classifier
            .train(&rows(&case["rows"]), &labels(&case["labels"]))
            .unwrap();
        let state = classifier.to_mlp_json().unwrap();
        compare_json(
            &serde_json::to_value(&state).unwrap(),
            &case["state"],
            ITERATIVE_TOLERANCE,
        );
        if config.max_iterations == 0 {
            assert_eq!(
                state.layer_weights.len(),
                case["state"]["layerWeights"].as_array().unwrap().len()
            );
            for (actual_layer, expected_layer) in state
                .layer_weights
                .iter()
                .zip(case["state"]["layerWeights"].as_array().unwrap())
            {
                let expected_layer = numbers(expected_layer);
                assert_eq!(actual_layer.len(), expected_layer.len());
                for (actual, expected) in actual_layer.iter().zip(expected_layer) {
                    assert_eq!((*actual as f32).to_bits(), (expected as f32).to_bits());
                }
            }
        }
        let loaded = MlpClassifier::from_json(serialized_mlp(&case["state"]), config).unwrap();
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["probabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        for ((probe, expected), loaded_expected) in case["probes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
            .zip(case["loadedProbabilities"].as_array().unwrap())
        {
            let probe = numbers(probe)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let actual = classifier.predict_proba(&probe).unwrap();
            let expected = expected.as_f64().unwrap();
            close(actual, expected, ITERATIVE_TOLERANCE);
            close(
                loaded.predict_proba(&probe).unwrap(),
                loaded_expected.as_f64().unwrap(),
                ITERATIVE_TOLERANCE,
            );
            if (expected - 0.5).abs() > ITERATIVE_TOLERANCE {
                assert_eq!(actual >= 0.5, expected >= 0.5, "{} decision", case["id"]);
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}

#[test]
fn mlp_invalid_malformed_nonfinite_extreme_and_prediction_rows_are_consumed() {
    let value = fixture("classifiers/mlp-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let config = mlp_config(&case["params"]);
        let result = match case["operation"].as_str().unwrap() {
            "construct" => MlpClassifier::new(config).map(|_| ()),
            "train" => {
                let mut classifier = MlpClassifier::new(config).unwrap();
                classifier.train(&rows(&case["rows"]), &labels(&case["labels"]))
            }
            "load" => MlpClassifier::from_json(serialized_mlp(&case["state"]), config).map(|_| ()),
            "predict" => {
                let classifier =
                    MlpClassifier::from_json(serialized_mlp(&case["state"]), config).unwrap();
                classifier
                    .predict_proba(
                        &encoded_numbers(&case["probe"])
                            .into_iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>(),
                    )
                    .map(|_| ())
            }
            operation => panic!("unconsumed MLP operation {operation}"),
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed);
}

#[test]
fn svm_iterative_checkpoints_and_round_trips_match_tensorflowjs() {
    let value = fixture("classifiers/svm-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let params = &case["params"];
        let config = SvmConfig {
            c: params["C"].as_f64().unwrap(),
            max_iterations: params["maxIterations"].as_u64().unwrap() as usize,
            use_class_weights: params["useClassWeights"].as_bool().unwrap_or(false),
        };
        let mut classifier = SvmClassifier::new(config).unwrap();
        classifier
            .train(&rows(&case["rows"]), &labels(&case["labels"]))
            .unwrap();
        let state = match classifier.to_json().unwrap() {
            slouch_ml::ported::types::SerializedClassifierState::Svm(state) => state,
            _ => unreachable!(),
        };
        compare_json(
            &serde_json::to_value(&state).unwrap(),
            &case["state"],
            ITERATIVE_TOLERANCE,
        );
        assert_eq!(
            state.class_weights,
            [
                case["state"]["classWeights"][0].as_f64().unwrap(),
                case["state"]["classWeights"][1].as_f64().unwrap()
            ],
            "{} class weights",
            case["id"]
        );
        let loaded = SvmClassifier::from_json(serialized_svm(&case["state"]), config).unwrap();
        let loaded_state = match loaded.to_json().unwrap() {
            slouch_ml::ported::types::SerializedClassifierState::Svm(state) => state,
            _ => unreachable!(),
        };
        assert_eq!(
            loaded_state
                .weights
                .iter()
                .map(|value| (*value as f32).to_bits())
                .collect::<Vec<_>>(),
            case["stateF32Bits"]["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u32)
                .collect::<Vec<_>>(),
            "{} serialized weight bits",
            case["id"]
        );
        assert_eq!(
            (loaded_state.bias as f32).to_bits(),
            case["stateF32Bits"]["bias"].as_u64().unwrap() as u32,
            "{} serialized bias bits",
            case["id"]
        );
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["probabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["decisionValues"].as_array().unwrap().len()
        );
        for (((probe, expected), loaded_expected), expected_decision) in case["probes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
            .zip(case["loadedProbabilities"].as_array().unwrap())
            .zip(case["decisionValues"].as_array().unwrap())
        {
            let probe = numbers(probe)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let actual = classifier.predict_proba(&probe).unwrap();
            let expected = expected.as_f64().unwrap();
            close(actual, expected, ITERATIVE_TOLERANCE);
            close(
                loaded.predict_proba(&probe).unwrap(),
                loaded_expected.as_f64().unwrap(),
                ITERATIVE_TOLERANCE,
            );
            let decision = state
                .weights
                .iter()
                .map(|value| *value as f32)
                .zip(&probe)
                .fold(state.bias as f32, |sum, (weight, feature)| {
                    sum + weight * feature
                });
            close(
                f64::from(decision),
                expected_decision.as_f64().unwrap(),
                ITERATIVE_TOLERANCE,
            );
            if (expected - 0.5).abs() > ITERATIVE_TOLERANCE {
                assert_eq!(actual >= 0.5, expected >= 0.5, "{} decision", case["id"]);
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}

#[test]
fn svm_invalid_malformed_nonfinite_extreme_and_prediction_rows_are_consumed() {
    let value = fixture("classifiers/svm-v1.json");
    let mut consumed = BTreeSet::new();
    for case in value["invalidCases"].as_array().unwrap() {
        let config = svm_config(&case["params"]);
        let result = match case["operation"].as_str().unwrap() {
            "construct" => SvmClassifier::new(config).map(|_| ()),
            "train" => {
                let mut classifier = SvmClassifier::new(config).unwrap();
                classifier.train(&rows(&case["rows"]), &labels(&case["labels"]))
            }
            "load" => SvmClassifier::from_json(serialized_svm(&case["state"]), config).map(|_| ()),
            "predict" => {
                let classifier =
                    SvmClassifier::from_json(serialized_svm(&case["state"]), config).unwrap();
                classifier
                    .predict_proba(
                        &encoded_numbers(&case["probe"])
                            .into_iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>(),
                    )
                    .map(|_| ())
            }
            operation => panic!("unconsumed SVM operation {operation}"),
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "invalidCases", consumed);
}

#[test]
fn kmeans_composite_classifiers_match_typescript() {
    let logistic = fixture("classifiers/kmeans-logistic-v1.json");
    let mut consumed = BTreeSet::new();
    for case in logistic["cases"].as_array().unwrap() {
        let params = &case["params"];
        let config = KMeansLogisticConfig {
            temperature: params["temperature"].as_f64().unwrap(),
            weight_decay: params["weightDecay"].as_f64().unwrap(),
            max_iterations: params["maxIterations"].as_u64().unwrap() as usize,
            use_class_weights: params["useClassWeights"].as_bool().unwrap_or(false),
            n_clusters: params["nClusters"].as_u64().unwrap() as usize,
        };
        let mut classifier = KMeansLogisticClassifier::with_config(config).unwrap();
        classifier
            .train(&rows(&case["rows"]), &labels(&case["labels"]))
            .unwrap();
        let actual_state = match classifier.to_json().unwrap() {
            slouch_ml::ported::types::SerializedClassifierState::KMeansLogistic(state) => state,
            _ => unreachable!(),
        };
        compare_json(
            &serde_json::to_value(&actual_state).unwrap(),
            &case["state"],
            ITERATIVE_TOLERANCE,
        );
        let state: SerializedKMeansLogistic =
            serde_json::from_value(case["state"].clone()).unwrap();
        assert_eq!(actual_state.temperature, state.temperature);
        assert_eq!(actual_state.centroids.len(), state.centroids.len());
        for (actual, expected) in actual_state.centroids.iter().zip(&state.centroids) {
            compare_f64(
                actual,
                &serde_json::to_value(expected).unwrap(),
                NUMERIC_TOLERANCE,
            );
        }
        assert_eq!(
            actual_state
                .cluster_models
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>(),
            state
                .cluster_models
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>()
        );
        let selected_k = case["trainingTrace"]["selectedK"].as_u64().unwrap() as usize;
        assert_eq!(actual_state.centroids.len(), selected_k);
        let clustering = kmeans::kmeans(&rows(&case["rows"]), selected_k, 100, 42.0).unwrap();
        assert_eq!(
            clustering.assignments,
            case["trainingTrace"]["assignments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        let loaded = KMeansLogisticClassifier::from_json(state.clone(), config).unwrap();
        let expected_count = case["probes"].as_array().unwrap().len();
        assert_eq!(
            expected_count,
            case["probabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            expected_count,
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            expected_count,
            case["routingWeights"].as_array().unwrap().len()
        );
        for (((probe, expected), loaded_expected), expected_routing) in case["probes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
            .zip(case["loadedProbabilities"].as_array().unwrap())
            .zip(case["routingWeights"].as_array().unwrap())
        {
            let probe = numbers(probe)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let actual = classifier.predict_proba(&probe).unwrap();
            let expected = expected.as_f64().unwrap();
            close(actual, expected, ITERATIVE_TOLERANCE);
            close(
                loaded.predict_proba(&probe).unwrap(),
                loaded_expected.as_f64().unwrap(),
                ITERATIVE_TOLERANCE,
            );
            compare_f64(
                &routing_weights(&state.centroids, &probe, state.temperature),
                expected_routing,
                NUMERIC_TOLERANCE,
            );
            if (expected - 0.5).abs() > ITERATIVE_TOLERANCE {
                assert_eq!(actual >= 0.5, expected >= 0.5, "{} decision", case["id"]);
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&logistic, "cases", consumed);

    let mut consumed_legacy = BTreeSet::new();
    for case in logistic["legacyCases"].as_array().unwrap() {
        assert_eq!(case["expectedWeightDecayDefault"], 1);
        assert!(case["params"].get("weightDecay").is_none());
        let config = KMeansLogisticConfig {
            temperature: case["params"]["temperature"].as_f64().unwrap_or(1.0),
            weight_decay: case["params"]["weightDecay"].as_f64().unwrap_or(1.0),
            max_iterations: case["params"]["maxIterations"].as_u64().unwrap_or(100) as usize,
            use_class_weights: case["params"]["useClassWeights"].as_bool().unwrap_or(false),
            n_clusters: case["params"]["nClusters"].as_u64().unwrap_or(0) as usize,
        };
        assert_eq!(config.weight_decay, 1.0);
        let loaded =
            KMeansLogisticClassifier::from_json(serialized_kmeans_logistic(&case["state"]), config)
                .unwrap();
        assert_eq!(
            case["probes"].as_array().unwrap().len(),
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        for (probe, expected) in case["probes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["loadedProbabilities"].as_array().unwrap())
        {
            let probe = numbers(probe)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            close(
                loaded.predict_proba(&probe).unwrap(),
                expected.as_f64().unwrap(),
                ITERATIVE_TOLERANCE,
            );
        }
        consumed_legacy.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&logistic, "legacyCases", consumed_legacy);

    let prototype = fixture("classifiers/kmeans-prototype-v1.json");
    let mut consumed = BTreeSet::new();
    for case in prototype["cases"].as_array().unwrap() {
        let params = &case["params"];
        let mut classifier = KMeansPrototypeClassifier::new(
            params["temperature"].as_f64().unwrap(),
            params["nClusters"].as_u64().unwrap() as usize,
        )
        .unwrap();
        classifier
            .train(&rows(&case["rows"]), &labels(&case["labels"]))
            .unwrap();
        let actual_state = match classifier.to_json().unwrap() {
            slouch_ml::ported::types::SerializedClassifierState::KMeansPrototype(state) => state,
            _ => unreachable!(),
        };
        compare_json(
            &serde_json::to_value(&actual_state).unwrap(),
            &case["state"],
            NUMERIC_TOLERANCE,
        );
        let selected_k = case["trainingTrace"]["selectedK"].as_u64().unwrap() as usize;
        assert_eq!(actual_state.clusters.len(), selected_k);
        let clustering = kmeans::kmeans(&rows(&case["rows"]), selected_k, 100, 42.0).unwrap();
        assert_eq!(
            clustering.assignments,
            case["trainingTrace"]["assignments"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_u64().unwrap() as usize)
                .collect::<Vec<_>>()
        );
        let state: SerializedKMeansPrototype =
            serde_json::from_value(case["state"].clone()).unwrap();
        let loaded = KMeansPrototypeClassifier::from_state(state.clone()).unwrap();
        let expected_count = case["probes"].as_array().unwrap().len();
        assert_eq!(
            expected_count,
            case["probabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            expected_count,
            case["loadedProbabilities"].as_array().unwrap().len()
        );
        assert_eq!(
            expected_count,
            case["routingWeights"].as_array().unwrap().len()
        );
        let centroids = state
            .clusters
            .iter()
            .map(|cluster| cluster.centroid.clone())
            .collect::<Vec<_>>();
        for (((probe, expected), loaded_expected), expected_routing) in case["probes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
            .zip(case["loadedProbabilities"].as_array().unwrap())
            .zip(case["routingWeights"].as_array().unwrap())
        {
            let probe = numbers(probe)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let actual = classifier.predict_proba(&probe).unwrap();
            let expected = expected.as_f64().unwrap();
            close(actual, expected, NUMERIC_TOLERANCE);
            close(
                loaded.predict_proba(&probe).unwrap(),
                loaded_expected.as_f64().unwrap(),
                NUMERIC_TOLERANCE,
            );
            compare_f64(
                &routing_weights(&centroids, &probe, state.temperature),
                expected_routing,
                NUMERIC_TOLERANCE,
            );
            if (expected - 0.5).abs() > NUMERIC_TOLERANCE {
                assert_eq!(actual >= 0.5, expected >= 0.5, "{} decision", case["id"]);
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&prototype, "cases", consumed);
}

#[test]
fn composite_invalid_state_and_input_rows_are_consumed() {
    let logistic = fixture("classifiers/kmeans-logistic-v1.json");
    let mut consumed = BTreeSet::new();
    for case in logistic["invalidCases"].as_array().unwrap() {
        let params = &case["params"];
        let config = KMeansLogisticConfig {
            temperature: params["temperature"].as_f64().unwrap_or(1.0),
            weight_decay: params["weightDecay"].as_f64().unwrap_or(1.0),
            max_iterations: params["maxIterations"].as_u64().unwrap_or(100) as usize,
            use_class_weights: params["useClassWeights"].as_bool().unwrap_or(false),
            n_clusters: params["nClusters"].as_u64().unwrap_or(0) as usize,
        };
        let result = match case["operation"].as_str().unwrap() {
            "construct" => KMeansLogisticClassifier::with_config(config).map(|_| ()),
            "train" => {
                let mut classifier = KMeansLogisticClassifier::with_config(config).unwrap();
                classifier.train(&rows(&case["rows"]), &labels(&case["labels"]))
            }
            "load" => KMeansLogisticClassifier::from_json(
                serialized_kmeans_logistic(&case["state"]),
                config,
            )
            .map(|_| ()),
            operation => panic!("unconsumed K-Means Logistic operation {operation}"),
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&logistic, "invalidCases", consumed);

    let prototype = fixture("classifiers/kmeans-prototype-v1.json");
    let mut consumed = BTreeSet::new();
    for case in prototype["invalidCases"].as_array().unwrap() {
        let params = &case["params"];
        let temperature = params["temperature"].as_f64().unwrap_or(1.0);
        let clusters = params["nClusters"].as_u64().unwrap_or(0) as usize;
        let result = match case["operation"].as_str().unwrap() {
            "construct" => KMeansPrototypeClassifier::new(temperature, clusters).map(|_| ()),
            "train" => {
                let mut classifier = KMeansPrototypeClassifier::new(temperature, clusters).unwrap();
                classifier.train(&rows(&case["rows"]), &labels(&case["labels"]))
            }
            "load" => KMeansPrototypeClassifier::from_json_with_clusters(
                serialized_kmeans_prototype(&case["state"]),
                temperature,
                clusters,
            )
            .map(|_| ()),
            operation => panic!("unconsumed K-Means Prototype operation {operation}"),
        };
        assert_native_error(result, case);
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&prototype, "invalidCases", consumed);
}

#[test]
fn deterministic_cross_validation_fixtures_execute_in_native_code() {
    let value = fixture("math/cross-validation-v1.json");
    assert_eq!(value["cases"].as_array().unwrap().len(), 2);
    let mut consumed = BTreeSet::new();
    for case in value["cases"].as_array().unwrap() {
        let labels = labels(&case["labels"]);
        let folds = if let Some(timestamps) = case.get("timestamps") {
            // The frozen generator intentionally calls the fourth positional
            // TypeScript argument with 42, so this case uses a 42 ms gap and
            // the source default seed of 42.
            create_temporal_block_k_fold(&numbers(timestamps), &labels, 4, 42.0, 42.0)
        } else {
            create_stratified_k_fold(&labels, 4, 42.0)
        };
        let expected_folds = case["folds"].as_array().unwrap();
        assert_eq!(folds.len(), expected_folds.len(), "{}", case["id"]);
        for (actual, expected) in folds.iter().zip(expected_folds) {
            assert_eq!(
                actual.train_indices,
                expected["trainIndices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual.test_indices,
                expected["testIndices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(validate_folds(&folds, labels.len()), case["valid"]);

        if let Some(timestamps) = case.get("timestamps") {
            let timestamps = numbers(timestamps);
            let blocks = detect_temporal_blocks(&timestamps, &labels, 15_000.0);
            let expected_blocks = case["blocks"].as_array().unwrap();
            assert_eq!(blocks.len(), expected_blocks.len());
            for (actual, expected) in blocks.iter().zip(expected_blocks) {
                assert_eq!(
                    actual.indices,
                    expected["indices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>()
                );
                assert_eq!(actual.label, expected["label"].as_i64().unwrap() as i32);
                assert_eq!(actual.start_time, expected["startTime"].as_f64().unwrap());
                assert_eq!(actual.end_time, expected["endTime"].as_f64().unwrap());
            }
        }
        consumed.insert(case["id"].as_str().unwrap().to_owned());
    }
    assert_consumed(&value, "cases", consumed);
}
