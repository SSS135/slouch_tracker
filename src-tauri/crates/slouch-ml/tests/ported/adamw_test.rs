use slouch_ml::ported::adamw::{
    adamw, adamw_with_parameters, AdamWConfig, AdamWOptimizer, NamedTensor,
};

fn variable(name: &str, values: &[f32]) -> NamedTensor {
    NamedTensor::new(name, values.to_vec())
}

fn apply(
    optimizer: &mut AdamWOptimizer,
    variables: &mut [NamedTensor],
    gradients: &[Option<NamedTensor>],
) {
    optimizer.apply_gradients(variables, gradients).unwrap();
}

fn close(actual: f32, expected: f32, digits: i32) -> bool {
    let scale = 0.5_f32 * 10_f32.powi(-digits);
    (actual - expected).abs() < scale
}

#[test]
fn creates_optimizer_with_default_parameters() {
    let optimizer = adamw();
    assert_eq!(AdamWOptimizer::class_name(), "AdamW");
    assert_eq!(
        optimizer.get_config(),
        AdamWConfig {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-7,
            weight_decay: 0.0,
        }
    );
}

#[test]
fn creates_optimizer_with_custom_parameters() {
    let optimizer = adamw_with_parameters(0.01, 0.95, 0.9999, 1e-8, 0.05).unwrap();
    assert_eq!(
        optimizer.get_config(),
        AdamWConfig {
            learning_rate: 0.01,
            beta1: 0.95,
            beta2: 0.9999,
            epsilon: 1e-8,
            weight_decay: 0.05,
        }
    );
}

#[test]
fn creates_optimizer_using_class_constructor() {
    let optimizer = AdamWOptimizer::new(0.001, 0.9, 0.999, 1e-7, 0.01).unwrap();
    assert_eq!(optimizer.get_config().weight_decay, 0.01);
}

#[test]
fn minimizes_simple_quadratic_function() {
    let mut values = vec![10.0_f32];
    let mut optimizer = adamw_with_parameters(0.5, 0.9, 0.999, 1e-7, 0.0).unwrap();

    for _ in 0..200 {
        let gradient = variable("x", &[2.0 * values[0]]);
        let mut parameters = vec![variable("x", &values)];
        apply(&mut optimizer, &mut parameters, &[Some(gradient)]);
        values = parameters.remove(0).values;
    }

    assert!(values[0].abs() < 1.0);
}

#[test]
fn minimizes_multi_dimensional_quadratic_function() {
    let target = [1.0_f32, 2.0, 3.0];
    let mut values = vec![0.0_f32; 3];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.0).unwrap();

    for _ in 0..200 {
        let gradients: Vec<f32> = values
            .iter()
            .zip(target)
            .map(|(value, target)| 2.0 * (*value - target))
            .collect();
        let mut parameters = vec![variable("w", &values)];
        apply(
            &mut optimizer,
            &mut parameters,
            &[Some(variable("w", &gradients))],
        );
        values = parameters.remove(0).values;
    }

    for (actual, expected) in values.iter().zip(target) {
        assert!(close(*actual, expected, 1), "{actual} != {expected}");
    }
}

#[test]
fn converges_faster_without_high_weight_decay() {
    let mut first = vec![0.0_f32];
    let mut optimizer1 = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.0).unwrap();
    for _ in 0..50 {
        let gradient = variable("x1", &[2.0 * (first[0] - 5.0)]);
        let mut parameters = vec![variable("x1", &first)];
        apply(&mut optimizer1, &mut parameters, &[Some(gradient)]);
        first = parameters.remove(0).values;
    }
    let loss1 = (first[0] - 5.0).abs();

    let mut second = vec![0.0_f32];
    let mut optimizer2 = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.1).unwrap();
    for _ in 0..50 {
        let gradient = variable("x2", &[2.0 * (second[0] - 5.0)]);
        let mut parameters = vec![variable("x2", &second)];
        apply(&mut optimizer2, &mut parameters, &[Some(gradient)]);
        second = parameters.remove(0).values;
    }
    let loss2 = (second[0] - 5.0).abs();

    assert!(loss2 > loss1, "{loss2} should be greater than {loss1}");
}

#[test]
fn applies_decoupled_weight_decay_correctly() {
    let mut parameters = vec![variable("w", &[1.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.5).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("w", &[0.0]))],
    );

    assert!(close(parameters[0].values[0], 0.95, 5));
}

#[test]
fn does_not_apply_weight_decay_when_decay_is_zero() {
    let mut parameters = vec![variable("w", &[1.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.0).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("w", &[0.0]))],
    );

    assert!(close(parameters[0].values[0], 1.0, 3));
}

#[test]
fn applies_weight_decay_over_multiple_steps() {
    let learning_rate = 0.01_f32;
    let decay = 0.1_f32;
    let mut parameters = vec![variable("w", &[1.0])];
    let mut optimizer =
        adamw_with_parameters(learning_rate as f64, 0.9, 0.999, 1e-7, decay as f64).unwrap();

    for _ in 0..10 {
        apply(
            &mut optimizer,
            &mut parameters,
            &[Some(variable("w", &[0.0]))],
        );
    }

    let expected = (1.0 - learning_rate * decay).powi(10);
    assert!(close(parameters[0].values[0], expected, 3));
}

#[test]
fn maintains_bounded_optimizer_state_during_optimization() {
    let mut parameters = vec![variable("x", &[5.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();

    for _ in 0..20 {
        let gradient = variable("x", &[2.0 * parameters[0].values[0]]);
        apply(&mut optimizer, &mut parameters, &[Some(gradient)]);
        assert_eq!(optimizer.get_weights().len(), 3);
    }

    optimizer.dispose();
    assert_eq!(optimizer.get_weights().len(), 1);
}

#[test]
fn clears_optimizer_state_when_disposed() {
    let mut parameters = vec![variable("x", &[1.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("x", &[2.0]))],
    );

    assert_eq!(optimizer.get_weights().len(), 3);
    optimizer.dispose();
    assert_eq!(optimizer.get_weights().len(), 1);
}

#[test]
fn maintains_and_clears_state_for_multiple_variables() {
    let mut parameters = vec![
        variable("x1", &[1.0]),
        variable("x2", &[2.0]),
        variable("x3", &[3.0]),
    ];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[
            Some(variable("x1", &[2.0])),
            Some(variable("x2", &[4.0])),
            Some(variable("x3", &[6.0])),
        ],
    );

    assert_eq!(optimizer.get_weights().len(), 7);
    optimizer.dispose();
    assert_eq!(optimizer.get_weights().len(), 1);
}

#[test]
fn works_with_array_style_gradient_specification() {
    let mut parameters = vec![variable("x", &[5.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("x", &[1.0]))],
    );

    assert_ne!(parameters[0].values[0], 5.0);
}

#[test]
fn works_with_ordered_gradients_for_multiple_named_variables() {
    let mut parameters = vec![variable("x", &[5.0]), variable("y", &[3.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();

    // The native API is intentionally positional: each gradient corresponds
    // to the variable at the same index. Distinct signs make a swapped order
    // observable instead of merely checking that both values changed.
    let gradients = [
        Some(variable("gradient-for-x", &[1.0])),
        Some(variable("gradient-for-y", &[-1.0])),
    ];
    apply(&mut optimizer, &mut parameters, &gradients);

    assert!(close(parameters[0].values[0], 4.895, 3));
    assert!(close(parameters[1].values[0], 3.097, 3));
}

#[test]
fn handles_null_gradients_gracefully() {
    let mut parameters = vec![variable("x", &[5.0]), variable("y", &[3.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("x", &[1.0])), None],
    );

    assert_ne!(parameters[0].values[0], 5.0);
    assert_eq!(parameters[1].values[0], 3.0);
}

#[test]
fn applies_bias_correction_in_early_iterations() {
    let mut parameters = vec![variable("x", &[0.0])];
    let mut optimizer = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.0).unwrap();
    apply(
        &mut optimizer,
        &mut parameters,
        &[Some(variable("x", &[1.0]))],
    );

    assert!(parameters[0].values[0].abs() > 0.09);
}

#[test]
fn serializes_configuration_correctly() {
    let optimizer = adamw_with_parameters(0.005, 0.95, 0.9999, 1e-8, 0.05).unwrap();
    assert_eq!(
        optimizer.get_config(),
        AdamWConfig {
            learning_rate: 0.005,
            beta1: 0.95,
            beta2: 0.9999,
            epsilon: 1e-8,
            weight_decay: 0.05,
        }
    );
}

#[test]
fn supports_get_weights_and_set_weights() {
    let mut parameters = vec![variable("x", &[5.0])];
    let mut optimizer1 = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();

    for _ in 0..5 {
        let gradient = variable("x", &[2.0 * parameters[0].values[0]]);
        apply(&mut optimizer1, &mut parameters, &[Some(gradient)]);
    }

    let weights = optimizer1.get_weights();
    assert!(!weights.is_empty());

    let mut optimizer2 = adamw_with_parameters(0.1, 0.9, 0.999, 1e-7, 0.01).unwrap();
    optimizer2.set_weights(&weights).unwrap();

    let value = parameters[0].values[0];
    let mut first_update_parameters = vec![variable("x", &[value])];
    apply(
        &mut optimizer1,
        &mut first_update_parameters,
        &[Some(variable("x", &[2.0 * value]))],
    );
    let update1 = (first_update_parameters[0].values[0] - value).abs();

    let mut second_update_parameters = vec![variable("x", &[value])];
    apply(
        &mut optimizer2,
        &mut second_update_parameters,
        &[Some(variable("x", &[2.0 * value]))],
    );
    let update2 = (second_update_parameters[0].values[0] - value).abs();

    assert!(close(update1, update2, 5), "{update1} != {update2}");
}

#[test]
fn matches_manual_weight_decay_and_adam_combination() {
    let learning_rate = 0.01_f32;
    let decay = 0.05_f32;
    let mut adamw_parameters = [variable("x1", &[10.0])];
    let mut optimizer =
        adamw_with_parameters(learning_rate as f64, 0.9, 0.999, 1e-7, decay as f64).unwrap();

    let mut manual_value = 10.0_f32;
    let mut first_moment = 0.0_f32;
    let mut second_moment = 0.0_f32;
    for step in 1..=10 {
        let adamw_gradient = 2.0 * adamw_parameters[0].values[0];
        apply(
            &mut optimizer,
            &mut adamw_parameters,
            &[Some(variable("x1", &[adamw_gradient]))],
        );

        let gradient = 2.0 * manual_value;
        first_moment = 0.9 * first_moment + 0.1 * gradient;
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient;
        let corrected_first = first_moment / (1.0 - 0.9_f32.powi(step));
        let corrected_second = second_moment / (1.0 - 0.999_f32.powi(step));
        manual_value -= learning_rate * corrected_first / (corrected_second.sqrt() + 1e-7_f32);
        manual_value *= 1.0 - learning_rate * decay;
    }

    let adamw_value = adamw_parameters[0].values[0];
    assert!(adamw_value > 0.0 && adamw_value < 10.0);
    assert!(manual_value > 0.0 && manual_value < 10.0);
    assert!((adamw_value - manual_value).abs() < 2.0);
}
