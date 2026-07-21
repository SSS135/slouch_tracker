use slouch_ml::ported::sgd::{
    default_sgd, sgd, Gradient, NamedGradientMap, NamedTensor, NamedVariableMap, OptimizerConfig,
    SgdError, SgdOptimizer, Variable,
};

fn apply_scalar(optimizer: &mut SgdOptimizer, variable: &mut Variable, gradient: f32) {
    optimizer
        .apply_gradients(
            std::slice::from_mut(variable),
            &[Gradient::new("x", vec![gradient])],
        )
        .unwrap();
}

fn minimize_quadratic(optimizer: &mut SgdOptimizer, variable: &mut Variable, target: f32) {
    let gradient = 2.0 * (variable.values[0] - target);
    apply_scalar(optimizer, variable, gradient);
}

fn minimize_vector_quadratic(
    optimizer: &mut SgdOptimizer,
    variable: &mut Variable,
    target: &[f32],
) {
    let gradient = variable
        .values
        .iter()
        .zip(target)
        .map(|(value, target)| 2.0 * (value - target))
        .collect::<Vec<_>>();
    optimizer
        .apply_gradients(
            std::slice::from_mut(variable),
            &[Gradient {
                name: "w".into(),
                values: Some(gradient),
            }],
        )
        .unwrap();
}

#[test]
fn creates_optimizer_with_default_parameters() {
    let optimizer = default_sgd();
    assert_eq!(optimizer.get_config().learning_rate, 0.01);
    assert_eq!(optimizer.get_config().momentum, 0.9);
    assert_eq!(optimizer.get_config().weight_decay, 0.0);
    assert!(!optimizer.get_config().nesterov);
    assert_eq!(SgdOptimizer::class_name(), "SGD");
}

#[test]
fn creates_optimizer_with_custom_parameters() {
    let optimizer = sgd(0.001, 0.95, 0.01, true);
    assert_eq!(
        optimizer.get_config(),
        OptimizerConfig {
            learning_rate: 0.001,
            momentum: 0.95,
            weight_decay: 0.01,
            nesterov: true,
        }
    );
}

#[test]
fn creates_optimizer_using_constructor() {
    let optimizer = SgdOptimizer::new(0.01, 0.9, 0.0, false);
    assert_eq!(optimizer.get_config().learning_rate, 0.01);
}

#[test]
fn minimizes_simple_quadratic_function_with_momentum() {
    let mut x = Variable::new("x", vec![10.0]);
    let mut optimizer = sgd(0.1, 0.9, 0.0, false);

    for _ in 0..100 {
        minimize_quadratic(&mut optimizer, &mut x, 0.0);
    }

    assert!(x.values[0].abs() < 0.5);
}

#[test]
fn minimizes_simple_quadratic_function_without_momentum() {
    let mut x = Variable::new("x", vec![10.0]);
    let mut optimizer = sgd(0.1, 0.0, 0.0, false);

    for _ in 0..200 {
        minimize_quadratic(&mut optimizer, &mut x, 0.0);
    }

    assert!(x.values[0].abs() < 1.0);
}

#[test]
fn minimizes_multi_dimensional_quadratic_function() {
    let mut weights = Variable::new("w", vec![0.0, 0.0, 0.0]);
    let mut optimizer = sgd(0.1, 0.9, 0.0, false);
    let target = [1.0, 2.0, 3.0];

    for _ in 0..200 {
        minimize_vector_quadratic(&mut optimizer, &mut weights, &target);
    }

    assert!((weights.values[0] - 1.0).abs() < 0.5);
    assert!((weights.values[1] - 2.0).abs() < 0.5);
    assert!((weights.values[2] - 3.0).abs() < 0.5);
}

#[test]
fn converges_faster_with_momentum_than_without() {
    let mut with_momentum = Variable::new("x", vec![0.0]);
    let mut momentum_optimizer = sgd(0.01, 0.9, 0.0, false);
    for _ in 0..50 {
        minimize_quadratic(&mut momentum_optimizer, &mut with_momentum, 5.0);
    }
    let momentum_loss = (with_momentum.values[0] - 5.0).abs();

    let mut without_momentum = Variable::new("x", vec![0.0]);
    let mut vanilla_optimizer = sgd(0.01, 0.0, 0.0, false);
    for _ in 0..50 {
        minimize_quadratic(&mut vanilla_optimizer, &mut without_momentum, 5.0);
    }
    let vanilla_loss = (without_momentum.values[0] - 5.0).abs();

    assert!(momentum_loss < vanilla_loss);
}

#[test]
fn applies_regular_momentum_correctly() {
    let mut x = Variable::new("x", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.5, 0.0, false);

    apply_scalar(&mut optimizer, &mut x, 1.0);
    assert!((x.values[0] - 0.9).abs() < 1e-5);

    apply_scalar(&mut optimizer, &mut x, 1.0);
    assert!((x.values[0] - 0.75).abs() < 1e-5);
}

#[test]
fn applies_nesterov_momentum_correctly() {
    let mut x = Variable::new("x", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.5, 0.0, true);

    apply_scalar(&mut optimizer, &mut x, 1.0);
    assert!((x.values[0] - 0.85).abs() < 1e-5);
}

#[test]
fn regular_and_nesterov_momentum_have_different_behavior() {
    let mut regular = Variable::new("x", vec![0.0]);
    let mut regular_optimizer = sgd(0.05, 0.9, 0.0, false);
    for _ in 0..30 {
        minimize_quadratic(&mut regular_optimizer, &mut regular, 5.0);
    }

    let mut nesterov = Variable::new("x", vec![0.0]);
    let mut nesterov_optimizer = sgd(0.05, 0.9, 0.0, true);
    for _ in 0..30 {
        minimize_quadratic(&mut nesterov_optimizer, &mut nesterov, 5.0);
    }

    assert!((regular.values[0] - nesterov.values[0]).abs() > 0.01);
}

#[test]
fn applies_decoupled_weight_decay() {
    let mut weight = Variable::new("w", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.0, 0.5, false);

    optimizer
        .apply_gradients(
            std::slice::from_mut(&mut weight),
            &[Gradient::new("w", vec![0.0])],
        )
        .unwrap();

    assert!((weight.values[0] - 0.95).abs() < 1e-5);
}

#[test]
fn does_not_apply_weight_decay_when_decay_is_zero() {
    let mut weight = Variable::new("w", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.0, 0.0, false);

    optimizer
        .apply_gradients(
            std::slice::from_mut(&mut weight),
            &[Gradient::new("w", vec![0.0])],
        )
        .unwrap();

    assert!((weight.values[0] - 1.0).abs() < 1e-5);
}

#[test]
fn applies_weight_decay_over_multiple_steps() {
    let mut weight = Variable::new("w", vec![1.0]);
    let learning_rate = 0.01;
    let decay = 0.1;
    let mut optimizer = sgd(learning_rate, 0.0, decay, false);

    for _ in 0..10 {
        optimizer
            .apply_gradients(
                std::slice::from_mut(&mut weight),
                &[Gradient::new("w", vec![0.0])],
            )
            .unwrap();
    }

    let expected = (1.0 - learning_rate * decay).powi(10);
    assert!((weight.values[0] - expected as f32).abs() < 1e-3);
}

#[test]
fn applies_weight_decay_with_momentum() {
    let mut weight = Variable::new("w", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.5, 0.1, false);

    optimizer
        .apply_gradients(
            std::slice::from_mut(&mut weight),
            &[Gradient::new("w", vec![1.0])],
        )
        .unwrap();

    assert!((weight.values[0] - 0.89).abs() < 1e-5);
}

#[test]
fn keeps_only_optimizer_state_during_optimization() {
    let mut x = Variable::new("x", vec![5.0]);
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);

    for _ in 0..20 {
        minimize_quadratic(&mut optimizer, &mut x, 0.0);
    }

    assert_eq!(optimizer.get_weights().len(), 2);
    optimizer.dispose();
    assert_eq!(optimizer.get_weights().len(), 1);
}

#[test]
fn dispose_releases_optimizer_state() {
    let mut x = Variable::new("x", vec![1.0]);
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);
    minimize_quadratic(&mut optimizer, &mut x, 0.0);

    optimizer.dispose();
    x.values[0] = 1.0;
    assert_eq!(optimizer.iterations(), 1);
    assert_eq!(
        optimizer.get_weights(),
        vec![NamedTensor::new("iter", vec![1.0])]
    );
}

#[test]
fn dispose_handles_multiple_variables() {
    let mut variables = vec![
        Variable::new("x1", vec![1.0]),
        Variable::new("x2", vec![2.0]),
        Variable::new("x3", vec![3.0]),
    ];
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);
    optimizer
        .apply_gradients(
            &mut variables,
            &[
                Gradient::new("x1", vec![2.0]),
                Gradient::new("x2", vec![4.0]),
                Gradient::new("x3", vec![6.0]),
            ],
        )
        .unwrap();

    optimizer.dispose();
    assert_eq!(optimizer.get_weights().len(), 1);
}

#[test]
fn supports_array_style_gradient_specification() {
    let mut x = Variable::new("x", vec![5.0]);
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);

    apply_scalar(&mut optimizer, &mut x, 1.0);
    assert_ne!(x.values[0], 5.0);
}

#[test]
fn supports_object_style_gradient_specification() {
    let mut variables: NamedVariableMap = [("x".into(), Variable::new("x", vec![5.0]))].into();
    let gradients: NamedGradientMap = [("x".into(), Some(vec![1.0]))].into();
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);

    optimizer
        .apply_gradients_map(&mut variables, &gradients)
        .unwrap();
    assert_ne!(variables.get("x").unwrap().values[0], 5.0);
}

#[test]
fn handles_missing_gradients_gracefully() {
    let mut variables = vec![Variable::new("x", vec![5.0]), Variable::new("y", vec![3.0])];
    let mut optimizer = sgd(0.1, 0.9, 0.01, false);

    optimizer
        .apply_gradients(
            &mut variables,
            &[Gradient::new("x", vec![1.0]), Gradient::missing("y")],
        )
        .unwrap();

    assert_ne!(variables[0].values[0], 5.0);
    assert_eq!(variables[1].values[0], 3.0);
}

#[test]
fn serializes_configuration_correctly() {
    let optimizer = sgd(0.005, 0.95, 0.05, true);
    assert_eq!(
        optimizer.get_config(),
        OptimizerConfig {
            learning_rate: 0.005,
            momentum: 0.95,
            weight_decay: 0.05,
            nesterov: true,
        }
    );
}

#[test]
fn supports_get_weights_and_set_weights() {
    let mut x = Variable::new("x", vec![5.0]);
    let mut optimizer1 = sgd(0.1, 0.9, 0.01, false);

    for _ in 0..5 {
        minimize_quadratic(&mut optimizer1, &mut x, 0.0);
    }

    let weights = optimizer1.get_weights();
    assert!(!weights.is_empty());

    let mut optimizer2 = sgd(0.1, 0.9, 0.01, false);
    optimizer2.set_weights(&weights).unwrap();

    let value = x.values[0];
    minimize_quadratic(&mut optimizer1, &mut x, 0.0);
    let update1 = (x.values[0] - value).abs();

    x.values[0] = value;
    minimize_quadratic(&mut optimizer2, &mut x, 0.0);
    let update2 = (x.values[0] - value).abs();

    assert!((update1 - update2).abs() < 1e-5);
}

#[test]
fn behaves_like_independently_computed_regular_momentum() {
    let learning_rate = 0.01_f32;
    let momentum = 0.9_f32;
    let mut actual = Variable::new("x", vec![10.0]);
    let mut optimizer = sgd(f64::from(learning_rate), f64::from(momentum), 0.0, false);
    let mut expected = 10.0_f32;
    let mut velocity = 0.0_f32;

    for _ in 0..20 {
        let gradient = 2.0 * expected;
        velocity = momentum * velocity + gradient;
        expected -= learning_rate * velocity;
        minimize_quadratic(&mut optimizer, &mut actual, 0.0);
    }

    assert_eq!(actual.values[0].to_bits(), expected.to_bits());
}

#[test]
fn object_form_preserves_non_lexical_order_and_positional_velocity_state() {
    let mut variables: NamedVariableMap = [
        ("b".into(), Variable::new("b", vec![10.0])),
        ("a".into(), Variable::new("a", vec![20.0])),
    ]
    .into();
    let first: NamedGradientMap =
        [("b".into(), Some(vec![1.0])), ("a".into(), Some(vec![2.0]))].into();
    let mut optimizer = sgd(0.1, 0.5, 0.0, false);

    optimizer
        .apply_gradients_map(&mut variables, &first)
        .unwrap();
    assert_eq!(
        optimizer
            .get_weights()
            .iter()
            .map(|weight| weight.name.as_str())
            .collect::<Vec<_>>(),
        vec!["iter", "b/velocity", "a/velocity"],
    );

    let second: NamedGradientMap =
        [("a".into(), Some(vec![3.0])), ("b".into(), Some(vec![4.0]))].into();
    optimizer
        .apply_gradients_map(&mut variables, &second)
        .unwrap();

    assert!((variables.get("a").unwrap().values[0] - 19.45).abs() < 2e-6);
    assert!((variables.get("b").unwrap().values[0] - 9.4).abs() < 2e-6);
    assert_eq!(optimizer.get_weights()[1].name, "b/velocity");
    assert_eq!(optimizer.get_weights()[2].name, "a/velocity");
}

#[test]
fn object_form_rejects_mismatched_variable_key_without_mutation() {
    let mut variables: NamedVariableMap =
        [("registered".into(), Variable::new("different", vec![5.0]))].into();
    let gradients: NamedGradientMap = [("registered".into(), Some(vec![1.0]))].into();
    let mut optimizer = sgd(0.1, 0.9, 0.0, false);

    assert_eq!(
        optimizer.apply_gradients_map(&mut variables, &gradients),
        Err(SgdError::VariableKeyNameMismatch {
            key: "registered".into(),
            name: "different".into(),
        }),
    );
    assert_eq!(variables.get("registered").unwrap().values, vec![5.0]);
    assert_eq!(optimizer.iterations(), 0);
}

#[test]
fn restored_velocity_length_mismatch_is_typed_and_non_mutating() {
    for velocity in [vec![0.25], vec![0.25, 0.5, 0.75]] {
        let mut optimizer = sgd(0.1, 0.9, 0.0, false);
        optimizer
            .set_weights(&[
                NamedTensor::new("iter", vec![7.0]),
                NamedTensor::new("x/velocity", velocity.clone()),
            ])
            .unwrap();
        let mut variable = Variable::new("x", vec![5.0, 6.0]);

        assert_eq!(
            optimizer.apply_gradients(
                std::slice::from_mut(&mut variable),
                &[Gradient::new("x", vec![1.0, 1.0])],
            ),
            Err(SgdError::VelocityLengthMismatch {
                name: "x".into(),
                variable: 2,
                velocity: velocity.len(),
            }),
        );
        assert_eq!(variable.values, vec![5.0, 6.0]);
        assert_eq!(optimizer.iterations(), 7);
    }
}
