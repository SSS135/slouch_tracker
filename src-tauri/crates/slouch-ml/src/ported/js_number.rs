//! JavaScript `Number.prototype.toString` formatting used by seeded RNGs.

/// Formats a finite f64 using JavaScript's decimal/exponent thresholds.
pub(super) fn to_string(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    if !value.is_finite() {
        return value.to_string();
    }

    let negative = value.is_sign_negative();
    let absolute = value.abs();
    let fixed = absolute.to_string();
    if (1e-6..1e21).contains(&absolute) {
        return if negative { format!("-{fixed}") } else { fixed };
    }

    let (digits, exponent) = if let Some(decimal) = fixed.strip_prefix("0.") {
        let leading_zeroes = decimal.bytes().take_while(|byte| *byte == b'0').count();
        (
            decimal[leading_zeroes..].trim_end_matches('0').to_owned(),
            -i32::try_from(leading_zeroes).expect("f64 decimal exponent fits i32") - 1,
        )
    } else {
        let mut digits = fixed.replace('.', "");
        let decimal_index = fixed.find('.').unwrap_or(fixed.len());
        while digits.ends_with('0') {
            digits.pop();
        }
        (
            digits,
            i32::try_from(decimal_index).expect("f64 decimal exponent fits i32") - 1,
        )
    };
    let mut result = String::new();
    if negative {
        result.push('-');
    }
    result.push_str(&digits[..1]);
    if digits.len() > 1 {
        result.push('.');
        result.push_str(&digits[1..]);
    }
    result.push('e');
    if exponent >= 0 {
        result.push('+');
    }
    result.push_str(&exponent.to_string());
    result
}

#[cfg(test)]
mod tests {
    use super::to_string;

    #[test]
    fn matches_javascript_number_string_boundaries() {
        assert_eq!(to_string(42.0), "42");
        assert_eq!(to_string(-0.0), "0");
        assert_eq!(to_string(1e-6), "0.000001");
        assert_eq!(to_string(1e-7), "1e-7");
        assert_eq!(to_string(1e20), "100000000000000000000");
        assert_eq!(to_string(1e21), "1e+21");
        assert_eq!(to_string(1.23e21), "1.23e+21");
    }
}
