//! Shared response types and error formatting for native worker boundaries.

use serde::{Deserialize, Serialize};

/// Fixed discriminator for the standard worker error envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorResponseType {
    #[serde(rename = "error")]
    Error,
}

/// Standard error response sent when a worker operation fails.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub response_type: ErrorResponseType,
    pub error: String,
}

/// Creates a standardized error response from any displayable error value.
pub fn create_error_response(error: impl std::fmt::Display) -> ErrorResponse {
    ErrorResponse {
        response_type: ErrorResponseType::Error,
        error: format_error_message(error),
    }
}

/// Formats an error value using its display representation.
pub fn format_error_message(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn creates_and_serializes_the_exact_error_envelope() {
        let response = create_error_response("failure");
        assert_eq!(
            serde_json::to_value(response).expect("serialize error response"),
            json!({ "type": "error", "error": "failure" })
        );
    }

    #[test]
    fn rejects_any_non_error_discriminator() {
        let error = serde_json::from_value::<ErrorResponse>(json!({
            "type": "result",
            "error": "failure"
        }))
        .expect_err("non-error discriminator must be rejected");
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn formats_error_and_primitive_display_values() {
        assert_eq!(
            format_error_message(std::io::Error::other("io failure")),
            "io failure"
        );
        assert_eq!(format_error_message(42), "42");
        assert_eq!(format_error_message(false), "false");
    }
}
