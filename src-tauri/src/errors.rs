use serde::Serialize;

/// Stable error envelope returned by every native command.
#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(tag = "kind", content = "message", rename_all = "camelCase")]
pub enum ApiError {
    InvalidRequest(String),
    NotFound(String),
    NotReady(String),
    Busy(String),
    Cancelled(String),
    DatasetChanged(String),
    Storage(String),
    Inference(String),
    Training(String),
    Ipc(String),
    Internal(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(value)
            | Self::NotFound(value)
            | Self::NotReady(value)
            | Self::Busy(value)
            | Self::Cancelled(value)
            | Self::DatasetChanged(value)
            | Self::Storage(value)
            | Self::Inference(value)
            | Self::Training(value)
            | Self::Ipc(value)
            | Self::Internal(value) => formatter.write_str(value),
        }
    }
}

impl std::error::Error for ApiError {}

#[cfg(test)]
mod tests {
    use super::ApiError;

    #[test]
    fn error_kinds_serialize_deterministically() {
        let cases = [
            (ApiError::InvalidRequest("x".into()), "invalidRequest"),
            (ApiError::NotFound("x".into()), "notFound"),
            (ApiError::DatasetChanged("x".into()), "datasetChanged"),
            (ApiError::Ipc("x".into()), "ipc"),
        ];
        for (error, kind) in cases {
            let value = serde_json::to_value(error).expect("serialize API error");
            assert_eq!(value["kind"], kind);
            assert_eq!(value["message"], "x");
        }
    }
}
