//! Adversarial tests for the raw IPC trust boundary.
//!
//! The app lib keeps its modules private, so these tests compile the exact
//! production sources via `#[path]` inclusion. Everything exercised here is
//! the same code `api.rs` and the inference actor run in production.

#[path = "../src/errors.rs"]
mod errors;
#[path = "../src/inference_cache.rs"]
mod inference_cache;
#[path = "../src/ipc_validation.rs"]
mod ipc_validation;

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use tauri::http::{HeaderMap, HeaderName, HeaderValue};

use errors::ApiError;
use inference_cache::{
    InferenceCache, INFERENCE_CACHE_CAPACITY, INFERENCE_CACHE_MAX_BYTES, INFERENCE_CACHE_TTL,
    TOMBSTONE_CAPACITY, TOMBSTONE_TTL,
};
use ipc_validation::{
    ensure_js_safe_timestamp, ensure_js_safe_u64, ensure_js_safe_usize, header_string_value,
    parse_frame_label, parse_header_value, parse_raw_image_from, require_ipc_version_header,
    validate_id, validate_image_dimensions, validate_image_layout, validate_page_limit,
    validate_thumbnail_size, validate_training_settings, MAX_IMAGE_BYTES, MAX_PAGE_SIZE,
    MAX_THUMBNAIL_BYTES,
};
use slouch_domain::{
    ClassifierConfig, ClassifierId, DimensionalityReductionConfig, DimensionalityReductionMethod,
    FeatureId, FeatureMap, FrameLabel, TrainingSettings,
};
use slouch_vision::ported::inference_worker::NativeInferenceResult;

fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (name, value) in pairs {
        map.append(
            HeaderName::from_bytes(name.as_bytes()).expect("header name"),
            HeaderValue::from_str(value).expect("header value"),
        );
    }
    map
}

fn image_headers(width: &str, height: &str, stride: &str) -> HeaderMap {
    headers(&[
        ("x-slouch-pixel-format", "rgba8"),
        ("x-slouch-width", width),
        ("x-slouch-height", height),
        ("x-slouch-stride", stride),
    ])
}

fn empty_result() -> NativeInferenceResult {
    NativeInferenceResult {
        person_found: false,
        bbox: None,
        keypoints: None,
        features: FeatureMap::new(),
        classification: None,
    }
}

fn sized_result(floats: usize) -> NativeInferenceResult {
    let mut result = empty_result();
    result
        .features
        .insert(FeatureId::BackboneFeatures, vec![0.0; floats]);
    result
}

fn valid_training_settings() -> TrainingSettings {
    TrainingSettings {
        classifier_config: ClassifierConfig {
            classifier_id: ClassifierId::GaussianNb,
            params: BTreeMap::new(),
        },
        dim_reduction_config: DimensionalityReductionConfig {
            method: DimensionalityReductionMethod::None,
            components: 1,
        },
        posture_feature_types: vec![FeatureId::GauFeatures],
        presence_feature_types: vec![FeatureId::RtmDetExtracted],
        feature_types: None,
        normalization_mode: None,
        cv_folds: 5,
        last_updated: 1.0,
    }
}

#[test]
fn api_error_wire_kinds_and_messages_cover_every_variant() {
    // The camelCase kind tags are the wire contract the TypeScript bridge
    // dispatches on; renaming any variant must fail this test.
    let cases = [
        (ApiError::InvalidRequest("m".into()), "invalidRequest"),
        (ApiError::NotFound("m".into()), "notFound"),
        (ApiError::NotReady("m".into()), "notReady"),
        (ApiError::Busy("m".into()), "busy"),
        (ApiError::Cancelled("m".into()), "cancelled"),
        (ApiError::DatasetChanged("m".into()), "datasetChanged"),
        (ApiError::Storage("m".into()), "storage"),
        (ApiError::Inference("m".into()), "inference"),
        (ApiError::Training("m".into()), "training"),
        (ApiError::Ipc("m".into()), "ipc"),
        (ApiError::Internal("m".into()), "internal"),
    ];
    for (error, kind) in cases {
        assert_eq!(error.to_string(), "m");
        let value = serde_json::to_value(&error).expect("serialize ApiError");
        assert_eq!(value["kind"], kind);
        assert_eq!(value["message"], "m");
    }
}

#[test]
fn image_dimensions_reject_zero_and_above_native_caps() {
    assert!(matches!(
        validate_image_dimensions(0, 1),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_image_dimensions(1, 0),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_image_dimensions(1921, 1080),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_image_dimensions(1920, 1081),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(validate_image_dimensions(1, 1).is_ok());
    assert!(validate_image_dimensions(1920, 1080).is_ok());
}

#[test]
fn image_layout_rejects_stride_and_length_mismatches_and_overflow() {
    // stride must equal the tightly packed 4-byte RGBA row
    assert!(matches!(
        validate_image_layout(2, 2, 7, 16),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_image_layout(2, 2, 9, 18),
        Err(ApiError::InvalidRequest(_))
    ));
    // off-by-one body sizes in both directions
    assert!(matches!(
        validate_image_layout(2, 2, 8, 15),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_image_layout(2, 2, 8, 17),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(validate_image_layout(2, 2, 8, 16).is_ok());
    // stride * height wrapping past usize must be caught, not accepted
    let overflow = validate_image_layout(u32::MAX, u32::MAX, u32::MAX as usize * 4, 16)
        .expect_err("overflow-shaped layout");
    assert!(overflow.to_string().contains("overflow"), "{overflow:?}");
}

#[test]
fn frame_payload_cap_sits_exactly_at_full_hd_rgba() {
    // 1920 * 1080 * 4 is both the cap constant and the largest legal frame
    let full_hd_bytes = 1920usize * 1080 * 4;
    assert_eq!(full_hd_bytes, MAX_IMAGE_BYTES);
    assert!(validate_image_layout(1920, 1080, 1920 * 4, full_hd_bytes).is_ok());
    assert!(validate_image_layout(1920, 1080, 1920 * 4, full_hd_bytes + 1).is_err());
    // a self-consistent layout that exceeds the cap is rejected by the cap
    let error =
        validate_image_layout(1921, 1080, 1921 * 4, 1921 * 4 * 1080).expect_err("over-cap frame");
    assert!(error.to_string().contains("8 MiB"), "{error:?}");
}

#[test]
fn thumbnail_cap_sits_exactly_at_two_mib() {
    assert_eq!(MAX_THUMBNAIL_BYTES, 2 * 1024 * 1024);
    assert!(matches!(
        validate_thumbnail_size(0),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(validate_thumbnail_size(1).is_ok());
    assert!(validate_thumbnail_size(MAX_THUMBNAIL_BYTES).is_ok());
    assert!(matches!(
        validate_thumbnail_size(MAX_THUMBNAIL_BYTES + 1),
        Err(ApiError::InvalidRequest(_))
    ));
}

#[test]
fn dataset_page_limit_is_bounded_between_1_and_100() {
    assert!(matches!(
        validate_page_limit(0),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(validate_page_limit(1).is_ok());
    assert!(validate_page_limit(MAX_PAGE_SIZE).is_ok());
    assert!(matches!(
        validate_page_limit(MAX_PAGE_SIZE + 1),
        Err(ApiError::InvalidRequest(_))
    ));
}

#[test]
fn ipc_version_header_must_be_exactly_the_supported_version() {
    assert!(require_ipc_version_header(&headers(&[("x-slouch-ipc-version", "1")])).is_ok());
    for wrong in ["0", "2", "", " 1", "1 ", "1.0", "one"] {
        assert!(
            matches!(
                require_ipc_version_header(&headers(&[("x-slouch-ipc-version", wrong)])),
                Err(ApiError::InvalidRequest(_))
            ),
            "version {wrong:?} must be rejected"
        );
    }
    // missing entirely
    assert!(require_ipc_version_header(&HeaderMap::new()).is_err());
    // duplicated header (smuggling attempt) is rejected even with equal values
    assert!(require_ipc_version_header(&headers(&[
        ("x-slouch-ipc-version", "1"),
        ("x-slouch-ipc-version", "1"),
    ]))
    .is_err());
}

#[test]
fn numeric_headers_enforce_u64_parsing_and_the_js_safe_range() {
    let parse =
        |value: &str| parse_header_value(&headers(&[("x-slouch-width", value)]), "x-slouch-width");
    assert_eq!(parse("0").expect("zero"), 0);
    assert_eq!(
        parse("9007199254740991").expect("largest JS-safe integer"),
        ipc_validation::MAX_SAFE_JS_INTEGER
    );
    for bad in [
        "9007199254740992",     // JS-safe boundary + 1
        "18446744073709551616", // u64::MAX + 1
        "-1",
        "1.5",
        "0x10",
        " 7",
        "",
        "abc",
    ] {
        assert!(
            matches!(parse(bad), Err(ApiError::InvalidRequest(_))),
            "header value {bad:?} must be rejected"
        );
    }
}

#[test]
fn header_strings_reject_missing_duplicate_and_non_utf8_values() {
    assert_eq!(
        header_string_value(
            &headers(&[("x-slouch-frame-id", "f1")]),
            "x-slouch-frame-id"
        )
        .expect("single header"),
        "f1"
    );
    assert!(header_string_value(&HeaderMap::new(), "x-slouch-frame-id").is_err());
    assert!(header_string_value(
        &headers(&[("x-slouch-frame-id", "a"), ("x-slouch-frame-id", "b")]),
        "x-slouch-frame-id"
    )
    .is_err());
    // opaque bytes are legal header values but must be refused as non-UTF-8
    let mut map = HeaderMap::new();
    map.append(
        HeaderName::from_static("x-slouch-frame-id"),
        HeaderValue::from_bytes(&[0xF0, 0x28]).expect("opaque header value"),
    );
    assert!(matches!(
        header_string_value(&map, "x-slouch-frame-id"),
        Err(ApiError::InvalidRequest(_))
    ));
}

#[test]
fn raw_image_parsing_round_trips_only_well_formed_rgba_frames() {
    let body = [7u8; 16];
    let image = parse_raw_image_from(&image_headers("2", "2", "8"), &body).expect("2x2 frame");
    assert_eq!((image.width, image.height), (2, 2));
    assert_eq!(image.data, body.to_vec());

    // pixel format is case-sensitive and rgba8-only
    for format in ["RGBA8", "rgb8", "bgra8", ""] {
        let mut map = image_headers("2", "2", "8");
        map.insert(
            HeaderName::from_static("x-slouch-pixel-format"),
            HeaderValue::from_str(format).expect("format value"),
        );
        assert!(
            parse_raw_image_from(&map, &body).is_err(),
            "pixel format {format:?} must be rejected"
        );
    }

    // u32 overflow disguised as a JS-safe integer
    assert!(parse_raw_image_from(&image_headers("4294967296", "2", "8"), &body).is_err());
    // integer-overflow-shaped width * height combination
    assert!(parse_raw_image_from(
        &image_headers("9007199254740991", "9007199254740991", "8"),
        &body
    )
    .is_err());
    // missing stride header
    let mut no_stride = image_headers("2", "2", "8");
    no_stride.remove(HeaderName::from_static("x-slouch-stride"));
    assert!(parse_raw_image_from(&no_stride, &body).is_err());
}

#[test]
fn js_safe_integer_guards_hold_at_the_exact_boundary() {
    let max = ipc_validation::MAX_SAFE_JS_INTEGER;
    // the validator and the token cache must agree on the JS-safe boundary
    assert_eq!(max, inference_cache::MAX_SAFE_JS_INTEGER);
    assert!(ensure_js_safe_u64(max, "value").is_ok());
    assert!(matches!(
        ensure_js_safe_u64(max + 1, "value"),
        Err(ApiError::Storage(_))
    ));
    assert!(matches!(
        ensure_js_safe_u64(u64::MAX, "value"),
        Err(ApiError::Storage(_))
    ));
    assert!(ensure_js_safe_usize(max as usize, "value").is_ok());
    assert!(matches!(
        ensure_js_safe_usize(usize::MAX, "value"),
        Err(ApiError::Storage(_))
    ));
    assert!(ensure_js_safe_timestamp(1.0, "value").is_ok());
    assert!(ensure_js_safe_timestamp(max as f64, "value").is_ok());
    for bad in [
        0.0,
        -1.0,
        1.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        (max + 1) as f64,
    ] {
        assert!(
            matches!(
                ensure_js_safe_timestamp(bad, "value"),
                Err(ApiError::Storage(_))
            ),
            "timestamp {bad} must be rejected"
        );
    }
}

#[test]
fn frame_ids_enforce_the_1_to_128_byte_contract() {
    assert!(validate_id("frame-1").is_ok());
    assert!(validate_id(&"x".repeat(128)).is_ok());
    assert!(matches!(validate_id(""), Err(ApiError::InvalidRequest(_))));
    assert!(matches!(
        validate_id("   "),
        Err(ApiError::InvalidRequest(_))
    ));
    assert!(matches!(
        validate_id(&"x".repeat(129)),
        Err(ApiError::InvalidRequest(_))
    ));
    // the limit counts BYTES, not characters: 33 four-byte crabs are 132 bytes
    assert!(validate_id(&"\u{1F980}".repeat(33)).is_err());
    // the contract is deliberately shape-agnostic: ids are opaque SQL
    // parameters downstream, so separators and control bytes pass here
    assert!(validate_id("../../etc/passwd").is_ok());
    assert!(validate_id("id\nwith\ncontrol").is_ok());
}

#[test]
fn frame_labels_parse_only_the_four_known_lowercase_labels() {
    assert!(matches!(parse_frame_label("good"), Ok(FrameLabel::Good)));
    assert!(matches!(parse_frame_label("bad"), Ok(FrameLabel::Bad)));
    assert!(matches!(parse_frame_label("away"), Ok(FrameLabel::Away)));
    assert!(matches!(
        parse_frame_label("unused"),
        Ok(FrameLabel::Unused)
    ));
    for bad in ["Good", "GOOD", " good", "good ", "excellent", ""] {
        assert!(
            matches!(parse_frame_label(bad), Err(ApiError::InvalidRequest(_))),
            "label {bad:?} must be rejected"
        );
    }
}

#[test]
fn training_settings_hyperparameters_are_range_checked() {
    assert!(validate_training_settings(&valid_training_settings()).is_ok());

    let with = |mutate: fn(&mut TrainingSettings)| {
        let mut settings = valid_training_settings();
        mutate(&mut settings);
        settings
    };

    // cv folds only carry an upper sanity bound; 0/1 mean "no CV" (worker-supported).
    assert!(validate_training_settings(&with(|s| s.cv_folds = 1)).is_ok());
    assert!(validate_training_settings(&with(|s| s.cv_folds = 101)).is_err());
    assert!(validate_training_settings(&with(|s| s.cv_folds = 2)).is_ok());
    assert!(validate_training_settings(&with(|s| s.cv_folds = 100)).is_ok());

    // lastUpdated must be a positive finite number
    assert!(validate_training_settings(&with(|s| s.last_updated = 0.0)).is_err());
    assert!(validate_training_settings(&with(|s| s.last_updated = -5.0)).is_err());
    assert!(validate_training_settings(&with(|s| s.last_updated = f64::NAN)).is_err());
    assert!(validate_training_settings(&with(|s| s.last_updated = f64::INFINITY)).is_err());

    // feature selections must be non-empty, unique, and in registry order
    assert!(validate_training_settings(&with(|s| s.posture_feature_types = Vec::new())).is_err());
    assert!(validate_training_settings(&with(|s| s.presence_feature_types = Vec::new())).is_err());
    assert!(validate_training_settings(&with(|s| {
        s.posture_feature_types = vec![FeatureId::GauFeatures, FeatureId::GauFeatures];
    }))
    .is_err());
    // descending order violates the registry-order requirement
    assert!(validate_training_settings(&with(|s| {
        s.posture_feature_types = vec![FeatureId::GauFeatures, FeatureId::BackboneFeatures];
        s.posture_feature_types.sort();
        s.posture_feature_types.reverse();
    }))
    .is_err());

    // dimensionality-reduction components are bounded to 1..=1_048_576
    assert!(validate_training_settings(&with(|s| s.dim_reduction_config.components = 0)).is_err());
    assert!(validate_training_settings(&with(|s| {
        s.dim_reduction_config.components = 1_048_577;
    }))
    .is_err());
    assert!(validate_training_settings(&with(|s| {
        s.dim_reduction_config.components = 1_048_576;
    }))
    .is_ok());
}

#[test]
fn inference_tokens_are_js_safe_nonzero_and_distinct() {
    let mut cache = InferenceCache::with_seed(7);
    let mut seen = HashSet::new();
    for request_id in 0..INFERENCE_CACHE_CAPACITY as u64 {
        let token = cache.insert(request_id, empty_result()).expect("insert");
        assert_ne!(token, 0, "tokens must be nonzero");
        assert!(
            token <= inference_cache::MAX_SAFE_JS_INTEGER,
            "token {token} exceeds the JS-safe range"
        );
        assert_ne!(token, request_id, "tokens must differ from request ids");
        assert!(seen.insert(token), "token {token} was issued twice");
    }
}

#[test]
fn a_token_is_consumed_exactly_once_and_reports_consumption_afterwards() {
    let mut cache = InferenceCache::with_seed(8);
    let token = cache.insert(11, empty_result()).expect("insert");
    // a mismatched request id neither consumes nor leaks the entry
    let mismatch = cache.checkout(token, 12).expect_err("wrong request id");
    assert!(
        matches!(&mismatch, ApiError::InvalidRequest(m) if m.contains("does not match")),
        "{mismatch:?}"
    );
    let result = cache.checkout(token, 11).expect("first checkout");
    drop(result);
    // while reserved, a concurrent checkout is refused as busy, not unknown
    assert!(matches!(cache.checkout(token, 11), Err(ApiError::Busy(_))));
    cache.commit(token, 11).expect("commit consumes the token");
    let consumed = cache.checkout(token, 11).expect_err("second consume");
    assert!(
        matches!(&consumed, ApiError::InvalidRequest(m) if m.contains("already consumed")),
        "{consumed:?}"
    );
}

#[test]
fn failed_saves_restore_the_token_and_mismatched_restores_keep_the_reservation() {
    let mut cache = InferenceCache::with_seed(9);
    let token = cache.insert(21, empty_result()).expect("insert");
    let result = cache.checkout(token, 21).expect("checkout");
    // restoring under a different request id must not hijack the reservation
    assert!(matches!(
        cache.restore(token, 22, empty_result()),
        Err(ApiError::InvalidRequest(_))
    ));
    cache.restore(token, 21, result).expect("restore");
    let result = cache.checkout(token, 21).expect("checkout after restore");
    drop(result);
    cache.commit(token, 21).expect("commit");
    // finalizing twice is an internal-contract violation, never silent success
    assert!(matches!(
        cache.commit(token, 21),
        Err(ApiError::Internal(_))
    ));
    assert!(matches!(
        cache.restore(token, 21, empty_result()),
        Err(ApiError::Internal(_))
    ));
}

#[test]
fn lru_eviction_at_entry_capacity_reports_the_evicted_reason() {
    let mut cache = InferenceCache::with_seed(10);
    let mut tokens = Vec::new();
    for request_id in 0..=INFERENCE_CACHE_CAPACITY as u64 {
        tokens.push(cache.insert(request_id, empty_result()).expect("insert"));
    }
    // the 33rd insert evicts exactly the oldest token
    let evicted = cache.checkout(tokens[0], 0).expect_err("evicted token");
    assert!(
        matches!(&evicted, ApiError::InvalidRequest(m) if m.contains("was evicted")),
        "{evicted:?}"
    );
    assert!(
        cache.checkout(tokens[1], 1).is_ok(),
        "second-oldest survives"
    );
    let newest = *tokens.last().expect("newest token");
    assert!(cache
        .checkout(newest, INFERENCE_CACHE_CAPACITY as u64)
        .is_ok());
}

#[test]
fn byte_cap_evicts_older_bundles_and_rejects_oversized_ones_outright() {
    let mut cache = InferenceCache::with_seed(12);
    // a single bundle above 64 MiB never enters the cache
    let error = cache
        .insert(1, sized_result(17_000_000))
        .expect_err("oversized bundle");
    assert!(
        matches!(&error, ApiError::InvalidRequest(m) if m.contains("64 MiB")),
        "{error:?}"
    );
    // two ~34 MiB bundles cannot coexist under the byte cap: LRU eviction
    let first = cache.insert(2, sized_result(9_000_000)).expect("first");
    let second = cache.insert(3, sized_result(9_000_000)).expect("second");
    assert!(cache.retained_bytes <= INFERENCE_CACHE_MAX_BYTES);
    let evicted = cache.checkout(first, 2).expect_err("evicted bundle");
    assert!(
        matches!(&evicted, ApiError::InvalidRequest(m) if m.contains("was evicted")),
        "{evicted:?}"
    );
    assert!(cache.checkout(second, 3).is_ok());
}

#[test]
fn expired_tokens_report_expiry_and_expired_tombstones_are_forgotten() {
    let mut cache = InferenceCache::with_seed(13);
    let token = cache.insert(5, empty_result()).expect("insert");
    cache.backdate_entries(INFERENCE_CACHE_TTL + Duration::from_secs(1));
    let expired = cache.checkout(token, 5).expect_err("expired token");
    assert!(
        matches!(&expired, ApiError::InvalidRequest(m) if m.contains("expired")),
        "{expired:?}"
    );
    assert_eq!(
        cache.retained_bytes, 0,
        "expiry must release retained bytes"
    );
    cache.backdate_tombstones(TOMBSTONE_TTL + Duration::from_secs(1));
    let unknown = cache.checkout(token, 5).expect_err("forgotten token");
    assert!(
        matches!(&unknown, ApiError::InvalidRequest(m) if m.contains("unknown")),
        "{unknown:?}"
    );
}

#[test]
fn tombstone_ring_capacity_forgets_the_oldest_consumed_token() {
    let mut cache = InferenceCache::with_seed(14);
    let first = cache.insert(0, empty_result()).expect("insert");
    let checked_out = cache.checkout(first, 0).expect("checkout");
    drop(checked_out);
    cache.commit(first, 0).expect("commit");
    for request_id in 1..=TOMBSTONE_CAPACITY as u64 {
        let token = cache.insert(request_id, empty_result()).expect("insert");
        let checked_out = cache.checkout(token, request_id).expect("checkout");
        drop(checked_out);
        cache.commit(token, request_id).expect("commit");
    }
    // 65 consumed tombstones overflow the 64-slot ring: the oldest token now
    // reads as unknown instead of consumed
    let unknown = cache.checkout(first, 0).expect_err("forgotten tombstone");
    assert!(
        matches!(&unknown, ApiError::InvalidRequest(m) if m.contains("unknown")),
        "{unknown:?}"
    );
}

#[test]
fn clearing_the_cache_forgets_entries_reservations_and_tombstones() {
    let mut cache = InferenceCache::with_seed(15);
    let live = cache.insert(1, empty_result()).expect("insert live");
    let consumed = cache.insert(2, empty_result()).expect("insert consumed");
    let checked_out = cache.checkout(consumed, 2).expect("checkout");
    drop(checked_out);
    cache.commit(consumed, 2).expect("commit");
    cache.clear();
    assert_eq!(cache.retained_bytes, 0);
    for (token, request_id) in [(live, 1), (consumed, 2)] {
        let error = cache
            .checkout(token, request_id)
            .expect_err("cleared token");
        assert!(
            matches!(&error, ApiError::InvalidRequest(m) if m.contains("unknown")),
            "{error:?}"
        );
    }
}

#[test]
fn time_seeded_cache_construction_issues_working_js_safe_tokens() {
    let mut cache = InferenceCache::new();
    let token = cache.insert(42, empty_result()).expect("insert");
    assert_ne!(token, 0);
    assert!(token <= inference_cache::MAX_SAFE_JS_INTEGER);
    assert!(cache.checkout(token, 42).is_ok());
}
